import json
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app import config, voice
from app.agent.graph import build_graph
from app.agent.llm import DEFAULT_MODEL, list_models
from app.rag import ingest
from app.schemas import ChatRequest, ChatResponse, ModelsResponse, SpeakRequest, ToolCallInfo, TranscriptResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = await run_in_threadpool(ingest.build_index)
    log.info("[startup] indexed %d policy chunks", count)
    yield


app = FastAPI(title="OmniCare Assistant", lifespan=lifespan)


def get_graph_builder():
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "GEMINI_API_KEY is not set on the backend. Add it to .env and restart.")
    return build_graph


def tool_result(message: ToolMessage) -> dict:
    if message.name == "search_policy":
        return {"sections": [source["section"] for source in message.artifact or []]}
    try:
        result = json.loads(message.text)
    except ValueError:
        result = None
    return result if isinstance(result, dict) else {"message": message.text}


def collect_tool_calls(messages: list[AnyMessage], new_messages: list[AnyMessage]) -> list[ToolCallInfo]:
    # a confirmed or cancelled claim was requested in the previous turn, so look calls up in the whole history
    calls = {call["id"]: call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls}

    tool_calls = []
    for message in new_messages:
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(message.tool_call_id, {})
        default_status = "rejected" if message.status == "error" else "executed"
        tool_calls.append(
            ToolCallInfo(
                name=message.name or call.get("name", ""),
                args=call.get("args", {}),
                status=message.additional_kwargs.get("call_status", default_status),
                result=tool_result(message),
            )
        )
    return tool_calls


def is_cited(section: str, response: str) -> bool:
    # a heading written as "Label: Title" also counts as cited when only its label is used
    label = section.split(":")[0].strip()
    return re.search(rf"{re.escape(label)}(?!\w)", response, re.IGNORECASE) is not None


def collect_sources(new_messages: list[AnyMessage], response: str) -> list[str]:
    retrieved = []
    for message in new_messages:
        if isinstance(message, ToolMessage) and message.name == "search_policy":
            for source in message.artifact or []:
                if source not in retrieved:
                    retrieved.append(source)

    cited = [source for source in retrieved if is_cited(source["section"], response)]
    return [f"{s['source']} — {s['section']}" for s in cited or retrieved]


def run_turn(graph, user_id: str, message: str) -> ChatResponse:
    config = {"configurable": {"thread_id": user_id}}
    snapshot = graph.get_state(config)
    seen = len(snapshot.values.get("messages", []))

    if snapshot.interrupts:
        log.info("[chat] resuming paused thread %s", user_id)
        result = graph.invoke(Command(resume=message), config)
    else:
        result = graph.invoke({"messages": [HumanMessage(message)]}, config)

    messages = result["messages"]
    new_messages = messages[seen:]
    tool_calls = collect_tool_calls(messages, new_messages)

    if "__interrupt__" in result:
        pause = result["__interrupt__"][0].value
        response = pause["message"]
        if pause["type"] == "confirm_claim":
            tool_calls.append(
                ToolCallInfo(name="submit_claim", args=pause["claim"], status="awaiting_confirmation")
            )
    else:
        response = messages[-1].text or "Sorry, I couldn't come up with an answer. Please try again."

    return ChatResponse(
        response=response, sources=collect_sources(new_messages, response), tool_calls=tool_calls
    )


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/api/v1/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(models=list_models(), default=DEFAULT_MODEL)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, builder=Depends(get_graph_builder)) -> ChatResponse:
    model = request.model or DEFAULT_MODEL
    log.info("[chat] user=%s model=%s", request.user_id, model)
    try:
        graph = builder(model)
        return await run_in_threadpool(run_turn, graph, request.user_id, request.message)
    except Exception as error:
        log.exception("[chat] request failed")
        if "RESOURCE_EXHAUSTED" in str(error):
            raise HTTPException(429, "The Gemini rate limit was reached. Wait a minute or choose another model.")
        raise HTTPException(502, "The language model request failed. Please try again in a moment.")


@app.post("/api/v1/voice/transcribe", response_model=TranscriptResponse)
async def transcribe(file: UploadFile, model: str | None = Form(None)) -> TranscriptResponse:
    # browsers send types like "audio/webm;codecs=opus"
    mime_type = (file.content_type or "").split(";")[0].strip().lower()
    if mime_type not in config.AUDIO_TYPES:
        raise HTTPException(415, "Unsupported audio type. Use audio/wav, audio/webm or audio/mpeg.")

    audio = await file.read(config.MAX_AUDIO_BYTES + 1)
    if not audio:
        raise HTTPException(400, "The audio file is empty.")
    if len(audio) > config.MAX_AUDIO_BYTES:
        raise HTTPException(413, "The audio file is too large. The limit is 10 MB.")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "GEMINI_API_KEY is not set on the backend. Add it to .env and restart.")
    if model and model not in list_models():
        raise HTTPException(422, "Unknown model.")

    try:
        text = await run_in_threadpool(voice.transcribe, audio, mime_type, model or DEFAULT_MODEL)
    except Exception:
        log.exception("[voice] transcription failed")
        raise HTTPException(502, "The audio could not be transcribed. Please try again.")

    log.info("[voice] transcribed %d bytes", len(audio))
    return TranscriptResponse(text=text)


@app.post("/api/v1/voice/speak")
async def speak(request: SpeakRequest) -> Response:
    try:
        audio = await run_in_threadpool(voice.speak, request.text)
    except Exception:
        log.exception("[voice] speech synthesis failed")
        raise HTTPException(502, "The reply could not be turned into speech.")
    return Response(content=audio, media_type="audio/mpeg")
