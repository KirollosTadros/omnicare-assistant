import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.llm import DEFAULT_MODEL
from app.rag import ingest
from app.schemas import ChatRequest, ChatResponse, ToolCallInfo

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


def collect_sources(new_messages: list[AnyMessage], response: str) -> list[str]:
    retrieved = []
    for message in new_messages:
        if isinstance(message, ToolMessage) and message.name == "search_policy":
            for source in message.artifact or []:
                if source not in retrieved:
                    retrieved.append(source)

    # "Section 1" on its own also counts as citing "Section 1: Home Water Damage Coverage"
    cited = [s for s in retrieved if s["section"] in response or s["section"].split(":")[0] in response]
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


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, builder=Depends(get_graph_builder)) -> ChatResponse:
    model = request.model or DEFAULT_MODEL
    log.info("[chat] user=%s model=%s", request.user_id, model)
    try:
        graph = builder(model)
        return await run_in_threadpool(run_turn, graph, request.user_id, request.message)
    except Exception:
        log.exception("[chat] request failed")
        raise HTTPException(502, "The language model request failed. Please try again in a moment.")
