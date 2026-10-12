import base64
import io
import re

from gtts import gTTS
from langchain_core.messages import HumanMessage

from app.agent.llm import DEFAULT_MODEL, get_llm

CITATION_REGEX = re.compile(r"\[[^\]]*\]")


def transcribe(audio: bytes, mime_type: str, model: str = DEFAULT_MODEL) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Transcribe this audio exactly. Return only the transcript."},
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": mime_type,
                "data": base64.b64encode(audio).decode("utf-8"),
            },
        ]
    )
    return get_llm(model).invoke([message]).text.strip()


def clean_for_speech(text: str) -> str:
    text = CITATION_REGEX.sub("", text)
    text = re.sub(r"[*_#`]", "", text)
    return re.sub(r"\s+([.,!?])", r"\1", text).strip()


def speak(text: str) -> bytes:
    buffer = io.BytesIO()
    gTTS(clean_for_speech(text), lang="en").write_to_fp(buffer)
    return buffer.getvalue()
