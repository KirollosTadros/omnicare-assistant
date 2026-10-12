import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

log = logging.getLogger(__name__)

# the only model name in the code: used when DEFAULT_MODEL isn't set in .env,
# and offered on its own when the model list can't be fetched (no API key, no network)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3.5-flash-lite")

# models that support generateContent but are not text chat models
NON_CHAT_MODELS = ["tts", "image", "audio", "live", "embedding", "robotics", "computer-use"]


def get_llm(model: str = DEFAULT_MODEL) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model=model,
        temperature=0,
    )


@lru_cache(maxsize=1)
def fetch_models() -> list[str]:
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"), http_options=types.HttpOptions(timeout=10_000)
    )
    models = []
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        if (
            name.startswith("gemini-")
            and "generateContent" in (model.supported_actions or [])
            and not any(word in name for word in NON_CHAT_MODELS)
        ):
            models.append(name)
    return sorted(models)


def list_models() -> list[str]:
    """The Gemini chat models this API key can use, or just the default one if they can't be fetched."""
    if not os.getenv("GEMINI_API_KEY"):
        return [DEFAULT_MODEL]
    try:
        # a failed fetch raises, so it is not cached and the next call tries again
        return fetch_models() or [DEFAULT_MODEL]
    except Exception:
        log.warning("[llm] could not fetch the model list, offering only the default model", exc_info=True)
        return [DEFAULT_MODEL]
