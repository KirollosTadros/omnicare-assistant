import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent.parent

# in the container BACKEND_DIR is /app, so these default to /app/data and /app/chroma
POLICY_DIR = Path(os.getenv("POLICY_DIR", BACKEND_DIR / "data"))
CLAIMS_FILE = Path(os.getenv("CLAIMS_FILE", POLICY_DIR / "mock_claims.json"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BACKEND_DIR / "chroma"))

COLLECTION_NAME = "policy_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GUARD_MODEL = os.getenv("GUARD_MODEL", "gemini-3.5-flash-lite")

MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_TYPES = ["audio/wav", "audio/x-wav", "audio/webm", "audio/mpeg"]
