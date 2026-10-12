import json
import os
from pathlib import Path

import pytest

from app import config
from app.rag import ingest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def pytest_collection_modifyitems(items):
    if os.getenv("GEMINI_API_KEY"):
        return
    skip = pytest.mark.skip(reason="GEMINI_API_KEY is not set")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def temp_data(tmp_path, monkeypatch):
    # only the two records that ship with the repo: inside Docker, data/ is a volume
    # that may also hold claims submitted while trying the app
    claims = json.loads((DATA_DIR / "mock_claims.json").read_text())[:2]
    claims_file = tmp_path / "mock_claims.json"
    claims_file.write_text(json.dumps(claims, indent=2))

    monkeypatch.setattr(config, "CLAIMS_FILE", claims_file)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "POLICY_DIR", DATA_DIR)
    ingest.get_store.cache_clear()
    yield
    ingest.get_store.cache_clear()


@pytest.fixture
def policy_index():
    return ingest.build_index()
