import json
import os
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.runnables import RunnableLambda

from app import config
from app.agent import assistant, guard
from app.agent.graph import build_graph
from app.rag import ingest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class FakeChatModel(FakeMessagesListChatModel):
    """Replays scripted AIMessages for the agent and scripted verdicts for the guard."""

    verdicts: list[str] = []

    def _generate(self, *args, **kwargs):
        # hand out a copy each time: a repeated message would keep its id and add_messages
        # would overwrite the earlier one instead of appending
        result = super()._generate(*args, **kwargs)
        result.generations[0].message = result.generations[0].message.model_copy(update={"id": None})
        return result

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def answer(_):
            verdict = self.verdicts.pop(0) if self.verdicts else "safe"
            return schema(verdict=verdict, reason="scripted")

        return RunnableLambda(answer)


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


@pytest.fixture
def scripted_graph(monkeypatch):
    def build(responses, verdicts=()):
        fake = FakeChatModel(responses=responses, verdicts=list(verdicts))
        monkeypatch.setattr(assistant, "get_llm", lambda model: fake)
        monkeypatch.setattr(guard, "get_llm", lambda model: fake)
        build_graph.cache_clear()
        return build_graph("fake-model")

    yield build
    build_graph.cache_clear()
