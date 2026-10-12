import shutil

import pytest

from app import config
from app.rag import ingest
from app.rag.textReader import MarkdownReader, PDFReader
from app.tools.policy import search_policy

SECTION_1 = "Section 1: Home Water Damage Coverage"
SECTION_2 = "Section 2: Personal Property Protection"


def test_markdown_chunker_makes_one_chunk_per_section():
    text = MarkdownReader(config.POLICY_DIR / "sample_policy.md").extract_text()
    chunks = ingest.get_embedder().markdownChunker(text, source="sample_policy.md")

    assert [chunk["section"] for chunk in chunks] == [SECTION_1, SECTION_2]
    assert all(chunk["source"] == "sample_policy.md" for chunk in chunks)
    assert chunks[0]["text"].startswith(f"OmniCare General Insurance Policy 2026 — {SECTION_1}\n")
    assert "pipe bursts" in chunks[0]["text"]
    assert "appraisal receipts" in chunks[1]["text"]


def test_markdown_chunker_splits_long_sections_and_keeps_the_heading():
    words = [f"w{i}" for i in range(250)]
    text = "# Title\n\n## Long Section\n\n" + " ".join(words) + "\n\n## Short Section\n\nA few words."
    chunks = ingest.get_embedder().markdownChunker(text, source="long.md")

    assert [chunk["section"] for chunk in chunks] == ["Long Section"] * 4 + ["Short Section"]
    bodies = [chunk["text"].split("\n", 1)[1].split() for chunk in chunks[:4]]
    assert bodies[0] == words[:100]
    assert bodies[1] == words[75:175]
    assert all(len(body) <= 100 for body in bodies)
    assert all(chunk["text"].startswith("Title — Long Section\n") for chunk in chunks[:4])


def test_long_sections_and_pdfs_use_the_configured_chunk_size(tmp_path, monkeypatch):
    (tmp_path / "long.md").write_text("# Title\n\n## Long Section\n\n" + " ".join(["word"] * 250))
    (tmp_path / "extra.pdf").write_bytes(b"%PDF-1.4")

    class FakePDFReader:
        def __init__(self, file_path):
            pass

        def extract_text(self):
            return " ".join(["word"] * 250)

    monkeypatch.setattr(ingest, "PDFReader", FakePDFReader)

    chunks = ingest.load_chunks(tmp_path)

    assert [chunk["id"] for chunk in chunks] == [
        "extra.pdf::chunk-1",
        "extra.pdf::chunk-2",
        "extra.pdf::chunk-3",
        "extra.pdf::chunk-4",
        "long.md::section-1",
        "long.md::section-1-part-2",
        "long.md::section-1-part-3",
        "long.md::section-1-part-4",
    ]
    assert all(len(chunk["text"].split()) <= config.CHUNK_SIZE + 10 for chunk in chunks)


def test_readers_check_the_file_extension():
    with pytest.raises(ValueError):
        MarkdownReader(config.POLICY_DIR / "mock_claims.json")
    with pytest.raises(ValueError):
        PDFReader(config.POLICY_DIR / "sample_policy.md")


def test_ingestion_picks_reader_by_extension_and_skips_other_files(tmp_path, monkeypatch):
    shutil.copy(config.POLICY_DIR / "sample_policy.md", tmp_path / "sample_policy.md")
    shutil.copy(config.POLICY_DIR / "mock_claims.json", tmp_path / "mock_claims.json")
    used = []

    class FakePDFReader:
        def __init__(self, file_path):
            used.append(file_path.name)

        def extract_text(self):
            return "Mobile phones are covered against theft when a police report is provided."

    (tmp_path / "extra_policy.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(ingest, "PDFReader", FakePDFReader)

    chunks = ingest.load_chunks(tmp_path)

    assert used == ["extra_policy.pdf"]
    assert [chunk["id"] for chunk in chunks] == [
        "extra_policy.pdf::chunk-1",
        "sample_policy.md::section-1",
        "sample_policy.md::section-2",
    ]
    assert chunks[0]["section"] == "chunk 1"
    assert {chunk["source"] for chunk in chunks} == {"extra_policy.pdf", "sample_policy.md"}


def test_build_index_stores_both_sections(policy_index):
    assert policy_index == 2
    assert ingest.get_store().collection.count() == 2


@pytest.mark.parametrize(
    "query, expected",
    [
        ("burst pipe", SECTION_1),
        ("jewelry appraisal", SECTION_2),
        ("flood", SECTION_1),
    ],
)
def test_retrieval_ranks_the_right_section_first(policy_index, query, expected):
    results = search_policy(query)

    assert results[0]["section"] == expected
    assert results[0]["source"] == "sample_policy.md"
    assert results[0]["distance"] <= results[1]["distance"]
