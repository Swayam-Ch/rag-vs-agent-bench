from pathlib import Path

import pytest

from ragbench.pipeline.fetch import parse_sources
from ragbench.pipeline.ingest import guess_title, ingest_dir
from ragbench.pipeline.metadata import (
    API_URL,
    PaperMeta,
    base_id,
    fetch_metadata,
    parse_feed,
    read_metadata,
    sync_metadata,
    write_metadata,
)

FEED = (Path(__file__).parent / "fixtures" / "arxiv_feed.xml").read_text()
DATA = Path(__file__).parent.parent / "data"

ERROR_FEED = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id>
<title>Error</title><summary>incorrect id format for 1234</summary></entry></feed>"""


# --------------------------------------------------------------------------- #
# Parsing the Atom feed
# --------------------------------------------------------------------------- #


def test_parse_feed_extracts_all_fields():
    rag, attention = parse_feed(FEED)
    assert rag == PaperMeta(
        arxiv_id="2005.11401",
        version="v4",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
        published="2020-05-22",
        category="cs.CL",
    )
    assert attention.title == "Attention Is All You Need"


def test_title_line_breaks_are_collapsed():
    assert "\n" not in parse_feed(FEED)[0].title
    assert "  " not in parse_feed(FEED)[0].title


def test_api_error_entry_raises():
    with pytest.raises(ValueError, match="incorrect id format"):
        parse_feed(ERROR_FEED)


@pytest.mark.parametrize(
    ("raw", "base"), [("2005.11401v4", "2005.11401"), ("1706.03762", "1706.03762")]
)
def test_base_id(raw, base):
    assert base_id(raw) == base


# --------------------------------------------------------------------------- #
# Fetching with a fake session
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, text=FEED):
        self.text = text
        self.calls: list[dict] = []

    def get(self, url, params=None, **kwargs):
        assert url == API_URL
        self.calls.append(params)
        return FakeResponse(self.text)


def test_fetch_sends_one_batched_request():
    session = FakeSession()
    papers = fetch_metadata(["2005.11401", "1706.03762"], session)
    assert len(papers) == 2
    assert session.calls == [{"id_list": "2005.11401,1706.03762", "max_results": 2}]


def test_fetch_raises_when_an_id_is_missing():
    with pytest.raises(ValueError, match="2104.08663"):
        fetch_metadata(["2005.11401", "1706.03762", "2104.08663"], FakeSession())


def test_sync_writes_once_then_makes_no_requests(tmp_path):
    path = tmp_path / "metadata.jsonl"
    session = FakeSession()
    assert sync_metadata(["2005.11401v4", "1706.03762"], path, session) == [
        "2005.11401",
        "1706.03762",
    ]
    assert set(read_metadata(path)) == {"2005.11401", "1706.03762"}

    assert sync_metadata(["2005.11401", "1706.03762"], path, session) == []
    assert len(session.calls) == 1  # second run: everything known, no network


def test_sync_drops_papers_removed_from_sources(tmp_path):
    path = tmp_path / "metadata.jsonl"
    sync_metadata(["2005.11401", "1706.03762"], path, FakeSession())
    sync_metadata(["1706.03762"], path, FakeSession())
    assert set(read_metadata(path)) == {"1706.03762"}


def test_metadata_file_round_trip_is_sorted(tmp_path):
    path = tmp_path / "metadata.jsonl"
    papers = {p.arxiv_id: p for p in parse_feed(FEED)}
    write_metadata(path, papers)
    lines = path.read_text().splitlines()
    assert lines[0].startswith('{"arxiv_id": "1706.03762"')
    assert read_metadata(path) == papers


# --------------------------------------------------------------------------- #
# Ingestion uses metadata; fallback title guess
# --------------------------------------------------------------------------- #


def test_ingest_uses_metadata_for_arxiv_files(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    (raw / "2005.11401.txt").write_text("arXiv:2005.11401v4 [cs.CL]\nsome extracted text")
    (raw / "notes.txt").write_text("Lab Notes On Retrieval\nbody")
    metadata = {p.arxiv_id: p for p in parse_feed(FEED)}

    rag, notes = ingest_dir(raw, out, metadata=metadata)

    assert rag.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert rag.manifest_entry()["authors"][0] == "Patrick Lewis"
    assert (rag.published, rag.category) == ("2020-05-22", "cs.CL")
    assert notes.title == "Lab Notes On Retrieval"
    assert notes.arxiv_id == ""


@pytest.mark.parametrize(
    "first_page",
    [
        "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023\nAttention Is All You Need\nAshish Vaswani",
        "Published as a conference paper at ICLR 2023\nAttention Is All You Need\n",
        "Provided proper attribution is provided, Google hereby grants\nAttention Is All You Need",
        "\n\n# Attention Is All You Need\n\nAbstract",
    ],
)
def test_guess_title_skips_stamps_and_venue_lines(first_page):
    assert guess_title(first_page, "fallback") == "Attention Is All You Need"


def test_guess_title_falls_back_when_nothing_fits():
    assert guess_title("12345\nhello\n", "paper_01") == "paper_01"


# --------------------------------------------------------------------------- #
# The committed corpus metadata (once data/metadata.jsonl exists)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not (DATA / "metadata.jsonl").exists(), reason="metadata not fetched yet")
def test_committed_metadata_covers_corpus_with_real_titles():
    ids = [base_id(i) for i in parse_sources((DATA / "sources.txt").read_text())]
    meta = read_metadata(DATA / "metadata.jsonl")
    assert sorted(meta) == sorted(ids)
    for arxiv_id, paper in meta.items():
        assert paper.title and paper.title != arxiv_id
        assert paper.authors and paper.published and paper.category
