import random
import re
import string

import pytest

from ragbench.pipeline.ingest import ingest_dir
from ragbench.pipeline.metadata import PaperMeta
from ragbench.search.agent_tools import (
    MAX_HITS,
    MAX_OUTPUT,
    MAX_READ,
    Corpus,
    Paper,
    Read,
)

ATTENTION = (
    "Attention Is All You Need\n\n"
    "The Transformer relies entirely on attention.\n\n"
    "Multi-head attention runs h = 8 attention heads in parallel.\n\n"
    "In Table 3 rows (A), we vary the number of attention heads."
)
RAG = (
    "Retrieval-Augmented Generation\n\n"
    "RAG combines a dense retriever with a generator.\n\n"
    "We evaluate on open-domain question answering."
)


@pytest.fixture
def corpus(tmp_path) -> Corpus:
    """A two-paper corpus built by the real ingestion code, with arXiv metadata."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1706.03762.txt").write_text(ATTENTION)
    (raw / "2005.11401.txt").write_text(RAG)
    meta = {
        "1706.03762": PaperMeta(
            "1706.03762",
            "v7",
            "Attention Is All You Need",
            ["Ashish Vaswani"],
            "2017-06-12",
            "cs.CL",
        ),
        "2005.11401": PaperMeta(
            "2005.11401",
            "v4",
            "Retrieval-Augmented Generation",
            ["Patrick Lewis"],
            "2020-05-22",
            "cs.CL",
        ),
    }
    ingest_dir(raw, tmp_path / "processed", metadata=meta)
    return Corpus.load(tmp_path / "processed")


def _id(corpus: Corpus, arxiv_id: str) -> str:
    return corpus.resolve(arxiv_id).doc_id


# --------------------------------------------------------------------------- #
# list_files
# --------------------------------------------------------------------------- #


def test_list_files_shows_ids_years_and_titles(corpus):
    out = corpus.list_files()
    assert out.startswith("2 papers")
    assert "1706.03762 | 2017" in out and "Attention Is All You Need" in out
    assert out.index("2017") < out.index("2020")  # oldest first


# --------------------------------------------------------------------------- #
# grep
# --------------------------------------------------------------------------- #


def test_grep_is_case_insensitive_and_reports_page_and_offset(corpus):
    out = corpus.grep("MULTI-HEAD")
    assert out.startswith("1 matches in 1 papers")
    doc = _id(corpus, "1706.03762")
    paper = corpus.papers[doc]
    offset = paper.text.lower().index("multi-head")
    assert f"{doc} p.1 @{offset}:" in out


def test_grep_counts_matches_per_paper(corpus):
    out = corpus.grep("attention")
    attention_count = corpus.papers[_id(corpus, "1706.03762")].text.lower().count("attention")
    assert f"Attention Is All You Need): {attention_count}" in out


def test_grep_supports_regex(corpus):
    assert "2 papers" in corpus.grep(r"retriev\w+|transformer")


def test_grep_can_be_limited_to_one_paper(corpus):
    out = corpus.grep("a", doc_id="2005.11401")
    assert "in 1 papers" in out and _id(corpus, "1706.03762") not in out


SNIPPET_LINE = re.compile(r"^\w+ p\.\d+ @\d+: ", re.MULTILINE)


def test_grep_shows_at_most_max_hits_snippets(corpus):
    assert len(SNIPPET_LINE.findall(corpus.grep("e", max_hits=3))) == 3
    # an absurd max_hits is silently clamped
    assert len(SNIPPET_LINE.findall(corpus.grep("e", max_hits=10_000))) <= MAX_HITS


def test_invalid_regex_falls_back_to_literal_search(corpus):
    out = corpus.grep("rows (A")  # unbalanced parenthesis
    assert "invalid regex" in out and "1 matches" in out


def test_grep_no_match_and_bad_input(corpus):
    assert corpus.grep("zebra").startswith("no matches")
    assert corpus.grep("  ").startswith("error:")
    assert corpus.grep("x" * 500).startswith("error:")
    assert corpus.grep("a", doc_id="nope").startswith("error:")


def test_zero_width_matches_are_ignored(corpus):
    assert corpus.grep("z*").startswith("no matches")


def test_huge_match_is_shortened(corpus):
    out = corpus.grep(".*", max_hits=1)
    assert len(out) < 1_000


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


def test_read_returns_the_exact_text_with_a_header(corpus):
    doc = _id(corpus, "2005.11401")
    text = corpus.papers[doc].text
    out = corpus.read_file(doc, start=5, length=20)
    assert out.splitlines()[0].startswith(f"{doc} (Retrieval-Augmented Generation) characters 5-25")
    assert text[5:25] in out
    assert "continue with start=25" in out


def test_read_accepts_the_arxiv_id(corpus):
    assert corpus.read_file("1706.03762", length=9).split("\n\n")[1].startswith("Attention")


def test_read_to_the_end_has_no_continue_hint(corpus):
    doc = _id(corpus, "2005.11401")
    out = corpus.read_file(doc, start=0, length=MAX_READ)
    assert "continue with" not in out


def test_read_length_is_clamped(corpus):
    doc = _id(corpus, "1706.03762")
    corpus.read_file(doc, start=0, length=10**9)
    assert corpus.reads[-1].end - corpus.reads[-1].start <= MAX_READ


@pytest.mark.parametrize(("start", "message"), [(-1, "0 or more"), (10**9, "past the end")])
def test_read_bad_offsets(corpus, start, message):
    out = corpus.read_file("1706.03762", start=start)
    assert out.startswith("error:") and message in out


@pytest.mark.parametrize(
    "name",
    ["../manifest.jsonl", "../../etc/passwd", "/etc/passwd", "manifest", "", "  ", "*"],
)
def test_names_never_reach_the_file_system(corpus, name):
    assert corpus.read_file(name).startswith("error: no paper")
    assert corpus.reads == []


# --------------------------------------------------------------------------- #
# Read log: the agent's evidence
# --------------------------------------------------------------------------- #


def test_successful_reads_are_logged_failures_are_not(corpus):
    doc = _id(corpus, "1706.03762")
    corpus.read_file(doc, start=0, length=10)
    corpus.read_file(doc, start=-5)
    corpus.read_file("unknown")
    corpus.grep("attention")  # searching is not reading
    assert corpus.reads == [Read(doc, 0, 10)]


# --------------------------------------------------------------------------- #
# Robustness: random input never raises and never floods the context
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(30))
def test_random_calls_never_raise_and_stay_bounded(seed):
    rng = random.Random(seed)
    long_text = "\n\n".join(
        " ".join(rng.choices(["attention", "head", "BLEU", "x" * 300], k=400)) for _ in range(30)
    )
    corpus = Corpus([Paper("doc", long_text, "A long paper", [0, len(long_text) // 2])])
    alphabet = string.ascii_letters + string.punctuation + " "
    for _ in range(20):
        pattern = "".join(rng.choices(alphabet, k=rng.randint(0, 12)))
        out = corpus.grep(pattern, max_hits=rng.randint(-5, 100))
        assert len(out) <= MAX_OUTPUT
        out = corpus.read_file(
            rng.choice(["doc", "nope"]), rng.randint(-10, 200_000), rng.randint(-10, 100_000)
        )
        assert len(out) <= MAX_OUTPUT
    assert len(corpus.list_files()) <= MAX_OUTPUT


# --------------------------------------------------------------------------- #
# Evidence log: reads *and* grep snippets the agent was shown
# --------------------------------------------------------------------------- #


def test_grep_snippets_are_logged_as_evidence(corpus):
    corpus.grep("number of attention heads")
    [snippet] = corpus.evidence
    paper = corpus.papers[snippet.doc_id]
    assert snippet.kind == "snippet"
    assert "number of attention heads" in paper.text[snippet.start : snippet.end]
    assert corpus.reads == []  # still not a read


def test_reads_and_snippets_share_one_ordered_log(corpus):
    doc = _id(corpus, "1706.03762")
    corpus.read_file(doc, length=10)
    corpus.grep("question answering")
    assert [e.kind for e in corpus.evidence] == ["read", "snippet"]


def test_failed_or_empty_searches_log_nothing(corpus):
    corpus.grep("zebra")
    corpus.grep("  ")
    corpus.grep("a", doc_id="nope")
    assert corpus.evidence == []


def test_snippets_cut_off_by_the_output_cap_are_not_evidence():
    long_text = " ".join(["needle " + "x" * 280] * 200)
    corpus = Corpus([Paper("doc", long_text, "Long", [0])])
    out = corpus.grep("needle", max_hits=MAX_HITS)
    assert "[output truncated" in out
    shown = len(SNIPPET_LINE.findall(out))
    assert len(corpus.evidence) <= shown < MAX_HITS
