import random
import re
import string

import pytest

from ragbench.pipeline.chunk import (
    Chunk,
    _split_long,
    chunk_corpus,
    chunk_fixed,
    chunk_paragraphs,
    main,
    output_name,
    page_at,
    read_chunks,
    stats,
)

# --------------------------------------------------------------------------- #
# page_at
# --------------------------------------------------------------------------- #

OFFSETS = [0, 100, 250]  # page 1 = [0,100), page 2 = [100,250), page 3 = [250, ...)


@pytest.mark.parametrize(
    ("offset", "page"), [(0, 1), (99, 1), (100, 2), (249, 2), (250, 3), (10_000, 3)]
)
def test_page_at(offset, page):
    assert page_at(offset, OFFSETS) == page


def test_page_at_without_offsets_is_page_one():
    assert page_at(42, []) == 1


# --------------------------------------------------------------------------- #
# chunk_fixed: exact positions on a small example
# --------------------------------------------------------------------------- #


def test_sizes_and_overlap():
    text = "abcdefghij" * 3  # 30 chars
    chunks = chunk_fixed("d", text, [0], size=10, overlap=4)
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (6, 16), (12, 22), (18, 28), (24, 30)]
    assert chunks[1].text[:4] == chunks[0].text[-4:]  # overlap really is shared


def test_last_chunk_is_never_only_overlap():
    # 0-10 then 6-16 would end exactly at the text end; no extra 12-16 chunk after it
    chunks = chunk_fixed("d", "x" * 16, [0], size=10, overlap=4)
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (6, 16)]


def test_text_shorter_than_size_is_one_chunk():
    [chunk] = chunk_fixed("d", "short", [0], size=100, overlap=10)
    assert chunk == Chunk("d", "d:0000", 0, 5, 1, 1, "short")


def test_empty_text_has_no_chunks():
    assert chunk_fixed("d", "", [0]) == []


def test_chunk_ids_are_ordered_and_unique():
    chunks = chunk_fixed("abc", "y" * 95, [0], size=10, overlap=0)
    assert [c.chunk_id for c in chunks] == [f"abc:{i:04d}" for i in range(10)]


def test_pages_follow_page_offsets():
    text = "a" * 100 + "b" * 150 + "c" * 50  # pages start at 0, 100, 250
    chunks = chunk_fixed("d", text, OFFSETS, size=120, overlap=20)
    assert [(c.page, c.page_end) for c in chunks] == [(1, 2), (2, 2), (2, 3)]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (10, 10), (10, 15), (10, -1)])
def test_invalid_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_fixed("d", "text", [0], size=size, overlap=overlap)


# --------------------------------------------------------------------------- #
# Invariants on random documents: these must hold for ANY input
# --------------------------------------------------------------------------- #


def _random_doc(rng: random.Random) -> tuple[str, list[int]]:
    length = rng.randint(1, 5_000)
    text = "".join(rng.choices(string.ascii_letters + " \n", k=length))
    n_pages = rng.randint(1, 10)
    offsets = sorted({0, *rng.sample(range(length), min(n_pages - 1, length))})
    return text, offsets


@pytest.mark.parametrize("seed", range(50))
def test_invariants_on_random_documents(seed):
    rng = random.Random(seed)
    text, offsets = _random_doc(rng)
    size = rng.randint(1, 1_200)
    overlap = rng.randint(0, size - 1)

    chunks = chunk_fixed("doc", text, offsets, size=size, overlap=overlap)

    # every chunk is a faithful slice of the document
    assert all(text[c.start : c.end] == c.text for c in chunks)
    # nothing is ever longer than the limit
    assert all(0 < len(c.text) <= size for c in chunks)
    # together they cover the whole document, from the first to the last character
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    assert len(covered) == len(text)
    # neighbours overlap by exactly `overlap` characters (except possibly the last pair)
    for a, b in zip(chunks, chunks[1:-1], strict=False):
        assert a.end - b.start == overlap
    # pages never go backwards
    assert [c.page for c in chunks] == sorted(c.page for c in chunks)


def test_page_end_ignores_trailing_blank_pages():
    # three emptied bibliography pages at the end leave only blank separators behind
    text = "a" * 50 + "\n\n" + "\n\n" * 3
    offsets = [0, 52, 54, 56]
    [chunk] = chunk_fixed("d", text, offsets, size=1000, overlap=0)
    assert chunk.page_end == 1


# --------------------------------------------------------------------------- #
# chunk_paragraphs
# --------------------------------------------------------------------------- #


def test_paragraphs_are_packed_until_the_next_one_would_not_fit():
    text = "Intro heading\n\nFirst paragraph here.\n\nSecond paragraph here.\n\nThird one."
    chunks = chunk_paragraphs("d", text, [0], max_size=40)
    assert [c.text for c in chunks] == [
        "Intro heading\n\nFirst paragraph here.",
        "Second paragraph here.\n\nThird one.",
    ]


def test_chunks_never_start_or_end_mid_paragraph():
    paragraphs = [f"Paragraph {i} " + "word " * (i % 7 + 3) for i in range(30)]
    text = "\n\n".join(p.strip() for p in paragraphs)
    for c in chunk_paragraphs("d", text, [0], max_size=120):
        assert c.start == 0 or text[c.start - 2 : c.start] == "\n\n"
        assert c.end == len(text) or text[c.end : c.end + 2] == "\n\n"


def test_long_paragraph_is_split_at_spaces():
    text = " ".join(f"w{i:03d}" for i in range(100))  # 499 chars, one paragraph
    chunks = chunk_paragraphs("d", text, [0], max_size=50)
    assert all(len(c.text) <= 50 for c in chunks)
    assert all(not c.text.startswith(" ") and not c.text.endswith(" ") for c in chunks)
    assert " ".join(c.text for c in chunks) == text  # no word cut in half, none lost


def test_split_long_hard_cuts_a_giant_word():
    text = "x" * 25
    assert list(_split_long(text, 0, 25, 10)) == [(0, 10), (10, 20), (20, 25)]


def test_paragraph_chunks_have_correct_pages():
    text = "Page one text.\n\nPage two text."
    chunks = chunk_paragraphs("d", text, [0, 16], max_size=14)
    assert [(c.text, c.page, c.page_end) for c in chunks] == [
        ("Page one text.", 1, 1),
        ("Page two text.", 2, 2),
    ]


def test_blank_text_has_no_paragraph_chunks():
    assert chunk_paragraphs("d", "\n\n  \n\n", [0]) == []


def test_invalid_max_size_raises():
    with pytest.raises(ValueError):
        chunk_paragraphs("d", "text", [0], max_size=0)


def _random_paragraph_doc(rng: random.Random) -> tuple[str, list[int]]:
    words = ["retrieval", "dense", "a", "the", "BM25", "x" * rng.randint(1, 80), "agent"]
    paragraphs = [
        " ".join(rng.choices(words, k=rng.randint(1, 120))) for _ in range(rng.randint(1, 40))
    ]
    text = "\n\n".join(paragraphs)
    starts = [0] + [m.end() for m in re.finditer("\n\n", text)]
    offsets = sorted(rng.sample(starts, min(len(starts), rng.randint(1, 8))) + [0])
    return text, sorted(set(offsets))


@pytest.mark.parametrize("seed", range(50))
def test_paragraph_invariants_on_random_documents(seed):
    rng = random.Random(seed)
    text, offsets = _random_paragraph_doc(rng)
    max_size = rng.randint(20, 1_500)

    chunks = chunk_paragraphs("doc", text, offsets, max_size=max_size)

    assert all(text[c.start : c.end] == c.text for c in chunks)
    assert all(0 < len(c.text) <= max_size for c in chunks)
    # in order and never overlapping
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:], strict=False))
    # coverage: every visible character of the document is inside some chunk
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    missing = [i for i, ch in enumerate(text) if not ch.isspace() and i not in covered]
    assert missing == []
    # what's left between chunks is only whitespace
    gaps = [text[a.end : b.start] for a, b in zip(chunks, chunks[1:], strict=False)]
    assert all(not g.strip() for g in gaps)


# --------------------------------------------------------------------------- #
# Corpus level + CLI
# --------------------------------------------------------------------------- #


@pytest.fixture
def processed(tmp_path):
    """A tiny processed/ folder, built by the real ingestion code."""
    from ragbench.pipeline.ingest import ingest_dir

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("Heading A\n\n" + "Alpha text. " * 50)
    (raw / "b.txt").write_text("Heading B\n\n" + "Beta text. " * 10)
    ingest_dir(raw, tmp_path / "processed")
    return tmp_path / "processed"


@pytest.mark.parametrize("method", ["fixed", "paragraph"])
def test_chunk_corpus_covers_every_document(processed, method):
    chunks = chunk_corpus(processed, method=method, size=200, overlap=50)
    assert len({c.doc_id for c in chunks}) == 2
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunk_corpus_requires_ingestion_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="ragbench-ingest"):
        chunk_corpus(tmp_path)


def test_unknown_method_raises(processed):
    with pytest.raises(ValueError, match="method"):
        chunk_corpus(processed, method="semantic")


def test_output_name_encodes_all_settings():
    assert output_name("paragraph", 1000, 200) == "chunks.paragraph.1000"
    assert output_name("fixed", 1000, 200) == "chunks.fixed.1000-200"
    assert output_name("fixed", 1000, 100) != output_name("fixed", 1000, 200)


def test_cli_writes_readable_jsonl_and_prints_stats(processed, capsys):
    assert main([str(processed), "--method", "paragraph", "--size", "300"]) == 0
    out_file = processed / "chunks.paragraph.300.jsonl"
    chunks = read_chunks(out_file)
    assert chunks and all(len(c.text) <= 300 for c in chunks)
    printed = capsys.readouterr().out
    assert "chunks" in printed and "max_chars" in printed


def test_cli_reports_missing_manifest(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "ragbench-ingest" in capsys.readouterr().err


def test_stats():
    chunks = chunk_paragraphs("d", "aa\n\nbbbb", [0], max_size=4)
    assert stats(chunks) == {
        "chunks": 2,
        "documents": 1,
        "per_document": 2.0,
        "mean_chars": 3,
        "median_chars": 3,
        "min_chars": 2,
        "max_chars": 4,
    }


# --------------------------------------------------------------------------- #
# Filling and small-leftover merging (#34)
# --------------------------------------------------------------------------- #


def test_long_paragraph_fills_the_current_chunk_first():
    heading = "2 Method"
    long_para = " ".join(f"word{i:02d}" for i in range(30))  # 209 chars
    text = f"{heading}\n\n{long_para}"
    chunks = chunk_paragraphs("d", text, [0], max_size=100, min_size=10)
    assert chunks[0].text.startswith(heading + "\n\nword00")  # heading isn't stranded
    assert all(len(c.text) <= 100 for c in chunks)


def test_small_leftover_is_merged_into_a_neighbour():
    # the tail of a paragraph that continued onto the next page
    text = "A" * 50 + "\n\n" + "tail." + "\n\n" + "B" * 90
    chunks = chunk_paragraphs("d", text, [0], max_size=100, min_size=20)
    assert [c.text for c in chunks] == ["A" * 50 + "\n\ntail.", "B" * 90]


def test_small_chunk_kept_when_no_neighbour_has_room():
    text = "A" * 99 + "\n\n" + "tail." + "\n\n" + "B" * 99
    chunks = chunk_paragraphs("d", text, [0], max_size=100, min_size=20)
    assert [len(c.text) for c in chunks] == [99, 5, 99]


def test_explicit_min_size_above_max_raises():
    with pytest.raises(ValueError, match="min_size"):
        chunk_paragraphs("d", "text", [0], max_size=100, min_size=200)


@pytest.mark.parametrize("seed", range(50))
def test_small_chunks_only_when_unavoidable(seed):
    rng = random.Random(seed)
    text, offsets = _random_paragraph_doc(rng)
    max_size = rng.randint(50, 1_500)
    min_size = max_size // 10
    chunks = chunk_paragraphs("doc", text, offsets, max_size=max_size)
    for i, c in enumerate(chunks):
        if len(c.text) < min_size and len(chunks) > 1:
            fits_prev = i > 0 and c.end - chunks[i - 1].start <= max_size
            fits_next = i + 1 < len(chunks) and chunks[i + 1].end - c.start <= max_size
            assert not fits_prev and not fits_next


def test_output_name_records_a_custom_min_size():
    assert output_name("paragraph", 1000, 200, None) == "chunks.paragraph.1000"
    assert output_name("paragraph", 1000, 200, 50) == "chunks.paragraph.1000-min50"


def test_cli_min_size_option(processed):
    assert main([str(processed), "--size", "300", "--min-size", "10"]) == 0
    assert (processed / "chunks.paragraph.300-min10.jsonl").exists()
