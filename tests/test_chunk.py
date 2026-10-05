import random
import string

import pytest

from ragbench.pipeline.chunk import Chunk, chunk_fixed, page_at

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
