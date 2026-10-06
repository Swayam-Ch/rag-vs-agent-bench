"""Split ingested documents into chunks for vector and hybrid search.

Two strategies are compared:

* `chunk_fixed`: fixed character windows with overlap, the classic baseline;
* `chunk_paragraphs`: whole paragraphs packed up to a size limit, so a chunk never
  starts or stops mid-sentence unless a single paragraph is longer than the limit.

Every chunk records exactly where it came from: character offsets into the document
text (so `text[start:end] == chunk.text` always holds) and the page(s) it spans. That
lets the benchmark check whether a retrieved chunk is on the right page, and lets
anyone audit what a search method actually saw.

The agent searches the full markdown files instead, so chunking only affects the
vector and hybrid methods. Chunk size is therefore a parameter of the comparison, and
every result records the settings it was run with.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SIZE = 1000  # characters, roughly 200-250 English words
DEFAULT_OVERLAP = 200  # characters shared between neighbouring chunks
MIN_SIZE_SHARE = 0.1  # paragraph chunks below 10% of the limit are merged into a neighbour


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str  # "<doc_id>:<index>", e.g. "bdfaa68d8984:0007"
    start: int  # character offset in the document text, inclusive
    end: int  # character offset, exclusive
    page: int  # 1-based page where the chunk starts
    page_end: int  # 1-based page where the chunk ends
    text: str


def page_at(offset: int, page_offsets: list[int]) -> int:
    """1-based page number containing character `offset`.

    `page_offsets[i]` is where page i+1 starts, so the page is the number of page
    starts at or before the offset.
    """
    return max(1, bisect_right(page_offsets, offset))


def _make_chunk(doc_id: str, index: int, text: str, start: int, end: int, offsets) -> Chunk:
    body = text[start:end]
    # page_end comes from the last visible character: trailing blank lines left by
    # emptied bibliography pages must not make a chunk "end" on an empty page
    last = start + len(body.rstrip()) - 1 if body.strip() else start
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}:{index:04d}",
        start=start,
        end=end,
        page=page_at(start, offsets),
        page_end=page_at(last, offsets),
        text=body,
    )


def chunk_fixed(
    doc_id: str,
    text: str,
    page_offsets: list[int],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Cut `text` into windows of `size` characters, each overlapping the previous one.

    The windows ignore words and sentences on purpose: this is the simplest baseline.
    The final chunk may be shorter than `size`, but it is never a pure repeat of the
    previous chunk's overlap.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got overlap={overlap}, size={size}")

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(_make_chunk(doc_id, len(chunks), text, start, end, page_offsets))
        if end == len(text):
            break
        start += step
    return chunks


# --------------------------------------------------------------------------- #
# Paragraph-aware chunking
# --------------------------------------------------------------------------- #

_PARAGRAPH = re.compile(r"[^\n]*\S[^\n]*")  # ingestion leaves one paragraph per line


def _split_long(text: str, start: int, end: int, max_size: int) -> Iterator[tuple[int, int]]:
    """Split text[start:end] into pieces of at most max_size, preferring word breaks."""
    while end - start > max_size:
        cut = text.rfind(" ", start + 1, start + max_size + 1)
        if cut <= start:  # one enormous "word" (e.g. a URL or a table row): hard cut
            cut = start + max_size
        yield start, cut
        start = cut
        while start < end and text[start] == " ":  # don't start the next piece on a space
            start += 1
    if start < end:
        yield start, end


def _last_space_before(text: str, start: int, limit: int) -> int | None:
    """Index of the last space in text[start+1 : limit+1], or None."""
    cut = text.rfind(" ", start + 1, limit + 1)
    return cut if cut > start else None


def _best_split(
    text: str, start: int, end: int, min_size: int, max_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Split text[start:end] into two spans, each between min_size and max_size long.

    Prefers a paragraph break, then a space, closest to the middle; a hard cut in the
    middle is the last resort (one giant "word").
    """
    middle = (start + end) // 2
    for separator in ("\n\n", " "):
        options = []
        pos = text.find(separator, start + 1, end)
        while pos != -1:
            left_end = len(text[start:pos].rstrip()) + start
            right_start = pos + len(text[pos:end]) - len(text[pos:end].lstrip())
            if (
                min_size <= left_end - start <= max_size
                and min_size <= end - right_start <= max_size
            ):
                options.append((abs(pos - middle), left_end, right_start))
            pos = text.find(separator, pos + 1, end)
        if options:
            _, left_end, right_start = min(options)
            return (start, left_end), (right_start, end)
    return (start, middle), (middle, end)


def _belongs_with_previous(chunk_text: str) -> bool:
    """A leftover starting in lowercase or with punctuation continues the text before it
    ("al., 2020).", "our filtering scheme..."); a heading or caption starts what follows."""
    first = chunk_text.lstrip()[:1]
    return not (first.isupper() or first.isdigit())


def _merge_small(
    text: str, spans: list[tuple[int, int]], min_size: int, max_size: int
) -> list[tuple[int, int]]:
    """Get rid of spans shorter than `min_size`.

    A small span is merged into a neighbour if the result fits. If neither neighbour has
    room, it is joined with the neighbour it belongs with (previous for a continuation,
    next for a heading) and the pair is re-split into two balanced chunks.
    """
    out = list(spans)
    i = 0
    while i < len(out):
        start, end = out[i]
        if end - start >= min_size or len(out) == 1:
            i += 1
            continue
        if i > 0 and end - out[i - 1][0] <= max_size:
            out[i - 1 : i + 1] = [(out[i - 1][0], end)]
            continue
        if i + 1 < len(out) and out[i + 1][1] - start <= max_size:
            out[i : i + 2] = [(start, out[i + 1][1])]
            continue
        # no room anywhere: rebalance with the neighbour this text belongs to
        use_previous = i > 0 and (i + 1 == len(out) or _belongs_with_previous(text[start:end]))
        j = i - 1 if use_previous else i
        left, right = _best_split(text, out[j][0], out[j + 1][1], min_size, max_size)
        out[j : j + 2] = [left, right]
        i = j + 2
    return out


def chunk_paragraphs(
    doc_id: str,
    text: str,
    page_offsets: list[int],
    max_size: int = DEFAULT_SIZE,
    min_size: int | None = None,
) -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most `max_size` characters.

    * A paragraph that fits is added to the current chunk; one that doesn't fit starts
      a new chunk, so chunks normally begin and end at paragraph boundaries.
    * A paragraph longer than `max_size` on its own is cut into `max_size` pieces at
      spaces. Only if the current chunk is still tiny (a lone heading) is it topped up
      with the paragraph's first words; otherwise the long paragraph starts fresh, so
      the chunk before it still ends at a paragraph boundary.
    * Leftovers shorter than `min_size` (default: 10% of `max_size`), e.g. the tail of
      a paragraph that continues on the next page, are merged into a neighbour, or
      rebalanced with it when neither neighbour has room (see `_merge_small`).

    Chunks don't overlap; only whitespace between them is left out.
    """
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    if min_size is None:
        min_size = int(MIN_SIZE_SHARE * max_size)
    if not 0 <= min_size <= max_size:
        raise ValueError(f"min_size must be in [0, max_size], got {min_size}")

    spans: list[tuple[int, int]] = []
    for m in _PARAGRAPH.finditer(text):
        start = m.start() + (len(m.group()) - len(m.group().lstrip()))
        end = m.start() + len(m.group().rstrip())

        if spans and end - spans[-1][0] <= max_size:
            spans[-1] = (spans[-1][0], end)  # the whole paragraph fits: extend
            continue
        if end - start > max_size and spans and spans[-1][1] - spans[-1][0] < min_size:
            # too long to ever fit whole, and the current chunk is a lone heading:
            # top it up with the paragraph's first words instead of stranding it
            room_end = spans[-1][0] + max_size
            cut = _last_space_before(text, start, room_end)
            if cut is not None and cut - start >= min_size:
                spans[-1] = (spans[-1][0], cut)
                start = cut + 1
        spans.extend(_split_long(text, start, end, max_size))

    spans = _merge_small(text, spans, min_size, max_size)
    return [_make_chunk(doc_id, i, text, a, b, page_offsets) for i, (a, b) in enumerate(spans)]


# --------------------------------------------------------------------------- #
# Corpus level: read ingested docs, write chunks.jsonl
# --------------------------------------------------------------------------- #

METHODS = ("fixed", "paragraph")


def chunk_corpus(
    processed_dir: Path,
    method: str = "paragraph",
    size: int = DEFAULT_SIZE,
    overlap: int = 0,
    min_size: int | None = None,
) -> list[Chunk]:
    """Chunk every document listed in `processed_dir/manifest.jsonl`."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    manifest = processed_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; run ragbench-ingest first")

    chunks: list[Chunk] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        text = (processed_dir / f"{entry['doc_id']}.md").read_text(encoding="utf-8")
        text = text.removesuffix("\n")  # ingest adds one trailing newline to the file
        if method == "fixed":
            chunks += chunk_fixed(entry["doc_id"], text, entry["page_offsets"], size, overlap)
        else:
            chunks += chunk_paragraphs(entry["doc_id"], text, entry["page_offsets"], size, min_size)
    return chunks


def output_name(method: str, size: int, overlap: int, min_size: int | None = None) -> str:
    """File name that encodes every setting, so different runs never overwrite each other.

    "chunks.paragraph.1000", "chunks.fixed.1000-200"
    """
    if method == "fixed":
        return f"chunks.fixed.{size}-{overlap}"
    return f"chunks.paragraph.{size}" + (f"-min{min_size}" if min_size is not None else "")


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def stats(chunks: list[Chunk]) -> dict[str, float]:
    sizes = [len(c.text) for c in chunks]
    docs = {c.doc_id for c in chunks}
    return {
        "chunks": len(chunks),
        "documents": len(docs),
        "per_document": round(len(chunks) / len(docs), 1) if docs else 0,
        "mean_chars": round(statistics.mean(sizes)) if sizes else 0,
        "median_chars": round(statistics.median(sizes)) if sizes else 0,
        "min_chars": min(sizes, default=0),
        "max_chars": max(sizes, default=0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk ingested documents into chunks.jsonl")
    parser.add_argument("processed", type=Path, help="folder with manifest.jsonl + <doc_id>.md")
    parser.add_argument("--method", choices=METHODS, default="paragraph")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="max characters per chunk")
    parser.add_argument(
        "--min-size",
        type=int,
        help="paragraph method: merge smaller chunks into a neighbour (default: 10%% of --size)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="characters shared by neighbouring chunks (fixed method only)",
    )
    parser.add_argument(
        "--out", type=Path, help="output file (default: <processed>/<settings name>.jsonl)"
    )
    args = parser.parse_args(argv)

    name = output_name(args.method, args.size, args.overlap, args.min_size)
    out = args.out or args.processed / f"{name}.jsonl"
    try:
        chunks = chunk_corpus(args.processed, args.method, args.size, args.overlap, args.min_size)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_chunks(out, chunks)

    print(f"wrote {out}")
    for key, value in stats(chunks).items():
        print(f"  {key:13} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
