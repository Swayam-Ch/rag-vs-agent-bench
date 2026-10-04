"""Turn raw source files (PDF, .txt, .md) into clean markdown + a metadata manifest.

Design decisions worth knowing about (and worth mentioning in the README):

* **Metadata lives in a sidecar manifest, not in the markdown.** If you put YAML
  front matter in every file, the agent's grep/read tools and the vector
  index both "see" that boilerplate, which inflates token counts and can
  create fake matches. Keeping the text body pure means every method searches
  exactly the same content.
* **Page boundaries are kept as character offsets** in the manifest, so later
  we can check whether an answer cites the right page without polluting the
  text with page markers.
* **Everything is deterministic.** Same input bytes -> same doc_id -> same
  output. That is what makes the benchmark reproducible.

Usage:
    ragbench-ingest data/raw data/processed
    python -m ragbench.pipeline.ingest data/raw data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

from ragbench.pipeline.metadata import PaperMeta, base_id, read_metadata

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

# PDF extractors often emit typographic ligatures as single code points.
# NFKC normalisation fixes most of them; this table covers the rest explicitly.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


@dataclass
class Document:
    """One ingested source file."""

    doc_id: str  # short content hash, stable across machines
    source: str  # original file name
    title: str
    text: str  # clean markdown body (no front matter)
    page_offsets: list[int] = field(default_factory=list)  # char index where each page starts
    sha256: str = ""  # full hash of the original bytes
    references_removed: int = 0  # raw characters cut by strip_references (0 = none found)
    boilerplate_lines_removed: int = 0  # repeated header/footer + page-number lines cut
    # from data/metadata.jsonl when the file is an arXiv paper; empty otherwise
    arxiv_id: str = ""
    authors: list[str] = field(default_factory=list)
    published: str = ""  # YYYY-MM-DD
    category: str = ""  # primary arXiv category, e.g. "cs.CL"

    def manifest_entry(self) -> dict:
        """Everything except the (large) text body."""
        entry = asdict(self)
        entry.pop("text")
        entry["n_chars"] = len(self.text)
        entry["n_pages"] = len(self.page_offsets)
        return entry


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #


def normalize_text(raw: str) -> str:
    """Clean text extracted from PDFs or plain files.

    Steps (each one is individually tested):
      1. Unicode NFKC + explicit ligature replacement
      2. Re-join words hyphenated across line breaks ("retrie-\\nval" -> "retrieval")
      3. Unwrap hard line breaks inside paragraphs
      4. Collapse runs of spaces and of blank lines
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = unicodedata.normalize("NFKC", text)

    # 2. de-hyphenate: a lowercase letter, hyphen, newline, lowercase letter
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # 3. a single newline between two non-empty lines is a soft wrap -> space
    text = re.sub(r"(?<=\S)[ \t]*\n(?=[ \t]*\S)", " ", text)

    # 4. tidy whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# A heading on a line of its own: "References", "REFERENCES", "7 References", "Bibliography".
# Requiring the whole line means the word inside a sentence never matches.
_REFERENCES_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(?:references|bibliography)[ \t]*$", re.IGNORECASE | re.MULTILINE
)
# Where appendices restart real content after the bibliography.
_APPENDIX_HEADING = re.compile(
    r"^[ \t]*(?:appendix|appendices|supplementary material)\b.*$", re.IGNORECASE | re.MULTILINE
)
# Lettered appendix sections: "A Additional results", "A.1 MMLU", "B.2.1 Details".
# No dot after the letter, so author initials ("A. Vaswani, N. Shazeer") never match.
_LETTERED_HEADING = re.compile(
    r"^[ \t]*(?P<label>[A-H](?:\.\d+)*)[ \t]+[A-Z][^\n]{2,80}$", re.MULTILINE
)


def _is_next_label(prev: str, nxt: str) -> bool:
    """True if `nxt` can directly follow `prev` in an appendix outline.

    From "A.1" the next heading may be "A.1.1" (go deeper), "A.2" (sibling), or
    "B" / "A" -> "B" etc. (back up a level and step on).
    """
    p, n = prev.split("."), nxt.split(".")
    if n == p + ["1"]:
        return True
    for depth in range(len(p)):
        head, last = p[: depth + 1], p[depth]
        stepped = chr(ord(last) + 1) if depth == 0 else str(int(last) + 1)
        if n == head[:-1] + [stepped]:
            return True
    return False


def _lettered_appendix_start(text: str, pos: int) -> int | None:
    """Position of the first lettered appendix heading after `pos`, or None.

    A candidate only counts if the next candidate continues the outline
    (A -> A.1 or B, A.1 -> A.2 ...). A wrapped reference line that happens to look like
    "A Survey of Dense Retrieval" is never followed by "A.1" or "B", so it is skipped.
    """
    found = [(m.start(), m["label"]) for m in _LETTERED_HEADING.finditer(text, pos)]
    for (start, label), (_, nxt) in pairwise(found):
        if label in ("A", "A.1") and _is_next_label(label, nxt):
            return start
    return None


def strip_references(pages: list[str]) -> tuple[list[str], int]:
    """Remove the bibliography from raw (not yet normalised) page texts.

    Cuts from the *last* References/Bibliography heading up to where the appendices
    begin: an "Appendix" heading or a lettered outline ("A ...", "A.1 ...", "B ...").
    With no appendix, it cuts to the end of the document. The number of pages never
    changes (emptied pages stay as ""), which keeps page numbers aligned with the PDF.

    Returns (pages, number_of_characters_removed).
    """
    text = "\n".join(pages)  # one string so headings can be found across page breaks
    heads = list(_REFERENCES_HEADING.finditer(text))
    if not heads:
        return pages, 0
    start = heads[-1].start()

    ends = [len(text)]
    if explicit := _APPENDIX_HEADING.search(text, start):
        ends.append(explicit.start())
    if (lettered := _lettered_appendix_start(text, start)) is not None:
        ends.append(lettered)
    end = min(ends)

    # cut [start, end) out of each page that overlaps it
    out: list[str] = []
    page_start = 0
    for page in pages:
        page_end = page_start + len(page)
        lo, hi = max(start, page_start), min(end, page_end)
        if lo < hi:
            page = page[: lo - page_start] + page[hi - page_start :]
        out.append(page)
        page_start = page_end + 1  # +1 for the "\n" joining pages
    removed = sum(map(len, pages)) - sum(map(len, out))
    return out, removed


EDGE_LINES = 3  # headers/footers live in the first/last few lines of a page
_PAGE_NUMBER = re.compile(r"^(?:page\s*)?(?P<n>\d{1,4})(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)


def _boilerplate_key(line: str) -> str:
    """Normalise a line so the same header on different pages compares equal.

    Case and spacing are ignored, and so is a page number standing as a word at the
    start or end ("3 Published at ICLR", "Under review 12"). Numbers inside the line
    still count, so "results on page 1." and "results on page 2." stay different.
    """
    tokens = line.lower().split()
    if tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    if tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return " ".join(tokens)


def _edge_indices(lines: list[str]) -> list[int]:
    """Indices of the first and last EDGE_LINES non-empty lines of a page."""
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return sorted(set(filled[:EDGE_LINES] + filled[-EDGE_LINES:]))


def _page_number_offset(split: list[list[str]], min_pages: float) -> int | None:
    """The shift between page position and printed page number, if one is consistent.

    Page 5 of the PDF may print "5" (offset 0), or "4" if the cover is unnumbered
    (offset -1). Chart tick labels and table values at a page edge don't follow the
    page sequence, so they never agree on an offset.
    """
    offsets: Counter[int] = Counter()
    for position, lines in enumerate(split, start=1):
        found = set()
        for i in _edge_indices(lines):
            if m := _PAGE_NUMBER.match(lines[i].strip()):
                found.add(int(m["n"]) - position)
        offsets.update(found)
    if not offsets:
        return None
    offset, count = offsets.most_common(1)[0]
    return offset if count >= min_pages else None


def remove_repeated_lines(pages: list[str], min_share: float = 0.5) -> tuple[list[str], list[str]]:
    """Remove running headers, footers and page numbers from raw page texts.

    Only lines near the top or bottom of a page are considered; the middle of a page
    is never touched. Two kinds of line are removed:

    * a header/footer: the same line (case, spacing and an edge page number ignored)
      at a page edge on at least `min_share` of the pages, and on at least 2 pages;
    * a page number: a standalone number ("7", "Page 7", "7 of 12") that matches its
      page's position, using the offset most pages agree on. A number that doesn't
      follow the page sequence (a chart tick label, a table value) is kept.

    Returns (pages, removed_lines).
    """
    split = [page.split("\n") for page in pages]
    threshold = max(2, min_share * len(pages))

    pages_with_key: Counter[str] = Counter()
    for lines in split:
        pages_with_key.update({_boilerplate_key(lines[i]) for i in _edge_indices(lines)})
    # "" is the key of a bare number; numbers are handled by the page-sequence rule below
    repeated = {key for key, n in pages_with_key.items() if key and n >= threshold}
    offset = _page_number_offset(split, min_pages=max(2, 0.25 * len(pages)))

    out: list[str] = []
    removed: list[str] = []
    for position, lines in enumerate(split, start=1):
        drop = set()
        for i in _edge_indices(lines):
            line = lines[i].strip()
            number = _PAGE_NUMBER.match(line)
            if number:
                if offset is not None and int(number["n"]) - position == offset:
                    drop.add(i)
            elif _boilerplate_key(line) in repeated:
                drop.add(i)
        removed.extend(lines[i].strip() for i in sorted(drop))
        out.append("\n".join(line for i, line in enumerate(lines) if i not in drop))
    return out, removed


# Lines at the top of a first page that are never the title.
_NOT_A_TITLE = re.compile(
    r"arxiv:|published as|under review|preprint|provided proper attribution|proceedings|"
    r"conference|workshop|copyright|©|@|https?://|^\W*\d",
    re.IGNORECASE,
)


def guess_title(first_page: str, fallback: str) -> str:
    """Best guess at a title from the *raw* (not yet normalised) first page.

    Only a fallback for files without arXiv metadata. Takes the first line that has
    at least two words, isn't venue/stamp/contact boilerplate and isn't a sentence.
    Runs on raw text because normalisation merges the opening lines into one paragraph.
    """
    for line in first_page.splitlines():
        line = line.strip().lstrip("#").strip()
        if (
            len(line.split()) >= 2
            and len(line) <= 200
            and not line.endswith(".")
            and not _NOT_A_TITLE.search(line)
        ):
            return line
    return fallback


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_pdf_pages(path: Path) -> list[str]:
    """Return raw text for each page of a PDF."""
    from pypdf import PdfReader  # imported lazily so .txt-only use needs no PDF lib

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # a single broken page shouldn't kill the whole doc
            log.warning("%s: page %d failed to extract (%s)", path.name, i + 1, exc)
            pages.append("")
    return pages


def extract_pages(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(path)
    if suffix in {".txt", ".md"}:
        return [path.read_text(encoding="utf-8", errors="replace")]
    raise ValueError(f"Unsupported file type: {path.suffix}")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def ingest_file(
    path: Path, keep_references: bool = False, meta: PaperMeta | None = None
) -> Document:
    """Read one file and return a cleaned Document.

    `meta` is the paper's arXiv metadata, if known; its title beats any guess.
    """
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    extracted = extract_pages(path)
    title = meta.title if meta else guess_title(extracted[0] if extracted else "", path.stem)

    # headers first: emptied reference pages would otherwise dilute the per-page counts
    raw_pages, boilerplate = remove_repeated_lines(extracted)
    if boilerplate:
        log.info("%s: removed %s", path.name, Counter(boilerplate).most_common(5))
    removed = 0
    if not keep_references:
        # must run before normalize_text, which joins the heading line onto the next one
        raw_pages, removed = strip_references(raw_pages)
    pages = [normalize_text(p) for p in raw_pages]

    # join pages with a blank line and remember where each one starts
    offsets: list[int] = []
    parts: list[str] = []
    cursor = 0
    for page in pages:
        offsets.append(cursor)
        parts.append(page)
        cursor += len(page) + 2  # +2 for the "\n\n" separator
    text = "\n\n".join(parts)

    if not text.strip():
        log.warning("%s: no text extracted (scanned PDF? needs OCR)", path.name)

    return Document(
        doc_id=digest[:12],
        source=path.name,
        title=title,
        text=text,
        page_offsets=offsets,
        sha256=digest,
        references_removed=removed,
        boilerplate_lines_removed=len(boilerplate),
        arxiv_id=meta.arxiv_id if meta else "",
        authors=list(meta.authors) if meta else [],
        published=meta.published if meta else "",
        category=meta.category if meta else "",
    )


def ingest_dir(
    src: Path,
    dest: Path,
    keep_references: bool = False,
    metadata: dict[str, PaperMeta] | None = None,
) -> list[Document]:
    """Ingest every supported file in `src`, writing `<doc_id>.md` + manifest.jsonl to `dest`.

    `metadata` maps arXiv IDs to PaperMeta; a file named `<arXiv ID>.pdf` gets its entry.
    """
    metadata = metadata or {}
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No {sorted(SUPPORTED_SUFFIXES)} files found in {src}")

    dest.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = []
    for path in files:
        meta = metadata.get(base_id(path.stem))
        doc = ingest_file(path, keep_references=keep_references, meta=meta)
        (dest / f"{doc.doc_id}.md").write_text(doc.text + "\n", encoding="utf-8")
        docs.append(doc)
        log.info(
            "ingested %-40s -> %s.md (%d chars, %d reference chars removed)",
            path.name,
            doc.doc_id,
            len(doc.text),
            doc.references_removed,
        )
        if not keep_references and doc.references_removed == 0:
            log.warning("%s: no References heading found", path.name)

    manifest = dest / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.manifest_entry(), ensure_ascii=False) + "\n")
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="folder of raw PDFs / text files")
    parser.add_argument("dest", type=Path, help="output folder for markdown + manifest")
    parser.add_argument(
        "--keep-references", action="store_true", help="don't strip the bibliography"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="arXiv metadata file (default: metadata.jsonl in the parent of src, if present)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    metadata_path = args.metadata or args.src.parent / "metadata.jsonl"
    metadata = read_metadata(metadata_path)
    if metadata:
        log.info("using titles and metadata from %s", metadata_path)
    try:
        docs = ingest_dir(
            args.src, args.dest, keep_references=args.keep_references, metadata=metadata
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Ingested {len(docs)} documents into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
