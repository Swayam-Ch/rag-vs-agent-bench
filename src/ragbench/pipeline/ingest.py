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
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


def guess_title(text: str, fallback: str) -> str:
    """First non-empty line, if it looks like a title; otherwise the file stem."""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line if len(line) <= 200 else fallback
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


def ingest_file(path: Path) -> Document:
    """Read one file and return a cleaned Document."""
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    pages = [normalize_text(p) for p in extract_pages(path)]

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
        title=guess_title(text, fallback=path.stem),
        text=text,
        page_offsets=offsets,
        sha256=digest,
    )


def ingest_dir(src: Path, dest: Path) -> list[Document]:
    """Ingest every supported file in `src`, writing `<doc_id>.md` + manifest.jsonl to `dest`."""
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No {sorted(SUPPORTED_SUFFIXES)} files found in {src}")

    dest.mkdir(parents=True, exist_ok=True)
    docs: list[Document] = []
    for path in files:
        doc = ingest_file(path)
        (dest / f"{doc.doc_id}.md").write_text(doc.text + "\n", encoding="utf-8")
        docs.append(doc)
        log.info("ingested %-40s -> %s.md (%d chars)", path.name, doc.doc_id, len(doc.text))

    manifest = dest / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.manifest_entry(), ensure_ascii=False) + "\n")
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="folder of raw PDFs / text files")
    parser.add_argument("dest", type=Path, help="output folder for markdown + manifest")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        docs = ingest_dir(args.src, args.dest)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Ingested {len(docs)} documents into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
