"""Download the arXiv PDFs listed in a sources file.

The sources file is the single, committed definition of the corpus: one arXiv ID
per line, `#` starts a comment. Downloaded PDFs are not committed (see .gitignore);
anyone can rebuild the exact corpus from this file.

Usage:
    ragbench-fetch data/sources.txt data/raw
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

# New-style arXiv IDs: YYMM.NNNNN with an optional version suffix, e.g. 2005.11401v4
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
PDF_URL = "https://arxiv.org/pdf/{id}"
DELAY_SECONDS = 3.0  # arXiv asks automated clients for at most ~1 request every 3 s
USER_AGENT = "rag-vs-agent-bench/0.1 (+https://github.com/Swayam-Ch/rag-vs-agent-bench)"


class HttpResponse(Protocol):
    content: bytes

    def raise_for_status(self) -> None: ...


class HttpSession(Protocol):
    """The tiny slice of `requests.Session` we use. Lets tests pass a fake instead."""

    def get(self, url: str, **kwargs) -> HttpResponse: ...


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #


def parse_sources(text: str) -> list[str]:
    """Return arXiv IDs from a sources file, in order, without duplicates.

    Blank lines and anything after `#` are ignored. Raises ValueError naming the
    line number if a line isn't a valid arXiv ID.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        candidate = line.split("#", 1)[0].strip()
        if not candidate:
            continue
        if not ARXIV_ID.match(candidate):
            raise ValueError(f"line {lineno}: {candidate!r} is not a valid arXiv ID")
        if candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def download_pdf(arxiv_id: str, dest_dir: Path, session: HttpSession) -> Path:
    """Download one PDF to `dest_dir/<id>.pdf` and return its path.

    Writes to `<id>.pdf.part` first and renames on success, so an interrupted run
    never leaves a truncated file that a later run would mistake for a finished one.
    """
    response = session.get(PDF_URL.format(id=arxiv_id), timeout=60)
    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):
        # arXiv serves an HTML page (e.g. a rate-limit notice) with status 200 sometimes
        raise ValueError(f"{arxiv_id}: response is not a PDF")

    final = dest_dir / f"{arxiv_id}.pdf"
    partial = final.with_name(final.name + ".part")
    partial.write_bytes(response.content)
    partial.replace(final)  # atomic on the same filesystem
    return final


def fetch_all(
    ids: list[str],
    dest_dir: Path,
    session: HttpSession,
    delay: float = DELAY_SECONDS,
) -> tuple[list[Path], list[str]]:
    """Download every ID not already in `dest_dir`.

    Returns (downloaded_paths, skipped_ids). Sleeps `delay` seconds between real
    downloads only, so a re-run where everything exists finishes instantly.
    """
    downloaded: list[Path] = []
    skipped: list[str] = []
    first = True
    for arxiv_id in ids:
        if (dest_dir / f"{arxiv_id}.pdf").exists():
            skipped.append(arxiv_id)
            continue
        if not first:
            time.sleep(delay)
        first = False
        path = download_pdf(arxiv_id, dest_dir, session)
        log.info("downloaded %s (%d KB)", path.name, path.stat().st_size // 1024)
        downloaded.append(path)
    return downloaded, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download arXiv PDFs listed in a sources file")
    parser.add_argument("sources", type=Path, help="text file with one arXiv ID per line")
    parser.add_argument("dest", type=Path, help="folder to save PDFs into")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    try:
        ids = parse_sources(args.sources.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import requests  # imported here so the pure functions above need no network library

    args.dest.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        try:
            downloaded, skipped = fetch_all(ids, args.dest, session)
        except (requests.RequestException, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(f"{len(downloaded)} downloaded, {len(skipped)} already present, in {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
