"""Paper metadata (title, authors, date, category) from the arXiv API.

Metadata is fetched once, at download time, and saved to `data/metadata.jsonl`, which
is committed. Ingestion only ever reads that file, so the pipeline after `fetch` never
needs the network and every run sees the same titles.

The API returns Atom XML: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

API_URL = "https://export.arxiv.org/api/query"
BATCH_SIZE = 20  # IDs per request; arXiv allows far more, small batches keep errors local
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_VERSION = re.compile(r"v\d+$")


class HttpResponse(Protocol):
    text: str

    def raise_for_status(self) -> None: ...


class HttpSession(Protocol):
    def get(self, url: str, **kwargs) -> HttpResponse: ...


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str  # without version, e.g. "2005.11401"
    version: str  # latest version at fetch time, e.g. "v4"
    title: str
    authors: list[str]
    published: str  # first submission date, YYYY-MM-DD
    category: str  # primary arXiv category, e.g. "cs.CL"


def base_id(arxiv_id: str) -> str:
    """'2005.11401v4' -> '2005.11401'."""
    return _VERSION.sub("", arxiv_id)


def _clean(text: str | None) -> str:
    """Titles in the feed contain line breaks and double spaces; collapse them."""
    return " ".join((text or "").split())


def parse_feed(xml_text: str) -> list[PaperMeta]:
    """Parse an arXiv API Atom response into PaperMeta records."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        entry_id = _clean(entry.findtext("atom:id", namespaces=NS))
        if "/api/errors" in entry_id:  # the API reports bad IDs as an "Error" entry
            raise ValueError(f"arXiv API error: {_clean(entry.findtext('atom:summary', '', NS))}")
        full_id = entry_id.rsplit("/abs/", 1)[-1]  # http://arxiv.org/abs/2005.11401v4
        version = _VERSION.search(full_id)
        category = entry.find("arxiv:primary_category", NS)
        papers.append(
            PaperMeta(
                arxiv_id=base_id(full_id),
                version=version.group() if version else "",
                title=_clean(entry.findtext("atom:title", namespaces=NS)),
                authors=[
                    _clean(a.findtext("atom:name", namespaces=NS))
                    for a in entry.findall("atom:author", NS)
                ],
                published=_clean(entry.findtext("atom:published", namespaces=NS))[:10],
                category=category.get("term", "") if category is not None else "",
            )
        )
    return papers


def fetch_metadata(ids: list[str], session: HttpSession) -> list[PaperMeta]:
    """Look up metadata for arXiv IDs, BATCH_SIZE at a time. Raises if any ID is missing."""
    papers: list[PaperMeta] = []
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start : start + BATCH_SIZE]
        response = session.get(
            API_URL,
            params={"id_list": ",".join(batch), "max_results": len(batch)},
            timeout=60,
        )
        response.raise_for_status()
        papers.extend(parse_feed(response.text))

    missing = {base_id(i) for i in ids} - {p.arxiv_id for p in papers}
    if missing:
        raise ValueError(f"arXiv API returned no metadata for: {', '.join(sorted(missing))}")
    return papers


def read_metadata(path: Path) -> dict[str, PaperMeta]:
    """Load metadata.jsonl into {base arXiv ID: PaperMeta}. A missing file means no entries."""
    if not path.exists():
        return {}
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {r["arxiv_id"]: PaperMeta(**r) for r in records}


def write_metadata(path: Path, papers: dict[str, PaperMeta]) -> None:
    """Write metadata sorted by ID, one JSON object per line, so git diffs stay readable."""
    lines = [json.dumps(asdict(papers[k]), ensure_ascii=False) + "\n" for k in sorted(papers)]
    path.write_text("".join(lines), encoding="utf-8")


def sync_metadata(ids: list[str], path: Path, session: HttpSession) -> list[str]:
    """Fetch metadata for IDs not yet in `path`, drop IDs no longer listed, save.

    Returns the base IDs that were newly fetched (empty when nothing changed, in
    which case no request is made at all).
    """
    wanted = [base_id(i) for i in ids]
    known = read_metadata(path)
    new = [i for i in wanted if i not in known]
    if new:
        known.update({p.arxiv_id: p for p in fetch_metadata(new, session)})
    kept = {i: known[i] for i in wanted}
    if new or kept.keys() != read_metadata(path).keys():
        write_metadata(path, kept)
    return new
