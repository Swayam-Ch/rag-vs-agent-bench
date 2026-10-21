"""The agent's tools: list the papers, grep them, read a range of one.

These are plain Python functions over the ingested markdown; the agent loop (#12) exposes
them to the model as tool calls. Three rules hold for every tool, because the caller is
a language model:

* **Never raise.** Bad input (unknown paper, broken regex, negative offset) returns a
  message starting with "error:" that tells the model what to do instead.
* **Never flood the context.** Every output is capped (MAX_OUTPUT characters); long
  results say how to get the rest.
* **Never touch anything outside the corpus.** Papers are looked up by ID in the
  manifest; a name is never joined onto a file path, so "../../etc/passwd" is just an
  unknown ID.

Everything the agent is shown is logged in `Corpus.evidence`, in order: each successful
`read_file` range, and each grep snippet that made it into the (capped) output. That is
the agent's "retrieved evidence", scored with the same page-level metrics as the
chunk-based methods. (An agent can answer from snippets alone, without ever calling
read_file; that is a fair use of its tools and must count.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ragbench.pipeline.chunk import page_at

MAX_OUTPUT = 6_000  # characters any single tool call may return
DEFAULT_READ, MAX_READ = 2_000, 4_000
DEFAULT_HITS, MAX_HITS = 10, 30
SNIPPET = 100  # characters of context on each side of a grep match
MAX_PATTERN = 200
MAX_MATCH_SHOWN = 200  # a longer match (e.g. from ".*") is shortened in its snippet
MAX_COUNTED = 10_000  # stop counting matches here; enough to say "very many"


@dataclass(frozen=True)
class Read:
    """A range of a paper the agent was shown."""

    doc_id: str
    start: int
    end: int
    kind: str = "read"  # "read" (read_file) or "snippet" (shown in grep results)


@dataclass
class Paper:
    doc_id: str
    text: str
    title: str
    page_offsets: list[int]
    arxiv_id: str = ""
    published: str = ""
    authors: list[str] = field(default_factory=list)

    def page(self, offset: int) -> int:
        return page_at(offset, self.page_offsets)


def _cap(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    note = f"\n[output truncated at {limit} characters]"
    return text[: limit - len(note)] + note


def _one_line(text: str) -> str:
    """Show paragraph breaks as ¶ so a snippet stays on one line."""
    return " ¶ ".join(part.strip() for part in text.split("\n\n")).replace("\n", " ")


class Corpus:
    def __init__(self, papers: list[Paper]):
        self.papers = {p.doc_id: p for p in papers}
        # friendlier names the model may use instead of the hash ID
        self._aliases: dict[str, str] = {}
        for p in papers:
            if p.arxiv_id:
                self._aliases[p.arxiv_id] = p.doc_id
        self.reads: list[Read] = []  # read_file calls only
        self.evidence: list[Read] = []  # everything shown: reads and grep snippets, in order

    @classmethod
    def load(cls, processed_dir: Path) -> Corpus:
        """Read manifest.jsonl and the <doc_id>.md files written by ragbench-ingest."""
        papers = []
        manifest = processed_dir / "manifest.jsonl"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            text = (processed_dir / f"{m['doc_id']}.md").read_text(encoding="utf-8")
            papers.append(
                Paper(
                    doc_id=m["doc_id"],
                    text=text.removesuffix("\n"),
                    title=m["title"],
                    page_offsets=m["page_offsets"],
                    arxiv_id=m.get("arxiv_id", ""),
                    published=m.get("published", ""),
                    authors=m.get("authors", []),
                )
            )
        return cls(papers)

    def resolve(self, name: str) -> Paper | None:
        """Find a paper by doc ID or arXiv ID. Only ever a dictionary lookup."""
        name = name.strip()
        doc_id = name if name in self.papers else self._aliases.get(name)
        return self.papers.get(doc_id) if doc_id else None

    def _unknown(self, name: str) -> str:
        return f"error: no paper with ID {name!r}. Call list_files to see the valid IDs."

    # ------------------------------------------------------------------ #
    # list_files
    # ------------------------------------------------------------------ #

    def list_files(self) -> str:
        """One line per paper: IDs, year, pages, length, title."""
        lines = [f"{len(self.papers)} papers (doc_id | arXiv | year | pages | chars | title)"]
        for p in sorted(self.papers.values(), key=lambda p: (p.published, p.doc_id)):
            lines.append(
                f"{p.doc_id} | {p.arxiv_id or '-'} | {p.published[:4] or '-'} | "
                f"{len(p.page_offsets)} | {len(p.text)} | {p.title}"
            )
        return _cap("\n".join(lines))

    # ------------------------------------------------------------------ #
    # grep
    # ------------------------------------------------------------------ #

    def grep(self, pattern: str, doc_id: str | None = None, max_hits: int = DEFAULT_HITS) -> str:
        """Case-insensitive regex search over all papers (or one).

        Shows a per-paper match count (useful for "which papers mention X" questions),
        then up to `max_hits` snippets with their page and character offset.

        Known limit: Python's `re` has no timeout, so a pathological pattern such as
        "(a+)+$" can backtrack for a very long time on some text. Models rarely write
        these; the agent loop's overall time limit is the backstop.
        """
        if not pattern.strip():
            return "error: empty pattern."
        if len(pattern) > MAX_PATTERN:
            return f"error: pattern longer than {MAX_PATTERN} characters; use a shorter one."
        note = ""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
            note = f"(invalid regex: {exc}; searched for the literal text instead)\n"

        if doc_id is not None:
            paper = self.resolve(doc_id)
            if paper is None:
                return self._unknown(doc_id)
            papers = [paper]
        else:
            papers = sorted(self.papers.values(), key=lambda p: p.doc_id)
        max_hits = max(1, min(max_hits, MAX_HITS))

        counts: dict[str, int] = {}
        hits: list[tuple[str, Read]] = []
        total = 0
        for paper in papers:
            for m in regex.finditer(paper.text):
                if m.end() == m.start():  # zero-width matches (e.g. "a*") carry no evidence
                    continue
                counts[paper.doc_id] = counts.get(paper.doc_id, 0) + 1
                total += 1
                if len(hits) < max_hits:
                    shown_end = min(m.end(), m.start() + MAX_MATCH_SHOWN)
                    a = max(0, m.start() - SNIPPET)
                    b = min(len(paper.text), shown_end + SNIPPET)
                    snippet = _one_line(paper.text[a:b])
                    line = f"{paper.doc_id} p.{paper.page(m.start())} @{m.start()}: …{snippet}…"
                    hits.append((line, Read(paper.doc_id, a, b, kind="snippet")))
                if total >= MAX_COUNTED:
                    break
            if total >= MAX_COUNTED:
                break

        if total == 0:
            return note + f"no matches for {pattern!r}."
        more = "+" if total >= MAX_COUNTED else ""
        lines = [note + f"{total}{more} matches in {len(counts)} papers:"]
        for d, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {d} ({self.papers[d].title[:60]}): {n}")
        lines.append(f"first {len(hits)} matches (read_file around @offset for more):")
        lines.extend(line for line, _ in hits)
        out = _cap("\n".join(lines))
        # a snippet counts as evidence only if it survived the output cap in full
        self.evidence.extend(read for line, read in hits if line in out)
        return out

    # ------------------------------------------------------------------ #
    # read_file
    # ------------------------------------------------------------------ #

    def read_file(self, doc_id: str, start: int = 0, length: int = DEFAULT_READ) -> str:
        """Read `length` characters (at most MAX_READ) of one paper, starting at `start`."""
        paper = self.resolve(doc_id)
        if paper is None:
            return self._unknown(doc_id)
        n = len(paper.text)
        if start < 0:
            return "error: start must be 0 or more."
        if start >= n:
            return f"error: start={start} is past the end; {paper.doc_id} has {n} characters."
        length = max(1, min(length, MAX_READ))
        end = min(n, start + length)
        self.reads.append(Read(paper.doc_id, start, end))
        self.evidence.append(self.reads[-1])

        header = (
            f"{paper.doc_id} ({paper.title[:80]}) characters {start}-{end} of {n}, "
            f"pages {paper.page(start)}-{paper.page(end - 1)}\n\n"
        )
        footer = f"\n\n[{n - end} more characters; continue with start={end}]" if end < n else ""
        return _cap(header + paper.text[start:end] + footer)
