"""Agentic search: a language model answers by exploring the papers with tools.

The loop, one *turn* at a time:

    1. send the conversation so far (+ the tool definitions) to the model
    2. if it asks for tools (stop_reason "tool_use"), run each one with the safe,
       bounded functions in agent_tools.py and send the results back
    3. repeat until it answers (stop_reason "end_turn") or a limit is hit

Limits keep a run cheap and bounded: `max_turns` model calls, `max_seconds` of wall
time, `max_tokens` per reply. A run never raises for a model-side problem (refusal, API
error, limits); it records how it ended in `AgentRun.stop` so the benchmark can count it.

Everything is recorded in an `AgentRun`: the answer, every tool call, the ranges read,
token usage and the full transcript, so any run can be audited later.

The model client is injected, so tests replay scripted conversations offline. The real
client needs `pip install -e ".[agent]"` and the ANTHROPIC_API_KEY environment variable
(plus ANTHROPIC_WORKSPACE_ID if the key is not tied to a single workspace).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragbench.search.agent_tools import DEFAULT_HITS, DEFAULT_READ, Corpus, Read
from ragbench.search.base import SearchResult, check_k

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """\
You answer questions about a fixed collection of research papers. You cannot see the \
papers directly; explore them with the tools:

- list_files: every paper with its IDs, year, length and title. Start here.
- grep: regex search across papers, with match counts per paper and snippets showing \
the page and character offset (@offset) of each match.
- read_file: read a range of characters of one paper. Read around a grep match \
(e.g. start = offset - 500) before relying on it.

Work like a careful researcher: search with several phrasings and synonyms, read the \
passages that matter, and check every paper that might be relevant, especially for \
questions about "which papers", "all", or "none".

End with your answer, then a line "Sources:" listing the evidence as \
"<arXiv ID> p.<page>", one per line. If the papers do not contain the answer, say so \
plainly instead of guessing.\
"""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List every paper: doc ID, arXiv ID, year, pages, length, title.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "grep",
        "description": (
            "Case-insensitive regex search over all papers, or one paper if doc_id is "
            "given. Returns match counts per paper, then snippets with page and @offset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "doc_id": {"type": "string", "description": "doc ID or arXiv ID (optional)"},
                "max_hits": {
                    "type": "integer",
                    "description": f"snippets to show (default {DEFAULT_HITS}, max 30)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read characters [start, start+length) of one paper (max 4000).",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "doc ID or arXiv ID"},
                "start": {"type": "integer", "description": "character offset, default 0"},
                "length": {
                    "type": "integer",
                    "description": f"characters to read (default {DEFAULT_READ})",
                },
            },
            "required": ["doc_id"],
        },
    },
]


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    output_chars: int
    is_error: bool


@dataclass
class Citation:
    """One entry of the answer's "Sources:" list, e.g. "1706.03762 p.9"."""

    ref: str  # as written by the model
    doc_id: str  # resolved paper, or "" if the reference matches no paper
    page: int


_SOURCES = re.compile(r"^\W*sources?\W*$|^\W*sources?\s*:", re.IGNORECASE | re.MULTILINE)
_CITATION = re.compile(
    r"(?P<ref>\d{4}\.\d{4,5}(?:v\d+)?|[0-9a-f]{12})"  # arXiv ID or internal doc ID
    r"[\s,;:()\-]*(?:p\.?|pp\.?|page|pages)\s*(?P<page>\d+)",
    re.IGNORECASE,
)


def parse_citations(answer: str, corpus: Corpus) -> list[Citation]:
    """Read the "Sources:" section of an answer into (paper, page) citations.

    Only text after the last "Sources:" heading counts, so IDs mentioned in the answer
    itself aren't mistaken for citations. "p.9", "p. 9", "page 9" and "pp. 9-10"
    (first page) are understood; a reference to an unknown paper keeps doc_id "".
    """
    headings = list(_SOURCES.finditer(answer))
    if not headings:
        return []
    section = answer[headings[-1].start() :]
    citations, seen = [], set()
    for m in _CITATION.finditer(section):
        ref, page = m["ref"], int(m["page"])
        paper = corpus.resolve(re.sub(r"v\d+$", "", ref))
        citation = Citation(ref, paper.doc_id if paper else "", page)
        if (citation.doc_id or ref, page) not in seen:
            seen.add((citation.doc_id or ref, page))
            citations.append(citation)
    return citations


@dataclass
class AgentRun:
    question: str
    model: str
    answer: str = ""
    # answered | max_turns | timeout | max_tokens | refusal | context_window | error
    stop: str = "answered"
    turns: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    reads: list[Read] = field(default_factory=list)  # read_file ranges only
    evidence: list[Read] = field(default_factory=list)  # reads + grep snippets, in order
    citations: list[Citation] = field(default_factory=list)  # parsed from "Sources:"
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def run_tool(corpus: Corpus, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Execute one tool call. Never raises: bad names or arguments become error text."""
    try:
        if name == "list_files":
            out = corpus.list_files()
        elif name == "grep":
            out = corpus.grep(
                str(args["pattern"]),
                doc_id=args.get("doc_id"),
                max_hits=int(args.get("max_hits", DEFAULT_HITS)),
            )
        elif name == "read_file":
            out = corpus.read_file(
                str(args["doc_id"]),
                start=int(args.get("start", 0)),
                length=int(args.get("length", DEFAULT_READ)),
            )
        else:
            return f"error: unknown tool {name!r}. Available: list_files, grep, read_file.", True
    except (KeyError, TypeError, ValueError) as exc:
        return f"error: bad arguments for {name}: {exc!r}", True
    return out, out.startswith("error:")


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Turn a response content block back into the plain dict the API expects."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump(exclude_none=True)  # other block types, passed back unchanged


_STOPS = {
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "model_context_window_exceeded": "context_window",
}


class AgentSearcher:
    name = "agent"

    def __init__(
        self,
        corpus: Corpus,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = 15,
        max_seconds: float = 180.0,
        max_tokens: int = 2048,
        transcript_dir: Path | None = None,
    ):
        if max_turns < 1 or max_tokens < 1 or max_seconds <= 0:
            raise ValueError("max_turns, max_tokens and max_seconds must be positive")
        self.corpus = corpus
        self.client = client if client is not None else _default_client()
        self.model, self.max_turns = model, max_turns
        self.max_seconds, self.max_tokens = max_seconds, max_tokens
        self.transcript_dir = transcript_dir
        self.last_run: AgentRun | None = None

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #

    def run(self, question: str) -> AgentRun:
        run = AgentRun(question=question, model=self.model)
        self.corpus.reads, self.corpus.evidence = [], []  # each run starts clean
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        started = time.monotonic()
        try:
            while True:
                if run.turns >= self.max_turns:
                    run.stop = "max_turns"
                    break
                if time.monotonic() - started > self.max_seconds:
                    run.stop = "timeout"
                    break
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
                run.turns += 1
                run.input_tokens += response.usage.input_tokens
                run.output_tokens += response.usage.output_tokens
                messages.append(
                    {"role": "assistant", "content": [_block_to_dict(b) for b in response.content]}
                )
                text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                if text:
                    run.answer = text  # the latest text is the answer so far

                if response.stop_reason != "tool_use":
                    run.stop = _STOPS.get(response.stop_reason, "answered")
                    break

                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    output, is_error = run_tool(self.corpus, block.name, block.input or {})
                    run.tool_calls.append(ToolCall(block.name, block.input, len(output), is_error))
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                            "is_error": is_error,
                        }
                    )
                messages.append({"role": "user", "content": results})
        except Exception as exc:  # an API failure must not stop a 20-question benchmark
            run.stop, run.error = "error", f"{type(exc).__name__}: {exc}"

        run.reads = list(self.corpus.reads)
        run.evidence = list(self.corpus.evidence)
        run.citations = parse_citations(run.answer, self.corpus)
        run.seconds = round(time.monotonic() - started, 2)
        run.messages = messages
        self.last_run = run
        if self.transcript_dir is not None:
            self._save(run)
        return run

    def _save(self, run: AgentRun) -> Path:
        digest = hashlib.sha256(run.question.encode()).hexdigest()[:10]
        slug = re.sub(r"[^a-z0-9]+", "-", run.question.lower())[:40].strip("-")
        path = self.transcript_dir / f"{slug}-{digest}.{self.model}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run.to_json(), indent=2, ensure_ascii=False, default=str))
        return path

    # ------------------------------------------------------------------ #
    # Searcher contract: the ranges the agent read are its "retrieved results"
    # ------------------------------------------------------------------ #

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Run the agent and return the distinct ranges it was shown, in order.

        That includes grep snippets, not just read_file ranges: an agent that answers
        from snippets (as on the attention-heads question) did retrieve that evidence.
        Scores are 1/rank; the agent gives no relevance score of its own, so the order
        in which it saw things is the only ranking available.
        """
        check_k(k)
        if not query.strip():
            return []
        run = self.run(query)
        results: list[SearchResult] = []
        seen: set[tuple[str, int, int]] = set()
        for read in run.evidence:
            key = (read.doc_id, read.start, read.end)
            if key in seen:
                continue
            seen.add(key)
            paper = self.corpus.papers[read.doc_id]
            results.append(
                SearchResult(
                    doc_id=read.doc_id,
                    start=read.start,
                    end=read.end,
                    page=paper.page(read.start),
                    page_end=paper.page(read.end - 1),
                    score=1.0 / (len(results) + 1),
                    text=paper.text[read.start : read.end],
                )
            )
        return results[:k]


def _default_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            'agent search needs the anthropic SDK: pip install -e ".[agent]"'
        ) from exc
    return anthropic.Anthropic(**client_options(os.environ))


def client_options(env: Mapping[str, str]) -> dict[str, Any]:
    """Extra client settings taken from environment variables.

    The SDK reads ANTHROPIC_API_KEY itself. A key that works across several workspaces
    must also name one per request (the `anthropic-workspace-id` header), otherwise the
    API answers "This API key is not scoped to a workspace". The SDK doesn't read
    ANTHROPIC_WORKSPACE_ID for plain API keys, so it is passed on here.
    """
    workspace = env.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    return {"default_headers": {"anthropic-workspace-id": workspace}} if workspace else {}
