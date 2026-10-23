"""The `ragbench` command: try any search method from the terminal.

    ragbench search "how many attention heads does the Transformer use?"
    ragbench search --method vector -k 3 "dense retrieval for open-domain QA"
    ragbench search --method agent "which papers evaluate on Natural Questions?"
    ragbench search --method hybrid --json "LoRA rank" > results.json

Every method searches the same ingested corpus (data/processed by default); the
chunk-based methods also need a chunks file from `ragbench-chunk`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ragbench.pipeline.chunk import read_chunks
from ragbench.search.base import KeywordSearcher, Searcher, SearchResult

METHODS = ("keyword", "bm25", "vector", "hybrid", "agent")
SNIPPET_CHARS = 160


def _titles(processed: Path) -> dict[str, tuple[str, str]]:
    """doc_id -> (arXiv ID or "", title), from the manifest."""
    out = {}
    for line in (processed / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        out[m["doc_id"]] = (m.get("arxiv_id", ""), m["title"])
    return out


DEFAULT_CHUNKS = "chunks.paragraph.1000.jsonl"


def find_chunks_file(processed: Path, explicit: Path | None = None) -> Path:
    """The chunks file to search: the one asked for, else the default, else the only one."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"{explicit} not found")
        return explicit
    default = processed / DEFAULT_CHUNKS
    if default.exists():
        return default
    found = sorted(processed.glob("chunks.*.jsonl"))
    if len(found) == 1:
        print(f"using {found[0].name}", file=sys.stderr)
        return found[0]
    if not found:
        raise FileNotFoundError(f"no chunks file in {processed}; run: ragbench-chunk {processed}")
    names = ", ".join(f.name for f in found)
    raise FileNotFoundError(
        f"several chunks files in {processed} ({names}); pick one with --chunks"
    )


def build_searcher(args: argparse.Namespace) -> Searcher:
    """Construct the requested method. Heavy dependencies are imported only when needed."""
    if args.method == "agent":
        from ragbench.search.agent_search import AgentSearcher
        from ragbench.search.agent_tools import Corpus

        return AgentSearcher(
            Corpus.load(args.processed),
            model=args.model,
            max_turns=args.max_turns,
            transcript_dir=args.processed / "transcripts",
        )

    chunks = read_chunks(find_chunks_file(args.processed, args.chunks))

    if args.method == "keyword":
        return KeywordSearcher(chunks)

    from ragbench.search.bm25 import BM25Searcher

    if args.method == "bm25":
        return BM25Searcher(chunks)

    from ragbench.search import vector_search

    vector = vector_search.VectorSearcher(
        chunks, vector_search.SentenceTransformerEmbedder(), cache_dir=args.processed / "cache"
    )
    if args.method == "vector":
        return vector

    from ragbench.search.hybrid_search import HybridSearcher

    return HybridSearcher([BM25Searcher(chunks), vector])


def _one_line(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def format_results(results: list[SearchResult], titles: dict[str, tuple[str, str]]) -> list[str]:
    lines = []
    for rank, r in enumerate(results, start=1):
        arxiv, title = titles.get(r.doc_id, ("", r.doc_id))
        pages = f"p.{r.page}" if r.page == r.page_end else f"pp.{r.page}-{r.page_end}"
        lines.append(f"{rank:>2}. {r.score:8.4f}  {arxiv or r.doc_id}  {pages}  {title[:60]}")
        lines.append(f"    {_one_line(r.text)}")
    return lines


def _search(args: argparse.Namespace) -> int:
    if not args.query.strip():
        print("error: the query is empty", file=sys.stderr)
        return 1
    manifest = args.processed / "manifest.jsonl"
    if not manifest.exists():
        print(f"error: {manifest} not found; run ragbench-ingest first", file=sys.stderr)
        return 1
    try:
        searcher = build_searcher(args)
    except (FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results = searcher.search(args.query, k=args.k)
    run = getattr(searcher, "last_run", None)

    if args.json:
        payload = {
            "method": args.method,
            "query": args.query,
            "results": [asdict(r) for r in results],
        }
        if run is not None:
            payload["agent"] = {k: v for k, v in run.to_json().items() if k != "messages"}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    titles = _titles(args.processed)
    if run is not None:
        _print_agent(run)
    if not results:
        print("no results")
        return 0
    print(f"{'evidence' if run is not None else 'results'} ({searcher.name}):")
    print("\n".join(format_results(results, titles)))
    return 0


def _print_agent(run) -> None:
    from ragbench.search.agent_search import estimate_cost

    print(run.answer or "(no answer)")
    print()
    cost = estimate_cost(run.model, run.input_tokens, run.output_tokens)
    cost_text = f"${cost:.3f}" if cost is not None else "unknown price"
    print(
        f"stop={run.stop} turns={run.turns} tokens={run.input_tokens}/{run.output_tokens} "
        f"cost≈{cost_text} time={run.seconds}s"
    )
    if run.error:
        print(f"error: {run.error}")
    calls = ", ".join(f"{c.name}({json.dumps(c.input)[:60]})" for c in run.tool_calls)
    print(f"tools: {calls or 'none'}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragbench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="run one query with one method")
    search.add_argument("query", help="the question or search terms (quote it)")
    search.add_argument("--method", "-m", choices=METHODS, default="bm25")
    search.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    search.add_argument("--processed", type=Path, default=Path("data/processed"))
    search.add_argument("--chunks", type=Path, help="chunks file (default: paragraph, 1000)")
    search.add_argument("--json", action="store_true", help="machine-readable output")
    agent = search.add_argument_group("agent options")
    agent.add_argument("--model", default="claude-sonnet-5-5")
    agent.add_argument("--max-turns", type=int, default=15)

    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("-k must be at least 1")
    return _search(args)


if __name__ == "__main__":
    raise SystemExit(main())
