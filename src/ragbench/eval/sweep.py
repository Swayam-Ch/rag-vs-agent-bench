"""Retrieval sweep: is the gap just a weak retriever? `ragbench sweep`

    ragbench sweep                          # every configuration below, ~10-20 min
    ragbench sweep --configs bm25,hybrid-bge+rerank

The first run used one setup: a small embedding model, paragraph chunks, no reranker.
This command scores retrieval only (no API calls, no cost) for several stronger setups,
on the same questions and metrics, and writes one comparison table:

* a larger embedding model (bge-base, with its query instruction)
* a cross-encoder reranker over the top 50 candidates of BM25 or hybrid
* fixed-size, overlapping chunks instead of paragraphs
* k = 20, so recall@20 shows how much more a wider net catches

Answers aren't generated: if no setup gets the gold pages into the top 20, a better
answer-writer can't fix it either. Each configuration is one method in a normal results
folder (run.json + <config>.jsonl), plus summary.md with the table.

Models are downloaded on first use (about 600 MB in total) and their exact revisions
are resolved and recorded in run.json.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragbench.eval.run import (
    config_hash,
    describe,
    git_info,
    new_run_dir,
    package_versions,
    run_benchmark,
    sha256_file,
)
from ragbench.pipeline.chunk import Chunk, chunk_corpus
from ragbench.search.base import Searcher

K = 20

MINILM = {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}  # revision: the pinned default
BGE = {
    "model_name": "BAAI/bge-base-en-v1.5",
    "revision": None,  # resolved at load time and recorded
    "query_prefix": "Represent this sentence for searching relevant passages: ",
}
EMBEDDERS = {"minilm": MINILM, "bge": BGE}
CHUNKINGS = {
    "paragraph": {"method": "paragraph", "size": 1000},
    "fixed": {"method": "fixed", "size": 1000, "overlap": 200},
}


@dataclass(frozen=True)
class Config:
    method: str  # bm25 | vector | hybrid
    chunks: str = "paragraph"
    embedder: str | None = None  # for vector / hybrid
    rerank: bool = False


CONFIGS: dict[str, Config] = {
    # the first run's setups, at k = 20
    "bm25": Config("bm25"),
    "vector-minilm": Config("vector", embedder="minilm"),
    "hybrid-minilm": Config("hybrid", embedder="minilm"),
    # a stronger embedding model
    "vector-bge": Config("vector", embedder="bge"),
    "hybrid-bge": Config("hybrid", embedder="bge"),
    # a cross-encoder reranker over the top 50
    "bm25+rerank": Config("bm25", rerank=True),
    "hybrid-bge+rerank": Config("hybrid", embedder="bge", rerank=True),
    # fixed-size overlapping chunks
    "bm25-fixed": Config("bm25", chunks="fixed"),
    "vector-minilm-fixed": Config("vector", chunks="fixed", embedder="minilm"),
    "hybrid-minilm-fixed": Config("hybrid", chunks="fixed", embedder="minilm"),
}


class Builder:
    """Builds the searchers, sharing chunks, models and vector indexes between configs."""

    def __init__(
        self,
        processed: Path,
        make_embedder: Callable[[dict[str, Any]], Any] | None = None,
        make_scorer: Callable[[], Any] | None = None,
    ):
        self.processed = processed
        self._make_embedder = make_embedder or _sentence_transformer
        self._make_scorer = make_scorer or _cross_encoder
        self._chunks: dict[str, list[Chunk]] = {}
        self._embedders: dict[str, Any] = {}
        self._vectors: dict[tuple[str, str], Searcher] = {}
        self._bm25: dict[str, Searcher] = {}
        self._scorer: Any = None

    def chunks(self, name: str) -> list[Chunk]:
        if name not in self._chunks:
            self._chunks[name] = chunk_corpus(self.processed, **CHUNKINGS[name])
        return self._chunks[name]

    def bm25(self, chunking: str) -> Searcher:
        from ragbench.search.bm25 import BM25Searcher

        if chunking not in self._bm25:
            self._bm25[chunking] = BM25Searcher(self.chunks(chunking))
        return self._bm25[chunking]

    def vector(self, embedder: str, chunking: str) -> Searcher:
        from ragbench.search.vector_search import VectorSearcher

        if embedder not in self._embedders:
            self._embedders[embedder] = self._make_embedder(EMBEDDERS[embedder])
        key = (embedder, chunking)
        if key not in self._vectors:
            self._vectors[key] = VectorSearcher(
                self.chunks(chunking), self._embedders[embedder], self.processed / "cache"
            )
        return self._vectors[key]

    def build(self, label: str, config: Config) -> Searcher:
        from ragbench.search.hybrid_search import HybridSearcher
        from ragbench.search.rerank_search import RerankSearcher

        if config.method == "bm25":
            searcher = self.bm25(config.chunks)
        elif config.method == "vector":
            searcher = self.vector(config.embedder or "minilm", config.chunks)
        elif config.method == "hybrid":
            searcher = HybridSearcher(
                [self.bm25(config.chunks), self.vector(config.embedder or "minilm", config.chunks)]
            )
        else:
            raise ValueError(f"unknown method {config.method!r}")
        if config.rerank:
            if self._scorer is None:
                self._scorer = self._make_scorer()
            searcher = RerankSearcher(searcher, self._scorer)
        return _Named(searcher, label)


class _Named:
    """A searcher under its configuration's label (the results file is <label>.jsonl)."""

    def __init__(self, inner: Searcher, name: str):
        self.inner, self.name = inner, name

    def search(self, query: str, k: int = 10):
        return self.inner.search(query, k=k)

    def __getattr__(self, attr: str):  # settings for run.json: k1, embedder, scorer, ...
        return getattr(self.inner, attr)


def _sentence_transformer(spec: dict[str, Any]) -> Any:
    from ragbench.search.vector_search import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(**spec)


def _cross_encoder() -> Any:
    from ragbench.search.rerank_search import CrossEncoderScorer

    return CrossEncoderScorer()


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

COLUMNS = (
    ("recall@5", "recall@5"),
    ("recall@10", "recall@10"),
    ("recall@20", "recall@20"),
    ("rr", "MRR"),
    ("paper_recall@all", "paper recall@20"),
)


def table(summary: dict[str, Any], configs: dict[str, Config]) -> list[str]:
    head = ["configuration", "chunks", *(label for _, label in COLUMNS), "time / q"]
    lines = ["| " + " | ".join(head) + " |", "|:---|:---|" + "---:|" * (len(COLUMNS) + 1)]
    for label, s in summary.items():
        cells = []
        for key, _ in COLUMNS:
            value = s["metrics"].get(key)
            cells.append("–" if value is None else f"{value:.2f}")
        per_q = s["seconds"] / (s["questions"] or 1)
        chunks = configs[label].chunks if label in configs else "?"
        lines.append(f"| {label} | {chunks} | " + " | ".join(cells) + f" | {per_q:.2f} s |")
    return lines


def render(run_dir: Path, info: dict[str, Any], configs: dict[str, Config]) -> str:
    n = len(info["inputs"]["questions"]["ids"])
    scored = next(iter(info["summary"].values()))["scored"] if info["summary"] else 0
    out = [
        f"### Retrieval sweep: `{run_dir.name}`",
        "",
        f"Retrieval only, k = {K}, on the {scored} of {n} questions that have evidence to "
        "find (negation and unanswerable questions have none). No answers, no API cost.",
        "",
        *table(info["summary"], configs),
        "",
        "recall@k: share of gold pages in the top k. MRR: 1 / rank of the first gold page. "
        "paper recall@20: share of gold papers anywhere in the top 20. Time includes "
        "embedding the query and reranking, not building the index.",
    ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def main_sweep(args: Any, builder: Builder | None = None) -> int:
    from ragbench.eval.questions import QuestionSetError, load
    from ragbench.eval.run import _split

    names = _split(args.configs) or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        print(
            f"error: unknown configuration(s) {', '.join(unknown)}; "
            f"choose from {', '.join(CONFIGS)}",
            file=sys.stderr,
        )
        return 1
    manifest = args.processed / "manifest.jsonl"
    if not manifest.exists():
        print(f"error: {manifest} not found; run ragbench-ingest first", file=sys.stderr)
        return 1
    try:
        questions = load(args.question_file).questions
    except (FileNotFoundError, QuestionSetError) as exc:
        problems = exc.problems if isinstance(exc, QuestionSetError) else [str(exc)]
        print("\n".join(f"error: {p}" for p in problems), file=sys.stderr)
        return 1

    builder = builder or Builder(args.processed)
    configs = {n: CONFIGS[n] for n in names}
    try:
        searchers = {}
        for name, config in configs.items():
            print(f"building {name} ...", file=sys.stderr)
            searchers[name] = builder.build(name, config)
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    arxiv_of = {m["doc_id"]: m.get("arxiv_id") or m["doc_id"] for m in entries}
    config = {
        "sweep": {n: vars(c) for n, c in configs.items()},
        "k": K,
        "chunkings": CHUNKINGS,
        "searchers": {n: describe(s) for n, s in searchers.items()},
    }
    inputs = {
        "questions": {
            "path": str(args.question_file),
            "sha256": sha256_file(args.question_file),
            "ids": [q.id for q in questions],
        },
        "manifest_sha256": sha256_file(manifest),
    }
    out_dir = new_run_dir(args.out, datetime.now(UTC), prefix="sweep-")
    info = {
        "config": config,
        "inputs": inputs,
        "config_hash": config_hash({"config": config, "inputs": inputs}),
        "command": ["ragbench", *sys.argv[1:]],
        "git": git_info(),
        "packages": package_versions(),
    }
    print(f"sweep {out_dir.name}: {len(configs)} configuration(s) x {len(questions)} question(s)")
    try:
        run_info = run_benchmark(
            questions,
            searchers,
            k=K,
            arxiv_of=arxiv_of,
            out_dir=out_dir,
            info=info,
            progress=lambda line: print(f"  {line}", file=sys.stderr),
        )
    except KeyboardInterrupt:
        print(f"\ninterrupted; partial results in {out_dir}", file=sys.stderr)
        return 130
    text = render(out_dir, run_info, configs)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\nsaved to {out_dir}")
    return 0 if run_info["status"] == "finished" else 2
