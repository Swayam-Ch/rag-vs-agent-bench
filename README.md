# rag-vs-agent-bench

[![CI](https://github.com/Swayam-Ch/rag-vs-agent-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Swayam-Ch/rag-vs-agent-bench/actions/workflows/ci.yml)

**Does agentic file search beat vector search?** A reproducible benchmark on public data.

This repo compares three ways of answering questions over a document corpus:

| Method  | How it finds context |
|---------|----------------------|
| Vector  | embed chunks, cosine similarity |
| Hybrid  | vector + BM25, fused |
| Agent   | an LLM with `list` / `grep` / `read` tools over the raw markdown |

All three search the *same* ingested text, answer the *same* versioned question set, and every run's raw output is committed under `results/`.

> Status: 🚧 week 1 — ingestion pipeline. No findings yet.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# download the corpus defined in data/sources.txt (~1 min, rate-limited), then ingest it
ragbench-fetch data/sources.txt data/raw -v
ragbench-ingest data/raw data/processed -v
```

## Methodology notes

- **No front matter in searchable text.** Metadata lives in `manifest.jsonl`, so no method is helped or hurt by boilerplate tokens.
- **Page offsets are recorded** so answers can be checked against the page they cite.
- **Deterministic IDs.** A document's ID is a hash of its bytes, so reruns line up.

## Layout

```
data/sources.txt       # the corpus: 20 arXiv IDs (PDFs themselves are not committed)  ✅
src/ragbench/
  pipeline/fetch.py    # download the PDFs listed in sources.txt                   ✅
  pipeline/ingest.py   # PDF/txt/md -> clean markdown + manifest   ✅
  pipeline/chunk.py    #                                            ⏳
  search/              # vector, hybrid, agent backends             ⏳
eval/                  # questions.yaml + run_eval.py               ⏳
results/               # committed run outputs                      ⏳
tests/
```

## License

MIT
