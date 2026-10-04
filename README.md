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
ragbench-fetch data/sources.txt data/raw --verify   # prove you have the exact same PDFs
ragbench-ingest data/raw data/processed -v
```

## Methodology notes

- **No front matter in searchable text.** Metadata lives in `manifest.jsonl`, so no method is helped or hurt by boilerplate tokens.
- **Bibliographies are stripped.** Reference lists repeat paper titles and keywords from across the field, which creates false matches for every method. They're cut at ingestion (appendices are kept); `manifest.jsonl` records how many characters were removed per paper so this can be audited.
- **Running headers and page numbers are stripped.** A line at the top or bottom of at least half the pages (e.g. *Published as a conference paper at ICLR 2023*) is boilerplate that would otherwise land in many chunks. Mid-page lines are never touched; the manifest records `boilerplate_lines_removed`.
- **Page offsets are recorded** so answers can be checked against the page they cite.
- **Pinned inputs.** `data/checksums.sha256` records the SHA-256 of every PDF. Any run on different bytes fails loudly instead of silently producing different numbers.
- **Metadata is a committed snapshot.** Titles, authors, dates and categories come from the arXiv API once, at download time, and live in `data/metadata.jsonl`. Ingestion never touches the network.
- **Deterministic IDs.** A document's ID is a hash of its bytes, so reruns line up.

## Layout

```
data/sources.txt         # the corpus: 20 arXiv IDs (PDFs themselves are not committed)  ✅
data/checksums.sha256    # SHA-256 of every PDF, checked on each run                     ✅
data/metadata.jsonl      # title, authors, date, category per paper (from the arXiv API) ✅
src/ragbench/
  pipeline/fetch.py      # download the PDFs listed in sources.txt                       ✅
  pipeline/checksums.py  # record / verify file hashes                                   ✅
  pipeline/metadata.py   # arXiv API client + metadata snapshot                          ✅
  pipeline/ingest.py     # PDF/txt/md -> clean markdown + manifest                       ✅
  pipeline/chunk.py      #                                                               ⏳
  search/                # vector, hybrid, agent backends                                ⏳
eval/                    # questions.yaml + run_eval.py                                  ⏳
results/                 # committed run outputs                                         ⏳
tests/
```

## License

MIT
