# rag-vs-agent-bench

[![CI](https://github.com/Swayam-Ch/rag-vs-agent-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Swayam-Ch/rag-vs-agent-bench/actions/workflows/ci.yml)

**Does agentic file search beat vector search?** A reproducible benchmark on public data.

> **Finding:** on 20 questions about 20 ML papers, an LLM agent that searches the papers
> with `grep` and `read` answered all 20 correctly (accuracy 1.00). Retrieval-augmented
> answers from the same model scored 0.55 (vector), 0.65 (hybrid) and 0.66 (BM25). The
> agent cost about 4x as much per question and took about 3x as long. Most of the gap is
> on questions that need *every* paper checked: "which papers …" lists, and above all
> "which papers do NOT …" (0.24–0.30 without the agent). On single facts, BM25 matched
> the agent. These are small numbers on
> famous papers with LLM-drafted questions: read [Limitations](#limitations) before
> quoting them.

The benchmark compares these ways of answering questions over a document collection:

| Method  | How it finds context |
|---------|----------------------|
| Keyword | count of query words in each chunk (a floor, not a contender) |
| BM25    | classic lexical ranking, implemented from scratch |
| Vector  | embed chunks, cosine similarity |
| Hybrid  | BM25 + vector, fused by reciprocal rank |
| Agent   | an LLM with `list_files` / `grep` / `read_file` tools over the raw text |

All of them search the *same* ingested text, answer the *same* versioned question set,
and every run's raw output is committed under `results/`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",vector" for vector search (installs PyTorch),
                                   # ",agent" for agentic search (needs ANTHROPIC_API_KEY)
pytest

# download the corpus defined in data/sources.txt (~1 min, rate-limited), then ingest it
ragbench-fetch data/sources.txt data/raw -v
ragbench-fetch data/sources.txt data/raw --verify   # prove you have the exact same PDFs
ragbench-ingest data/raw data/processed -v
ragbench-chunk data/processed                       # -> data/processed/chunks.paragraph.1000.jsonl
ragbench-chunk data/processed --method fixed        # -> chunks.fixed.1000-200.jsonl

# search with any method
ragbench search "how many attention heads does the Transformer use?"            # BM25
ragbench search -m vector -k 3 "dense retrieval for open-domain QA"            # needs [vector]
ragbench search -m hybrid "LoRA rank"
ragbench search -m agent "which papers evaluate on Natural Questions?"          # needs [agent]
ragbench search -m bm25 --json "BM25 baseline" > results.json

# check the question set (also runs in CI)
ragbench questions

# run the benchmark: every method on every question -> results/<UTC time>/
ragbench run                                        # keyword, bm25, vector, hybrid
ragbench run --methods bm25,agent                   # the agent costs money: opt in
ragbench run --methods bm25 --questions q001,q006 -k 5
ragbench run --methods bm25,vector,hybrid,agent --judge   # + graded answers (costs money)
ragbench summarize results/<run>                    # markdown tables for the README
ragbench chart results/<run>                        # accuracy_by_type.png (needs [plot])
```

Each run gets its own folder: `run.json` (config, input checksums, git commit, package
and model versions, a summary per method) plus one `<method>.jsonl` with a row per
question (ranked results with pages, retrieval metrics, timing; for the agent also the
answer, tool calls, tokens and cost). Runs with the same `config_hash` are comparable.

With `--judge`, every method also has to *answer*. The chunk-based methods get an answer
written from their top-k passages by the agent's model, so only retrieval differs. An LLM
judge (Opus by default, a different model from the writer) grades each answer against
the reference as correct, partial or wrong, with a one-line reason. "Which papers"
questions are scored by F1 of the papers named against the gold papers (computed in
code from the papers the judge extracts), so 1 of 10 and 9 of 10 no longer both count
as "partial". Answers that say "the papers don't say" are counted apart from confident
wrong answers. Answer and judge tokens, time and cost are recorded per question. The rubric, its pitfalls (self-
preference, verbosity, the judge's own knowledge) and their mitigations are in
`src/ragbench/eval/grade.py`; spot-check the verdicts' reasons before trusting them.

## Results

First full run ([`results/2026-10-26T120551Z`](results/2026-10-26T120551Z/summary.md)),
from commit `eb37542`: 4 methods x 20 questions, top 10 passages per question, answers
graded by an LLM judge. The chunk-based methods' answers were written by the agent's
model (`claude-sonnet-5-5`) from their passages, so only retrieval differs between them.
Full tables, per-type and per-author breakdowns are in the run's `summary.md`.

![Accuracy by question type, per method](results/2026-10-26T120551Z/accuracy_by_type.png)

| method | recall@5 | MRR | paper recall | accuracy | cost / q | time / q |
|:---|---:|---:|---:|---:|---:|---:|
| **bm25** | 0.61 | 0.58 | 0.88 | 0.66 | $0.012 | 3.9 s |
| **vector** | 0.48 | 0.49 | 0.87 | 0.55 | $0.011 | 4.2 s |
| **hybrid** | 0.56 | 0.55 | 0.93 | 0.65 | $0.011 | 3.9 s |
| **agent** | 0.45 | 0.40 | 1.00 | 1.00 | $0.045 | 11.0 s |

recall@5 and MRR: did the gold *page* appear in the top 5, and how high was the first one.
Paper recall: the share of gold papers found anywhere in what a method returned.
Accuracy: correct 1, partial 0.5, wrong 0, except "which papers" questions, which score
the F1 of the papers named. Cost and time are per question and include writing the
answer; judging the run cost another $0.76.

### What the numbers show

1. **Exhaustive questions are where top-k retrieval breaks.** On "which papers do NOT
   …" (6 questions), BM25, vector and hybrid scored 0.24–0.30. Ten passages can't show
   that a paper *lacks* something; the agent greps every paper and gets 1.00. Questions
   that need two papers' tables at once (q008) fail the same way: one search can't fetch
   both.
2. **For a single fact, BM25 is enough.** BM25 answered all 5 single-fact questions
   correctly, like the agent, at a quarter of the cost. Vector and hybrid got 4 of 5.
3. **BM25 ranks best; hybrid finds the most.** The questions share exact terms with the
   papers, which BM25 rewards (MRR 0.58 against vector's 0.49). Hybrid found the most gold
   papers (0.93) but often put a near-miss page first.
4. **The agent's rank metrics are the lowest, and that's expected.** Its "ranking" is
   the order in which it searched, not relevance; paper recall (1.00) is its meaningful
   retrieval number. It answered from `grep` snippets alone on 17 of 20 questions;
   only 3 times did it open a paper with `read_file`.
5. **Failing honestly vs failing confidently.** Of the 14 answers that scored 0, 10
   were abstentions ("the passages don't contain this") and 4 were confident wrong
   answers (1–2 per chunk-based method). The agent scored 0 on none.

### Limitations

- **Small.** 20 papers and 20 questions. One question moves accuracy by 0.05, and two
  question types (cross_doc, unanswerable) have a single question each. One run per
  method: the agent and the judge are not deterministic, and no variance is measured.
- **Famous papers.** The corpus is well-known ML papers that the model has read. The
  agent sometimes says it is answering from memory (q020: "I did not grep these papers.
  I'm relying on what I know of them"), and the judge counts a correct remembered fact
  as correct. A corpus the model has never seen (internal documents, papers newer than
  its training data) is the real test.
- **Question bias.** 18 of 20 questions were drafted with an LLM by grepping the corpus,
  the agent's own strategy, then verified by hand (`author: llm`). Only 2 are
  hand-written, too few for the by-author split to say anything yet.
- **LLM judge.** The judge (`claude-opus-5-5`) is from the same model family as the
  writer of every answer. The rubric grades facts against a human reference, and every
  verdict has a reason that was spot-checked, but judge bias isn't measured.
- **One configuration.** One answering model, one embedding model (all-MiniLM-L6-v2),
  one chunking (paragraphs, 1000 characters), k = 10. A larger embedding model, a
  reranker or more passages could close part of the gap; the agent was not tuned either.
- **Keyword is missing from the committed run.** A later run that included it ran out
  of API credit partway and was discarded; its retrieval-only numbers are free to
  reproduce with `ragbench run --methods keyword`.
- **Known wording issue.** q016 and q019 say "(see q006)", which methods can't see; the
  agent noted it and answered correctly anyway.

## Reproduce the results

Retrieval metrics (recall, MRR) are deterministic. Answers and verdicts come from LLMs
and can differ a little between runs. A full judged run costs about $2.50 in API usage
(prices checked 2026-10-21) and takes about 10 minutes.

```bash
git clone https://github.com/Swayam-Ch/rag-vs-agent-bench && cd rag-vs-agent-bench
git checkout eb37542            # the commit the committed run was made from
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,vector,agent]"

ragbench-fetch data/sources.txt data/raw -v
ragbench-fetch data/sources.txt data/raw --verify   # same PDFs, byte for byte
ragbench-ingest data/raw data/processed -v
ragbench-chunk data/processed                       # paragraph chunks, 1000 characters
ragbench questions                                  # 20 questions OK

export ANTHROPIC_API_KEY=...                        # and ANTHROPIC_WORKSPACE_ID if your
                                                    # key spans several workspaces
ragbench run --methods bm25,vector,hybrid,agent --judge

git checkout main               # the reporting commands came after the run
pip install -e ".[dev,vector,agent,plot]"
ragbench summarize results/<new run>                # compare with the committed summary.md
ragbench chart results/<new run>
```

Run at that commit, the new `run.json` should show the same `config_hash` as the
committed one (`0e31feb80e3f`): it fingerprints the settings, the models and the
checksums of the question file and the chunks. If it differs, an input differs; the
usual cause is a different PDF-extraction library version (the committed `run.json` lists
every package version, e.g. pypdf 6.19.0).

## Methodology notes

- **No front matter in searchable text.** Metadata lives in `manifest.jsonl`, so no method is helped or hurt by boilerplate tokens.
- **Bibliographies are stripped.** Reference lists repeat paper titles and keywords from across the field, which creates false matches for every method. They're cut at ingestion (appendices are kept); `manifest.jsonl` records how many characters were removed per paper so this can be audited.
- **Running headers and page numbers are stripped.** A line at the top or bottom of at least half the pages (e.g. *Published as a conference paper at ICLR 2023*) is boilerplate that would otherwise land in many chunks. Mid-page lines are never touched; the manifest records `boilerplate_lines_removed`.
- **Paragraphs come from the line layout.** PDF extractors rarely leave blank lines between paragraphs, so a short line that ends a sentence, followed by one that starts a new block, or a section heading, marks a paragraph break. Without this, every page was one "paragraph".
- **Page offsets are recorded** so answers can be checked against the page they cite.
- **Pinned inputs.** `data/checksums.sha256` records the SHA-256 of every PDF. Any run on different bytes fails loudly instead of silently producing different numbers.
- **Metadata is a committed snapshot.** Titles, authors, dates and categories come from the arXiv API once, at download time, and live in `data/metadata.jsonl`. Ingestion never touches the network.
- **One contract for every method.** Each search backend returns results that point into the documents by position (document, character range, pages), and passes the same contract tests: best first, deterministic ties, no duplicates, results that match the source text. The agent reports the ranges it read, so it can be scored with the same metrics as the chunk-based methods.
- **Deterministic IDs.** A document's ID is a hash of its bytes, so reruns line up.

## Layout

```
data/sources.txt         # the corpus: 20 arXiv IDs (PDFs themselves are not committed)  ✅
data/checksums.sha256    # SHA-256 of every PDF, checked on each run                     ✅
data/metadata.jsonl      # title, authors, date, category per paper (from the arXiv API) ✅
src/ragbench/
  cli.py                 # `ragbench search --method {keyword,bm25,vector,hybrid,agent}` ✅
  pipeline/fetch.py      # download the PDFs listed in sources.txt                       ✅
  pipeline/checksums.py  # record / verify file hashes                                   ✅
  pipeline/metadata.py   # arXiv API client + metadata snapshot                          ✅
  pipeline/ingest.py     # PDF/txt/md -> clean markdown + manifest                       ✅
  pipeline/chunk.py      # fixed windows or whole paragraphs, page-aware                 ✅
  search/base.py         # SearchResult, Searcher contract, keyword baseline             ✅
  search/bm25.py         # BM25 from scratch, cross-checked against rank_bm25            ✅
  search/vector_search.py # embeddings + cosine similarity, cached on disk               ✅
  search/hybrid_search.py # reciprocal rank fusion of any searchers (BM25 + vector)     ✅
  search/agent_tools.py  # list_files / grep / read_file: safe, bounded, logged          ✅
  search/agent_search.py # tool-calling agent loop: limits, transcripts, token usage     ✅
eval/questions.yaml      # the answer key: 20 questions, gold papers + pages (see below)  ✅
  src/ragbench/eval/questions.py # format + validator (`ragbench questions`)            ✅
  src/ragbench/eval/metrics.py   # recall@k, MRR, page hits (page and paper level)     ✅
  src/ragbench/eval/run.py       # `ragbench run`: every method x question, saved      ✅
  src/ragbench/eval/grade.py     # answers from passages + LLM judge (`--judge`)         ✅
  src/ragbench/eval/summarize.py # `ragbench summarize`: markdown tables from a run      ✅
  src/ragbench/eval/chart.py     # `ragbench chart`: accuracy by question type (PNG)    ✅
results/                 # committed run outputs (agent transcripts stay local)         ✅
tests/
```

**Where the questions come from.** q001–q002 were written by Swayam. q003–q020 were
*drafted with an LLM and verified by Swayam* (`author: llm`): drafted from the ingested
text with a quote and page for each, then checked by hand against the PDFs. The same
model family powers the agent under test, so the questions lean on details that are hard
to know without reading this collection (setup details, table values, counts across
papers), and every run reports accuracy by author so the effect of who wrote a question
stays visible.

## License

MIT
