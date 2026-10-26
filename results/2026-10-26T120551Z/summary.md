### Results: `2026-10-26T120551Z`

| method | recall@5 | MRR | paper recall | accuracy | cost / q | time / q |
|:---|---:|---:|---:|---:|---:|---:|
| **bm25** | 0.61 | 0.58 | 0.88 | 0.66 | $0.012 | 3.9 s |
| **vector** | 0.48 | 0.49 | 0.87 | 0.55 | $0.011 | 4.2 s |
| **hybrid** | 0.56 | 0.55 | 0.93 | 0.65 | $0.011 | 3.9 s |
| **agent** | 0.45 | 0.40 | 1.00 | 1.00 | $0.045 | 11.0 s |

recall@5, MRR: the gold *pages* in the top 5 / rank of the first one (questions with evidence to find). Paper recall: share of gold papers found in everything a method returned. Accuracy: judged answers; F1 of the papers named for list questions, else correct 1, partial 0.5, wrong 0. Cost and time per question include writing the answer, not judging it.

#### Accuracy by question type

| question type | bm25 | vector | hybrid | agent |
|:---|---:|---:|---:|---:|
| single_fact (5) | 1.00 | 0.80 | 0.80 | 1.00 |
| multi_filter (7) | 0.75 | 0.54 | 0.84 | 1.00 |
| cross_doc (1) | 0.50 | 0.50 | 0.50 | 1.00 |
| negation (6) | 0.24 | 0.30 | 0.28 | 1.00 |
| unanswerable (1) | 1.00 | 1.00 | 1.00 | 1.00 |

#### Accuracy by question author

| written by | bm25 | vector | hybrid | agent |
|:---|---:|---:|---:|---:|
| swayam (2) | 1.00 | 0.50 | 0.50 | 1.00 |
| claude (18) | 0.62 | 0.56 | 0.67 | 1.00 |

#### Run details

- run: `2026-10-26T12:06:01+00:00`, status **finished**, commit `eb37542`, config `0e31feb80e3f`
- questions: 20 from `eval/questions.yaml`, k = 10
- embeddings: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- agent: `claude-sonnet-5-5`, at most 15 turns
- answers written by `claude-sonnet-5-5` from the top-k passages (agent: its own); judged by `claude-opus-5-5` (judge-v2), judging cost $0.759
- problems: hybrid: check q014 by hand
