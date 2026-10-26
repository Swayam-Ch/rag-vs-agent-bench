### Results: `2026-10-26T162319Z`

| method | recall@5 | MRR | paper recall | accuracy | cost / q | time / q |
|:---|---:|---:|---:|---:|---:|---:|
| **bm25** | 0.61 | 0.58 | 0.88 | 0.64 | $0.012 | 4.3 s |

recall@5, MRR: the gold *pages* in the top 5 / rank of the first one (questions with evidence to find). Paper recall: share of gold papers found in everything a method returned. Accuracy: judged answers; F1 of the papers named for list questions, else correct 1, partial 0.5, wrong 0. Cost and time per question include writing the answer, not judging it.

#### Accuracy by question type

| question type | bm25 |
|:---|---:|
| single_fact (5) | 0.90 |
| multi_filter (7) | 0.75 |
| cross_doc (1) | 0.50 |
| negation (6) | 0.24 |
| unanswerable (1) | 1.00 |

#### Accuracy by question author

| written by | bm25 |
|:---|---:|
| swayam (2) | 0.75 |
| llm (18) | 0.62 |

#### Run details

- run: `2026-10-26T16:23:19+00:00`, status **finished**, commit `cef77d2`, config `0fe5bb1c758d`
- questions: 20 from `eval/questions.yaml`, k = 10
- answers written by `claude-sonnet-5-5` from the top-k passages (agent: its own); judged by `claude-opus-5-5` (judge-v2), judging cost $0.185
- problems: bm25: check q014 by hand
