### Results: `2026-10-26T161300Z`

| method | recall@5 | MRR | paper recall | accuracy | cost / q | time / q |
|:---|---:|---:|---:|---:|---:|---:|
| **hybrid-bge+rerank** | 0.66 | 0.77 | 0.93 | 0.62 | $0.012 | 4.1 s |

recall@5, MRR: the gold *pages* in the top 5 / rank of the first one (questions with evidence to find). Paper recall: share of gold papers found in everything a method returned. Accuracy: judged answers; F1 of the papers named for list questions, else correct 1, partial 0.5, wrong 0. Cost and time per question include writing the answer, not judging it.

#### Accuracy by question type

| question type | hybrid-bge+rerank |
|:---|---:|
| single_fact (5) | 1.00 |
| multi_filter (7) | 0.74 |
| cross_doc (1) | 0.50 |
| negation (6) | 0.11 |
| unanswerable (1) | 1.00 |

#### Accuracy by question author

| written by | hybrid-bge+rerank |
|:---|---:|
| swayam (2) | 1.00 |
| llm (18) | 0.58 |

#### Run details

- run: `2026-10-26T16:13:00+00:00`, status **finished**, commit `cef77d2`, config `3be3c1d9a546`
- questions: 20 from `eval/questions.yaml`, k = 10
- answers written by `claude-sonnet-5-5` from the top-k passages (agent: its own); judged by `claude-opus-5-5` (judge-v2), judging cost $0.185
- problems: hybrid-bge+rerank: check q015 by hand
