### Results: `2026-10-26T153448Z`

| method | recall@5 | MRR | paper recall | accuracy | cost / q | time / q |
|:---|---:|---:|---:|---:|---:|---:|
| **agent** | 0.14 | 0.26 | 0.98 | 0.97 | $0.0038 | 11.1 s |

recall@5, MRR: the gold *pages* in the top 5 / rank of the first one (questions with evidence to find). Paper recall: share of gold papers found in everything a method returned. Accuracy: judged answers; F1 of the papers named for list questions, else correct 1, partial 0.5, wrong 0. Cost and time per question include writing the answer, not judging it.

#### Accuracy by question type

| question type | agent |
|:---|---:|
| single_fact (5) | 1.00 |
| multi_filter (7) | 0.95 |
| cross_doc (1) | 1.00 |
| negation (6) | 0.94 |
| unanswerable (1) | 1.00 |

#### Accuracy by question author

| written by | agent |
|:---|---:|
| swayam (2) | 1.00 |
| llm (18) | 0.96 |

#### Run details

- run: `2026-10-26T15:34:48+00:00`, status **finished**, commit `cef77d2`, config `85b975b239cf`
- questions: 20 from `eval/questions.yaml`, k = 10
- agent: `claude-haiku-5-5`, at most 15 turns
- answers written by `claude-haiku-5-5` from the top-k passages (agent: its own); judged by `claude-opus-5-5` (judge-v2), judging cost $0.204
- problems: none
