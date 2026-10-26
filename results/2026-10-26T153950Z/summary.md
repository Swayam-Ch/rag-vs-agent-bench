### Results: `2026-10-26T153950Z`

| method | recall@5 | MRR | paper recall | accuracy | cost / q | time / q |
|:---|---:|---:|---:|---:|---:|---:|
| **agent** | 0.45 | 0.43 | 0.97 | 1.00 | $0.044 | 10.3 s |

recall@5, MRR: the gold *pages* in the top 5 / rank of the first one (questions with evidence to find). Paper recall: share of gold papers found in everything a method returned. Accuracy: judged answers; F1 of the papers named for list questions, else correct 1, partial 0.5, wrong 0. Cost and time per question include writing the answer, not judging it.

#### Accuracy by question type

| question type | agent |
|:---|---:|
| single_fact (5) | 1.00 |
| multi_filter (7) | 1.00 |
| cross_doc (1) | 1.00 |
| negation (6) | 1.00 |
| unanswerable (1) | 1.00 |

#### Accuracy by question author

| written by | agent |
|:---|---:|
| swayam (2) | 1.00 |
| llm (18) | 1.00 |

#### Run details

- run: `2026-10-26T15:39:50+00:00`, status **finished**, commit `cef77d2`, config `2b035d9c4391`
- questions: 20 from `eval/questions.yaml`, k = 10
- agent: `claude-sonnet-5-5`, at most 15 turns
- answers written by `claude-sonnet-5-5` from the top-k passages (agent: its own); judged by `claude-opus-5-5` (judge-v2), judging cost $0.202
- problems: none
