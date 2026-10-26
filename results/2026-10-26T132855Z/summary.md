### Results: `2026-10-26T132855Z`

| method | recall@5 | MRR | paper recall | accuracy | cost / q | time / q |
|:---|---:|---:|---:|---:|---:|---:|
| **keyword** | 0.00 | 0.01 | 0.39 | – | $0.0000 | 0.0 s |

recall@5, MRR: the gold *pages* in the top 5 / rank of the first one (questions with evidence to find). Paper recall: share of gold papers found in everything a method returned. Accuracy: judged answers; F1 of the papers named for list questions, else correct 1, partial 0.5, wrong 0. Cost and time per question include writing the answer, not judging it.

#### Run details

- run: `2026-10-26T13:28:55+00:00`, status **finished**, commit `a128bbf`, config `3ea39deeb0dd`
- questions: 20 from `eval/questions.yaml`, k = 10
- problems: none
