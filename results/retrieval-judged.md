### Judged runs on the current question set

| setup | runs | accuracy per run | mean | 95% interval | recall@5 | MRR | cost / q |
|:---|---:|:---|---:|:---:|---:|---:|---:|
| hybrid-bge+rerank | 1 | 0.62 | 0.62 | 0.44 to 0.79 | 0.66 | 0.77 | $0.012 |
| bm25 | 1 | 0.64 | 0.64 | 0.47 to 0.79 | 0.61 | 0.58 | $0.012 |
| agent (claude-haiku-5-5) | 3 | 0.97, 0.99, 0.97 | 0.98 | 0.95 to 1.00 | 0.17 | 0.28 | $0.004 |
| agent (claude-sonnet-5-5) | 2 | 1.00, 1.00 | 1.00 | 1.00 to 1.00 | 0.45 | 0.45 | $0.043 |

#### Accuracy by question type (mean over runs)

| question type | hybrid-bge+rerank | bm25 | agent (claude-haiku-5-5) | agent (claude-sonnet-5-5) |
|:---|---:|---:|---:|---:|
| single_fact (5) | 1.00 | 0.90 | 1.00 | 1.00 |
| multi_filter (7) | 0.74 | 0.75 | 0.97 | 1.00 |
| cross_doc (1) | 0.50 | 0.50 | 1.00 | 1.00 |
| negation (6) | 0.11 | 0.24 | 0.96 | 1.00 |
| unanswerable (1) | 1.00 | 1.00 | 1.00 | 1.00 |

#### Paired differences

| comparison | difference | 95% interval | better / same / worse | clear? |
|:---|---:|:---:|:---:|:---:|
| hybrid-bge+rerank vs bm25 | -0.02 | -0.18 to +0.13 | 6 / 9 / 5 | no |
| hybrid-bge+rerank vs agent (claude-haiku-5-5) | -0.36 | -0.53 to -0.20 | 0 / 8 / 12 | yes |
| hybrid-bge+rerank vs agent (claude-sonnet-5-5) | -0.38 | -0.56 to -0.21 | 0 / 8 / 12 | yes |
| bm25 vs agent (claude-haiku-5-5) | -0.34 | -0.50 to -0.19 | 1 / 6 / 13 | yes |
| bm25 vs agent (claude-sonnet-5-5) | -0.36 | -0.53 to -0.21 | 0 / 7 / 13 | yes |
| agent (claude-haiku-5-5) vs agent (claude-sonnet-5-5) | -0.02 | -0.05 to -0.00 | 0 / 16 / 4 | yes |

Runs: hybrid-bge+rerank: 2026-10-26T161300Z; bm25: 2026-10-26T162319Z; agent (claude-haiku-5-5): 2026-10-26T153448Z, 2026-10-26T154432Z, 2026-10-26T155449Z; agent (claude-sonnet-5-5): 2026-10-26T153950Z, 2026-10-26T154949Z
