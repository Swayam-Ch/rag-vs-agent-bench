### Agent by model

| model | runs | accuracy per run | mean | 95% interval | max_turns stops | cost / q | time / q |
|:---|---:|:---|---:|:---:|---:|---:|---:|
| claude-haiku-5-5 | 3 | 0.97, 0.99, 0.97 | 0.98 | 0.95 to 1.00 | 0 | $0.004 | 11.1 s |
| claude-sonnet-5-5 | 2 | 1.00, 1.00 | 1.00 | 1.00 to 1.00 | 0 | $0.043 | 10.5 s |

#### Accuracy by question type (mean over runs)

| question type | claude-haiku-5-5 | claude-sonnet-5-5 |
|:---|---:|---:|
| single_fact (5) | 1.00 | 1.00 |
| multi_filter (7) | 0.97 | 1.00 |
| cross_doc (1) | 1.00 | 1.00 |
| negation (6) | 0.96 | 1.00 |
| unanswerable (1) | 1.00 | 1.00 |

#### Paired differences (points averaged over each model's runs)

| comparison | difference | 95% interval | better / same / worse | clear? |
|:---|---:|:---:|:---:|:---:|
| claude-haiku-5-5 vs claude-sonnet-5-5 | -0.02 | -0.05 to -0.00 | 0 / 16 / 4 | yes |

Run folders: claude-haiku-5-5: 2026-10-26T153448Z, 2026-10-26T154432Z, 2026-10-26T155449Z; claude-sonnet-5-5: 2026-10-26T153950Z, 2026-10-26T154949Z
