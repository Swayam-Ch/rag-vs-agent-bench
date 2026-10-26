### How sure are the numbers? `2026-10-26T120551Z`

95% bootstrap intervals over the judged questions (10,000 resamples, seed 0). With this few questions, read the intervals, not the decimals.

| method | accuracy | 95% interval | questions |
|:---|---:|:---:|---:|
| bm25 | 0.66 | 0.49 to 0.82 | 20 |
| vector | 0.55 | 0.38 to 0.72 | 20 |
| hybrid | 0.65 | 0.48 to 0.81 | 20 |
| agent | 1.00 | 1.00 to 1.00 | 20 |

Paired differences: both methods on the same questions. *Clear* means the interval excludes 0.

| comparison | difference | 95% interval | better / same / worse | clear? |
|:---|---:|:---:|:---:|:---:|
| bm25 vs vector | +0.11 | -0.05 to +0.27 | 6 / 12 / 2 | no |
| bm25 vs hybrid | +0.01 | -0.10 to +0.14 | 2 / 14 / 4 | no |
| agent vs bm25 | +0.34 | +0.18 to +0.51 | 12 / 8 / 0 | yes |
| vector vs hybrid | -0.10 | -0.21 to -0.01 | 1 / 12 / 7 | yes |
| agent vs vector | +0.45 | +0.28 to +0.62 | 14 / 6 / 0 | yes |
| agent vs hybrid | +0.35 | +0.19 to +0.52 | 13 / 7 / 0 | yes |

#### The agent's citations

Each page in an answer's "Sources:" list, checked against the gold evidence and against the pages the agent was shown (grep snippets and read_file ranges). Unanswerable questions are skipped; negation questions are left out of the "shown" check, because evidence of absence is a grep with no matches.

- citations on a gold page: 56 of 106 (53%); of the rest, 25 are other pages of a gold paper and 25 are papers outside the answer (in list questions, mostly the near-misses an answer cites to rule them out)
- gold evidence entries the answers cited: 45 of 57 (79%)
- citations of a page the agent had been shown: 64 of 64 (100%)

Citations only cover what an answer cites. A claim made from memory without a citation (as in q020, where the agent wrote that it relied on what it knew of six papers) is not caught by this check.
