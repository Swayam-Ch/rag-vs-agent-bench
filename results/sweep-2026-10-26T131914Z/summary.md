### Retrieval sweep: `sweep-2026-10-26T131914Z`

Retrieval only, k = 20, on the 13 of 20 questions that have evidence to find (negation and unanswerable questions have none). No answers, no API cost.

| configuration | chunks | recall@5 | recall@10 | recall@20 | MRR | paper recall@20 | time / q |
|:---|:---|---:|---:|---:|---:|---:|---:|
| bm25 | paragraph | 0.61 | 0.68 | 0.75 | 0.58 | 0.96 | 0.00 s |
| vector-minilm | paragraph | 0.48 | 0.61 | 0.76 | 0.50 | 0.98 | 0.02 s |
| hybrid-minilm | paragraph | 0.56 | 0.68 | 0.78 | 0.56 | 0.99 | 0.02 s |
| vector-bge | paragraph | 0.54 | 0.64 | 0.78 | 0.62 | 0.94 | 0.07 s |
| hybrid-bge | paragraph | 0.56 | 0.74 | 0.79 | 0.67 | 0.95 | 0.03 s |
| bm25+rerank | paragraph | 0.65 | 0.76 | 0.79 | 0.76 | 0.97 | 0.51 s |
| hybrid-bge+rerank | paragraph | 0.66 | 0.80 | 0.83 | 0.77 | 0.98 | 0.57 s |
| bm25-fixed | fixed | 0.62 | 0.76 | 0.84 | 0.62 | 0.95 | 0.01 s |
| vector-minilm-fixed | fixed | 0.51 | 0.63 | 0.80 | 0.63 | 0.96 | 0.01 s |
| hybrid-minilm-fixed | fixed | 0.58 | 0.67 | 0.84 | 0.59 | 0.95 | 0.02 s |

recall@k: share of gold pages in the top k. MRR: 1 / rank of the first gold page. paper recall@20: share of gold papers anywhere in the top 20. Time includes embedding the query and reranking, not building the index.
