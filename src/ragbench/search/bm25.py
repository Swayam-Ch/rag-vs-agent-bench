"""BM25: the standard lexical (word-matching) ranking function, written from scratch.

For a query q and a chunk d, BM25 adds up one score per query word t:

    score(q, d) = sum over t in q of  idf(t) * tf(t, d) * (k1 + 1)
                                       -------------------------------------------
                                       tf(t, d) + k1 * (1 - b + b * len(d) / avglen)

* tf(t, d)   how often word t occurs in chunk d.
* k1         saturation: the 2nd occurrence adds less than the 1st, and the 10th
             almost nothing. With k1 = 1.5 a word's contribution can never exceed
             2.5 * idf, however often it repeats. (Plain counting has no ceiling;
             that's how "heads, heads, heads..." in a coin-flip appendix beat real
             matches in the keyword baseline.)
* b          length normalisation: long chunks contain more words by chance, so their
             tf is discounted relative to the average chunk length (b = 0 turns it off,
             b = 1 normalises fully).
* idf(t)     rarity: a word found in few chunks is strong evidence, a word found in
             most chunks ("model", "the") is weak.

Two idf variants are offered:

* "lucene" (default, as in Lucene/Elasticsearch): log(1 + (N - n + 0.5) / (n + 0.5)),
  always positive.
* "okapi" (as in the `rank_bm25` library): log((N - n + 0.5) / (n + 0.5)), which goes
  negative for words in more than half the chunks; those are replaced by a floor of
  epsilon * (average idf of all words). In a tiny corpus where most words are common,
  that average, and so the floor, can itself be negative. Kept so the tests can check
  our numbers against rank_bm25 exactly; "lucene" is the default for this reason.

N is the number of chunks, n the number of chunks containing t.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ragbench.pipeline.chunk import Chunk
from ragbench.search.base import SearchResult, check_k, tokenize, top_k

IDF_VARIANTS = ("lucene", "okapi")


class BM25Searcher:
    name = "bm25"

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        idf: str = "lucene",
        epsilon: float = 0.25,
    ):
        if idf not in IDF_VARIANTS:
            raise ValueError(f"idf must be one of {IDF_VARIANTS}, got {idf!r}")
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        self.chunks = chunks
        self.k1, self.b = k1, b

        # Inverted index: word -> [(chunk index, term frequency), ...]. Scoring a query
        # then only touches chunks that contain at least one of its words.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.avglen = sum(self.lengths) / len(chunks) if chunks else 0.0
        self.idf = self._idf(idf, epsilon)

    def _idf(self, variant: str, epsilon: float) -> dict[str, float]:
        n_chunks = len(self.chunks)
        doc_freq = {term: len(posting) for term, posting in self.postings.items()}
        if variant == "lucene":
            return {t: math.log(1 + (n_chunks - n + 0.5) / (n + 0.5)) for t, n in doc_freq.items()}
        idf = {t: math.log((n_chunks - n + 0.5) / (n + 0.5)) for t, n in doc_freq.items()}
        if idf:
            floor = epsilon * sum(idf.values()) / len(idf)
            idf = {t: (v if v >= 0 else floor) for t, v in idf.items()}
        return idf

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score of every chunk that shares a word with the query: {chunk index: score}.

        Each distinct query word counts once ("attention attention" == "attention").
        """
        totals: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:  # word never seen in the corpus
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avglen)
                totals[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return totals

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        check_k(k)
        scored = self.scores(query)
        return top_k(((s, self.chunks[i]) for i, s in scored.items()), k)
