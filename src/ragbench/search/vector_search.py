"""Dense (vector) search: compare meaning, not words.

Every chunk is turned into a vector by an embedding model, once. A query is embedded the
same way, and chunks are ranked by cosine similarity to it: the cosine of the angle
between the two vectors, 1.0 for "pointing the same way", ~0 for unrelated.

    cosine(q, d) = (q . d) / (|q| * |d|)

All vectors are normalised to length 1 when they are created, so at search time cosine
is just a dot product, and scoring every chunk is a single matrix-vector multiplication.

Embedding the corpus is the slow part (minutes on a laptop), so the vectors are cached
on disk. The cache file name is a fingerprint of the model name plus every chunk's ID
and text: change the model, the chunking or a single character, and a new file is
computed instead of silently reusing stale vectors.

The model is pluggable (anything with `name` and `embed`), which keeps the heavy
dependency out of the tests: they use a small deterministic stand-in.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from ragbench.pipeline.chunk import Chunk
from ragbench.search.base import SearchResult, check_k, top_k

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small (80 MB), fast on a CPU
# The exact model version (a Hugging Face commit). Without it, "the model" is whatever the
# authors last uploaded, so vectors could change between runs with nothing in the repo
# showing why. The revision is part of the embedder's name, and so of the cache fingerprint.
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class Embedder(Protocol):
    name: str  # identifies the model; part of the cache fingerprint

    def embed(self, texts: list[str]) -> np.ndarray:
        """One row per text, shape (len(texts), dimensions)."""
        ...


class SentenceTransformerEmbedder:
    """A local model from the sentence-transformers library (`pip install -e ".[vector]"`).

    Note: all-MiniLM-L6-v2 reads at most 256 word pieces (~1000 characters of English)
    and silently ignores the rest, which is one reason chunk size matters.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                'vector search needs sentence-transformers: pip install -e ".[vector]"'
            ) from exc
        self.name = f"{model_name}@{revision}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, revision=revision)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so a dot product is a cosine. All-zero rows stay zero."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def fingerprint(model_name: str, chunks: list[Chunk]) -> str:
    """Changes whenever the model, the chunk IDs or any chunk text changes."""
    digest = hashlib.sha256(model_name.encode())
    for chunk in chunks:
        digest.update(b"\0" + chunk.chunk_id.encode() + b"\0" + chunk.text.encode())
    return digest.hexdigest()[:16]


class VectorSearcher:
    name = "vector"

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache_dir: Path | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.matrix = self._load_or_embed(cache_dir)  # shape (n_chunks, dimensions)

    def _load_or_embed(self, cache_dir: Path | None) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, 0), dtype=np.float32)
        path = None
        if cache_dir is not None:
            path = cache_dir / f"embeddings.{fingerprint(self.embedder.name, self.chunks)}.npy"
            if path.exists():
                log.info("loading cached embeddings from %s", path)
                matrix = np.load(path)
                if matrix.shape[0] == len(self.chunks):
                    return matrix
                log.warning(
                    "%s has %d rows for %d chunks; recomputing",
                    path,
                    matrix.shape[0],
                    len(self.chunks),
                )

        log.info("embedding %d chunks with %s", len(self.chunks), self.embedder.name)
        matrix = normalize(self.embedder.embed([c.text for c in self.chunks]))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
        return matrix

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        check_k(k)
        if not query.strip() or not self.chunks:
            return []
        q = normalize(self.embedder.embed([query]))[0]
        similarities = self.matrix @ q  # cosine similarity with every chunk at once
        return top_k(zip(similarities.tolist(), self.chunks, strict=True), k)
