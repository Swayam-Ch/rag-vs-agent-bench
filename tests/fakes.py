"""Test doubles shared by several test files."""

import hashlib

import numpy as np

from ragbench.search.base import tokenize


class HashEmbedder:
    """A stand-in for a real embedding model: deterministic, instant, no download.

    Each word adds 1 at a fixed position (from a hash of the word) in a 256-number
    vector, so texts that share words point in similar directions. It knows nothing
    about meaning, so it can test the plumbing but not the quality of vector search.
    """

    def __init__(self, name: str = "hash-256", dims: int = 256, scale: float = 1.0):
        self.name, self.dims, self.scale = name, dims, scale
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in tokenize(text):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1
        return out * self.scale
