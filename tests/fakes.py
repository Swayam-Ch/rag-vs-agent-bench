"""Test doubles shared by several test files."""

import copy
import hashlib
import re
from types import SimpleNamespace

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


# --------------------------------------------------------------------------- #
# Fake Anthropic clients for the agent loop: same shapes as the real SDK objects
# --------------------------------------------------------------------------- #


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(tool_id: str, name: str, **args) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)


def reply(*blocks, stop: str | None = None, tokens: tuple[int, int] = (100, 20)):
    if stop is None:
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


class _Messages:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(copy.deepcopy(kwargs))
        return self.owner.next_reply(kwargs)


class ScriptedClient:
    """Replays a fixed list of replies, one per call; records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = _Messages(self)

    def next_reply(self, request):
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class PolicyClient:
    """Answers *any* question with a fixed strategy, like a very simple agent:
    grep for the question's words, read around the first match, then answer."""

    def __init__(self):
        self.calls: list[dict] = []
        self.messages = _Messages(self)

    def next_reply(self, request):
        messages = request["messages"]
        # this conversation's turn number, from the request itself (not a global counter)
        turn = 1 + sum(m["role"] == "assistant" for m in messages)
        if turn == 1:
            words = [w for w in re.findall(r"\w+", messages[0]["content"]) if len(w) > 2]
            return reply(tool_block("t1", "grep", pattern="|".join(words) or "x"))
        last = messages[-1]["content"][0]["content"]
        hit = re.search(r"^(\w+) p\.\d+ @(\d+):", last, re.MULTILINE)
        if turn == 2 and hit:
            start = max(0, int(hit.group(2)) - 20)
            return reply(tool_block("t2", "read_file", doc_id=hit.group(1), start=start, length=60))
        return reply(text_block("Done.\nSources:"))
