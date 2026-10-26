"""One model call, measured: the text, tokens, wall time and estimated cost.

Used for generating answers from retrieved passages and for judging answers (#18). Like
the agent loop, a call never raises: an API failure is recorded in `Completion.error`,
so one bad request can't end a benchmark run.

Request settings follow the current models (checked 2026-10-26 against the Sonnet 5.5
migration guide): no `temperature` (Sonnet 5.5 rejects sampling parameters, and SDK 1.x
no longer sends them), no forced `tool_choice`, no assistant prefill. Machine-readable
output comes from structured outputs (`output_config.format`) instead.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from ragbench.search.agent_search import estimate_cost


@dataclass
class Completion:
    model: str
    text: str = ""
    stop: str = ""  # the API's stop_reason: end_turn, max_tokens, refusal, ...
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float | None = 0.0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def complete(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    output_config: dict[str, Any] | None = None,
) -> Completion:
    """Send one user message and return the final text (thinking blocks are skipped)."""
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if output_config:
        request["output_config"] = output_config
    started = time.monotonic()
    try:
        response = client.messages.create(**request)
    except Exception as exc:  # recorded, never raised
        return Completion(
            model=model,
            seconds=round(time.monotonic() - started, 2),
            cost_usd=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    text = "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    usage = response.usage
    return Completion(
        model=model,
        text=text.strip(),
        stop=response.stop_reason or "",
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        seconds=round(time.monotonic() - started, 2),
        cost_usd=estimate_cost(model, usage.input_tokens, usage.output_tokens),
    )
