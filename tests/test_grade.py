"""Answering from passages and the LLM judge (#18), against fake API clients."""

import json
from types import SimpleNamespace

import pytest
from fakes import GradingClient, ScriptedClient, reply, text_block

from ragbench.eval import grade
from ragbench.eval.grade import (
    JUDGE_SYSTEM,
    VERDICT_SCHEMA,
    Grader,
    Judgement,
    format_passages,
    parse_verdict,
    prompt_fingerprint,
)
from ragbench.eval.questions import parse
from ragbench.llm import complete
from ragbench.search.base import SearchResult

QS = parse(
    """\
version: 1
questions:
  - id: q1
    type: single_fact
    question: How many heads?
    answer: 8 heads.
    gold:
      - paper: "1706.03762"
        pages: [5]
  - id: q2
    type: unanswerable
    question: What did training cost in dollars?
    answer: The papers don't say.
    gold: []
"""
).questions
PAPERS = {"bdfaa68d8984": ("1706.03762", "Attention Is All You Need")}
RESULTS = [
    SearchResult("bdfaa68d8984", 0, 50, 5, 5, 2.0, "  We use h = 8 parallel heads.  "),
    SearchResult("bdfaa68d8984", 60, 90, 8, 9, 1.0, "Table 3 rows (A)"),
]


# --------------------------------------------------------------------------- #
# Answering from passages
# --------------------------------------------------------------------------- #


def test_passages_are_numbered_with_paper_pages_and_title():
    text = format_passages(RESULTS, PAPERS)
    assert text == (
        "[1] 1706.03762 p.5 (Attention Is All You Need)\nWe use h = 8 parallel heads.\n\n"
        "[2] 1706.03762 pp.8-9 (Attention Is All You Need)\nTable 3 rows (A)"
    )
    assert format_passages([], PAPERS) == "(no passages were retrieved)"


def test_answer_request_and_cost():
    client = GradingClient()
    c = Grader(client, PAPERS).answer(QS[0], RESULTS)
    [request] = client.calls
    assert request["model"] == "claude-sonnet-5-5"
    assert request["system"] == grade.ANSWER_SYSTEM
    user = request["messages"][0]["content"]
    assert user.startswith("Question: How many heads?") and "[1] 1706.03762 p.5" in user
    # the current models reject these (400): never sent
    assert not {"temperature", "top_p", "tool_choice", "output_config"} & set(request)
    assert c.text.startswith("Eight heads.") and c.error == ""
    assert (c.input_tokens, c.output_tokens) == (1000, 100)
    assert c.cost_usd == pytest.approx((1000 * 2 + 100 * 10) / 1e6)  # Sonnet 5.5 prices


# --------------------------------------------------------------------------- #
# The judge
# --------------------------------------------------------------------------- #


def test_judge_request_uses_structured_output_and_the_rubric():
    client = GradingClient(verdict="Partial")  # enum case isn't guaranteed by the API
    j = Grader(client, PAPERS).judge(QS[0], "Eight, I think.")
    [request] = client.calls
    assert request["model"] == "claude-opus-5-5" and request["system"] == JUDGE_SYSTEM
    assert request["output_config"] == {
        "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
        "effort": "medium",
    }
    assert "temperature" not in request and "tool_choice" not in request
    user = request["messages"][0]["content"]
    assert "Reference answer: 8 heads." in user
    assert user.endswith("<answer>\nEight, I think.\n</answer>")  # the answer is data
    assert (j.verdict, j.points, j.reason) == ("partial", 0.5, "compared with the reference")
    out = j.to_json()
    assert out["cost_usd"] == pytest.approx((500 * 4 + 50 * 20) / 1e6)  # Opus 5.5 prices
    assert "text" not in out and out["verdict"] == "partial"


def test_unanswerable_questions_tell_the_judge_a_specific_answer_is_wrong():
    client = GradingClient(verdict="wrong")
    Grader(client, PAPERS).judge(QS[1], "About $4.6 million.")
    user = client.calls[0]["messages"][0]["content"]
    assert "Question type: unanswerable" in user and "any specific answer" in user


def test_empty_answer_is_wrong_without_calling_the_judge():
    client = GradingClient()
    j = Grader(client, PAPERS).judge(QS[0], "   ")
    assert (j.verdict, j.points, client.calls) == ("wrong", 0.0, [])
    assert j.to_json()["cost_usd"] == 0.0


@pytest.mark.parametrize(
    ("client_args", "reason"),
    [
        ({"fail": "judge"}, "judge call failed: RuntimeError: judge API down"),
        ({"judge_stop": "refusal"}, "judge stopped early (refusal)"),
        ({"judge_stop": "max_tokens"}, "judge stopped early (max_tokens)"),
    ],
)
def test_judge_failures_are_unjudged_not_wrong(client_args, reason):
    j = Grader(GradingClient(**client_args), PAPERS).judge(QS[0], "8")
    assert (j.verdict, j.points, j.reason) == (None, None, reason)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"reason": "ok", "verdict": "correct"}', ("correct", "ok")),
        ('{"reason": "ok", "verdict": " WRONG "}', ("wrong", "ok")),
        ('{"reason": "ok", "verdict": "maybe"}', (None, "unknown verdict 'maybe'")),
        ('{"reason": "ok"}', (None, 'unreadable judge output: \'{"reason": "ok"}\'')),
        ("not json", (None, "unreadable judge output: 'not json'")),
        ("[1, 2]", (None, "unreadable judge output: '[1, 2]'")),
    ],
)
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_verdict_schema_follows_the_structured_output_rules():
    # every object needs additionalProperties: false; unsupported keywords are a 400
    assert VERDICT_SCHEMA["additionalProperties"] is False
    assert set(VERDICT_SCHEMA["required"]) == set(VERDICT_SCHEMA["properties"])
    assert list(VERDICT_SCHEMA["properties"]) == ["reason", "verdict"]  # reason first
    text = json.dumps(VERDICT_SCHEMA)
    for keyword in ("minLength", "maxLength", "minimum", "maximum", "pattern"):
        assert keyword not in text


def test_points():
    assert [Judgement(v, "", None).points for v in ("correct", "partial", "wrong", None)] == [
        1.0,
        0.5,
        0.0,
        None,
    ]


def test_editing_a_prompt_changes_the_fingerprint(monkeypatch):
    before = prompt_fingerprint()
    monkeypatch.setattr(grade, "JUDGE_SYSTEM", JUDGE_SYSTEM + " Be strict.")
    assert prompt_fingerprint() != before


def test_grader_describes_itself_for_the_run_config():
    info = Grader(GradingClient(), PAPERS, judge_model="claude-sonnet-5-5").describe()
    assert info["judge_model"] == "claude-sonnet-5-5" and info["answer_model"]
    assert info["judge_prompt"] == "judge-v1" and len(info["prompts_sha"]) == 12


# --------------------------------------------------------------------------- #
# complete(): one measured call
# --------------------------------------------------------------------------- #


def test_complete_skips_thinking_blocks_and_records_usage():
    thinking = SimpleNamespace(type="thinking", thinking="")
    client = ScriptedClient([reply(thinking, text_block(" hi "), tokens=(10, 5))])
    c = complete(client, model="claude-haiku-5-5", system="s", user="u", max_tokens=5)
    assert (c.text, c.stop, c.input_tokens, c.output_tokens) == ("hi", "end_turn", 10, 5)
    assert c.cost_usd == pytest.approx((10 * 0.10 + 5 * 0.50) / 1e6)


def test_complete_records_an_api_error_instead_of_raising():
    client = ScriptedClient([ConnectionError("offline")])
    c = complete(client, model="claude-sonnet-5-5", system="s", user="u", max_tokens=5)
    assert c.error == "ConnectionError: offline" and c.text == "" and c.cost_usd is None


def test_unknown_model_price_is_none_not_zero():
    client = ScriptedClient([reply(text_block("x"))])
    c = complete(client, model="some-new-model", system="s", user="u", max_tokens=5)
    assert c.cost_usd is None
