import json
import os

import pytest
from fakes import PolicyClient, ScriptedClient, reply, text_block, tool_block

from ragbench.search.agent_search import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    TOOLS,
    AgentSearcher,
    Citation,
    client_options,
    parse_citations,
    run_tool,
)
from ragbench.search.agent_tools import Corpus, Paper, Read

TEXT = (
    "Attention Is All You Need\n\n"
    "Multi-head attention runs h = 8 attention heads in parallel.\n\n"
    "In Table 3 rows (A), we vary the number of attention heads."
)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus([Paper("doc1", TEXT, "Attention Is All You Need", [0, 60], "1706.03762")])


def _agent(corpus, replies, **kwargs) -> tuple[AgentSearcher, ScriptedClient]:
    client = ScriptedClient(replies)
    return AgentSearcher(corpus, client=client, **kwargs), client


# --------------------------------------------------------------------------- #
# A normal run, step by step
# --------------------------------------------------------------------------- #


def test_full_run_list_grep_read_answer(corpus):
    offset = TEXT.index("In Table 3")
    agent, client = _agent(
        corpus,
        [
            reply(text_block("Let me look."), tool_block("t1", "list_files")),
            reply(tool_block("t2", "grep", pattern="attention heads")),
            reply(tool_block("t3", "read_file", doc_id="1706.03762", start=offset, length=58)),
            reply(text_block("They vary the number of heads.\nSources:\n1706.03762 p.2")),
        ],
    )
    run = agent.run("Do they vary the number of heads?")

    assert run.stop == "answered"
    assert run.answer.startswith("They vary the number of heads.")
    assert run.turns == 4
    assert [c.name for c in run.tool_calls] == ["list_files", "grep", "read_file"]
    assert not any(c.is_error for c in run.tool_calls)
    assert run.reads == [Read("doc1", offset, offset + 58)]
    assert (run.input_tokens, run.output_tokens) == (400, 80)


def test_request_carries_model_system_prompt_and_tools(corpus):
    agent, client = _agent(corpus, [reply(text_block("No idea."))])
    agent.run("q")
    request = client.calls[0]
    assert request["model"] == DEFAULT_MODEL
    assert request["system"] == SYSTEM_PROMPT
    assert [t["name"] for t in request["tools"]] == ["list_files", "grep", "read_file"]
    assert request["messages"] == [{"role": "user", "content": "q"}]


def test_tool_results_follow_their_tool_use_and_come_first(corpus):
    agent, client = _agent(
        corpus,
        [
            reply(
                text_block("Two at once."),
                tool_block("a", "grep", pattern="heads"),
                tool_block("b", "list_files"),
            ),
            reply(text_block("Done.")),
        ],
    )
    agent.run("q")
    second = client.calls[1]["messages"]
    assistant, results = second[1], second[2]
    assert assistant["role"] == "assistant"
    assert [b["type"] for b in assistant["content"]] == ["text", "tool_use", "tool_use"]
    assert results["role"] == "user"
    assert [b["type"] for b in results["content"]] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in results["content"]] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Bad tool calls are errors for the model, not exceptions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("delete_files", {}),
        ("read_file", {}),  # missing doc_id
        ("read_file", {"doc_id": "doc1", "start": "the beginning"}),
        ("grep", {"pattern": "heads", "max_hits": None}),
        ("read_file", {"doc_id": "../../etc/passwd"}),
    ],
)
def test_bad_tool_calls_become_error_results(corpus, name, args):
    output, is_error = run_tool(corpus, name, args)
    assert is_error and output.startswith("error:")


def test_loop_continues_after_a_tool_error(corpus):
    agent, client = _agent(
        corpus,
        [reply(tool_block("t1", "read_file", doc_id="nope")), reply(text_block("Recovered."))],
    )
    run = agent.run("q")
    assert run.stop == "answered" and run.answer == "Recovered."
    assert run.tool_calls[0].is_error
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "list_files" in result["content"]


# --------------------------------------------------------------------------- #
# Every way a run can end
# --------------------------------------------------------------------------- #


def test_max_turns_stops_the_loop(corpus):
    endless = [reply(tool_block(f"t{i}", "list_files")) for i in range(10)]
    agent, client = _agent(corpus, endless, max_turns=3)
    run = agent.run("q")
    assert (run.stop, run.turns, len(client.calls)) == ("max_turns", 3, 3)


def test_timeout_stops_the_loop(corpus, monkeypatch):
    from ragbench.search import agent_search

    clock = iter([0.0, 0.0, 500.0, 500.0, 500.0])
    monkeypatch.setattr(agent_search.time, "monotonic", lambda: next(clock))
    agent, _ = _agent(corpus, [reply(tool_block("t1", "list_files"))] * 3, max_seconds=60)
    assert agent.run("q").stop == "timeout"


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "answered"),
        ("max_tokens", "max_tokens"),
        ("refusal", "refusal"),
        ("model_context_window_exceeded", "context_window"),
    ],
)
def test_stop_reasons_are_recorded(corpus, stop_reason, expected):
    agent, _ = _agent(corpus, [reply(text_block("partial"), stop=stop_reason)])
    assert agent.run("q").stop == expected


def test_api_error_is_recorded_not_raised(corpus):
    agent, _ = _agent(corpus, [reply(tool_block("t1", "list_files")), ConnectionError("down")])
    run = agent.run("q")
    assert run.stop == "error" and "ConnectionError: down" in run.error
    assert run.turns == 1


def test_invalid_limits_raise(corpus):
    for kwargs in ({"max_turns": 0}, {"max_tokens": 0}, {"max_seconds": 0}):
        with pytest.raises(ValueError):
            AgentSearcher(corpus, client=ScriptedClient([]), **kwargs)


# --------------------------------------------------------------------------- #
# Transcripts and the Searcher view
# --------------------------------------------------------------------------- #


def test_transcript_is_saved_as_json(corpus, tmp_path):
    agent, _ = _agent(
        corpus,
        [reply(tool_block("t1", "grep", pattern="heads")), reply(text_block("Yes."))],
        transcript_dir=tmp_path,
    )
    agent.run("How many heads?")
    [path] = tmp_path.glob("how-many-heads-*.json")
    saved = json.loads(path.read_text())
    assert saved["answer"] == "Yes." and saved["tool_calls"][0]["name"] == "grep"
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "user", "assistant"]


def test_search_returns_distinct_reads_in_order(corpus):
    agent, _ = _agent(
        corpus,
        [
            reply(tool_block("t1", "read_file", doc_id="doc1", start=0, length=20)),
            reply(tool_block("t2", "read_file", doc_id="doc1", start=70, length=30)),
            reply(tool_block("t3", "read_file", doc_id="doc1", start=0, length=20)),  # repeat
            reply(text_block("Done.")),
        ],
    )
    results = agent.search("q", k=10)
    assert [(r.start, r.end) for r in results] == [(0, 20), (70, 100)]
    assert [r.score for r in results] == [1.0, 0.5]
    assert [r.page for r in results] == [1, 2]
    assert all(r.text == TEXT[r.start : r.end] for r in results)


def test_blank_query_makes_no_api_call(corpus):
    agent, client = _agent(corpus, [])
    assert agent.search("  ", k=3) == []
    assert client.calls == []


def test_each_run_starts_with_an_empty_read_log(corpus):
    agent = AgentSearcher(corpus, client=PolicyClient())
    first = agent.run("attention heads")
    second = agent.run("attention heads")
    assert first.reads == second.reads and len(second.reads) == 1


def test_tool_schemas_are_valid_json_schema_objects():
    for tool in TOOLS:
        assert tool["input_schema"]["type"] == "object"
        assert set(tool.get("input_schema", {}).get("required", [])) <= set(
            tool["input_schema"]["properties"]
        )


# --------------------------------------------------------------------------- #
# The real model (opt-in: costs a few cents, needs ANTHROPIC_API_KEY)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get("RAGBENCH_REAL_AGENT"), reason="set RAGBENCH_REAL_AGENT=1 to run"
)
def test_real_agent_finds_and_cites_the_passage(corpus):
    agent = AgentSearcher(corpus, max_turns=8)
    run = agent.run("How many attention heads does the model run in parallel?")
    assert run.stop == "answered", run.error
    assert "8" in run.answer
    assert any(call.name == "grep" for call in run.tool_calls)
    assert run.reads, "the agent should read the passage before answering"


# --------------------------------------------------------------------------- #
# Client settings from the environment
# --------------------------------------------------------------------------- #


def test_workspace_id_becomes_a_request_header():
    assert client_options({"ANTHROPIC_WORKSPACE_ID": "wrkspc_01abc"}) == {
        "default_headers": {"anthropic-workspace-id": "wrkspc_01abc"}
    }


@pytest.mark.parametrize(
    "env", [{}, {"ANTHROPIC_WORKSPACE_ID": ""}, {"ANTHROPIC_WORKSPACE_ID": "  "}]
)
def test_no_workspace_id_means_no_extra_settings(env):
    assert client_options(env) == {}


def test_default_client_passes_the_workspace_header(monkeypatch, corpus):
    import sys
    import types

    created = []
    fake = types.SimpleNamespace(Anthropic=lambda **kwargs: created.append(kwargs) or object())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_01abc")

    AgentSearcher(corpus)

    assert created == [{"default_headers": {"anthropic-workspace-id": "wrkspc_01abc"}}]


# --------------------------------------------------------------------------- #
# Evidence from snippets, and citations
# --------------------------------------------------------------------------- #


def test_answer_from_grep_snippets_alone_still_has_evidence(corpus):
    """What the real agent did on the attention-heads question: grep, grep, answer."""
    agent, _ = _agent(
        corpus,
        [
            reply(tool_block("t1", "grep", pattern="number of attention heads")),
            reply(text_block("They vary it.\nSources:\n1706.03762 p.2")),
        ],
    )
    run = agent.run("q")
    assert run.reads == []
    assert [e.kind for e in run.evidence] == ["snippet"]
    assert run.citations == [Citation("1706.03762", "doc1", 2)]

    agent, _ = _agent(
        corpus,
        [
            reply(tool_block("t1", "grep", pattern="number of attention heads")),
            reply(text_block("They vary it.")),
        ],
    )
    [result] = agent.search("q", k=5)
    assert "number of attention heads" in result.text
    # the snippet's leading context starts on page 1; the match itself is on page 2.
    # Page-level metrics must therefore count every page in [page, page_end].
    assert result.page <= 2 <= result.page_end


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        ("Sources:\n1706.03762 p.9", [("1706.03762", 9)]),
        ("Sources:\n- 1706.03762, p. 9", [("1706.03762", 9)]),
        ("Sources:\n1706.03762v7 page 9", [("1706.03762v7", 9)]),
        ("Sources:\n1706.03762 pp. 9-10", [("1706.03762", 9)]),
        ("Sources: 1706.03762 p.9; 1706.03762 p.3", [("1706.03762", 9), ("1706.03762", 3)]),
        ("Sources:\n1706.03762 p.9\n1706.03762 p.9", [("1706.03762", 9)]),
        ("Sources:\ndoc1 p.1", []),  # not an arXiv or hash-style ID
        ("No sources section, but 1706.03762 p.9 is mentioned.", []),
    ],
)
def test_parse_citations(corpus, sources, expected):
    got = parse_citations(sources, corpus)
    assert [(c.ref, c.page) for c in got] == expected
    assert all(c.doc_id == "doc1" for c in got)


def test_ids_mentioned_before_the_sources_heading_are_not_citations(corpus):
    answer = "Unlike 2005.11401 p.4, this paper...\n\nSources:\n1706.03762 p.9"
    assert [c.ref for c in parse_citations(answer, corpus)] == ["1706.03762"]


def test_citation_of_an_unknown_paper_is_kept_but_unresolved(corpus):
    [c] = parse_citations("Sources:\n9999.99999 p.1", corpus)
    assert (c.ref, c.doc_id, c.page) == ("9999.99999", "", 1)


def test_evidence_and_citations_are_in_the_transcript(corpus, tmp_path):
    agent, _ = _agent(
        corpus,
        [
            reply(tool_block("t1", "grep", pattern="heads")),
            reply(text_block("Yes.\nSources:\n1706.03762 p.2")),
        ],
        transcript_dir=tmp_path,
    )
    agent.run("q")
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["evidence"][0]["kind"] == "snippet"
    assert saved["citations"] == [{"ref": "1706.03762", "doc_id": "doc1", "page": 2}]
