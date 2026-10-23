"""Research can start before a batch; batch actions still require supplied candidates."""

import json

import pytest

from src.prediction import cyp as cyp_tools
from src.workbench.session import SessionError
from test_external_search_session import result as external_result
from test_workbench_assistant import Harness, completion, final, intent, keys, tool
from test_workbench_gateway import Harness as GatewayHarness, keys as gateway_keys


def test_empty_session_accepts_chat_through_http():
    def answer(session, token, message, *, deadline):
        assert session.intent_context(token)["candidates"] == []
        session.interpret(token, message, {"decision": "question", "span": message})
        return {"status": "complete", "explanation": "Tell me which molecule to investigate."}

    h = GatewayHarness(assistant=answer)
    operation = h.complete(h.post("/api/chat", {"keys": gateway_keys(h.initial), "message": "Hello"}))
    assert operation["status"] == "complete"
    assert h.state()["dataset"] is None
    assert h.state()["turns"][-1]["answer"]["explanation"] == "Tell me which molecule to investigate."


def test_empty_session_research_runs_science_search_and_evidence(monkeypatch):
    monkeypatch.setattr(cyp_tools, "predict_cyp", lambda smiles: {"query_smiles": smiles, "enzymes": {}})

    def answer(request, body):
        context = json.loads(body["messages"][1]["content"])
        assert context["candidates"] == [] and context["constraints"] is None
        results = [json.loads(row["content"]) for row in body["messages"] if row["role"] == "tool"]
        assert all(row["citation"]["candidate_ids"] == [] for row in results)
        return completion(final(citations=[row["citation"] for row in results]))

    h = Harness([completion(intent()), completion(tools=[
        tool("predict_cyp", {"smiles": "CCO"}),
        tool("search_external_analogs", {"smiles": "CCO", "max_results": 3, "threshold": 70}, identifier="search"),
        tool("lookup_structure", {"smiles": "CCN", "refresh": False}, identifier="evidence")]), answer], candidates=[])
    monkeypatch.setattr(h.source, "search_analogs", lambda *a, **k: external_result(), raising=False)
    outcome = h.run()
    assert outcome["status"] == "complete"
    assert [block["kind"] for block in outcome["attachments"]] == ["scores", "external_analogs", "evidence"]
    assert h.events()[1]["result"]["data"]["analogs"][0]["candidate_ids"] == []
    snapshot = h.state()["snapshots"][0]
    assert snapshot["candidate_id"] is None and snapshot["dataset_digest"] is None
    assert snapshot["model_digest"] == h.state()["model_digest"]
    assert h.state()["dataset"] is None and h.state()["revision"] == 0
    assert h.state()["evidence_version"] == 1
    assert h.requests[-1]["response_format"]["json_schema"]["schema"]["$defs"]["Citation"]["properties"]["candidate_ids"]["maxItems"] == 0


def test_empty_session_explicit_batch_change_requests_a_batch():
    message = "Set the budget to 2"
    h = Harness([completion(intent(message, decision="explicit_change", budget=2))], candidates=[])
    outcome = h.run(message)
    assert outcome["status"] == "complete"
    assert h.state()["turns"][-1]["interpretation"]["decision"] == "clarification_needed"
    assert h.state()["constraints"] is None and h.state()["revision"] == 0
    assert len(h.requests) == 1


def test_empty_session_rejects_batch_operations():
    h = Harness([], candidates=[])
    state = h.state()
    for action in (
        lambda: h.session.start_operation(keys(state), "evidence"),
        lambda: h.session.revise(keys(state), {"budget": 2}, "Set budget"),
        lambda: h.session.mark_reviewed(keys(state, evidence=True)),
    ):
        with pytest.raises(SessionError, match="dataset"):
            action()
    assert not h.state()["busy"] and h.state()["revision"] == 0
