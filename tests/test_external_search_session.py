"""External discovery stays inside session admission, provenance and tool budgets."""

from copy import deepcopy

import pytest

from src.workbench.assistant import operation_attachments
from src.workbench.session import SessionError
from test_workbench_assistant import Harness, completion, final, intent, tool
from test_workbench_state import current, keys, session


def result(smiles="CCO"):
    return {"status": "ok", "query_smiles": smiles, "source": "https://www.ebi.ac.uk/chembl/api/data",
            "retrieved_at": "2026-10-22T22:00:00+00:00", "threshold_percent": 70,
            "candidate_count": 1, "truncated": False, "error": None,
            "analogs": [{"compound_id": "CHEMBL545", "name": "ethylamine", "smiles": "CCN",
                         "chembl_similarity_percent": 72.0, "morgan_similarity": .33,
                         "compound_url": "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL545"}]}


def test_search_retains_provenance_and_structure_without_changing_batch(monkeypatch):
    s, source, _ = session()
    before = current(s)
    calls = []

    def search(smiles, **kwargs):
        calls.append((smiles, kwargs))
        return result(smiles)

    monkeypatch.setattr(source, "search_analogs", search, raising=False)
    token = s.start_operation(keys(before), "evidence", timeout=30)
    event = s.search_external_analogs(token, "OCC", max_results=3, threshold=70)
    data = event["result"]["data"]
    assert data["query_smiles"] == "CCO" and data["origin"] == "live"
    assert data["analogs"][0]["candidate_ids"] == ["b"]
    assert "<svg" in data["analogs"][0]["svg"]
    assert calls[0][1]["deadline"] == 30 and calls[0][1]["clock"]() == 0
    assert event["result"]["candidate_ids"] == [" a\n"]
    assert operation_attachments(s.poll(before["session_id"], token), [])[0]["kind"] == "external_analogs"
    s.finish_operation(token, {"status": "complete"})
    after = current(s)
    for key in ("dataset", "constraints", "selection", "revision", "evidence_version", "snapshots"):
        assert after[key] == before[key]
    exported = s.capture_export(keys(after, evidence=True))
    assert exported["operations"][-1]["events"][0] == event


def test_recorded_mode_never_starts_live_search(monkeypatch):
    s, source, _ = session()
    s._mode = s._state["source_mode"] = "recorded"
    monkeypatch.setattr(source, "search_analogs", lambda *a, **k: pytest.fail("live request"), raising=False)
    token = s.start_operation(keys(current(s)), "evidence")
    event = s.search_external_analogs(token, "CCO", max_results=3, threshold=70)
    assert event["result"]["data"]["status"] == "unavailable"
    assert event["result"]["data"]["origin"] == "recorded"
    assert event["result"]["data"]["analogs"] == []


def test_late_search_is_not_attached_and_lookup_ownership_is_released(monkeypatch):
    s, source, now = session()
    def search(*args, **kwargs):
        now[0] = 31
        return result()
    monkeypatch.setattr(source, "search_analogs", search, raising=False)
    token = s.start_operation(keys(current(s)), "evidence", timeout=30)
    with pytest.raises(SessionError, match="deadline"):
        s.search_external_analogs(token, "CCO", max_results=3, threshold=70)
    assert not s._operations[token]["_in_flight"]
    assert not s.poll(current(s)["session_id"], token)["events"]


def test_search_requires_intent_before_network(monkeypatch):
    s, source, _ = session()
    monkeypatch.setattr(source, "search_analogs", lambda *a, **k: pytest.fail("live request"), raising=False)
    token = s.start_operation(keys(current(s)), "chat", message="Find analogues")
    with pytest.raises(SessionError, match="Interpret"):
        s.search_external_analogs(token, "CCO", max_results=3, threshold=70)


def test_agent_dispatches_search_and_cites_retained_result(monkeypatch):
    h = Harness([completion(intent()), completion(tools=[tool("search_external_analogs",
        {"smiles": "CCO", "max_results": 3, "threshold": 70})]),
        lambda *_: completion(final(citations=[{"kind": "result",
            "reference_id": h.events()[0]["result_id"], "candidate_ids": ["a"]}]))])
    calls = []
    def search(smiles, **kwargs):
        calls.append(smiles)
        return result(smiles)
    monkeypatch.setattr(h.source, "search_analogs", search, raising=False)
    before = deepcopy(h.state())
    outcome = h.run()
    assert outcome["status"] == "complete" and calls == ["CCO"]
    assert outcome["attachments"][0]["kind"] == "external_analogs"
    assert outcome["result_refs"] == [h.events()[0]["result_id"]]
    assert h.state()["constraints"] == before["constraints"]
    assert len(h.ledger.settlements) == 3


def test_agent_cannot_omit_search_citation(monkeypatch):
    h = Harness([completion(intent()), completion(tools=[tool("search_external_analogs",
        {"smiles": "CCO", "max_results": 3, "threshold": 70})]), completion(final())])
    monkeypatch.setattr(h.source, "search_analogs", lambda *a, **k: result(), raising=False)
    with pytest.raises(SessionError, match="cite"):
        h.run()
    assert len(h.events()) == 1


def test_provider_citation_schema_allows_only_actual_dataset_ids():
    h = Harness([completion(intent()), completion(final())], candidates=[
        {"id": 'query "α"', "smiles": "CCO"}, {"id": "control", "smiles": "CCN"}])
    h.run()
    schema = h.requests[-1]["response_format"]["json_schema"]["schema"]
    candidate_ids = schema["$defs"]["Citation"]["properties"]["candidate_ids"]
    assert set(candidate_ids["items"]["enum"]) == {'query "α"', "control"}
    assert candidate_ids.get("minItems", 0) == 0  # External structures have no dataset ID.
    assert "Citation" not in h.requests[0]["response_format"]["json_schema"]["schema"].get("$defs", {})


def test_tool_context_supplies_exact_citations_without_discarding_retained_images(monkeypatch):
    import json

    def answer(request, body):
        results = [json.loads(row["content"]) for row in body["messages"] if row["role"] == "tool"]
        search, evidence = results
        assert "svg" not in search["result"]["data"]["analogs"][0]
        assert search["result"]["data"]["analogs"][0]["smiles"] == "CCN"
        assert search["result"]["data"]["service_match_count"] == 2
        assert search["result"]["data"]["returned_analog_count"] == 1
        assert "candidate_count" not in search["result"]["data"]
        assert search["citation"]["candidate_ids"] == ["a"]
        assert evidence["citation"]["candidate_ids"] == []
        assert evidence["citation"]["reference_id"] == evidence["snapshot_id"]
        return completion(final(citations=[search["citation"], evidence["citation"]]))

    h = Harness([completion(intent()), completion(tools=[
        tool("search_external_analogs", {"smiles": "CCO", "max_results": 3, "threshold": 70}),
        tool("lookup_structure", {"smiles": "CCCO", "refresh": False}, identifier="external-evidence")]), answer])
    monkeypatch.setattr(h.source, "search_analogs", lambda *a, **k: {**result(), "candidate_count": 2})
    outcome = h.run()
    assert outcome["status"] == "complete"
    assert "<svg" in h.events()[0]["result"]["data"]["analogs"][0]["svg"]
    assert h.events()[0]["result"]["data"]["candidate_count"] == 2
    assert "<svg" in outcome["attachments"][0]["data"]["analogs"][0]["svg"]
