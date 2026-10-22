"""Publication context is retrieved only for retained assay document references."""

import pytest

from src.workbench.session import SessionError
from test_workbench_assistant import Harness, completion, final, intent, tool
from test_workbench_state import session, current, keys


def snapshot(s):
    token = s.start_operation(keys(current(s)), "evidence")
    evidence = s.lookup(token, "b")
    s.finish_operation(token, {"status": "complete"})
    retained = s._state["snapshots"][-1]
    retained["result"]["records"] = [{"document_id": "CHEMBL1148425"}]
    return evidence["snapshot_id"]


def papers():
    return {"status": "ok", "source": "https://www.ebi.ac.uk/chembl/api/data",
            "retrieved_at": "2026-10-22T22:00:00+00:00", "document_ids": ["CHEMBL1148425"],
            "papers": [{"document_id": "CHEMBL1148425", "title": "Retained source",
                        "document_type": "PUBLICATION", "abstract": "Source abstract."}],
            "missing_ids": [], "error": None}


def test_enrichment_uses_only_documents_in_retained_snapshot(monkeypatch):
    s, source, _ = session()
    sid = snapshot(s)
    before = current(s)
    calls = []
    def documents(ids, **kwargs):
        calls.append((ids, kwargs))
        return papers()
    monkeypatch.setattr(source, "documents", documents, raising=False)
    token = s.start_operation(keys(before), "evidence", timeout=30)
    event = s.enrich_papers(token, sid, max_results=3)
    data = event["result"]["data"]
    assert calls[0][0] == ["CHEMBL1148425"] and calls[0][1]["deadline"] == 30
    assert data["snapshot_id"] == sid and data["query_smiles"] == "CCN"
    assert event["result"]["candidate_ids"] == ["b"]
    assert data["papers"][0]["document_type"] == "PUBLICATION"
    after = current(s)
    for key in ("constraints", "selection", "revision", "evidence_version", "snapshots"):
        assert after[key] == before[key]


def test_unknown_snapshot_and_recorded_mode_make_no_request(monkeypatch):
    s, source, _ = session()
    sid = snapshot(s)
    monkeypatch.setattr(source, "documents", lambda *a, **k: pytest.fail("network"), raising=False)
    token = s.start_operation(keys(current(s)), "evidence")
    with pytest.raises(SessionError, match="snapshot"):
        s.enrich_papers(token, "invented", max_results=3)
    s.finish_operation(token, {"status": "complete"})
    s._mode = s._state["source_mode"] = "recorded"
    token = s.start_operation(keys(current(s)), "evidence")
    assert s.enrich_papers(token, sid, max_results=3)["result"]["data"]["status"] == "unavailable"


def test_missing_document_references_are_not_an_invented_search(monkeypatch):
    s, source, _ = session()
    sid = snapshot(s)
    s._state["snapshots"][-1]["result"]["records"] = []
    monkeypatch.setattr(source, "documents", lambda *a, **k: pytest.fail("network"), raising=False)
    token = s.start_operation(keys(current(s)), "evidence")
    data = s.enrich_papers(token, sid, max_results=3)["result"]["data"]
    assert data["status"] == "not_found" and data["papers"] == [] and data["document_ids"] == []


def test_agent_enriches_snapshot_and_attaches_cited_source_documents(monkeypatch):
    h = Harness([completion(intent()),
        lambda *_: completion(tools=[tool("enrich_papers", {"snapshot_id": sid, "max_results": 3})]),
        lambda *_: completion(final(citations=[{"kind": "result",
            "reference_id": h.events()[0]["result_id"], "candidate_ids": ["b"]}]))])
    sid = snapshot(h.session)
    monkeypatch.setattr(h.source, "documents", lambda *a, **k: papers(), raising=False)
    outcome = h.run()
    assert outcome["attachments"][0]["kind"] == "papers"
    assert outcome["attachments"][0]["data"]["snapshot_id"] == sid
    assert outcome["result_refs"] == [h.events()[0]["result_id"]]
