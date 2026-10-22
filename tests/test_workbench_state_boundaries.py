"""Session contract checks through the real scientific and evidence boundaries."""

from collections import deque
from copy import deepcopy
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from src import chembl as evidence
from src.screening import load_candidates
from src.workbench.science import ScientificOperations
from src.workbench.session import Session, SessionError


@pytest.fixture(scope="module")
def science():
    if not Path("artifacts/models/CYP3A4.joblib").exists():
        pytest.skip("Local frozen CYP3A4 artifact is unavailable")
    return ScientificOperations("artifacts")


@pytest.fixture
def candidates():
    return load_candidates("examples/cyp3a4-candidates.csv")


@pytest.fixture
def transport(monkeypatch):
    """Use real EvidenceClient parsing; an unexpected request fails locally."""
    responses, requests = deque(), []

    def open_request(request, timeout):
        requests.append(request.full_url)
        payload = responses.popleft()
        if callable(payload):
            payload = payload()
        if isinstance(payload, Exception):
            raise payload
        return io.BytesIO(json.dumps(payload, allow_nan=False).encode("utf-8"))

    monkeypatch.setattr(evidence, "urlopen", open_request)
    monkeypatch.setattr(evidence.time, "sleep", lambda _: None)
    return responses, requests


def keys(state, *, evidence_version=False):
    result = {"session_id": state["session_id"],
              "dataset_digest": state["dataset_digest"],
              "base_revision": state["revision"]}
    if evidence_version:
        result["evidence_version"] = state["evidence_version"]
    return result


def loaded(science, candidates, *, clock=None):
    options = {} if clock is None else {"clock": clock}
    session = Session(science, evidence.EvidenceClient(), **options)
    state = session.replace_dataset(keys(session.bootstrap()), candidates, 4, "load sample")
    return session, state


def page(key, rows):
    return {key: rows, "page_meta": {"total_count": len(rows), "next": None}}


def measured_lookup(smiles, activity_id=900001):
    return [
        page("molecules", [{"molecule_chembl_id": "CHEMBL999001",
                            "molecule_structures": {"canonical_smiles": smiles}}]),
        page("activities", [{"activity_id": activity_id,
                             "molecule_chembl_id": "CHEMBL999001",
                             "canonical_smiles": smiles, "assay_chembl_id": "CHEMBL999002",
                             "target_chembl_id": "CHEMBL340", "target_organism": "Homo sapiens",
                             "standard_type": "IC50", "standard_relation": ">",
                             "standard_value": "10000", "standard_units": "nM",
                             "type": "IC50", "relation": ">", "value": "10", "units": "uM"}]),
        page("assays", [{"assay_chembl_id": "CHEMBL999002", "target_chembl_id": "CHEMBL340",
                         "src_id": 1, "src_assay_id": "independent-session-fixture",
                         "description": "Reported observation; no causal interpretation"}]),
    ]


@pytest.mark.parametrize("reorder", [False, True])
def test_replacing_identical_dataset_rejects_old_keys_despite_same_digest_and_versions(
        science, candidates, transport, reorder):
    session, original = loaded(science, candidates)
    old_keys = keys(original)
    message = "Consider capacity three"
    token = session.start_operation(old_keys, "chat", message=message)
    session.interpret(token, message, {"decision": "question", "span": message})
    proposal = session.propose(token, {"budget": 3}, "Possible capacity change", [])
    session.finish_operation(token, {"status": "complete"})
    session.mark_reviewed(keys(original, evidence_version=True))
    replacement = list(reversed(candidates)) if reorder else deepcopy(candidates)
    current = session.replace_dataset(old_keys, replacement, 4, "replace sample")

    assert current["dataset_digest"] == original["dataset_digest"]
    assert current["model_digest"] == original["model_digest"]
    assert current["revision"] == original["revision"]
    assert current["evidence_version"] == original["evidence_version"]
    assert current["session_id"] != original["session_id"]
    with pytest.raises(SessionError):
        session.read(original["session_id"])
    with pytest.raises(SessionError):
        session.revise(old_keys, {"budget": 3}, "stale tab change")
    with pytest.raises(SessionError):
        session.start_operation(old_keys, "evidence")
    with pytest.raises(SessionError):
        session.mark_reviewed(keys(original, evidence_version=True))
    with pytest.raises(SessionError):
        session.capture_export(keys(original, evidence_version=True))
    with pytest.raises(SessionError):
        session.apply_proposal(keys(original, evidence_version=True),
                               proposal["proposal_id"], "stale proposal action")
    assert session.read(current["session_id"]) == current
    assert transport[1] == []


def test_restart_has_empty_state_and_rejects_prior_process_keys(science, candidates, transport):
    old_session, previous = loaded(science, candidates)
    old_token = old_session.start_operation(keys(previous), "evidence")
    old_session.finish_operation(old_token, {"status": "complete"})
    restarted = Session(science, evidence.EvidenceClient())
    empty = restarted.bootstrap()

    assert empty["session_id"] != previous["session_id"]
    assert not empty["dataset"] and not empty["snapshots"] and not empty["turns"]
    assert transport[1] == []
    reloaded = restarted.replace_dataset(keys(empty), candidates, 4, "load after restart")
    assert reloaded["dataset_digest"] == previous["dataset_digest"]
    assert reloaded["revision"] == previous["revision"]
    with pytest.raises(SessionError):
        restarted.poll(previous["session_id"], old_token)
    with pytest.raises(SessionError):
        restarted.revise(keys(previous), {"budget": 3}, "stale process change")
    with pytest.raises(SessionError):
        restarted.mark_reviewed(keys(previous, evidence_version=True))
    with pytest.raises(SessionError):
        restarted.capture_export(keys(previous, evidence_version=True))
    assert restarted.read(reloaded["session_id"]) == reloaded


def test_expired_lookup_cannot_attach_after_remote_request_finishes(science, candidates, transport):
    now = [10.0]
    session, before = loaded(science, candidates, clock=lambda: now[0])
    token = session.start_operation(keys(before), "evidence", timeout=30)

    def remote_finishes_after_deadline():
        now[0] = 41.0
        return page("molecules", [])

    transport[0].append(remote_finishes_after_deadline)
    with pytest.raises(SessionError):
        session.lookup(token, candidates[0]["id"])
    after = session.read(before["session_id"])
    assert len(transport[1]) == 1
    assert after["snapshots"] == before["snapshots"]
    assert after["evidence_version"] == before["evidence_version"]
    assert after["selection"] == before["selection"]
    assert after["busy"]
    with pytest.raises(SessionError):
        session.start_operation(keys(after), "evidence")
    session.finish_operation(token, {"status": "failed", "error": {
        "code": "operation_expired", "message": "turn deadline exceeded"}})
    assert not session.read(before["session_id"])["busy"]


def test_completed_measurement_survives_later_operation_failure(science, candidates, transport):
    session, before = loaded(science, candidates)
    transport[0].extend(measured_lookup(candidates[0]["smiles"]))
    token = session.start_operation(keys(before), "evidence")
    snapshot = session.lookup(token, candidates[0]["id"])
    observed = session.read(before["session_id"])
    assert snapshot["result"]["status"] == "ok"
    assert snapshot["result"]["records"][0]["standard_value"] == "10000"
    assert observed["evidence_version"] == before["evidence_version"] + 1

    session.finish_operation(token, {"status": "failed", "error": {
        "code": "provider_failure", "message": "provider failed after lookup"}})
    failed = session.read(before["session_id"])
    assert failed["snapshots"] == observed["snapshots"]
    assert failed["current_evidence"] == observed["current_evidence"]
    assert failed["evidence_version"] == observed["evidence_version"]
    assert failed["selection"] == before["selection"]
    assert not failed["busy"]
    retry = session.start_operation(keys(failed), "evidence")
    assert session.lookup(retry, candidates[0]["id"]) == snapshot
    session.finish_operation(retry, {"status": "complete"})
    assert len(transport[1]) == 3


def test_later_exclusion_and_budget_change_preserve_existing_pin_and_exclusions(
        science, candidates, transport):
    session, state = loaded(science, candidates)
    state = session.revise(keys(state), {"include_ids": ["candidate-20"],
                                        "exclude_ids": ["candidate-15"]}, "require20 exclude15")
    assert [row["id"] for row in state["selection"]["selected"]] == [
        "candidate-10", "candidate-08", "candidate-12", "candidate-20"]
    expanded_exclusions = [*state["constraints"]["exclude_ids"], "candidate-08"]
    state = session.revise(keys(state), {"exclude_ids": expanded_exclusions}, "also exclude08")
    assert state["constraints"]["include_ids"] == ["candidate-20"]
    assert set(state["constraints"]["exclude_ids"]) == {"candidate-15", "candidate-08"}
    assert [row["id"] for row in state["selection"]["selected"]] == [
        "candidate-10", "candidate-09", "candidate-12", "candidate-20"]
    preserved = deepcopy(state["constraints"])
    state = session.revise(keys(state), {"budget": 3}, "reduce capacity")
    assert state["constraints"] == {**preserved, "budget": 3}
    assert [row["id"] for row in state["selection"]["selected"]] == [
        "candidate-10", "candidate-09", "candidate-20"]
    assert transport[1] == []


def test_unselected_lookup_invalidates_review_proposal_and_export_without_changing_batch(
        science, candidates, transport):
    session, initial = loaded(science, candidates)
    reviewed = session.mark_reviewed(keys(initial, evidence_version=True))
    assert reviewed["review"]
    message = "Consider a smaller batch"
    token = session.start_operation(keys(reviewed), "chat", message=message)
    session.interpret(token, message, {"decision": "question", "span": message})
    proposal = session.propose(token, {"budget": 3}, "For the user's review", [])
    session.finish_operation(token, {"status": "complete"})
    before_lookup = session.read(initial["session_id"])
    stale_keys = keys(before_lookup, evidence_version=True)
    selected_ids = {row["id"] for row in initial["selection"]["selected"]}
    unselected = next(row for row in candidates if row["id"] not in selected_ids)
    transport[0].extend(measured_lookup(unselected["smiles"]))
    token = session.start_operation(keys(before_lookup), "evidence")
    snapshot = session.lookup(token, unselected["id"])
    session.finish_operation(token, {"status": "complete"})
    current = session.read(initial["session_id"])

    assert current["revision"] == initial["revision"]
    assert current["constraints"] == initial["constraints"]
    assert current["selection"] == initial["selection"]
    assert current["evidence_version"] == initial["evidence_version"] + 1
    assert current["review"] is None
    assert current["current_evidence"][unselected["id"]] == snapshot["snapshot_id"]
    with pytest.raises(SessionError):
        session.mark_reviewed(stale_keys)
    with pytest.raises(SessionError):
        session.capture_export(stale_keys)
    with pytest.raises(SessionError):
        session.apply_proposal(stale_keys, proposal["proposal_id"], "apply stale suggestion")
    with pytest.raises(SessionError):
        session.apply_proposal(keys(current, evidence_version=True),
                               proposal["proposal_id"], "current tab, stale suggestion")
    assert session.read(initial["session_id"])["selection"] == initial["selection"]
    exported = session.capture_export(keys(current, evidence_version=True))
    assert snapshot in exported["snapshots"]
    assert exported["review"] is None
    exported["snapshots"].clear()
    assert session.read(initial["session_id"])["snapshots"] == current["snapshots"]


def test_failed_refresh_is_current_and_cached_without_erasing_success(science, candidates, transport):
    session, initial = loaded(science, candidates)
    target = candidates[0]
    transport[0].extend(measured_lookup(target["smiles"]))
    token = session.start_operation(keys(initial), "evidence")
    successful = session.lookup(token, target["id"])
    session.finish_operation(token, {"status": "complete"})
    previous = session.read(initial["session_id"])
    session.mark_reviewed(keys(previous, evidence_version=True))

    transport[0].append(URLError("independent test: remote service unavailable"))
    token = session.start_operation(keys(previous), "evidence")
    failure = session.lookup(token, target["id"], refresh=True)
    session.finish_operation(token, {"status": "complete"})
    current = session.read(initial["session_id"])
    assert failure["result"]["status"] == "error"
    assert failure["result"]["records"] == []
    assert current["current_evidence"][target["id"]] == failure["snapshot_id"]
    assert current["snapshots"] == [successful, failure]
    assert current["review"] is None
    assert current["evidence_version"] == previous["evidence_version"] + 1

    token = session.start_operation(keys(current), "evidence")
    assert session.lookup(token, target["id"]) == failure
    session.finish_operation(token, {"status": "complete"})
    reused = session.read(initial["session_id"])
    assert reused["evidence_version"] == current["evidence_version"]
    assert reused["snapshots"] == [successful, failure]
    assert len(transport[1]) == 4
