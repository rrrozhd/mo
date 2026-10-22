from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import threading

import pytest

from src.chembl import EvidenceClient
from src.workbench.science import RecordedEvidence, ScientificOperations


def state_module():
    from src.workbench import session as workbench_state
    return workbench_state


class Screen:
    def score(self, rows):
        return [{"id": row["id"], "canonical_smiles": row["smiles"], "score": .5,
                 "validation_percentile": .5, "nearest_training_similarity": .7,
                 "similarity_flag": False, "confidence": "uncalibrated"} for row in rows]


class Evidence(EvidenceClient):
    def __init__(self):
        self.calls = []
        self.status = "ok"
        self.entered: threading.Event | None = None
        self.release: threading.Event | None = None

    def lookup(self, smiles, limit=10):
        self.calls.append((smiles, limit))
        if self.entered is not None:
            self.entered.set()
            assert self.release is not None
            assert self.release.wait(3)
        return {"query_smiles": smiles, "status": self.status, "records": [],
                "retrieved_at": "2026-10-22T00:00:00+00:00", "truncated": False,
                "error": "offline" if self.status == "error" else None}


def rows():
    return [{"id": " a\n", "smiles": "CCO"}, {"id": "b", "smiles": "CCN"},
            {"id": "c", "smiles": "c1ccccc1"}]


def keys(state, evidence=False):
    result = {"session_id": state["session_id"], "dataset_digest": state["dataset_digest"],
              "base_revision": state["revision"]}
    if evidence:
        result["evidence_version"] = state["evidence_version"]
    return result


def session(loaded=True):
    source = Evidence()
    now = [0.0]
    instance = state_module().Session(ScientificOperations(screen=Screen(), model_digest="a" * 64),
                                      source, clock=lambda: now[0])
    if loaded:
        instance.replace_dataset(keys(instance.bootstrap()), rows(), 2, "Upload candidates")
    return instance, source, now


def current(instance):
    return instance.bootstrap()


def assert_error(code, fn):
    with pytest.raises(state_module().SessionError) as caught:
        fn()
    assert caught.value.code == code


def test_bootstrap_reconnects_empty_then_loaded_state_and_returns_copies():
    s, source, _ = session(False)
    empty = current(s)
    assert empty["dataset"] is None and empty["revision"] == empty["evidence_version"] == 0
    loaded = s.replace_dataset(keys(empty), rows(), 2, "Upload candidates")
    assert loaded["session_id"] != empty["session_id"] and loaded["revision"] == 1
    assert loaded["model_digest"] == "a" * 64 and loaded["source_mode"] == "live"
    assert loaded["constraints"] == {"budget": 2, "include_ids": [], "exclude_ids": [], "max_similarity": .8}
    loaded["selection"]["selected"].clear()
    assert len(s.read(current(s)["session_id"])["selection"]["selected"]) == 2
    assert not source.calls


@pytest.mark.parametrize("mutation", [lambda k: k.pop("session_id"),
    lambda k: k.update(base_revision=True), lambda k: k.update(base_revision="1"),
    lambda k: k.update(dataset_digest=1), lambda k: k.update(extra="ignored")])
def test_invalid_request_keys_are_rejected_without_changes(mutation):
    s, _, _ = session()
    before = current(s)
    request = keys(before)
    mutation(request)
    assert_error("invalid_request", lambda: s.revise(request, {"budget": 1}, "Change budget"))
    assert current(s) == before


def test_revisions_are_atomic_preserve_lists_and_explain_actual_user_action():
    s, _, _ = session()
    first = current(s)
    revised = s.revise(keys(first), {"include_ids": ["c"], "exclude_ids": ["b"]}, "Pin c and exclude b")
    final = s.revise(keys(revised), {"budget": 1}, "Reduce budget")
    assert final["constraints"]["include_ids"] == ["c"] and final["constraints"]["exclude_ids"] == ["b"]
    assert [row["id"] for row in final["selection"]["selected"]] == ["c"]
    revision = final["revisions"][-1]
    assert revision["user_action"] == "Reduce budget" and revision["actor"] == "user"
    assert revision["removed_ids"] == [" a\n"] and revision["retained_ids"] == ["c"]
    assert_error("stale_revision", lambda: s.revise(keys(first), {"budget": 1}, "Stale click"))
    before = current(s)
    assert_error("invalid_constraints", lambda: s.revise(keys(before), {"exclude_ids": ["c"]}, "Conflict"))
    assert current(s) == before
    cleared = s.revise(keys(before), {"include_ids": [], "exclude_ids": []}, "Clear constraints")
    assert cleared["constraints"]["include_ids"] == cleared["constraints"]["exclude_ids"] == []


def test_valid_greedy_shortfall_is_committed():
    s, _, _ = session()
    result = s.revise(keys(current(s)), {"budget": 4}, "Request four slots")
    assert result["revision"] == 2 and not result["selection"]["complete"]
    assert result["selection"]["shortfall_reason"]


@pytest.mark.parametrize("bad_rows,budget", [([], 2), ([{"id": "x", "smiles": "bad"}], 2),
    (rows(), None), (rows(), True), (rows() * 68, 2),
    ([{"id": "x" * 4001, "smiles": "CCO"}], 1)])
def test_failed_replacement_preserves_current_dataset(bad_rows, budget):
    s, _, _ = session()
    before = current(s)
    with pytest.raises(state_module().SessionError):
        s.replace_dataset(keys(before), bad_rows, budget, "Replace")
    after = current(s)
    for field in ("session_id", "dataset", "revision", "selection", "constraints"):
        assert after[field] == before[field]
    assert not after["busy"]


def test_unexpected_scorer_failure_releases_worker_and_preserves_valid_state():
    class FailingScreen(Screen):
        fail = False
        def score(self, rows):
            if self.fail:
                raise RuntimeError("scorer fault")
            return super().score(rows)
    screen = FailingScreen()
    s = state_module().Session(ScientificOperations(screen=screen, model_digest="a" * 64), Evidence())
    s.replace_dataset(keys(current(s)), rows(), 2, "Initial dataset")
    before = current(s)
    screen.fail = True
    with pytest.raises(RuntimeError, match="scorer fault"):
        s.replace_dataset(keys(before), [{"id": "new", "smiles": "CC"}], 1, "Failed replacement")
    after = current(s)
    assert not after["busy"] and after["session_id"] == before["session_id"]
    assert after["dataset"] == before["dataset"] and after["selection"] == before["selection"]
    assert "scorer fault" not in str(after["operations"])
    assert s.revise(keys(after), {"budget": 1}, "Recover")["revision"] == 2


def test_same_and_reordered_dataset_replacements_invalidate_every_old_key():
    s, _, _ = session()
    for replacement in (rows(), list(reversed(rows()))):
        before = current(s)
        after = s.replace_dataset(keys(before), replacement, 2, "Replace same dataset")
        assert after["dataset_digest"] == before["dataset_digest"]
        assert after["session_id"] != before["session_id"]
        assert_error("stale_session", lambda: s.read(before["session_id"]))
        assert_error("stale_session", lambda: s.mark_reviewed(keys(before, True)))
        assert_error("stale_session", lambda: s.capture_export(keys(before, True)))
        assert_error("stale_session", lambda: s.start_operation(keys(before), "evidence"))
    restarted, _, _ = session(False)
    assert_error("stale_session", lambda: restarted.read(current(s)["session_id"]))


def test_cached_failure_and_refresh_history_are_current_and_invalidate_review():
    s, source, _ = session()
    token = s.start_operation(keys(current(s)), "evidence")
    first = s.lookup(token, "b")
    assert s.lookup(token, "b") == first and len(source.calls) == 1
    s.finish_operation(token, {"status": "complete"})
    reviewed = s.mark_reviewed(keys(current(s), True))
    assert reviewed["review"]["evidence_version"] == 1
    token = s.start_operation(keys(current(s)), "evidence")
    source.status = "error"
    failed = s.lookup(token, "b", refresh=True)
    assert failed["result"]["status"] == "error" and failed["snapshot_id"] != first["snapshot_id"]
    assert s.lookup(token, "b") == failed and len(source.calls) == 2
    s.finish_operation(token, {"status": "failed", "error": {"code": "provider_failure", "message": "Offline"}})
    state = current(s)
    assert state["evidence_version"] == 2 and state["review"] is None
    assert state["current_evidence"]["b"] == failed["snapshot_id"]
    assert [row["snapshot_id"] for row in state["snapshots"]] == [first["snapshot_id"], failed["snapshot_id"]]
    assert_error("stale_evidence", lambda: s.capture_export(keys(reviewed, True)))


def test_structure_lookup_shares_candidate_cache_and_retains_other_structures():
    s, source, _ = session()
    token = s.start_operation(keys(current(s)), "evidence")
    candidate = s.lookup_structure(token, "OCC")
    assert candidate["candidate_id"] == " a\n"
    assert s.lookup(token, " a\n") == candidate
    other = s.lookup_structure(token, "ClCC")
    assert other["candidate_id"] is None and other["canonical_query"] == "CCCl"
    assert source.calls == [("CCO", 10), ("CCCl", 10)]
    assert current(s)["current_evidence"] == {" a\n": candidate["snapshot_id"]}
    s.finish_operation(token, {"status": "complete"})
    reviewed = s.mark_reviewed(keys(current(s), True))

    token = s.start_operation(keys(reviewed), "evidence")
    assert s.lookup_structure(token, "CCCl") == other
    assert current(s)["review"] == reviewed["review"]
    assert current(s)["evidence_version"] == 2
    source.status = "error"
    refreshed = s.lookup_structure(token, "CCCl", refresh=True)
    assert refreshed["snapshot_id"] != other["snapshot_id"]
    assert refreshed["result"]["status"] == "error"
    assert current(s)["review"] is None and current(s)["evidence_version"] == 3
    assert s.lookup_structure(token, "ClCC") == refreshed
    assert len(source.calls) == 3


def test_recorded_structure_lookup_never_falls_back_to_live_and_clears_review(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "workbench-evidence.json"
    bundle = json.loads(path.read_text())
    science = ScientificOperations(screen=Screen(), model_digest=bundle["model_digest"])
    s = state_module().Session(science, RecordedEvidence(path, science.model_digest))
    s.replace_dataset(keys(current(s)), rows(), 2, "Load local candidates")
    reviewed = s.mark_reviewed(keys(current(s), True))
    monkeypatch.setattr(EvidenceClient, "lookup", lambda *a, **kw: pytest.fail("Live lookup in recorded mode"))
    token = s.start_operation(keys(reviewed), "evidence")
    entry = bundle["entries"][0]
    captured = s.lookup_structure(token, entry["canonical_query"])
    assert captured["candidate_id"] is None and captured["origin"] == "recorded"
    assert captured["result"] == entry["result"]
    assert captured["fixture_hash"] == sha256(path.read_bytes()).hexdigest()
    assert s.lookup_structure(token, entry["canonical_query"]) == captured
    missing = s.lookup_structure(token, "[13CH3]CCCl")
    assert missing["result"]["status"] == "not_recorded"
    assert missing["candidate_id"] is None and missing["origin"] == "recorded"
    assert current(s)["review"] is None and current(s)["evidence_version"] == 2


def test_chemistry_results_bind_exact_dataset_structures_and_survive_export():
    s, _, now = session()
    token = s.start_operation(keys(current(s)), "chat", message="Compare these molecules")
    args = {"first_smiles": "OCC", "second_smiles": "ClCC"}
    data = {"fingerprint_tanimoto": .3}
    assert_error("invalid_intent", lambda: s.record_tool_result(token, "compare_structures", args, data))
    s.interpret(token, "Compare these molecules", {"decision": "question", "span": ""})
    event = s.record_tool_result(token, "compare_structures", args, data)
    assert event["kind"] == "chemistry"
    assert event["result"] == {"tool": "compare_structures", "arguments": args,
                              "data": data, "candidate_ids": [" a\n"]}
    args["first_smiles"] = "CCN"
    data["fingerprint_tanimoto"] = 0
    retained = s.capture_export(keys(current(s), True))["operations"][-1]["events"][-1]
    assert retained == event
    unrelated = s.record_tool_result(token, "predict_cyp", {"smiles": "CCCl"}, {"score": .2})
    assert unrelated["result"]["candidate_ids"] == []
    now[0] = 121
    assert_error("operation_expired", lambda: s.record_tool_result(token, "predict_cyp", {"smiles": "CCN"}, {}))


def test_live_structure_lookup_uses_session_deadline_and_discards_late_result(monkeypatch):
    import io
    from src import chembl as evidence

    now, requests = [40.0], []
    source = EvidenceClient()
    s = state_module().Session(ScientificOperations(screen=Screen(), model_digest="a" * 64),
                              source, clock=lambda: now[0])
    s.replace_dataset(keys(current(s)), rows(), 2, "Load local candidates")
    token = s.start_operation(keys(current(s)), "evidence", timeout=.5)

    def respond(request, timeout):
        requests.append((request, timeout))
        now[0] = 41.0
        return io.BytesIO(json.dumps({"molecules": [{"molecule_chembl_id": "CHEMBL1",
            "molecule_structures": {"canonical_smiles": "CCCl"}}],
            "page_meta": {"total_count": 1, "next": None}}).encode())

    monkeypatch.setattr(evidence, "urlopen", respond)
    assert_error("operation_expired", lambda: s.lookup_structure(token, "CCCl"))
    assert len(requests) == 1 and requests[0][1] == pytest.approx(.5)
    assert not current(s)["snapshots"] and current(s)["evidence_version"] == 0


def test_network_does_not_hold_lock_and_expired_result_cannot_attach():
    s, source, now = session()
    source.entered, source.release = threading.Event(), threading.Event()
    before = current(s)
    token = s.start_operation(keys(before), "evidence", timeout=1)
    failures = []
    def lookup():
        try:
            s.lookup(token, "b")
        except state_module().SessionError as exc:
            failures.append(exc.code)
    worker = threading.Thread(target=lookup)
    worker.start()
    assert source.entered.wait(2)
    state = s.read(before["session_id"])
    assert state["busy"] and not state["snapshots"]
    assert_error("busy", lambda: s.revise(keys(state), {"budget": 1}, "Conflicting click"))
    assert_error("busy", lambda: s.replace_dataset(keys(state), rows(), 2, "Replace during lookup"))
    now[0] = 2
    assert s.poll(before["session_id"], token)["status"] == "failed"
    assert current(s)["busy"]
    source.release.set()
    worker.join(3)
    assert failures == ["operation_expired"] and not current(s)["snapshots"]
    s.finish_operation(token, {"status": "complete"})
    assert not current(s)["busy"]
    assert s.poll(before["session_id"], token)["status"] == "failed"


def test_explicit_intent_adds_exclusions_and_later_budget_preserves_all_constraints():
    s, _, _ = session()
    s.revise(keys(current(s)), {"include_ids": ["c"], "exclude_ids": ["b"]}, "Initial constraints")
    message = "Also exclude a"
    token = s.start_operation(keys(current(s)), "chat", message=message)
    raw = {"decision": "explicit_change", "span": message, "add_exclude_ids": [" a\n"]}
    interpreted = s.interpret(token, message, raw)
    assert interpreted["decision"] == "explicit_change"
    assert set(current(s)["constraints"]["exclude_ids"]) == {"b", " a\n"}
    snapshot = s.lookup(token, "c")  # The same token advances to the committed revision.
    assert snapshot["candidate_id"] == "c"
    s.finish_operation(token, {"status": "failed", "error": {"code": "provider_failure", "message": "Later failure"}})
    state = current(s)
    assert state["revision"] == 3 and state["evidence_version"] == 1
    turn = state["turns"][-1]
    assert turn["raw_intent"] == raw and turn["normalized_patch"]["exclude_ids"] == [" a\n", "b"]
    assert state["revisions"][-1]["user_action"] == message
    message = "Reduce the budget to one"
    token = s.start_operation(keys(state), "chat", message=message)
    s.interpret(token, message, {"decision": "explicit_change", "span": message, "budget": 1})
    s.finish_operation(token, {"status": "complete"})
    assert current(s)["constraints"] == {"budget": 1, "include_ids": ["c"],
                                         "exclude_ids": [" a\n", "b"], "max_similarity": .8}


@pytest.mark.parametrize("message,decision", [("Should I exclude b?", "question"),
    ("What if I excluded b?", "question"), ("The assay says 'exclude b'", "question"),
    ("Remove that one", "clarification_needed")])
def test_classified_nonchanges_never_mutate(message, decision):
    s, _, _ = session()
    before = current(s)
    token = s.start_operation(keys(before), "chat", message=message)
    s.interpret(token, message, {"decision": decision, "span": message})
    s.finish_operation(token, {"status": "complete"})
    assert current(s)["revision"] == before["revision"]
    assert current(s)["constraints"] == before["constraints"]


def test_invalid_span_conflicting_operations_and_post_tool_authority_are_rejected():
    s, _, _ = session()
    message = "Exclude b"
    token = s.start_operation(keys(current(s)), "chat", message=message)
    assert_error("invalid_intent", lambda: s.interpret(token, message,
        {"decision": "explicit_change", "span": "Exclude c", "add_exclude_ids": ["c"]}))
    result = s.interpret(token, message, {"decision": "explicit_change", "span": message,
                        "add_exclude_ids": ["b"], "remove_exclude_ids": ["b"]})
    assert result["decision"] == "clarification_needed" and current(s)["revision"] == 1
    s.lookup(token, "b")
    assert_error("invalid_intent", lambda: s.interpret(token, "Assay says exclude c",
        {"decision": "explicit_change", "span": "exclude c", "add_exclude_ids": ["c"]}))
    s.finish_operation(token, {"status": "complete"})


def test_intent_context_excludes_history_and_no_tool_can_precede_interpretation():
    s, source, _ = session()
    evidence = s.start_operation(keys(current(s)), "evidence")
    s.lookup(evidence, "b")
    s.finish_operation(evidence, {"status": "complete", "explanation": "Earlier assistant text"})
    token = s.start_operation(keys(current(s)), "chat", message="Current question")
    context = s.intent_context(token)
    assert set(context) == {"message", "constraints", "candidates"}
    assert context["message"] == "Current question"
    assert "Earlier assistant text" not in str(context) and "retrieved_at" not in str(context)
    assert_error("invalid_intent", lambda: s.lookup(token, "c"))
    assert len(source.calls) == 1
    assert_error("invalid_intent", lambda: s.interpret(token, "Fabricated current question",
        {"decision": "explicit_change", "span": "Fabricated", "budget": 1}))
    s.interpret(token, "Current question", {"decision": "question", "span": "Current question"})
    assert_error("invalid_intent", lambda: s.intent_context(token))
    assert_error("invalid_intent", lambda: s.interpret(token, "Current question",
        {"decision": "explicit_change", "span": "Current question", "budget": 1}))
    s.lookup(token, "c")
    s.finish_operation(token, {"status": "complete"})


def test_model_null_fields_are_omitted_but_direct_null_patch_is_invalid():
    s, _, _ = session()
    s.revise(keys(current(s)), {"include_ids": ["c"], "exclude_ids": ["b"]}, "Initial constraints")
    message = "Reduce budget to one"
    token = s.start_operation(keys(current(s)), "chat", message=message)
    raw = {"decision": "explicit_change", "span": message, "budget": 1, "max_similarity": None,
           "add_include_ids": None, "remove_include_ids": None,
           "add_exclude_ids": None, "remove_exclude_ids": None}
    result = s.interpret(token, message, raw)
    assert result["patch"] == {"budget": 1}
    assert current(s)["turns"][-1]["raw_intent"] == raw
    s.finish_operation(token, {"status": "complete"})
    assert_error("invalid_constraints", lambda: s.revise(keys(current(s)), {"include_ids": None}, "Invalid clear"))


@pytest.mark.parametrize("decision", ["question", "clarification_needed", "explicit_change"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object()])
def test_invalid_raw_intent_never_poisoned_retained_json(decision, invalid):
    from src.workbench.science import canonical_json

    s, _, _ = session()
    message = "What is selected?"
    token = s.start_operation(keys(current(s)), "chat", message=message)
    assert_error("invalid_intent", lambda: s.interpret(token, message,
        {"decision": decision, "span": message, "budget": invalid}))
    assert current(s)["turns"][-1]["raw_intent"] is None
    canonical_json(current(s))
    raw = {"decision": "question", "span": message, "budget": None, "add_include_ids": None}
    s.interpret(token, message, raw)
    assert current(s)["turns"][-1]["raw_intent"] == raw
    canonical_json(current(s))
    s.finish_operation(token, {"status": "complete"})


def test_same_token_cannot_overlap_network_or_finish_before_worker_unwinds():
    s, source, _ = session()
    source.entered, source.release = threading.Event(), threading.Event()
    token = s.start_operation(keys(current(s)), "evidence")
    worker = threading.Thread(target=lambda: s.lookup(token, "b"))
    worker.start()
    assert source.entered.wait(2)
    assert_error("busy", lambda: s.lookup(token, "c"))
    assert_error("busy", lambda: s.finish_operation(token, {"status": "complete"}))
    source.release.set()
    worker.join(3)
    s.finish_operation(token, {"status": "complete"})
    assert len(source.calls) == 1


@pytest.mark.parametrize("next_call", ["finish", "lookup"])
def test_completed_lookup_attaches_before_releasing_lock_to_other_calls(monkeypatch, next_call):
    s, source, _ = session()
    token = s.start_operation(keys(current(s)), "evidence")
    observations = []
    real_lock = threading.RLock()

    class ReleaseBoundary:
        """Interleave a second call at the first lock release after the HTTP return."""
        def __enter__(self):
            real_lock.acquire()

        def __exit__(self, *_):
            real_lock.release()
            if source.calls and not observations:
                observations.append(True)  # Disable the hook before the nested call.
                observations[0] = bool(current(s)["snapshots"])
                if next_call == "finish":
                    s.finish_operation(token, {"status": "complete"})
                else:
                    s.lookup(token, "c")

    monkeypatch.setattr(s, "_lock", ReleaseBoundary())
    failures = []
    try:
        first = s.lookup(token, "b")
    except state_module().SessionError as exc:
        failures.append(exc.code)
        first = None
    assert observations == [True] and not failures
    state = current(s)
    assert first == state["snapshots"][0]
    assert [snapshot["candidate_id"] for snapshot in state["snapshots"]] == (
        ["b"] if next_call == "finish" else ["b", "c"])
    if next_call == "lookup":
        s.finish_operation(token, {"status": "complete"})


def test_failed_snapshot_validation_releases_lookup_flag_without_erasing_completed_work(monkeypatch):
    s, source, _ = session()
    token = s.start_operation(keys(current(s)), "evidence")
    first = s.lookup(token, "b")
    original = source.lookup

    def mismatched_query(smiles, limit=10):
        result = original(smiles, limit)
        result["query_smiles"] = "CC"
        return result

    monkeypatch.setattr(source, "lookup", mismatched_query)
    with pytest.raises(ValueError, match="query does not match"):
        s.lookup(token, "c")
    assert current(s)["snapshots"] == [first]
    s.finish_operation(token, {"status": "failed", "error": {"code": "invalid_evidence", "message": "Invalid query"}})
    assert not current(s)["busy"]


@pytest.mark.parametrize("budget", [1, 2])
def test_intent_rechecks_deadline_after_selection_before_commit(monkeypatch, budget):
    s, _, now = session()
    token = s.start_operation(keys(current(s)), "evidence")
    first = s.lookup(token, "b")
    s.finish_operation(token, {"status": "complete"})
    before = current(s)
    message = "Set the budget"
    token = s.start_operation(keys(before), "chat", message=message, timeout=1)
    select = s._science.select

    def late_selection(dataset, constraints):
        result = select(dataset, constraints)
        now[0] = 1
        return result

    monkeypatch.setattr(s._science, "select", late_selection)
    assert_error("operation_expired", lambda: s.interpret(token, message,
        {"decision": "explicit_change", "span": message, "budget": budget}))
    after = current(s)
    assert after["revision"] == before["revision"] and after["constraints"] == before["constraints"]
    assert after["snapshots"] == [first] and after["turns"][-1]["raw_intent"] is None
    assert after["busy"] and after["operation"]["status"] == "failed"
    s.finish_operation(token, {"status": "failed"})
    assert not current(s)["busy"]


@pytest.mark.parametrize("budget", [1, 2])
def test_post_commit_expiry_keeps_intent_record_only_when_a_revision_committed(monkeypatch, budget):
    s, _, now = session()
    before = current(s)
    message = "Set the budget"
    token = s.start_operation(keys(before), "chat", message=message, timeout=1)
    raw = {"decision": "explicit_change", "span": message, "budget": budget}
    commit = s._commit

    def expire_after_commit(*args, **kwargs):
        commit(*args, **kwargs)
        now[0] = 2

    monkeypatch.setattr(s, "_commit", expire_after_commit)
    if budget == 1:
        interpretation = s.interpret(token, message, raw)
        assert interpretation == {"decision": "explicit_change", "patch": {"budget": 1},
                                  "span": message, "clarification": None}
        after = current(s)
        assert after["revision"] == before["revision"] + 1
        assert after["turns"][-1]["raw_intent"] == raw
        assert after["turns"][-1]["normalized_patch"] == {"budget": 1}
        assert after["turns"][-1]["interpretation"] == interpretation
    else:
        assert_error("operation_expired", lambda: s.interpret(token, message, raw))
        after = current(s)
        assert after["revision"] == before["revision"]
        assert after["turns"][-1]["raw_intent"] is None
    assert s.poll(after["session_id"], token)["status"] == "failed"
    assert after["busy"]
    s.finish_operation(token, {"status": "complete"})
    assert not current(s)["busy"]
    assert s.poll(after["session_id"], token)["status"] == "failed"


@pytest.mark.parametrize("kind", ["proposal", "preview", "comparison", "inspection", "cached_lookup"])
def test_local_results_recheck_deadline_without_erasing_completed_work(monkeypatch, kind):
    s, _, now = session()
    token = s.start_operation(keys(current(s)), "evidence", timeout=1)
    s.lookup(token, "b")
    s.propose(token, {"budget": 1}, "Completed proposal", [])
    before = current(s)
    target, method = ((s._science, "compare") if kind == "comparison" else
                      (s, "_known_ids") if kind in {"inspection", "cached_lookup"} else
                      (s._science, "select"))
    original = getattr(target, method)

    def late_calculation(*args):
        result = original(*args)
        now[0] = 1
        return result

    monkeypatch.setattr(target, method, late_calculation)
    actions = {"proposal": lambda: s.propose(token, {"budget": 3}, "Late proposal", []),
               "preview": lambda: s.preview_constraints(token, {"budget": 3}),
               "comparison": lambda: s.compare(token, "b", "c"),
               "inspection": lambda: s.inspect(token, ["c"]),
               "cached_lookup": lambda: s.lookup(token, "b")}
    assert_error("operation_expired", actions[kind])
    after = current(s)
    for field in ("revision", "constraints", "snapshots", "evidence_version", "proposals"):
        assert after[field] == before[field]
    assert after["operation"]["events"] == before["operation"]["events"]
    s.finish_operation(token, {"status": "complete"})
    assert s.poll(after["session_id"], token)["status"] == "failed"


def test_recorded_session_provenance_cache_and_live_restart_are_bound_to_launch_mode():
    path = Path(__file__).resolve().parents[1] / "examples" / "workbench-evidence.json"
    bundle = json.loads(path.read_text())
    science = ScientificOperations(screen=Screen(), model_digest=bundle["model_digest"])
    source = RecordedEvidence(path, science.model_digest)
    s = state_module().Session(science, source)
    entry = bundle["entries"][0]
    candidate = {"id": entry["candidate_id"], "smiles": entry["canonical_query"]}
    s.replace_dataset(keys(current(s)), [candidate], 1, "Load recorded candidate")
    assert current(s)["source_mode"] == "recorded"
    token = s.start_operation(keys(current(s)), "evidence")
    first = s.lookup(token, candidate["id"])
    assert first["origin"] == "recorded" and first["fixture_hash"] == sha256(path.read_bytes()).hexdigest()
    assert first["playback_at"] and first["retrieved_at"] == entry["result"]["retrieved_at"]
    assert first["result"] == entry["result"]
    assert s.lookup(token, candidate["id"]) == first
    assert current(s)["evidence_version"] == 1
    s.finish_operation(token, {"status": "complete"})
    reviewed = s.mark_reviewed(keys(current(s), True))
    assert reviewed["review"] is not None

    live = state_module().Session(science, Evidence())
    restarted = current(live)
    assert restarted["source_mode"] == "live" and restarted["session_id"] != reviewed["session_id"]
    assert restarted["dataset"] is None and restarted["revision"] == restarted["evidence_version"] == 0
    assert restarted["review"] is None and not restarted["snapshots"] and not restarted["turns"]
    assert_error("stale_session", lambda: live.read(reviewed["session_id"]))
    assert_error("stale_session", lambda: live.mark_reviewed(keys(reviewed, True)))
    assert_error("stale_session", lambda: live.capture_export(keys(reviewed, True)))
    assert_error("stale_session", lambda: live.start_operation(keys(reviewed), "evidence"))
    other_model = ScientificOperations(screen=Screen(), model_digest="a" * 64)
    assert_error("invalid_request", lambda: state_module().Session(other_model, source))


def test_proposals_need_explicit_application_and_current_evidence():
    s, _, _ = session()
    token = s.start_operation(keys(current(s)), "evidence")
    snapshot = s.lookup(token, "b")
    proposal = s.propose(token, {"exclude_ids": ["b"]}, "Review this observation", [snapshot["snapshot_id"]])
    assert current(s)["revision"] == 1
    s.finish_operation(token, {"status": "complete"})
    state = s.apply_proposal(keys(current(s), True), proposal["proposal_id"], "Apply proposal")
    assert state["constraints"]["exclude_ids"] == ["b"]
    assert state["revisions"][-1]["proposal_id"] == proposal["proposal_id"]
    token = s.start_operation(keys(state), "evidence")
    stale = s.propose(token, {"budget": 1}, "Consider one slot", [])
    s.lookup(token, "c")
    s.finish_operation(token, {"status": "complete"})
    assert_error("stale_proposal", lambda: s.apply_proposal(keys(current(s), True), stale["proposal_id"], "Apply stale"))


def test_export_during_operation_is_immutable_and_review_invalidation_is_exact():
    s, _, _ = session()
    reviewed = s.mark_reviewed(keys(current(s), True))
    token = s.start_operation(keys(reviewed), "evidence")
    export = s.capture_export(keys(reviewed, True))
    s.lookup(token, "c")  # Unselected evidence also invalidates review.
    s.finish_operation(token, {"status": "complete"})
    assert export["review"] and export["evidence_version"] == 0 and not export["snapshots"]
    assert current(s)["review"] is None and current(s)["evidence_version"] == 1
    reviewed = s.mark_reviewed(keys(current(s), True))
    assert s.revise(keys(reviewed), {"budget": 1}, "Reduce")["review"] is None


def test_read_tools_return_bound_results_without_constraint_changes():
    s, _, _ = session()
    token = s.start_operation(keys(current(s)), "evidence")
    assert [r["id"] for r in s.inspect(token, ["b", "c"])["result"]] == ["b", "c"]
    compared = s.compare(token, "b", "c")
    assert compared["result"]["a"]["id"] == "b" and compared["result"]["evidence_refs"] == {"b": [], "c": []}
    preview = s.preview_constraints(token, {"budget": 1})
    assert preview["result"]["constraints"]["budget"] == 1
    assert current(s)["constraints"]["budget"] == 2
    s.finish_operation(token, {"status": "complete"})


def test_turn_and_revision_caps_reject_excess_without_erasing_history():
    s, _, _ = session()
    for index in range(100):
        token = s.start_operation(keys(current(s)), "chat", message=f"Question {index}")
        s.finish_operation(token, {"status": "failed", "error": {"code": "unavailable", "message": "No model"}})
    assert_error("limit_exceeded", lambda: s.start_operation(keys(current(s)), "chat", message="One more"))
    assert len(current(s)["turns"]) == 100
    for index in range(99):
        s.revise(keys(current(s)), {"budget": 1 if index % 2 == 0 else 2}, "Change budget")
    assert current(s)["revision"] == 100 and len(current(s)["revisions"]) == 100
    assert_error("limit_exceeded", lambda: s.revise(keys(current(s)), {"budget": 2}, "One more"))


def test_safe_action_and_message_validation_do_not_store_arbitrary_objects():
    s, _, _ = session()
    assert_error("invalid_request", lambda: s.revise(keys(current(s)), {"budget": 1}, {"secret": "not an action"}))
    assert_error("limit_exceeded", lambda: s.start_operation(keys(current(s)), "chat", message="x" * 8001))
    token = s.start_operation(keys(current(s)), "evidence")
    assert_error("invalid_request", lambda: s.finish_operation(token, {"status": "complete", "hidden_reasoning": "private"}))
    s.finish_operation(token, {"status": "complete"})
