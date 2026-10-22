"""Starting a fresh conversation retains the scientific audit and invalidates old requests."""

import json

import pytest

from test_workbench_assistant import Harness, completion, final, intent, keys
from test_workbench_gateway import Harness as GatewayHarness, keys as gateway_keys
from test_workbench_state import assert_error, current, session


@pytest.mark.parametrize("loaded", [False, True])
def test_new_chat_archives_conversation_and_retains_scientific_state(loaded):
    s, _, _ = session(loaded)
    message = "Research this molecule"
    token = s.start_operation(keys(current(s)), "chat", message=message)
    s.interpret(token, message, {"decision": "question", "span": message})
    snapshot = s.lookup_structure(token, "CCO")
    if loaded:
        s.propose(token, {"budget": 1}, "Investigate one compound", [snapshot["snapshot_id"]])
    s.finish_operation(token, {"status": "complete", "explanation": "Retained answer"})
    if loaded:
        s.mark_reviewed(keys(current(s), evidence=True))
    before = current(s)
    after = s.new_chat(keys(before, evidence=True))
    assert after["session_id"] != before["session_id"]
    assert after["turns"] == after["proposals"] == after["operations"] == []
    assert not after["busy"]
    for name in ("dataset", "constraints", "selection", "revision", "revisions", "snapshots",
                 "current_evidence", "evidence_version", "dataset_digest", "model_digest", "source_mode"):
        assert after[name] == before[name]
    if loaded:
        assert after["review"] == {**before["review"], "session_id": after["session_id"]}
    archive = after["archived_chats"][0]
    assert archive == {name: before[name] for name in ("chat_id", "session_id", "turns", "proposals", "operations")}
    assert not any(name.startswith("_") for operation in archive["operations"] for name in operation)
    exported = s.capture_export(keys(after, evidence=True))
    assert exported["archived_chats"] == [archive]
    assert_error("stale_session", lambda: s.new_chat(keys(before, evidence=True)))
    assert_error("stale_session", lambda: s.read(before["session_id"]))
    if loaded:
        assert_error("stale_proposal", lambda: s.apply_proposal(keys(after, evidence=True),
                     before["proposals"][0]["proposal_id"], "Apply old proposal"))


def test_new_chat_requires_idle_current_evidence_and_complete_keys():
    s, _, now = session(False)
    before = current(s)
    assert_error("invalid_request", lambda: s.new_chat(keys(before)))
    stale = {**keys(before, evidence=True), "evidence_version": 1}
    assert_error("stale_evidence", lambda: s.new_chat(stale))
    token = s.start_operation(keys(before), "chat", message="Question", timeout=1)
    assert_error("busy", lambda: s.new_chat(keys(current(s), evidence=True)))
    now[0] = 2
    assert_error("busy", lambda: s.new_chat(keys(current(s), evidence=True)))
    s.finish_operation(token, {"status": "failed"})
    assert s.new_chat(keys(current(s), evidence=True))["turns"] == []


def test_new_chat_http_route_rejects_repeated_request_and_retains_empty_chats():
    h = GatewayHarness()
    body = {"keys": gateway_keys(h.initial, reviewed=True)}
    response = h.post("/api/chat/new", body)
    assert response.status_code == 200
    archive = response.get_json()["state"]["archived_chats"][0]
    assert archive["chat_id"] == h.initial["chat_id"]
    assert archive["turns"] == archive["operations"] == []
    repeated = h.post("/api/chat/new", body)
    assert repeated.status_code == 409
    assert repeated.get_json()["error"]["code"] == "stale_session"


def test_new_chat_keeps_provider_context_and_paid_ledger_separate():
    h = Harness([completion(intent()), completion(final(explanation="OLD CHAT TEXT")),
                 completion(intent()), completion(final())], candidates=[])
    h.run()
    settled = list(h.ledger.settlements)
    h.session.new_chat(keys(h.state(), evidence=True))
    assert h.ledger.settlements == settled
    h.run()
    context = json.loads(h.requests[-1]["messages"][1]["content"])
    assert context["earlier_exchanges"] == []
    assert "OLD CHAT TEXT" not in json.dumps(h.requests[-1])
    assert len(h.ledger.settlements) == 4


def test_new_chat_cannot_reset_session_turn_or_archive_limits(monkeypatch):
    monkeypatch.setattr("src.workbench.session.MAX_TURNS", 2)
    h = Harness([completion(intent()), completion(final()), completion(intent()), completion(final())], candidates=[])
    for _ in range(2):
        h.run()
        h.session.new_chat(keys(h.state(), evidence=True))
    assert_error("limit_exceeded", lambda: h.session.start_operation(keys(h.state()), "chat", message="Third"))
    assert len(h.state()["archived_chats"]) == 2


def test_archive_limit_also_bounds_chats_with_only_direct_operations(monkeypatch):
    monkeypatch.setattr("src.workbench.session.MAX_TURNS", 1)
    s, _, _ = session()
    for _ in range(2):
        token = s.start_operation(keys(current(s)), "evidence")
        s.finish_operation(token, {"status": "complete"})
        if not current(s)["archived_chats"]:
            s.new_chat(keys(current(s), evidence=True))
    before = current(s)
    assert_error("limit_exceeded", lambda: s.new_chat(keys(before, evidence=True)))
    assert current(s) == before


def test_switch_chat_restores_conversation_and_preserves_shared_science():
    s, _, _ = session()
    first_id = current(s)["chat_id"]
    token = s.start_operation(keys(current(s)), "chat", message="Investigate ethanol")
    s.interpret(token, "Investigate ethanol", {"decision": "question", "span": "Investigate ethanol"})
    snapshot = s.lookup_structure(token, "CCO")
    proposal = s.propose(token, {"budget": 1}, "One compound", [snapshot["snapshot_id"]])
    s.finish_operation(token, {"status": "complete", "explanation": "First answer"})
    original = current(s)
    second = s.new_chat(keys(original, evidence=True))
    s.revise(keys(second), {"budget": 3}, "Keep current shared batch")
    before_switch = current(s)
    first = s.switch_chat(keys(before_switch, evidence=True), first_id)
    assert first["chat_id"] == first_id
    assert first["session_id"] not in {original["session_id"], second["session_id"]}
    assert first["turns"] == original["turns"] and first["operations"] == original["operations"]
    assert first["constraints"]["budget"] == 3 and first["snapshots"] == original["snapshots"]
    assert first["proposals"][0]["session_id"] == original["session_id"]
    assert_error("stale_proposal", lambda: s.apply_proposal(keys(first, evidence=True), proposal["proposal_id"], "Old"))
    assert_error("stale_session", lambda: s.switch_chat(keys(before_switch, evidence=True), first_id))
    restored_second = s.switch_chat(keys(first, evidence=True), second["chat_id"])
    assert restored_second["turns"] == [] and restored_second["chat_id"] == second["chat_id"]
    assert len(restored_second["archived_chats"]) == 1
    assert restored_second["archived_chats"][0]["chat_id"] == first_id


def test_switch_chat_http_checks_identity_evidence_and_busy_state():
    h = GatewayHarness()
    first_id = h.initial["chat_id"]
    h.session.new_chat(gateway_keys(h.initial, reviewed=True))
    before = h.state()
    for changed, status in (({"chat_id": "unknown"}, 400),
                            ({"keys": gateway_keys(before)}, 400),
                            ({"keys": {**gateway_keys(before, reviewed=True), "evidence_version": 1}}, 409)):
        payload = {"keys": gateway_keys(before, reviewed=True), "chat_id": first_id, **changed}
        assert h.post("/api/chat/switch", payload).status_code == status
        assert h.state() == before
    token = h.session.start_operation(gateway_keys(before), "chat", message="Working")
    assert h.post("/api/chat/switch", {"keys": gateway_keys(h.state(), reviewed=True), "chat_id": first_id}).status_code == 409
    h.session.finish_operation(token, {"status": "complete"})
    response = h.post("/api/chat/switch", {"keys": gateway_keys(h.state(), reviewed=True), "chat_id": first_id})
    assert response.status_code == 200 and response.get_json()["state"]["chat_id"] == first_id


def test_restored_chat_continues_its_own_provider_context_without_duplicate_turns(monkeypatch):
    monkeypatch.setattr("src.workbench.session.MAX_TURNS", 2)
    h = Harness([completion(intent()), completion(final(explanation="First chat answer")),
                 completion(intent()), completion(final())], candidates=[])
    first_id = h.state()["chat_id"]
    h.run()
    second_id = h.session.new_chat(keys(h.state(), evidence=True))["chat_id"]
    for _ in range(3):
        h.session.switch_chat(keys(h.state(), evidence=True), first_id)
        h.session.switch_chat(keys(h.state(), evidence=True), second_id)
    h.session.switch_chat(keys(h.state(), evidence=True), first_id)
    h.run()
    context = json.loads(h.requests[-1]["messages"][1]["content"])
    assert context["earlier_exchanges"][-1]["content"] == "First chat answer"
    assert len(h.state()["turns"]) == 2 and len(h.state()["archived_chats"]) == 1
    assert h.state()["archived_chats"][0]["turns"] == []
    assert len(h.ledger.settlements) == 4
    assert_error("limit_exceeded", lambda: h.session.start_operation(keys(h.state()), "chat", message="Third"))
