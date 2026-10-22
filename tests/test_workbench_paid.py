from decimal import Decimal
import fcntl
import json
import multiprocessing
import os
from pathlib import Path

import pytest


def paid():
    from src.workbench import paid_ledger as workbench_paid
    return workbench_paid


def usage(cost=.001, prompt=100, completion=20):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "cost": cost}


def test_pending_reservations_survive_restart_and_seventeenth_is_refused(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    for number in range(16):
        ledger.reserve(f"operation-{number}", "intent")
    resumed = paid().FixedPaidLedger(path)
    assert Decimal(resumed.summary()["reserved_usd"]) == Decimal("4.98")
    assert Decimal(resumed.summary()["remaining_usd"]) == Decimal(".02")
    with pytest.raises(paid().LedgerError) as caught:
        resumed.reserve("last", "investigation")
    assert caught.value.code == "budget_exceeded"


def test_valid_settlement_replaces_reserve_and_is_idempotent(tmp_path):
    ledger = paid().FixedPaidLedger(tmp_path / "calls.jsonl")
    reservation = ledger.reserve("operation", "intent")
    ledger.settle(reservation, "request-1", "resolved-model-alias", usage())
    ledger.settle(reservation, "request-1", "resolved-model-alias", usage())
    summary = ledger.summary()
    assert Decimal(summary["settled_usd"]) == Decimal(".001")
    assert Decimal(summary["reserved_usd"]) == 0
    assert summary["settled_count"] == 1 and not summary["blocked"]


def test_missing_cost_blocks_future_calls_across_restart(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    reservation = ledger.reserve("operation", "intent")
    unknown = usage()
    del unknown["cost"]
    with pytest.raises(paid().LedgerError) as caught:
        ledger.settle(reservation, "request-1", paid().MODEL, unknown)
    assert caught.value.code == "invalid_usage"
    resumed = paid().FixedPaidLedger(path)
    assert resumed.summary()["blocked"]
    assert Decimal(resumed.summary()["reserved_usd"]) == Decimal(".31125")
    with pytest.raises(paid().LedgerError):
        resumed.reserve("another", "intent")


def test_reservation_is_fsynced_before_provider_can_run(tmp_path, monkeypatch):
    path = tmp_path / "new" / "calls.jsonl"
    synced = []
    original = os.fsync

    def sync(descriptor):
        original(descriptor)
        synced.append(os.fstat(descriptor).st_ino)

    monkeypatch.setattr(paid().os, "fsync", sync)
    ledger = paid().FixedPaidLedger(path)
    assert path.parent.stat().st_ino in synced
    synced.clear()
    reservation = ledger.reserve("operation", "intent")
    # This is the boundary a provider callback observes, without another ledger
    # transaction obscuring whether reserve itself synced the write.
    assert path.stat().st_ino in synced
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["reservation_id"] == reservation
    assert records[-1]["kind"] == "reserve"


def test_failed_sync_never_returns_permission_to_call_and_never_refunds(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    original = os.fsync
    calls = []

    def broken_sync(descriptor):
        if os.fstat(descriptor).st_ino == path.stat().st_ino:
            raise OSError("private filesystem detail")
        original(descriptor)

    monkeypatch.setattr(paid().os, "fsync", broken_sync)
    with pytest.raises(paid().LedgerError) as caught:
        ledger.reserve("operation", "intent")
        calls.append("provider invoked")
    assert caught.value.code == "ledger_io_error"
    assert str(caught.value) == "ledger_io_error" and not calls
    monkeypatch.setattr(paid().os, "fsync", original)
    assert paid().FixedPaidLedger(path).summary()["unsettled_count"] == 1


def _contender(path, ready, start, result):
    ready.put(True)
    start.wait(10)
    try:
        ledger = paid().FixedPaidLedger(path)
        ledger.reserve(f"worker-{os.getpid()}", "intent")
        result.put("reserved")
    except paid().LedgerError as exc:
        result.put(exc.code)


def test_two_processes_cannot_spend_the_same_final_reservation(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    for number in range(15):
        ledger.reserve(str(number), "intent")
    context = multiprocessing.get_context("spawn")
    ready, result, start = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=_contender, args=(str(path), ready, start, result)) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        assert ready.get(timeout=10) and ready.get(timeout=10)
        start.set()
        outcomes = [result.get(timeout=10), result.get(timeout=10)]
        assert outcomes.count("reserved") == 1
        assert set(outcomes) <= {"reserved", "budget_exceeded", "ledger_busy"}
    finally:
        for worker in workers:
            worker.join(10)
            if worker.is_alive():
                worker.terminate()
                worker.join(10)
    assert ledger.summary()["reservation_count"] == 16
    assert Decimal(ledger.summary()["committed_usd"]) == Decimal("4.98")


def test_separate_instances_reload_and_lock_remains_exclusive(tmp_path):
    path = tmp_path / "calls.jsonl"
    first, second = paid().FixedPaidLedger(path), paid().FixedPaidLedger(path)
    first.reserve("first", "intent")
    assert second.summary()["unsettled_count"] == 1
    with first.lock_path.open("a+b") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(paid().LedgerError) as caught:
                second.reserve("second", "intent")
            assert caught.value.code == "ledger_busy"
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
    second.reserve("second", "intent")
    assert first.summary()["unsettled_count"] == 2


@pytest.mark.parametrize("cost", [None, "0.001", True, -.1, float("nan"), float("inf")])
def test_invalid_cost_is_never_zero_or_a_refund_and_blocks_restart(tmp_path, cost):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    reservation = ledger.reserve("op", "intent")
    with pytest.raises(paid().LedgerError) as caught:
        ledger.settle(reservation, "request", paid().MODEL, usage(cost))
    assert caught.value.code == "invalid_usage"
    summary = paid().FixedPaidLedger(path).summary()
    assert summary["blocked"] and summary["block_codes"] == ["invalid_usage"]
    assert Decimal(summary["reserved_usd"]) == paid().RESERVATION
    assert Decimal(summary["settled_usd"]) == 0


@pytest.mark.parametrize("change", [{"prompt_tokens": True}, {"completion_tokens": -1},
                                   {"total_tokens": 999}, {"prompt_tokens": "100"}])
def test_invalid_token_metadata_keeps_known_cost_and_blocks(tmp_path, change):
    ledger = paid().FixedPaidLedger(tmp_path / "calls.jsonl")
    reservation = ledger.reserve("op", "intent")
    malformed = usage(.02)
    malformed.update(change)
    with pytest.raises(paid().LedgerError) as caught:
        ledger.settle(reservation, "request", paid().MODEL, malformed)
    assert caught.value.code == "invalid_usage"
    summary = ledger.summary()
    assert summary["blocked"] and Decimal(summary["settled_usd"]) == Decimal(".02")
    assert Decimal(summary["reserved_usd"]) == 0


def test_overrun_retains_actual_cost_and_blocks_instead_of_capping_it(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    reservation = ledger.reserve("op", "intent")
    with pytest.raises(paid().LedgerError) as caught:
        ledger.settle(reservation, "request", paid().MODEL, usage(6))
    assert caught.value.code == "cost_overrun"
    summary = paid().FixedPaidLedger(path).summary()
    assert Decimal(summary["settled_usd"]) == 6 and Decimal(summary["remaining_usd"]) == 0
    assert summary["block_codes"] == ["cost_overrun"]
    with pytest.raises(paid().LedgerError) as caught:
        ledger.reserve("another", "intent")
    assert caught.value.code == "billing_blocked"


def test_timeout_and_invalid_answer_outcomes_do_not_refund_costs(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    uncertain = ledger.reserve("op1", "intent")
    ledger.note_failure(uncertain, "timeout")
    ledger.note_failure(uncertain, "timeout")
    billed = ledger.reserve("op2", "intent")
    ledger.settle(billed, "request", paid().MODEL, usage(.002))
    ledger.note_failure(billed, "invalid_output")
    summary = paid().FixedPaidLedger(path).summary()
    assert not summary["blocked"]
    assert Decimal(summary["settled_usd"]) == Decimal(".002")
    assert Decimal(summary["reserved_usd"]) == paid().RESERVATION
    assert Decimal(summary["remaining_usd"]) == Decimal("4.68675")


def test_foreign_settlements_are_rejected_without_inventing_an_admitted_call(tmp_path):
    ledger = paid().FixedPaidLedger(tmp_path / "calls.jsonl")
    before = ledger.path.read_bytes()
    with pytest.raises(paid().LedgerError) as caught:
        ledger.settle("unknown", "another", paid().MODEL, usage())
    assert caught.value.code == "unknown_reservation" and ledger.path.read_bytes() == before


@pytest.mark.parametrize("same_reservation", [False, True])
def test_billing_conflicts_retain_known_cost_and_block_future_admission(tmp_path, same_reservation):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    first = ledger.reserve("first", "intent")
    ledger.settle(first, "request", paid().MODEL, usage(.001))
    second = first if same_reservation else ledger.reserve("second", "intent")
    with pytest.raises(paid().LedgerError) as caught:
        ledger.settle(second, "request", paid().MODEL, usage(.70))
    assert caught.value.code == "settlement_conflict"
    summary = paid().FixedPaidLedger(path).summary()
    assert summary["blocked"] and "settlement_conflict" in summary["block_codes"]
    assert Decimal(summary["settled_usd"]) == Decimal(".70" if same_reservation else ".701")
    assert json.loads(path.read_text().splitlines()[-1])["request_id"] == "request"
    with pytest.raises(paid().LedgerError) as caught:
        ledger.reserve("third", "intent")
    assert caught.value.code == "billing_blocked"


def test_replay_rejects_reservations_that_could_never_have_been_admitted(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    for index in range(16):
        ledger.reserve(str(index), "intent")
    lines = path.read_text().splitlines()
    forged = json.loads(lines[-1])
    forged["reservation_id"] = "unadmitted"
    with path.open("a") as handle:
        handle.write(json.dumps(forged) + "\n")
    with pytest.raises(paid().LedgerError) as caught:
        ledger.summary()
    assert caught.value.code == "ledger_corrupt"


@pytest.mark.parametrize("mutation", [
    lambda text: "", lambda text: text[:-1], lambda text: text + "{broken}\n",
    lambda text: text + "\n", lambda text: text.replace('"version":1', '"version":true'),
    lambda text: text.replace('"version":1', '"version":1,"version":1'),
    lambda text: text.replace('"input_usd_per_million":"0.75"', '"input_usd_per_million":"0.1"'),
    lambda text: text + text.splitlines()[1] + "\n",
    lambda text: text.replace('"kind":"reserve"', '"kind":"refund"'),
])
def test_corrupt_history_is_never_repaired_or_skipped(tmp_path, mutation):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    ledger.reserve("operation", "intent")
    damaged = mutation(path.read_text())
    path.write_text(damaged)
    with pytest.raises(paid().LedgerError) as caught:
        paid().FixedPaidLedger(path)
    assert caught.value.code == "ledger_corrupt"
    assert path.read_text() == damaged


def test_records_contain_only_accounting_fields_not_arbitrary_response_content(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    reservation = ledger.reserve("operation", "intent")
    detail = usage()
    detail.update(api_key="secret-value", prompt="private message", nested={"secret": "other-secret"})
    ledger.settle(reservation, "request", paid().MODEL, detail)
    with pytest.raises(paid().LedgerError):
        ledger.note_failure(reservation, "provider raw error with a secret-value")
    text = path.read_text()
    assert "secret" not in text and "private message" not in text
    assert json.loads(text.splitlines()[0]) == paid().HEADER
    assert Path(paid().DEFAULT_LEDGER).name == "paid-calls.jsonl"


def test_missing_existing_ledger_is_not_recreated_as_a_fresh_allowance(tmp_path):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    ledger.reserve("operation", "intent")
    path.unlink()
    with pytest.raises(paid().LedgerError) as caught:
        paid().FixedPaidLedger(path)
    assert caught.value.code == "ledger_corrupt" and not path.exists()


@pytest.mark.parametrize("request_id,model", [(None, "model"), ("request", None)])
def test_missing_response_metadata_retains_cost_but_blocks(tmp_path, request_id, model):
    ledger = paid().FixedPaidLedger(tmp_path / "calls.jsonl")
    reservation = ledger.reserve("operation", "intent")
    with pytest.raises(paid().LedgerError) as caught:
        ledger.settle(reservation, request_id, model, usage(.002))
    assert caught.value.code == "invalid_usage"
    summary = ledger.summary()
    assert summary["blocked"] and Decimal(summary["settled_usd"]) == Decimal(".002")


def test_partial_write_fails_closed_after_restart(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)

    def partial(handle, event):
        handle.seek(0, os.SEEK_END)
        handle.write(b'{"kind":"reserve"')
        handle.flush()
        raise OSError("partial write")

    monkeypatch.setattr(ledger, "_append", partial)
    with pytest.raises(paid().LedgerError) as caught:
        ledger.reserve("operation", "intent")
    assert caught.value.code == "ledger_io_error"
    with pytest.raises(paid().LedgerError) as caught:
        paid().FixedPaidLedger(path)
    assert caught.value.code == "ledger_corrupt"


@pytest.mark.parametrize("cost", [0, -0.0])
def test_reported_zero_is_valid_without_poisoning_history(tmp_path, cost):
    path = tmp_path / "calls.jsonl"
    ledger = paid().FixedPaidLedger(path)
    reservation = ledger.reserve("operation", "intent")
    ledger.settle(reservation, "request", paid().MODEL, usage(cost))
    summary = paid().FixedPaidLedger(path).summary()
    assert Decimal(summary["settled_usd"]) == 0 and not summary["blocked"]
