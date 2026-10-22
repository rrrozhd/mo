"""Durable, process-shared accounting for this workbench's fixed paid allowance.

The adapter must enforce the pinned model, provider route and token limits. This
ledger neither limits unrelated account activity nor guarantees provider billing.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path
import re
from uuid import uuid4

MODEL = "openai/gpt-5.4-mini"
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "runs/workbench/paid-calls.jsonl"
ALLOWANCE = Decimal("5")
RESERVATION = Decimal(".31125")
HEADER = {"kind": "header", "version": 1, "model": MODEL, "provider": "openai",
          "input_limit": 400000, "output_limit": 2500, "input_usd_per_million": "0.75",
          "output_usd_per_million": "4.5", "request_usd": "0",
          "allowance_usd": "5", "reservation_usd": "0.31125"}
FAILURE_CODES = {"timeout", "provider_failure", "transport_error", "operation_expired",
                 "cancelled", "invalid_output", "invalid_usage"}


class LedgerError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate field")
        result[key] = value
    return result


def _text(value, maximum=200):
    return type(value) is str and 0 < len(value) <= maximum and value.isprintable()


def _identifier(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is not None


def _money(value):
    if type(value) is not str or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None:
        raise ValueError("Invalid money")
    return Decimal(value)


def _tokens(value):
    if type(value) is not dict or set(value) != {"prompt_tokens", "completion_tokens", "total_tokens"}:
        return False
    return (all(type(count) is int and count >= 0 for count in value.values())
            and value["total_tokens"] == value["prompt_tokens"] + value["completion_tokens"])


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_directory(path):
    if path.is_dir():
        return
    _make_directory(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    _sync_directory(path.parent)


class FixedPaidLedger:
    """Append-only transactions; no reset, automatic refund or repair operation."""

    def __init__(self, path=DEFAULT_LEDGER):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        # Establish and validate the header before accepting any paid work.
        with self._locked() as handle:
            self._read(handle)

    @contextmanager
    def _locked(self):
        try:
            _make_directory(self.path.parent)
            try:
                lock_handle = self.lock_path.open("x+b")
                new_lock = True
            except FileExistsError:
                lock_handle = self.lock_path.open("r+b")
                new_lock = False
            with lock_handle as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise LedgerError("ledger_busy") from exc
                try:
                    if not new_lock and not self.path.exists():
                        raise LedgerError("ledger_corrupt")
                    try:
                        handle = self.path.open("x+b")
                        created = True
                    except FileExistsError:
                        handle = self.path.open("r+b")
                        created = False
                    with handle:
                        if created:
                            self._append(handle, HEADER)
                        # Retry this durability boundary even if an earlier
                        # creation wrote its header but directory sync failed.
                        _sync_directory(self.path.parent)
                        yield handle
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError as exc:
            raise LedgerError("ledger_io_error") from exc

    @staticmethod
    def _append(handle, event):
        handle.seek(0, os.SEEK_END)
        handle.write((_json(event) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())

    @staticmethod
    def _read(handle):
        handle.seek(0)
        raw = handle.read()
        try:
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("Missing or unfinished record")
            events = [json.loads(line, object_pairs_hook=_pairs) for line in raw.decode("utf-8").splitlines()]
            if _json(events[0]) != _json(HEADER):
                raise ValueError("Incompatible ledger header")
            reservations, requests, blocked = {}, set(), set()
            charged = Decimal(0)
            for event in events[1:]:
                if type(event) is not dict or not _text(event.get("at"), 64):
                    raise ValueError("Invalid record")
                reservation_id = event.get("reservation_id")
                if not _identifier(reservation_id):
                    raise ValueError("Invalid reservation ID")
                kind = event.get("kind")
                if kind == "reserve":
                    if (set(event) != {"kind", "at", "reservation_id", "operation_id", "phase", "amount_usd"}
                            or reservation_id in reservations or not _identifier(event["operation_id"])
                            or not _identifier(event["phase"]) or event["amount_usd"] != str(RESERVATION)):
                        raise ValueError("Invalid reservation")
                    if blocked or charged + RESERVATION > ALLOWANCE:
                        raise ValueError("Reservation could not have been admitted")
                    charged += RESERVATION
                    reservations[reservation_id] = {"settlement": None, "cost": None, "failures": set()}
                    continue
                if reservation_id not in reservations:
                    raise ValueError("Unknown reservation")
                entry = reservations[reservation_id]
                if kind == "settle":
                    if set(event) != {"kind", "at", "reservation_id", "request_id", "returned_model", "usage", "cost_usd"}:
                        raise ValueError("Invalid settlement")
                    cost = _money(event["cost_usd"])
                    request_id, model, usage = event["request_id"], event["returned_model"], event["usage"]
                    if request_id is not None and not _text(request_id):
                        raise ValueError("Invalid request ID")
                    if model is not None and not _text(model):
                        raise ValueError("Invalid returned model")
                    if usage is not None and not _tokens(usage):
                        raise ValueError("Invalid usage record")
                    if request_id is None or model is None or usage is None:
                        blocked.add("invalid_usage")
                    if entry["settlement"] is not None or request_id in requests:
                        blocked.add("settlement_conflict")
                    if request_id is not None:
                        requests.add(request_id)
                    if cost > RESERVATION:
                        blocked.add("cost_overrun")
                    previous_charge = RESERVATION if entry["cost"] is None else entry["cost"]
                    entry["cost"] = cost if entry["cost"] is None else max(cost, entry["cost"])
                    charged += entry["cost"] - previous_charge
                    entry["settlement"] = event
                elif kind == "failure":
                    if (set(event) != {"kind", "at", "reservation_id", "code"}
                            or event["code"] not in FAILURE_CODES or event["code"] in entry["failures"]):
                        raise ValueError("Invalid failure")
                    entry["failures"].add(event["code"])
                    if event["code"] == "invalid_usage":
                        blocked.add("invalid_usage")
                else:
                    raise ValueError("Unknown event")
            return reservations, requests, blocked
        except (ValueError, TypeError, KeyError, UnicodeError, InvalidOperation, RecursionError) as exc:
            raise LedgerError("ledger_corrupt") from exc

    @staticmethod
    def _summary(reservations, blocked):
        settled = [entry for entry in reservations.values() if entry["settlement"] is not None]
        actual = sum((entry["cost"] for entry in settled), Decimal(0))
        pending_count = len(reservations) - len(settled)
        pending = pending_count * RESERVATION
        return {"limit_usd": str(ALLOWANCE), "settled_usd": str(actual), "reserved_usd": str(pending),
                "committed_usd": str(actual + pending), "remaining_usd": str(max(Decimal(0), ALLOWANCE - actual - pending)),
                "reservation_count": len(reservations), "settled_count": len(settled),
                "unsettled_count": pending_count, "blocked": bool(blocked), "block_codes": sorted(blocked)}

    def summary(self):
        with self._locked() as handle:
            reservations, _, blocked = self._read(handle)
            return self._summary(reservations, blocked)

    def reserve(self, operation_id, phase):
        if not _identifier(operation_id) or not _identifier(phase):
            raise LedgerError("invalid_request")
        with self._locked() as handle:
            reservations, _, blocked = self._read(handle)
            if blocked:
                raise LedgerError("billing_blocked")
            if Decimal(self._summary(reservations, blocked)["remaining_usd"]) < RESERVATION:
                raise LedgerError("budget_exceeded")
            reservation_id = uuid4().hex
            self._append(handle, {"kind": "reserve", "at": _stamp(), "reservation_id": reservation_id,
                                  "operation_id": operation_id, "phase": phase, "amount_usd": str(RESERVATION)})
            return reservation_id

    def settle(self, reservation_id, request_id, returned_model, usage):
        """Persist reported cost before the adapter validates answer/model content.

        A known cost is retained even when token metadata is invalid. Unknown cost
        keeps the whole reservation. Both cases durably block subsequent calls.
        """
        with self._locked() as handle:
            reservations, requests, _ = self._read(handle)
            if not _identifier(reservation_id) or reservation_id not in reservations:
                raise LedgerError("unknown_reservation")
            entry = reservations[reservation_id]
            reported = usage.get("cost") if type(usage) is dict else None
            try:
                cost = Decimal(str(reported)) if type(reported) in (int, float) else None
            except (ValueError, InvalidOperation):
                cost = None
            if cost is None or not cost.is_finite() or cost < 0:
                self._failure(handle, entry, reservation_id, "invalid_usage")
                raise LedgerError("invalid_usage")
            cost_text = "0" if cost == 0 else format(cost, "f")
            if "." in cost_text:
                cost_text = cost_text.rstrip("0").rstrip(".")
            tokens = {name: usage.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
            event = {"kind": "settle", "at": _stamp(), "reservation_id": reservation_id,
                     "request_id": request_id if _text(request_id) else None,
                     "returned_model": returned_model if _text(returned_model) else None,
                     "usage": tokens if _tokens(tokens) else None, "cost_usd": cost_text}
            existing = entry["settlement"]
            identical = existing is not None and {key: value for key, value in event.items() if key != "at"} == {key: value for key, value in existing.items() if key != "at"}
            conflict = not identical and (existing is not None or event["request_id"] in requests)
            if not identical:
                self._append(handle, event)
            if conflict:
                raise LedgerError("settlement_conflict")
            if event["request_id"] is None or event["returned_model"] is None or event["usage"] is None:
                raise LedgerError("invalid_usage")
            if Decimal(event["cost_usd"]) > RESERVATION:
                raise LedgerError("cost_overrun")

    @staticmethod
    def _failure(handle, entry, reservation_id, code):
        if code not in entry["failures"]:
            FixedPaidLedger._append(handle, {"kind": "failure", "at": _stamp(),
                "reservation_id": reservation_id, "code": code})

    def note_failure(self, reservation_id, code):
        if type(code) is not str or code not in FAILURE_CODES:
            raise LedgerError("invalid_request")
        with self._locked() as handle:
            reservations, _, _ = self._read(handle)
            if not _identifier(reservation_id) or reservation_id not in reservations:
                raise LedgerError("unknown_reservation")
            self._failure(handle, reservations[reservation_id], reservation_id, code)
