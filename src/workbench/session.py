"""One in-memory research session, with explicit authority and serialized operations."""

from copy import deepcopy
from math import isfinite
import threading
import time
from typing import Any
from uuid import uuid4

from ..chemistry import canonical_smiles
from ..chembl import EvidenceClient
from .science import RecordedEvidence, canonical_json, evidence_snapshot

MAX_TURNS = MAX_REVISIONS = 100
CONSTRAINT_FIELDS = {"budget", "include_ids", "exclude_ids", "max_similarity"}
INTENT_OPERATIONS = {"add_include_ids", "remove_include_ids", "add_exclude_ids", "remove_exclude_ids"}


class SessionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(message)


def _text(value, name, limit=8000):
    if type(value) is not str or not value.strip():
        raise SessionError("invalid_request", f"{name} must be a nonempty string")
    if len(value) > limit:
        raise SessionError("limit_exceeded", f"{name} exceeds {limit} characters")
    return value


def _ids(values):
    if type(values) is not list or any(type(value) is not str for value in values):
        raise SessionError("invalid_request", "Candidate references must be a list of exact IDs")
    if len(set(values)) != len(values):
        raise SessionError("invalid_request", "Duplicate candidate references")
    return values


class Session:
    def __init__(self, science, source, *, clock=time.monotonic):
        if not isinstance(source, (RecordedEvidence, EvidenceClient)):
            raise SessionError("invalid_request", "Unsupported evidence source")
        if isinstance(source, RecordedEvidence) and source.model_digest != science.model_digest:
            raise SessionError("invalid_request", "Recorded evidence model differs")
        self._science, self._source, self._clock = science, source, clock
        self._mode = "recorded" if isinstance(source, RecordedEvidence) else "live"
        self._lock = threading.RLock()
        self._state: dict[str, Any] = self._empty_state()
        self._operations: dict[str, dict[str, Any]] = {}
        self._active = None

    def _empty_state(self):
        return {"session_id": uuid4().hex, "chat_id": uuid4().hex, "dataset_digest": None,
                "model_digest": self._science.model_digest, "source_mode": self._mode,
                "revision": 0, "evidence_version": 0, "dataset": None,
                "constraints": None, "selection": None, "revisions": [],
                "snapshots": [], "current_evidence": {}, "turns": [],
                "proposals": [], "review": None, "archived_chats": []}

    def _session(self, session_id):
        if type(session_id) is not str or not session_id:
            raise SessionError("invalid_request", "session_id is required")
        if session_id != self._state["session_id"]:
            raise SessionError("stale_session", "This request belongs to an earlier session")

    def _keys(self, keys, *, evidence=False):
        required = {"session_id", "dataset_digest", "base_revision"}
        if evidence:
            required.add("evidence_version")
        if (type(keys) is not dict or not required <= keys.keys()
                or keys.keys() - {"session_id", "dataset_digest", "base_revision", "evidence_version"}):
            raise SessionError("invalid_request", "Missing or unknown request identity fields")
        self._session(keys["session_id"])
        for name in ("base_revision", "evidence_version"):
            if name in keys and (type(keys[name]) is not int or keys[name] < 0):
                raise SessionError("invalid_request", f"{name} must be a nonnegative integer")
        if keys["dataset_digest"] is not None and type(keys["dataset_digest"]) is not str:
            raise SessionError("invalid_request", "dataset_digest must be a string or null")
        if keys["dataset_digest"] != self._state["dataset_digest"]:
            raise SessionError("stale_dataset", "The dataset has changed")
        if keys["base_revision"] != self._state["revision"]:
            raise SessionError("stale_revision", "The batch revision has changed")
        if "evidence_version" in keys and keys["evidence_version"] != self._state["evidence_version"]:
            raise SessionError("stale_evidence", "The evidence version has changed")

    def _expire(self):
        if self._active is not None:
            op = self._operations[self._active]
            if self._clock() >= op["_deadline"] and op["status"] == "running":
                op.update(status="failed", error={"code": "operation_expired", "message": "Operation deadline reached"})
                if op["_turn"] is not None:
                    op["_turn"].update(status="failed", error=deepcopy(op["error"]))

    def _idle(self):
        self._expire()
        if self._active is not None:
            raise SessionError("busy", "Another operation still occupies the worker")

    @staticmethod
    def _public_operation(op):
        return deepcopy({key: value for key, value in op.items() if not key.startswith("_")})

    def _view(self):
        self._expire()
        return deepcopy({**self._state, "busy": self._active is not None,
                         "operation": self._public_operation(self._operations[self._active])
                         if self._active is not None else None,
                         "operations": [self._public_operation(op) for op in self._operations.values()]})

    def bootstrap(self):
        with self._lock:
            return self._view()

    def read(self, session_id):
        with self._lock:
            self._session(session_id)
            return self._view()

    def _dataset(self):
        if self._state["dataset"] is None:
            raise SessionError("invalid_request", "Choose or upload a dataset first")
        return self._state["dataset"]

    def _selection(self, patch):
        if type(patch) is not dict or patch.keys() - CONSTRAINT_FIELDS:
            raise SessionError("invalid_constraints", "Unknown or malformed constraint patch")
        for name in ("include_ids", "exclude_ids"):
            if name in patch and type(patch[name]) is not list:
                raise SessionError("invalid_constraints", f"{name} must be a replacement list")
        constraints = deepcopy({**(self._state["constraints"] or {}), **patch})
        try:
            selection = self._science.select(self._dataset(), constraints)
        except (TypeError, ValueError) as exc:
            raise SessionError("invalid_constraints", str(exc)) from exc
        return constraints, selection

    def _commit(self, patch, action, *, op=None, proposal_id=None, evidence_refs=None, prepared=None):
        constraints, selection = self._selection(patch) if prepared is None else prepared
        if constraints == self._state["constraints"]:
            if op is not None:
                self._token(op["operation_id"])
            return
        if len(self._state["revisions"]) >= MAX_REVISIONS:
            raise SessionError("limit_exceeded", "The session has reached 100 batch revisions")
        old_ids = [row["id"] for row in (self._state["selection"] or {}).get("selected", [])]
        new_ids = [row["id"] for row in selection["selected"]]
        revision = self._state["revision"] + 1
        record = {"revision": revision,
            "old_constraints": deepcopy(self._state["constraints"]), "new_constraints": deepcopy(constraints),
            "added_ids": [value for value in new_ids if value not in old_ids],
            "removed_ids": [value for value in old_ids if value not in new_ids],
            "retained_ids": [value for value in new_ids if value in old_ids],
            "actor": "user", "user_action": action, "proposal_id": proposal_id,
            "evidence_refs": deepcopy(evidence_refs or [])}
        if op is not None:
            self._token(op["operation_id"])
        self._state["revisions"].append(record)
        self._state.update(constraints=constraints, selection=selection, revision=revision, review=None)
        if op is not None:
            op["revision"] = revision

    def replace_dataset(self, keys, rows, budget, action):
        _text(action, "User action")
        if type(rows) is not list or not rows or len(rows) > 200:
            raise SessionError("invalid_request", "A dataset needs 1 to 200 candidate rows")
        for row in rows:
            if type(row) is not dict:
                raise SessionError("invalid_request", "Candidate rows must be objects")
            for field in ("id", "smiles"):
                _text(row.get(field), field, 4000)
        if type(budget) is not int or budget <= 0:
            raise SessionError("invalid_constraints", "Choose a positive integer budget")
        with self._lock:
            self._keys(keys)
            self._idle()
            token = self._start("dataset", None, 120)
        try:
            dataset = self._science.prepare(rows)
            constraints = {"budget": budget, "include_ids": [], "exclude_ids": [], "max_similarity": .8}
            selection = self._science.select(dataset, constraints)
            with self._lock:
                self._token(token)
                self._state = self._empty_state()
                self._state.update(dataset=dataset, dataset_digest=dataset["dataset_digest"])
                self._commit(constraints, action, prepared=(constraints, selection))
                self._operations.clear()
                self._active = None
                return self._view()
        except (ValueError, TypeError) as exc:
            if isinstance(exc, SessionError):
                raise
            raise SessionError("invalid_dataset", str(exc)) from exc
        finally:
            with self._lock:
                if self._active == token:
                    self._operations[token].update(status="failed", error={
                        "code": "invalid_dataset", "message": "Dataset replacement did not complete"})
                    self._active = None

    def revise(self, keys, patch, action):
        _text(action, "User action")
        with self._lock:
            self._keys(keys)
            self._idle()
            self._commit(patch, action)
            return self._view()

    def _chat_snapshot(self):
        return {"chat_id": self._state["chat_id"], "session_id": self._state["session_id"],
                "turns": deepcopy(self._state["turns"]),
                "proposals": deepcopy(self._state["proposals"]),
                "operations": [self._public_operation(op) for op in self._operations.values()]}

    def _activate_chat(self, chat):
        self._state.update(session_id=uuid4().hex, chat_id=chat["chat_id"],
                           turns=deepcopy(chat["turns"]), proposals=deepcopy(chat["proposals"]))
        self._operations = {op["operation_id"]: deepcopy(op) for op in chat["operations"]}
        # Shared batch review survives, but old chat proposals retain stale authority.
        if self._state["review"] is not None:
            self._state["review"]["session_id"] = self._state["session_id"]

    def new_chat(self, keys):
        """Start another conversation while retaining shared science and earlier chats."""
        with self._lock:
            self._keys(keys, evidence=True)
            self._idle()
            if len(self._state["archived_chats"]) >= MAX_TURNS:
                raise SessionError("limit_exceeded", "The session has reached its archived chat limit")
            self._state["archived_chats"].append(self._chat_snapshot())
            self._activate_chat({"chat_id": uuid4().hex, "turns": [], "proposals": [], "operations": []})
            return self._view()

    def switch_chat(self, keys, chat_id):
        _text(chat_id, "Chat ID", 80)
        with self._lock:
            self._keys(keys, evidence=True)
            self._idle()
            if chat_id == self._state["chat_id"]:
                return self._view()
            chats = self._state["archived_chats"]
            position = next((index for index, chat in enumerate(chats) if chat["chat_id"] == chat_id), None)
            if position is None:
                raise SessionError("invalid_request", "Unknown chat ID")
            saved = self._chat_snapshot()
            selected = chats.pop(position)
            chats.append(saved)
            self._activate_chat(selected)
            return self._view()

    def _start(self, kind, message, timeout):
        token = uuid4().hex
        turn = None
        if kind == "chat":
            turn = {"operation_id": token, "message": message, "status": "running",
                    "raw_intent": None, "normalized_patch": None, "interpretation": None,
                    "answer": None, "error": None}
            self._state["turns"].append(turn)
        self._operations[token] = {"operation_id": token, "kind": kind,
            "session_id": self._state["session_id"], "dataset_digest": self._state["dataset_digest"],
            "model_digest": self._state["model_digest"], "source_mode": self._mode,
            "revision": self._state["revision"], "status": "running", "error": None,
            "events": [], "outcome": None, "_deadline": self._clock() + timeout,
            "_turn": turn, "_in_flight": False}
        self._active = token
        return token

    def start_operation(self, keys, kind, message=None, timeout=120):
        if type(kind) is not str or kind not in {"evidence", "chat"}:
            raise SessionError("invalid_request", "Operation kind must be evidence or chat")
        if type(timeout) not in (int, float) or not isfinite(timeout) or not 0 < timeout <= 120:
            raise SessionError("invalid_request", "Operation timeout must be in (0, 120]")
        if kind == "chat":
            _text(message, "User message")
        elif message is not None:
            raise SessionError("invalid_request", "Only chat operations accept a message")
        with self._lock:
            self._keys(keys)
            self._idle()
            if kind != "chat":
                self._dataset()
            turn_count = len(self._state["turns"]) + sum(
                len(chat["turns"]) for chat in self._state["archived_chats"])
            if kind == "chat" and turn_count >= MAX_TURNS:
                raise SessionError("limit_exceeded", "The session has reached 100 chat turns")
            return self._start(kind, message, timeout)

    def _token(self, token, *, expired=False, tool=False):
        if type(token) is not str or token != self._active:
            raise SessionError("stale_operation", "This operation is no longer active")
        self._expire()
        op = self._operations[token]
        if any(op[name] != self._state[name] for name in
               ("session_id", "dataset_digest", "model_digest", "source_mode", "revision")):
            raise SessionError("stale_operation", "The operation context has changed")
        if op["status"] != "running" and not expired:
            raise SessionError("operation_expired", "Operation deadline reached")
        if tool and op["_in_flight"]:
            raise SessionError("busy", "A lookup is already in flight for this operation")
        if tool and op["_turn"] is not None and op["_turn"]["interpretation"] is None:
            raise SessionError("invalid_intent", "Interpret the current user message before using tools")
        return op

    def poll(self, session_id, operation_id):
        with self._lock:
            self._session(session_id)
            self._expire()
            if type(operation_id) is not str or operation_id not in self._operations:
                raise SessionError("stale_operation", "Unknown operation")
            return self._public_operation(self._operations[operation_id])

    def _event(self, op, kind, result):
        event = {"result_id": uuid4().hex, "kind": kind, "result": deepcopy(result)}
        self._token(op["operation_id"], tool=True)
        op["events"].append(event)
        return deepcopy(event)

    def lookup(self, token, candidate_id, refresh=False):
        with self._lock:
            self._token(token, tool=True)
            self._known_ids([candidate_id])
            query = next(row["canonical_smiles"] for row in self._dataset()["scored"]
                         if row["id"] == candidate_id)
        return self._lookup(token, candidate_id, query, refresh)

    def lookup_structure(self, token, smiles, refresh=False):
        """Use this session's evidence source for a candidate or another exact structure."""
        _text(smiles, "SMILES", 4000)
        try:
            query = canonical_smiles(smiles)
        except ValueError as exc:
            raise SessionError("invalid_request", str(exc)) from exc
        with self._lock:
            self._token(token, tool=True)
            candidate_id = next((row["id"] for row in (self._state["dataset"] or {}).get("scored", [])
                                 if row["canonical_smiles"] == query), None)
        return self._lookup(token, candidate_id, query, refresh)

    def _lookup(self, token, candidate_id, query, refresh):
        if type(refresh) is not bool:
            raise SessionError("invalid_request", "refresh must be a boolean")
        with self._lock:
            op = self._token(token, tool=True)
            snapshot = next((row for row in reversed(self._state["snapshots"])
                             if row["canonical_query"] == query), None)
            if snapshot is not None and not refresh:
                self._event(op, "lookup_evidence", {"candidate_id": candidate_id,
                    "canonical_query": query, "snapshot_id": snapshot["snapshot_id"], "reused": True})
                return deepcopy(snapshot)
            # Structure research has model provenance even before a batch exists.
            dataset = deepcopy(self._state["dataset"] or {
                "dataset_digest": None, "model_digest": self._state["model_digest"]})
            op["_in_flight"] = True
        try:
            snapshot = evidence_snapshot(self._source, candidate_id, dataset, smiles=query,
                                         deadline=op["_deadline"], clock=self._clock)
        except BaseException:
            with self._lock:
                op["_in_flight"] = False
            raise
        with self._lock:
            # Release lookup ownership and attach its result atomically.
            op["_in_flight"] = False
            op = self._token(token, tool=True)
            if snapshot["origin"] != self._mode or snapshot["dataset_digest"] != self._state["dataset_digest"]:
                raise SessionError("stale_operation", "Evidence belongs to another context")
            retained = deepcopy(snapshot)
            self._event(op, "lookup_evidence", {"candidate_id": candidate_id,
                "canonical_query": query, "snapshot_id": snapshot["snapshot_id"], "reused": False})
            self._state["snapshots"].append(retained)
            if candidate_id is not None:
                self._state["current_evidence"][candidate_id] = snapshot["snapshot_id"]
            self._state["evidence_version"] += 1
            self._state["review"] = None
            return deepcopy(snapshot)

    def analyze_view(self, name, arguments):
        """Read-only views use the same trusted artifacts as the session scorer."""
        return self._science.analyze(name, arguments)

    def search_external_analogs(self, token, smiles, max_results=3, threshold=70):
        """Discover structures through the session's live source without changing its dataset."""
        _text(smiles, "SMILES", 4000)
        query = canonical_smiles(smiles)
        arguments = {"smiles": query, "max_results": max_results, "threshold": threshold}
        with self._lock:
            op = self._token(token, tool=True)
            if self._mode == "recorded":
                return self.record_tool_result(token, "search_external_analogs", arguments, {
                    "status": "unavailable", "origin": "recorded", "query_smiles": query,
                    "analogs": [], "error": "External analogue search requires live source mode; recorded mode makes no network requests."})
            op["_in_flight"] = True
        try:
            assert isinstance(self._source, EvidenceClient)
            result = self._source.search_analogs(query, max_results=max_results, threshold=threshold,
                                                 deadline=op["_deadline"], clock=self._clock)
            from ..depictions import depict
            for analog in result["analogs"]:
                analog["svg"] = depict(analog["smiles"])
        finally:
            with self._lock:
                op["_in_flight"] = False
        with self._lock:
            self._token(token, tool=True)
            for analog in result["analogs"]:
                analog["candidate_ids"] = [row["id"] for row in (self._state["dataset"] or {}).get("scored", [])
                                           if row["canonical_smiles"] == analog["smiles"]]
            result["origin"] = "live"
            return self.record_tool_result(token, "search_external_analogs", arguments, result)

    def analyze(self, token, name, arguments):
        with self._lock:
            op = self._token(token, tool=True)
            op["_in_flight"] = True
        try:
            result = self.analyze_view(name, arguments)
        finally:
            with self._lock:
                op["_in_flight"] = False
        return self.record_tool_result(token, name, arguments, result)

    def enrich_papers(self, token, snapshot_id, max_results=3):
        """Read source documents referenced by an existing exact-structure snapshot."""
        if type(max_results) is not int or not 1 <= max_results <= 3:
            raise SessionError("invalid_request", "Choose one to three source documents")
        with self._lock:
            op = self._token(token, tool=True)
            snapshot = next((row for row in self._state["snapshots"]
                             if row["snapshot_id"] == snapshot_id), None)
            if snapshot is None:
                raise SessionError("invalid_request", "Unknown evidence snapshot")
            query = snapshot["canonical_query"]
            ids = list(dict.fromkeys(row["document_id"] for row in snapshot["result"]["records"]
                                     if row.get("document_id")))
            arguments = {"snapshot_id": snapshot_id, "max_results": max_results, "smiles": query}
            context = {"snapshot_id": snapshot_id, "query_smiles": query, "origin": self._mode,
                       "documents_truncated": len(ids) > max_results}
            if self._mode == "recorded" or not ids:
                return self.record_tool_result(token, "enrich_papers", arguments, {**context,
                    "status": "unavailable" if self._mode == "recorded" else "not_found",
                    "papers": [], "document_ids": [], "missing_ids": [],
                    "error": "Source document enrichment requires live source mode." if self._mode == "recorded" else None})
            op["_in_flight"] = True
        try:
            assert isinstance(self._source, EvidenceClient)
            result = self._source.documents(ids[:max_results], deadline=op["_deadline"], clock=self._clock)
        finally:
            with self._lock:
                op["_in_flight"] = False
        with self._lock:
            self._token(token, tool=True)
            return self.record_tool_result(token, "enrich_papers", arguments, {**result, **context})

    def record_tool_result(self, token, name, arguments, result):
        """Retain scientific tool output and its exact inputs for citations and export."""
        _text(name, "Tool name", 80)
        if type(arguments) is not dict or type(result) is not dict:
            raise SessionError("invalid_request", "Tool arguments and result must be objects")
        canonical_json({"arguments": arguments, "data": result})
        with self._lock:
            op = self._token(token, tool=True)
            queries = {canonical_smiles(arguments[key])
                       for key in ("smiles", "first_smiles", "second_smiles") if key in arguments}
            ids = [row["id"] for row in (self._state["dataset"] or {}).get("scored", [])
                   if row["canonical_smiles"] in queries]
            return self._event(op, "chemistry", {"tool": name, "arguments": arguments,
                                                "data": result, "candidate_ids": ids})

    def _known_ids(self, ids):
        _ids(ids)
        known = {row["id"] for row in self._dataset()["rows"]}
        if set(ids) - known:
            raise SessionError("invalid_request", "Unknown candidate ID")

    def inspect(self, token, ids):
        with self._lock:
            op = self._token(token, tool=True)
            self._known_ids(ids)
            by_id = {row["id"]: row for row in self._dataset()["scored"]}
            return self._event(op, "inspect_candidates", [by_id[identifier] for identifier in ids])

    def compare(self, token, a_id, b_id):
        with self._lock:
            op = self._token(token, tool=True)
            self._known_ids([a_id] if a_id == b_id else [a_id, b_id])
            refs = {identifier: [self._state["current_evidence"][identifier]]
                    if identifier in self._state["current_evidence"] else [] for identifier in (a_id, b_id)}
            return self._event(op, "compare_candidates", self._science.compare(self._dataset(), a_id, b_id, refs))

    def preview_constraints(self, token, patch):
        with self._lock:
            op = self._token(token, tool=True)
            constraints, selection = self._selection(patch)
            return self._event(op, "preview_constraints", {"constraints": constraints, "selection": selection})

    def _intent_turn(self, token):
        op = self._token(token)
        turn = op["_turn"]
        if turn is None or turn["interpretation"] is not None or op["events"] or op["_in_flight"]:
            raise SessionError("invalid_intent", "Interpretation must precede investigation tools")
        return op, turn

    def intent_context(self, token):
        """The provider's intent pass receives no assistant history or external assay text."""
        with self._lock:
            _, turn = self._intent_turn(token)
            return deepcopy({"message": turn["message"], "constraints": self._state["constraints"],
                             "candidates": (self._state["dataset"] or {}).get("rows", [])})

    def interpret(self, token, message, raw_intent):
        with self._lock:
            op, turn = self._intent_turn(token)
            base_revision = op["revision"]
            allowed = {"decision", "span", "budget", "max_similarity"} | INTENT_OPERATIONS
            if message != turn["message"] or type(raw_intent) is not dict or raw_intent.keys() - allowed:
                raise SessionError("invalid_intent", "Interpretation must precede tools and refer only to this user message")
            try:
                canonical_json(raw_intent)
            except (TypeError, ValueError) as exc:
                raise SessionError("invalid_intent", "Interpretation must contain finite JSON values") from exc
            decision, span = raw_intent.get("decision"), raw_intent.get("span")
            if (type(decision) is not str or decision not in {"question", "explicit_change", "clarification_needed"}
                    or type(span) is not str):
                raise SessionError("invalid_intent", "Invalid interpretation classification or literal span")
            if (span and span not in message) or (decision == "explicit_change" and not span):
                raise SessionError("invalid_intent", "The supporting span must occur verbatim in the current message")
            patch, clarification = {}, None
            if decision == "explicit_change":
                try:
                    patch = {name: raw_intent[name] for name in ("budget", "max_similarity")
                             if raw_intent.get(name) is not None}
                    for field in ("include_ids", "exclude_ids"):
                        add, remove = raw_intent.get("add_" + field), raw_intent.get("remove_" + field)
                        add, remove = [] if add is None else add, [] if remove is None else remove
                        self._known_ids(add)
                        self._known_ids(remove)
                        if set(add) & set(remove):
                            raise SessionError("invalid_constraints", "The same ID cannot be added and removed")
                        if raw_intent.get("add_" + field) is not None or raw_intent.get("remove_" + field) is not None:
                            patch[field] = sorted((set(self._state["constraints"][field]) | set(add)) - set(remove))
                    if not patch:
                        raise SessionError("invalid_constraints", "No concrete constraint change was supplied")
                    self._commit(patch, message, op=op)
                except SessionError as exc:
                    if exc.code not in {"invalid_request", "invalid_constraints"}:
                        raise
                    decision, patch, clarification = "clarification_needed", {}, str(exc)
            interpretation = {"decision": decision, "patch": deepcopy(patch), "span": span,
                              "clarification": clarification}
            # A committed revision and its intent record form one atomic change.
            if op["revision"] == base_revision:
                self._token(token)
            turn.update(raw_intent=deepcopy(raw_intent), normalized_patch=deepcopy(patch),
                        interpretation=deepcopy(interpretation))
            return deepcopy(interpretation)

    def propose(self, token, patch, rationale, evidence_refs):
        _text(rationale, "Proposal rationale")
        _ids(evidence_refs)
        with self._lock:
            op = self._token(token, tool=True)
            snapshots = {row["snapshot_id"] for row in self._state["snapshots"]}
            if set(evidence_refs) - snapshots:
                raise SessionError("invalid_request", "Unknown evidence reference")
            self._selection(patch)
            proposal = {"proposal_id": uuid4().hex, "session_id": self._state["session_id"],
                        "base_revision": self._state["revision"], "evidence_version": self._state["evidence_version"],
                        "patch": deepcopy(patch), "rationale": rationale,
                        "evidence_refs": deepcopy(evidence_refs), "applied": False}
            self._event(op, "propose_constraints", {"proposal_id": proposal["proposal_id"]})
            self._state["proposals"].append(proposal)
            return deepcopy(proposal)

    def apply_proposal(self, keys, proposal_id, action):
        _text(action, "User action")
        with self._lock:
            self._keys(keys, evidence=True)
            self._idle()
            proposal = next((p for p in self._state["proposals"] if p["proposal_id"] == proposal_id), None)
            if (proposal is None or proposal["applied"] or proposal["session_id"] != self._state["session_id"]
                    or proposal["base_revision"] != self._state["revision"]
                    or proposal["evidence_version"] != self._state["evidence_version"]):
                raise SessionError("stale_proposal", "The proposal context has changed")
            self._commit(proposal["patch"], action, proposal_id=proposal_id, evidence_refs=proposal["evidence_refs"])
            proposal["applied"] = True
            return self._view()

    def mark_reviewed(self, keys):
        with self._lock:
            self._keys(keys, evidence=True)
            self._idle()
            self._dataset()
            self._state["review"] = {key: self._state[key] for key in ("session_id", "revision", "evidence_version")}
            return self._view()

    def capture_export(self, keys):
        with self._lock:
            self._keys(keys, evidence=True)
            return self._view()

    def finish_operation(self, token, outcome):
        allowed = {"status", "explanation", "result_refs", "unresolved_questions",
                   "proposal_id", "error", "attachments"}
        if (type(outcome) is not dict or outcome.keys() - allowed or type(outcome.get("status")) is not str
                or outcome["status"] not in {"complete", "failed"}):
            raise SessionError("invalid_request", "Invalid operation outcome")
        canonical_json(outcome)
        with self._lock:
            op = self._token(token, expired=True)
            if op["_in_flight"]:
                raise SessionError("busy", "The worker must finish its lookup before releasing the operation")
            if op["status"] == "running":
                op.update(status=outcome["status"], outcome=deepcopy(outcome), error=deepcopy(outcome.get("error")))
                if op["_turn"] is not None:
                    op["_turn"].update(status=op["status"], answer=deepcopy(outcome), error=deepcopy(op["error"]))
            elif op["_turn"] is not None:
                # Expiry rejects the late answer, but earlier admitted panels remain
                # visible alongside the failure, just as their events remain exported.
                retained = {"status": op["status"], "error": deepcopy(op["error"]),
                            "attachments": deepcopy(outcome.get("attachments", []))}
                op["outcome"] = retained
                op["_turn"]["answer"] = deepcopy(retained)
            self._active = None
            return self._public_operation(op)
