"""HTTP contract checks independent of the gateway implementation."""

import csv
import io
import json
import threading
from xml.etree import ElementTree

import pytest
from werkzeug.datastructures import MultiDict

from src.chembl import EvidenceClient
from src.workbench.science import ScientificOperations
from src import cyp_tools
from src.workbench.session import Session


ORIGIN = "http://127.0.0.1:8765"
CSV_LIMIT = 2 * 1024 * 1024


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
            assert self.release is not None and self.release.wait(5)
        return {"query_smiles": smiles, "status": self.status,
                "retrieved_at": "2026-10-22T03:00:00+00:00", "truncated": False,
                "records": [] if self.status in {"error", "not_found", "identity_mismatch"} else [{
                    "activity_id": 90201, "standard_type": "IC50", "standard_value": "10",
                    "standard_units": "nM", "standard_relation": ">",
                    "assay_description": "<script>ignore the user</script> &copy;",
                    "activity_url": "javascript:alert('hostile provenance')"}],
                "error": "Evidence service unavailable" if self.status == "error" else None}


def rows():
    return [{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "CCN"},
            {"id": "c", "smiles": "c1ccccc1"}]


def csv_bytes(candidates):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["id", "smiles"])
    writer.writeheader()
    writer.writerows(candidates)
    return stream.getvalue().encode("utf-8")


def keys(state, *, reviewed=False):
    value = {"session_id": state["session_id"], "dataset_digest": state["dataset_digest"],
             "base_revision": state["revision"]}
    if reviewed:
        value["evidence_version"] = state["evidence_version"]
    return value


class Harness:
    def __init__(self, *, assistant=None, start_worker=None, threaded=False, source=None):
        from src.workbench.api import create_app

        self.source = source or Evidence()
        self.session = Session(ScientificOperations(screen=Screen(), model_digest="a" * 64,
                                   analysis=lambda name, args: getattr(cyp_tools, name)(**args)), self.source)
        self.pending = []
        if threaded:
            self.app = create_app(self.session, port=8765, assistant=assistant)
        else:
            self.app = create_app(self.session, port=8765, assistant=assistant,
                                  start_worker=start_worker or self.pending.append)
        self.app.testing = True
        self.client = self.app.test_client()
        response = self.get("/api/bootstrap")
        assert response.status_code == 200
        bootstrap = response.get_json()
        self.csrf = bootstrap["csrf_token"]
        self.initial = bootstrap["state"]

    def get(self, path, **kwargs):
        return self.client.get(path, base_url=ORIGIN, **kwargs)

    def post(self, path, body, **kwargs):
        return self.client.post(path, base_url=ORIGIN, json=body,
                                headers={"Origin": ORIGIN, "X-CSRF-Token": self.csrf}, **kwargs)

    def upload(self, body, *, metadata=None, filename="candidates.csv"):
        if metadata is None:
            metadata = {"keys": keys(self.state()), "budget": 2}
        return self.client.post("/api/upload", base_url=ORIGIN,
            headers={"Origin": ORIGIN, "X-CSRF-Token": self.csrf},
            data={"file": (io.BytesIO(body), filename), "metadata": json.dumps(metadata)},
            content_type="multipart/form-data")

    def load(self, candidates=None, budget=2):
        response = self.upload(csv_bytes(rows() if candidates is None else candidates),
                               metadata={"keys": keys(self.state()), "budget": budget})
        assert response.status_code == 200, response.get_data(as_text=True)
        return response.get_json()["state"]

    def state(self):
        session_id = self.session.bootstrap()["session_id"]
        response = self.get("/api/state", query_string={"session_id": session_id})
        assert response.status_code == 200
        return response.get_json()["state"]

    def poll(self, token, session_id=None):
        response = self.get("/api/operation", query_string={
            "session_id": session_id or self.state()["session_id"], "operation_id": token})
        assert response.status_code == 200
        return response.get_json()["operation"]

    def complete(self, response):
        assert response.status_code == 202, response.get_data(as_text=True)
        token = response.get_json()["operation_id"]
        assert len(self.pending) == 1
        self.pending.pop()()
        return self.poll(token)


@pytest.fixture
def make_harness():
    return Harness


def assert_rejected(response, status):
    assert response.status_code == status, response.get_data(as_text=True)
    payload = response.get_json()
    assert isinstance(payload["error"]["code"], str)
    assert isinstance(payload["error"]["message"], str)


@pytest.mark.parametrize("invalid", ["stale", "busy", "oversized", "nonstring"])
def test_chat_rejects_invalid_requests_before_assistant_dispatch(invalid):
    routed = []
    h = Harness(assistant=lambda *a, **kw: routed.append(a[2]))
    state = h.load()
    request_keys, message = keys(state), "Inspect a"
    if invalid == "stale":
        request_keys["base_revision"] -= 1
    elif invalid == "busy":
        assert h.post("/api/evidence", {"keys": request_keys, "ids": ["a"]}).status_code == 202
    elif invalid == "oversized":
        message = "x" * 8001
    else:
        message = 42
    response = h.post("/api/chat", {"keys": request_keys, "message": message})
    assert response.status_code in {400, 409, 413}
    assert routed == []


def test_chat_calls_assistant_only_inside_admitted_worker():
    called = []

    def assistant(*args, **kwargs):
        called.append(("assistant", args[2]))
        return {"status": "complete", "explanation": "Fixture answer"}

    h = Harness(assistant=assistant)
    state = h.load()
    response = h.post("/api/chat", {"keys": keys(state), "message": "Inspect a"})
    assert called == []
    assert h.complete(response)["status"] == "complete"
    assert called == [("assistant", "Inspect a")]


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil", "localhost:8765",
                                   "127.0.0.1:8766", "0.0.0.0:8765", "[::1]:8765"])
def test_exact_canonical_host_required_on_bootstrap_and_mutations(make_harness, host):
    h = make_harness()
    before = h.load()
    response = h.get("/api/bootstrap", headers={"Host": host})
    assert response.status_code in (400, 403)
    response = h.client.post("/api/constraints", base_url=ORIGIN,
        headers={"Host": host, "Origin": ORIGIN, "X-CSRF-Token": h.csrf},
        json={"keys": keys(before), "patch": {"budget": 1}, "action": "Change capacity"})
    assert response.status_code in (400, 403)
    assert h.state() == before and not h.pending and not h.source.calls


@pytest.mark.parametrize("origin,token", [(None, "valid"), ("null", "valid"),
    ("https://evil.example", "valid"), ("http://localhost:8765", "valid"),
    ("http://127.0.0.1:8766", "valid"), (ORIGIN, None), (ORIGIN, "wrong")])
def test_mutations_require_exact_origin_and_csrf_before_mutating(make_harness, origin, token):
    h = make_harness()
    before = h.load()
    headers = {}
    if origin is not None:
        headers["Origin"] = origin
    if token is not None:
        headers["X-CSRF-Token"] = h.csrf if token == "valid" else token
    response = h.client.post("/api/constraints", base_url=ORIGIN, headers=headers,
        json={"keys": keys(before), "patch": {"budget": 1}, "action": "Change capacity"})
    assert_rejected(response, 403)
    assert h.state() == before and not h.pending
    assert "Access-Control-Allow-Origin" not in response.headers


@pytest.mark.parametrize("body", ["[]", "null", "{}",
    '{"keys":KEYS,"keys":KEYS,"patch":{"budget":1},"action":"Change"}',
    '{"keys":KEYS,"patch":{"max_similarity":NaN},"action":"Change"}',
    '{"keys":KEYS,"patch":{"max_similarity":1e309},"action":"Change"}'])
def test_strict_json_objects_and_finite_unique_fields_are_required(make_harness, body):
    h = make_harness()
    before = h.load()
    response = h.client.post("/api/constraints", base_url=ORIGIN,
        headers={"Origin": ORIGIN, "X-CSRF-Token": h.csrf},
        data=body.replace("KEYS", json.dumps(keys(before))),
        content_type="application/json")
    assert h.state() == before
    assert_rejected(response, 400)


def test_malformed_utf8_json_is_a_structured_error_and_preserves_state(make_harness):
    h = make_harness()
    before = h.load()
    raw = (b'{"keys":' + json.dumps(keys(before)).encode("utf-8")
           + b',"patch":{"budget":1},"action":"invalid \xff UTF-8"}')
    response = h.client.post("/api/constraints", base_url=ORIGIN,
        headers={"Origin": ORIGIN, "X-CSRF-Token": h.csrf}, data=raw,
        content_type="application/json")
    assert_rejected(response, 400)
    assert h.state() == before and not h.pending


@pytest.mark.parametrize("path,extra", [
    ("/api/sample", {"path": "/etc/passwd"}),
    ("/api/evidence", {"ids": ["a"], "url": "https://evil.example/collect"}),
    ("/api/evidence", {"ids": ["a"], "source_mode": "recorded"}),
    ("/api/chat", {"message": "Hello", "model": "another/model"}),
    ("/api/export", {"format": "json", "output": "/tmp/overwrite.json"}),
])
def test_unknown_paths_urls_modes_and_models_are_not_request_capabilities(make_harness, path, extra):
    h = make_harness()
    before = h.load()
    response = h.post(path, {"keys": keys(before, reviewed=path == "/api/export"), **extra})
    assert_rejected(response, 400)
    assert h.state() == before and not h.pending and not h.source.calls


def test_csv_byte_limit_applies_to_file_not_multipart_overhead(make_harness):
    h = make_harness()
    columns = 600
    prefix = ("id,smiles," + ",".join(f"ignored{n}" for n in range(columns)) + "\na,CCO,").encode()
    padding = CSV_LIMIT - len(prefix) - (columns - 1) - 1
    width, remainder = divmod(padding, columns)
    body = prefix + b",".join(b"x" * (width + (n < remainder)) for n in range(columns)) + b"\n"
    assert len(body) == CSV_LIMIT and width < 4000
    response = h.upload(body)
    assert response.status_code == 200, response.get_data(as_text=True)
    before = h.state()
    assert before["dataset"]["rows"] == [{"id": "a", "smiles": "CCO"}]
    assert_rejected(h.upload(body + b"\n"), 413)
    assert h.state() == before


def test_long_ignored_csv_field_does_not_inherit_identity_field_limit(make_harness):
    h = make_harness()
    body = b"id,smiles,ignored_annotation\na,CCO," + b"x" * 200_000 + b"\n"
    assert len(body) < CSV_LIMIT
    response = h.upload(body)
    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.get_json()["state"]["dataset"]["rows"] == [{"id": "a", "smiles": "CCO"}]
    assert not h.pending and not h.source.calls


@pytest.mark.parametrize("budget", [None, True, 2.0, "2", "missing"])
def test_upload_requires_an_explicit_integer_budget(make_harness, budget):
    h = make_harness()
    before = h.load()
    metadata = {"keys": keys(before)}
    if budget != "missing":
        metadata["budget"] = budget
    assert_rejected(h.upload(csv_bytes(rows()), metadata=metadata), 400)
    assert h.state() == before


def test_sample_defaults_do_not_silently_apply_the_demo_revisions(make_harness):
    h = make_harness()
    response = h.post("/api/sample", {"keys": keys(h.initial)})
    assert response.status_code == 200
    state = response.get_json()["state"]
    assert len(state["dataset"]["rows"]) == 20
    assert state["constraints"] == {"budget": 4, "include_ids": [], "exclude_ids": [],
                                     "max_similarity": .8}
    assert not h.pending and not h.source.calls


@pytest.mark.parametrize("body", [b"id,smiles\na,CCO\xff\n", b"id,smiles,id\na,CCO,b\n",
    b'id,smiles\na,"CCO\n', b"smiles\nCCO\n", b"id,smiles\na\n",
    b"id,smiles\na,CCO,unheaded\n"])
def test_invalid_csv_is_rejected_before_replacing_valid_state(make_harness, body):
    h = make_harness()
    before = h.load()
    assert_rejected(h.upload(body), 400)
    after = h.state()
    assert after == before and not after["busy"]


@pytest.mark.parametrize("duplicate", ["file", "metadata"])
def test_duplicate_multipart_fields_are_not_silently_selected(make_harness, duplicate):
    h = make_harness()
    before = h.load()
    metadata = json.dumps({"keys": keys(before), "budget": 1})
    parts = [("file", (io.BytesIO(csv_bytes(rows())), "data.csv")), ("metadata", metadata)]
    parts.append(("file", (io.BytesIO(b"id,smiles\nother,CO\n"), "other.csv"))
                 if duplicate == "file" else ("metadata", metadata))
    response = h.client.post("/api/upload", base_url=ORIGIN,
        headers={"Origin": ORIGIN, "X-CSRF-Token": h.csrf}, data=MultiDict(parts),
        content_type="multipart/form-data")
    # A two-part parser cap may reject before the duplicate-field schema check.
    assert response.status_code in (400, 413)
    assert_rejected(response, response.status_code)
    assert h.state() == before


def test_candidate_row_limit_has_inclusive_boundary(make_harness):
    h = make_harness()
    candidates = [{"id": str(n), "smiles": f"[{n}CH4]"} for n in range(1, 201)]
    accepted = h.load(candidates)
    assert len(accepted["dataset"]["rows"]) == 200
    response = h.upload(csv_bytes([*candidates, {"id": "201", "smiles": "[201CH4]"}]))
    assert h.state() == accepted
    assert_rejected(response, 413)


@pytest.mark.parametrize("candidate", [{"id": "x" * 4001, "smiles": "CCO"},
                                      {"id": "long-smiles", "smiles": "C" * 4001}],
                         ids=["id", "smiles"])
def test_candidate_field_limits_preserve_valid_boundary_length(make_harness, candidate):
    h = make_harness()
    boundary = h.load([{"id": "x" * 4000, "smiles": "CCO"}])
    assert boundary["dataset"]["rows"][0]["id"] == "x" * 4000
    response = h.upload(csv_bytes([candidate]))
    assert h.state() == boundary
    assert_rejected(response, 413)


def test_exact_hostile_ids_round_trip_through_controls_evidence_depiction_and_export(make_harness):
    h = make_harness()
    identifiers = [" a ", "a\nb", r"a\nb", "a&copy;", '<img src=x onerror="alert(1)">']
    candidates = [{"id": identifier, "smiles": smiles} for identifier, smiles in
                  zip(identifiers, ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "C1CCCCC1"])]
    initial = h.load(candidates, budget=2)
    assert {r["id"] for r in initial["dataset"]["rows"]} == set(identifiers)
    response = h.post("/api/constraints", {"keys": keys(initial),
        "patch": {"include_ids": [identifiers[-1]], "exclude_ids": [identifiers[1]]},
        "action": "Pin exact hostile ID and exclude literal newline ID"})
    assert response.status_code == 200
    current = response.get_json()["state"]
    assert current["constraints"]["include_ids"] == [identifiers[-1]]
    assert current["constraints"]["exclude_ids"] == [identifiers[1]]
    op = h.complete(h.post("/api/evidence", {"keys": keys(current), "ids": [identifiers[-1]]}))
    assert op["status"] == "complete"
    current = h.state()
    assert current["snapshots"][0]["candidate_id"] == identifiers[-1]
    depiction = h.get("/api/depiction", query_string={
        "session_id": current["session_id"], "candidate_id": identifiers[-1]})
    assert depiction.status_code == 200
    assert depiction.mimetype == "image/svg+xml"
    svg = ElementTree.fromstring(depiction.get_data())
    assert identifiers[-1] not in "".join(svg.itertext())
    assert not any(element.tag.rsplit("}", 1)[-1] == "script" for element in svg.iter())
    response = h.post("/api/export", {"keys": keys(current, reviewed=True), "format": "json"})
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].startswith("attachment;")
    exported = response.get_json()
    assert {r["id"] for r in exported["dataset"]["rows"]} == set(identifiers)
    assert exported["snapshots"] == current["snapshots"]
    assert exported["constraints"] == current["constraints"]
    json.dumps(exported, allow_nan=False)


def test_identical_reupload_invalidates_every_old_http_identity(make_harness):
    h = make_harness()
    old = h.load()
    operation = h.complete(h.post("/api/evidence", {"keys": keys(old), "ids": ["a"]}))
    old = h.state()
    current = h.load(list(reversed(rows())))
    assert current["dataset_digest"] == old["dataset_digest"]
    assert current["session_id"] != old["session_id"]
    assert_rejected(h.get("/api/state", query_string={"session_id": old["session_id"]}), 409)
    assert_rejected(h.get("/api/operation", query_string={"session_id": old["session_id"],
                    "operation_id": operation["operation_id"]}), 409)
    for path, payload in (("/api/evidence", {"ids": ["a"]}),
                          ("/api/review", {}), ("/api/export", {"format": "json"})):
        response = h.post(path, {"keys": keys(old, reviewed=path != "/api/evidence"), **payload})
        assert_rejected(response, 409)
    assert h.state() == current and not h.pending


def test_stale_revision_and_evidence_requests_preserve_current_http_state(make_harness):
    h = make_harness()
    initial = h.load()
    response = h.post("/api/constraints", {"keys": keys(initial),
        "patch": {"budget": 1}, "action": "Reduce capacity"})
    assert response.status_code == 200
    revised = response.get_json()["state"]
    assert revised["revision"] == initial["revision"] + 1

    stale_change = h.post("/api/constraints", {"keys": keys(initial),
        "patch": {"exclude_ids": ["a"]}, "action": "Outdated change"})
    assert_rejected(stale_change, 409)
    assert stale_change.get_json()["error"]["code"] == "stale_revision"
    assert h.state() == revised and not h.pending and not h.source.calls

    operation = h.complete(h.post("/api/evidence", {"keys": keys(revised), "ids": ["a"]}))
    assert operation["status"] == "complete"
    observed = h.state()
    assert observed["revision"] == revised["revision"]
    assert observed["evidence_version"] == revised["evidence_version"] + 1
    assert observed["review"] is None and len(observed["snapshots"]) == 1
    for path, payload in (("/api/review", {}), ("/api/export", {"format": "json"})):
        rejected = h.post(path, {"keys": keys(revised, reviewed=True), **payload})
        assert_rejected(rejected, 409)
        assert rejected.get_json()["error"]["code"] == "stale_evidence"
        assert "Content-Disposition" not in rejected.headers
        assert h.state() == observed and h.state()["review"] is None
    assert not h.pending and len(h.source.calls) == 1


def test_evidence_cache_and_failed_refresh_share_the_same_http_path(make_harness):
    h = make_harness()
    state = h.load()
    first = h.complete(h.post("/api/evidence", {"keys": keys(state), "ids": ["a"]}))
    assert first["status"] == "complete"
    observed = h.state()
    h.complete(h.post("/api/evidence", {"keys": keys(observed), "ids": ["a"]}))
    assert len(h.source.calls) == 1
    assert h.state()["snapshots"] == observed["snapshots"]
    assert h.state()["evidence_version"] == observed["evidence_version"]
    reviewed = h.post("/api/review", {"keys": keys(h.state(), reviewed=True)})
    assert reviewed.status_code == 200 and reviewed.get_json()["state"]["review"]
    h.source.status = "error"
    refreshed = h.complete(h.post("/api/evidence", {"keys": keys(h.state()), "ids": ["a"], "refresh": True}))
    assert refreshed["status"] == "failed" and refreshed["error"]["code"] == "partial_evidence"
    failed = h.state()
    assert failed["review"] is None and len(failed["snapshots"]) == 2
    assert failed["snapshots"][0] == observed["snapshots"][0]
    assert failed["snapshots"][1]["result"]["status"] == "error"
    h.complete(h.post("/api/evidence", {"keys": keys(failed), "ids": ["a"]}))
    assert len(h.source.calls) == 2
    assert h.state()["evidence_version"] == failed["evidence_version"]


@pytest.mark.parametrize("status", ["not_found", "identity_mismatch"])
def test_completed_empty_identity_lookups_are_not_provider_failures(make_harness, status):
    h = make_harness()
    h.source.status = status
    state = h.load()
    operation = h.complete(h.post("/api/evidence", {"keys": keys(state), "ids": ["a"]}))
    assert operation["status"] == "complete"
    assert operation["error"] is None
    current = h.state()
    assert not current["busy"]
    assert current["snapshots"][0]["result"]["status"] == status
    assert current["snapshots"][0]["result"]["records"] == []


@pytest.mark.parametrize("failure", ["unexpected", "malformed"])
def test_worker_failure_releases_busy_but_preserves_committed_and_retrieved_work(make_harness, failure):
    def assistant(session, token, message, *, deadline):
        assert isinstance(deadline, (int, float)) and deadline > 0
        session.interpret(token, message, {"decision": "explicit_change", "span": message, "budget": 1})
        session.lookup(token, "a")
        if failure == "unexpected":
            raise RuntimeError("private diagnostic should not reach HTTP")
        return {"status": "complete", "unexpected_key": "unvalidated provider answer"}

    h = make_harness(assistant=assistant)
    initial = h.load()
    operation = h.complete(h.post("/api/chat", {"keys": keys(initial), "message": "Reduce budget to one"}))
    current = h.state()
    assert operation["status"] == "failed" and not current["busy"]
    if failure == "malformed":
        assert operation["error"]["code"] == "invalid_output"
    assert current["constraints"]["budget"] == 1
    assert current["revision"] == initial["revision"] + 1
    assert current["evidence_version"] == 1 and len(current["snapshots"]) == 1
    assert current["turns"][-1]["interpretation"]["decision"] == "explicit_change"
    assert "private diagnostic" not in json.dumps(current)
    restored = h.post("/api/constraints", {"keys": keys(current), "patch": {"budget": 2}, "action": "Restore capacity"})
    assert restored.status_code == 200


def test_worker_submission_failure_releases_operation_without_erasing_dataset(make_harness):
    def cannot_submit(worker):
        raise RuntimeError("thread creation failed")

    h = make_harness(start_worker=cannot_submit)
    before = h.load()
    response = h.post("/api/evidence", {"keys": keys(before), "ids": ["a"]})
    current = h.state()
    assert not current["busy"] and current["dataset"] == before["dataset"]
    assert current["constraints"] == before["constraints"]
    assert not current["snapshots"] and not h.source.calls
    assert h.post("/api/constraints", {"keys": keys(current), "patch": {"budget": 1}, "action": "Direct change"}).status_code == 200
    # Failure to create the worker is service unavailability, before work begins.
    assert_rejected(response, 503)
    assert response.get_json()["error"]["code"] == "worker_unavailable"


def test_no_assistant_keeps_direct_controls_and_comparison_available_without_replay(make_harness):
    h = make_harness()
    state = h.load()
    bootstrap = h.get("/api/bootstrap").get_json()
    assert bootstrap["assistant_available"] is False
    assert_rejected(h.post("/api/chat", {"keys": keys(state), "message": "Help"}), 503)
    assert h.state() == state and not h.pending
    revised = h.post("/api/constraints", {"keys": keys(state), "patch": {"budget": 1}, "action": "One slot"})
    assert revised.status_code == 200
    current = revised.get_json()["state"]
    operation = h.complete(h.post("/api/compare", {"keys": keys(current), "a_id": "a", "b_id": "b"}))
    assert operation["status"] == "complete"
    assert operation["events"][0]["result"]["a"]["id"] == "a"
    assert operation["events"][0]["result"]["b"]["id"] == "b"
    before_reload = h.state()
    for _ in range(2):
        assert h.get("/api/bootstrap").get_json()["state"] == before_reload
    assert not h.pending and not h.source.calls
    restarted = make_harness()
    assert restarted.initial["dataset"] is None
    assert restarted.csrf != h.csrf
    assert not restarted.pending and not restarted.source.calls


def test_configured_assistant_is_not_replayed_on_bootstrap_or_process_restart(make_harness):
    calls = []

    def assistant(session, token, message, *, deadline):
        calls.append(message)
        session.interpret(token, message, {"decision": "question", "span": message})
        return {"status": "complete", "explanation": "The batch remains unchanged.",
                "result_refs": [], "unresolved_questions": [], "proposal_id": None}

    h = make_harness(assistant=assistant)
    state = h.load()
    operation = h.complete(h.post("/api/chat", {"keys": keys(state), "message": "What is the current batch?"}))
    assert operation["status"] == "complete" and len(calls) == 1
    current = h.state()
    for _ in range(2):
        bootstrap = h.get("/api/bootstrap").get_json()
        assert bootstrap["assistant_available"] is True and bootstrap["state"] == current
    assert len(calls) == 1 and not h.pending
    restarted = make_harness(assistant=assistant)
    assert restarted.initial["dataset"] is None and not restarted.initial["turns"]
    assert len(calls) == 1 and not restarted.pending


@pytest.mark.parametrize("tool_name,kind,data", [
    ("search_external_analogs", "external_analogs", {
        "status": "ok", "query_smiles": "CCO", "analogs": [{"smiles": "CCN"}]}),
    ("enrich_papers", "papers", {
        "status": "ok", "query_smiles": "CCO", "papers": [{"document_id": "CHEMBL1148425"}]}),
])
def test_research_attachment_answer_survives_gateway_validation(make_harness, tool_name, kind, data):
    from src.workbench.assistant import operation_attachments

    def assistant(session, token, message, *, deadline):
        session.interpret(token, message, {"decision": "question", "span": message})
        event = session.record_tool_result(token, tool_name, {"smiles": "CCO"}, data)
        operation = session.poll(session.bootstrap()["session_id"], token)
        return {"status": "complete", "explanation": "These retained research results need review.",
                "result_refs": [event["result_id"]], "unresolved_questions": [], "proposal_id": None,
                "attachments": operation_attachments(operation, [])}

    h = make_harness(assistant=assistant)
    state = h.load()
    operation = h.complete(h.post("/api/chat", {"keys": keys(state), "message": "Show research results"}))
    assert operation["status"] == "complete" and operation["error"] is None
    answer = h.state()["turns"][-1]["answer"]
    assert answer["explanation"] == "These retained research results need review."
    assert answer["result_refs"] == [operation["events"][0]["result_id"]]
    assert answer["attachments"] == [{"kind": kind, "data": data}]


def test_chat_character_limit_rejects_excess_without_truncating_or_starting_work(make_harness):
    calls = []

    def assistant(session, token, message, *, deadline):
        calls.append(message)
        session.interpret(token, message, {"decision": "question", "span": message})
        return {"status": "complete", "explanation": "No constraints changed.",
                "result_refs": [], "unresolved_questions": [], "proposal_id": None}

    h = make_harness(assistant=assistant)
    initial = h.load()
    response = h.post("/api/chat", {"keys": keys(initial), "message": "x" * 8001})
    assert_rejected(response, 413)
    assert response.get_json()["error"]["code"] == "limit_exceeded"
    assert h.state() == initial and not h.pending and not calls and not h.source.calls

    boundary = "x" * 8000
    operation = h.complete(h.post("/api/chat", {"keys": keys(initial), "message": boundary}))
    assert operation["status"] == "complete" and calls == [boundary]
    assert h.state()["turns"][-1]["message"] == boundary


def test_default_worker_rejects_conflicting_click_while_network_is_in_flight(make_harness, monkeypatch):
    source = Evidence()
    source.entered, source.release = threading.Event(), threading.Event()
    h = make_harness(threaded=True, source=source)
    state = h.load()
    finished = threading.Event()
    finish_operation = h.session.finish_operation

    def finish_and_signal(token, outcome):
        try:
            return finish_operation(token, outcome)
        finally:
            finished.set()

    monkeypatch.setattr(h.session, "finish_operation", finish_and_signal)
    response = h.post("/api/evidence", {"keys": keys(state), "ids": ["a"]})
    assert response.status_code == 202
    token = response.get_json()["operation_id"]
    try:
        assert source.entered.wait(3)
        assert h.poll(token)["status"] == "running"
        assert_rejected(h.post("/api/constraints", {"keys": keys(state), "patch": {"budget": 1},
                                                  "action": "Conflicting click"}), 409)
        assert h.state()["constraints"] == state["constraints"]
    finally:
        source.release.set()
        assert finished.wait(3)
    assert h.poll(token)["status"] == "complete"
    assert not h.state()["busy"] and len(source.calls) == 1


@pytest.mark.parametrize("configuration", ["configured", "missing", "blank", "ledger_failure", "provider_failure"])
def test_module_launcher_configures_assistant_from_environment_and_fails_closed(monkeypatch, tmp_path, capsys, configuration):
    from src.workbench import api as workbench, assistant as workbench_assistant, paid_ledger as workbench_paid

    calls, launched = [], {}
    monkeypatch.setattr("sys.argv", ["workbench", "--port", "8787"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=must-not-be-loaded\n")
    if configuration == "missing":
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", " " if configuration == "blank" else "private-test-key")
    science = ScientificOperations(screen=Screen(), model_digest="a" * 64,
                                   analysis=lambda name, args: getattr(cyp_tools, name)(**args))
    monkeypatch.setattr(workbench, "ScientificOperations", lambda path: science)
    monkeypatch.setattr(workbench, "EvidenceClient", Evidence)
    ledger = object()

    def make_ledger():
        calls.append("ledger")
        if configuration == "ledger_failure":
            raise workbench_paid.LedgerError("private-ledger-error")
        return ledger

    def verify_provider():
        calls.append("public-pricing")
        if configuration == "provider_failure":
            raise RuntimeError("private-provider-error")
        return {"verified": True}

    def run(app, **kwargs):
        launched.update(app=app, options=kwargs)

    monkeypatch.setattr(workbench_paid, "FixedPaidLedger", make_ledger)
    monkeypatch.setattr(workbench_assistant, "verify_provider", verify_provider)
    monkeypatch.setattr(workbench.Flask, "run", run)
    workbench.main()
    assert launched["options"] == {"host": "127.0.0.1", "port": 8787, "debug": False,
                                   "use_reloader": False, "load_dotenv": False}
    bootstrap = launched["app"].test_client().get("/api/bootstrap", headers={"Host": "127.0.0.1:8787"}).get_json()
    assert bootstrap["assistant_available"] is (configuration == "configured")
    assert bootstrap["state"]["dataset"] is None
    assert calls == ([] if configuration in {"missing", "blank"} else
                     ["ledger"] if configuration == "ledger_failure" else ["ledger", "public-pricing"])
    output = capsys.readouterr()
    assert "private-" not in output.out + output.err and "must-not-be-loaded" not in output.out + output.err
    if configuration != "configured":
        assert "Mo is unavailable" in output.err


@pytest.mark.parametrize("fail", [False, True])
def test_standalone_chat_uses_session_and_shared_ledger(tmp_path, monkeypatch, capsys, fail):
    from src.workbench import chat

    candidates = tmp_path / "candidates.csv"
    candidates.write_text("id,smiles\na,CCO\n")
    monkeypatch.setattr("sys.argv", ["chat", "Profile a", "--candidates", str(candidates)])
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    science = ScientificOperations(screen=Screen(), model_digest="a" * 64)
    monkeypatch.setattr(chat, "ScientificOperations", lambda artifacts: science)
    monkeypatch.setattr(chat, "EvidenceClient", Evidence)
    ledger, used = object(), []
    monkeypatch.setattr(chat, "FixedPaidLedger", lambda: ledger)

    def make_assistant(key, paid):
        assert key == "fixture-key" and paid is ledger

        def answer(session, token, message, *, deadline):
            used.append(session)
            session.interpret(token, message, {"decision": "question", "span": message})
            session.record_tool_result(token, "predict_cyp", {"smiles": "CCO"}, {"query_smiles": "CCO"})
            if fail:
                raise ValueError("private provider detail")
            return {"status": "complete", "explanation": "Fixture profile"}
        return answer

    monkeypatch.setattr(chat, "Assistant", make_assistant)
    if fail:
        with pytest.raises(SystemExit, match="1"):
            chat.main()
    else:
        chat.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == ("failed" if fail else "complete")
    assert result["attachments"] == [{"kind": "scores", "data": {"query_smiles": "CCO"}}]
    assert "private provider detail" not in json.dumps(result)
    assert not used[0].bootstrap()["busy"]


@pytest.mark.parametrize("path,name,arguments", [
    ("/api/score?smiles=CCO", "predict_cyp", {"smiles": "CCO"}),
    ("/api/diff?a=CCO&b=CCN", "show_structures", {"first_smiles": "CCO", "second_smiles": "CCN"}),
    ("/api/analogs?smiles=CCO&enzyme=CYP3A4", "find_analogs", {"smiles": "CCO", "enzyme": "CYP3A4", "measured_label": 0}),
    ("/api/attribution?smiles=CCO&enzyme=CYP3A4", "show_attribution", {"smiles": "CCO", "enzyme": "CYP3A4"}),
    ("/api/neighbourhood?smiles=CCO&enzyme=CYP3A4", "show_neighbourhood", {"smiles": "CCO", "enzyme": "CYP3A4"}),
])
def test_analysis_http_routes_use_session_artifacts(monkeypatch, path, name, arguments):
    h = Harness()
    calls = []
    monkeypatch.setattr(h.session._science, "analyze", lambda tool, args: calls.append((tool, args)) or {"fixture": True})
    response = h.get(path)
    assert response.status_code == 200 and response.get_json() == {"fixture": True}
    assert calls == [(name, arguments)]
