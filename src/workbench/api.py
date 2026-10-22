"""Loopback-only HTTP boundary for the in-memory screening workbench."""

import argparse
from collections.abc import Callable
import csv
import io
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time

from flask import Flask, Request, Response, request, send_from_directory
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from ..chembl import EvidenceClient
from .science import RecordedEvidence, ScientificOperations, canonical_json
from .session import Session, SessionError

MAX_UPLOAD = 2 * 1024 * 1024
MAX_OVERHEAD = 64 * 1024
ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parents[1] / "workbench_web"
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
FAILURES = {
    "operation_failed": "The operation could not complete. Earlier results are retained.",
    "operation_expired": "The operation deadline was reached. Earlier results are retained.",
    "partial_evidence": "Some evidence lookups are incomplete. Inspect their retained statuses.",
    "provider_failure": "The assistant provider could not complete this request.",
    "invalid_output": "The assistant returned an invalid answer. Earlier results are retained.",
    "limit_exceeded": "An operation limit was reached. Earlier results are retained.",
    "invalid_request": "The operation received an invalid request.",
    "invalid_constraints": "The requested constraints are invalid.",
    "invalid_intent": "The current message could not be interpreted safely.",
    "stale_operation": "The operation belongs to an earlier session or revision.",
    "unavailable_configuration": "The assistant is not configured for this process.",
}


class MemoryUploadRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        # The whole request is bounded; uploaded names are never used as paths.
        return io.BytesIO()


def _object(value, required, optional=()):
    if type(value) is not dict or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise SessionError("invalid_request", "Missing or unknown request fields")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _json(raw):
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="strict")
        value = json.loads(raw, object_pairs_hook=_pairs)
        canonical_json(value)
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SessionError("invalid_request", "Expected finite UTF-8 JSON without duplicate fields") from exc


def _csv(raw):
    if len(raw) > MAX_UPLOAD:
        raise RequestEntityTooLarge()
    try:
        text = raw.decode("utf-8", errors="strict")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        headers = next(reader)
        if (len(set(headers)) != len(headers) or not {"id", "smiles"} <= set(headers)
                or any(not header for header in headers)):
            raise ValueError("Invalid CSV header")
        indices = [headers.index(name) for name in ("id", "smiles")]
        rows = []
        for values in reader:
            if len(values) != len(headers):
                raise ValueError("CSV row width differs from header")
            row = {name: values[index] for name, index in zip(("id", "smiles"), indices)}
            if any(len(value) > 4000 for value in row.values()):
                raise SessionError("limit_exceeded", "Candidate fields may contain at most 4,000 characters")
            if any(not value.strip() for value in row.values()):
                raise ValueError("Invalid candidate field length")
            rows.append(row)
            if len(rows) > 200:
                raise SessionError("limit_exceeded", "A dataset may contain at most 200 candidate rows")
        if not rows:
            raise ValueError("Empty CSV")
        return rows
    except SessionError:
        raise
    except (UnicodeError, csv.Error, ValueError, StopIteration) as exc:
        raise SessionError("invalid_request", "CSV needs unique headers including id and smiles, and 1–200 valid rows with fields up to 4,000 characters") from exc


def _failure(code="operation_failed"):
    code = code if code in FAILURES else "operation_failed"
    return {"status": "failed", "error": {"code": code, "message": FAILURES[code]}}


ATTACHMENT_KINDS = ("scores", "analogs", "external_analogs", "structures", "map", "attribution", "evidence", "papers")


def _attachments(value):
    """Rendered blocks a turn carries. Each is a known kind plus the tool's own output."""
    if type(value) is not list or len(value) > 12:
        raise ValueError("Attachments must be a list of at most twelve blocks")
    for block in value:
        item = _object(block, {"kind", "data"})
        if item["kind"] not in ATTACHMENT_KINDS or type(item["data"]) is not dict:
            raise ValueError("Invalid attachment kind or data")
    return value


def _outcome(value):
    _object(value, {"status"},
            {"explanation", "result_refs", "unresolved_questions", "proposal_id", "error",
             "attachments"})
    canonical_json(value)
    if value["status"] not in ("complete", "failed"):
        raise ValueError("Invalid outcome status")
    if "explanation" in value and type(value["explanation"]) is not str:
        raise ValueError("Invalid explanation")
    for name in ("result_refs", "unresolved_questions"):
        if name in value and (type(value[name]) is not list or any(type(item) is not str for item in value[name])):
            raise ValueError("Invalid outcome references or questions")
    if "proposal_id" in value and value["proposal_id"] is not None and type(value["proposal_id"]) is not str:
        raise ValueError("Invalid proposal ID")
    if value.get("attachments") is not None:
        _attachments(value["attachments"])
    if value.get("error") is not None:
        error = _object(value["error"], {"code", "message"})
        if type(error["code"]) is not str or type(error["message"]) is not str:
            raise ValueError("Invalid outcome error")
        value = {**value, "error": _failure(error["code"])["error"]}
    return value


def create_app(session, *, port=8765, assistant: Callable | None = None,
               start_worker=None):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    csv.field_size_limit(MAX_UPLOAD)
    app = Flask(__name__, static_folder=None)
    app.request_class = MemoryUploadRequest
    app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD + MAX_OVERHEAD,
                      MAX_FORM_MEMORY_SIZE=MAX_UPLOAD + MAX_OVERHEAD, MAX_FORM_PARTS=2)
    origin, host = f"http://127.0.0.1:{port}", f"127.0.0.1:{port}"
    csrf = secrets.token_urlsafe(32)
    launch_worker = start_worker or (lambda worker: threading.Thread(target=worker, daemon=True).start())

    def response(value, status=200):
        return Response(canonical_json(value), status=status, mimetype="application/json")

    def error(code, message, status):
        return response({"error": {"code": code, "message": message}}, status)

    @app.before_request
    def guard():
        if request.headers.get("Host") != host:
            return error("invalid_host", "Use the configured loopback address", 403)
        if any(len(values) != 1 for _, values in request.args.lists()):
            raise SessionError("invalid_request", "Duplicate query fields are not allowed")
        if request.method == "POST":
            if request.headers.get("Origin") != origin:
                return error("invalid_origin", "A same-origin request is required", 403)
            supplied = request.headers.get("X-CSRF-Token", "").encode("utf-8")
            if not secrets.compare_digest(supplied, csrf.encode("ascii")):
                return error("invalid_csrf", "A valid CSRF token is required", 403)
            if request.args:
                raise SessionError("invalid_request", "Mutation requests do not accept query fields")
            if request.path != "/api/upload":
                request.max_content_length = MAX_UPLOAD
        return None

    @app.after_request
    def headers(result):
        result.headers["Content-Security-Policy"] = CSP
        result.headers["Cache-Control"] = "no-store"
        result.headers["X-Content-Type-Options"] = "nosniff"
        result.headers["Referrer-Policy"] = "no-referrer"
        return result

    @app.errorhandler(SessionError)
    def session_error(exc):
        status = 409 if exc.code == "busy" or exc.code.startswith("stale_") else 400
        if exc.code == "limit_exceeded":
            status = 413
        return error(exc.code, exc.message, status)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        code = "too_large" if exc.code == 413 else "invalid_request"
        message = "The request exceeds the size limit" if exc.code == 413 else "The HTTP request is invalid"
        return error(code, message, exc.code or 400)

    @app.errorhandler(Exception)
    def unexpected_error(_exc):
        return error("internal_error", "The request could not complete. The last valid state is retained.", 500)

    def query(names):
        return _object(request.args.to_dict(), set(names))

    def body(required, optional=()):
        if request.mimetype != "application/json":
            raise SessionError("invalid_request", "Content-Type must be application/json")
        value = _object(_json(request.get_data(cache=False)), required, optional)
        if "keys" in value and type(value["keys"]) is not dict:
            raise SessionError("invalid_request", "keys must be an object")
        return value

    def known_ids(keys, ids):
        if (type(ids) is not list or not 1 <= len(ids) <= 200
                or any(type(identifier) is not str for identifier in ids) or len(set(ids)) != len(ids)):
            raise SessionError("invalid_request", "Choose unique exact candidate IDs")
        state = session.read(keys.get("session_id"))
        known = {row["id"] for row in (state["dataset"] or {}).get("rows", [])}
        if set(ids) - known:
            raise SessionError("invalid_request", "Unknown candidate ID")

    def job(keys, task, *, kind="evidence", message=None):
        deadline = time.monotonic() + 120
        token = session.start_operation(keys, kind, message=message)

        def worker():
            outcome = _failure()
            try:
                if time.monotonic() >= deadline or session.poll(keys["session_id"], token)["status"] != "running":
                    outcome = _failure("operation_expired")
                else:
                    value = task(token, deadline)
                    try:
                        outcome = _outcome(value)
                    except (SessionError, ValueError, TypeError):
                        outcome = _failure("invalid_output")
                    if time.monotonic() >= deadline:
                        outcome = _failure("operation_expired")
            except SessionError as exc:
                outcome = _failure(exc.code)
            except Exception:
                outcome = _failure()
            finally:
                try:
                    if kind == "chat":
                        from .assistant import operation_attachments
                        outcome["attachments"] = operation_attachments(session.poll(keys["session_id"], token), session.read(keys["session_id"])["snapshots"])
                    session.finish_operation(token, outcome)
                except SessionError:
                    # A stale/expired worker never acquires authority over a new operation.
                    pass

        try:
            launch_worker(worker)
        except Exception:
            session.finish_operation(token, _failure())
            return error("worker_unavailable", "The worker could not start", 503)
        return response({"operation_id": token}, 202)

    @app.get("/")
    def index():
        query(())
        return send_from_directory(WEB, "index.html")

    @app.get("/assets/<path:name>")
    def asset(name):
        query(())
        if name not in {"app.js", "style.css"}:
            return error("not_found", "Asset not found", 404)
        return send_from_directory(WEB, name)

    @app.get("/api/bootstrap")
    def bootstrap():
        query(())
        return response({"state": session.bootstrap(), "csrf_token": csrf, "assistant_available": assistant is not None})

    @app.get("/api/state")
    def state():
        values = query({"session_id"})
        return response({"state": session.read(values["session_id"])})

    @app.post("/api/sample")
    def sample():
        values = body({"keys"})
        rows = _csv((ROOT / "examples" / "cyp3a4-candidates.csv").read_bytes())
        return response({"state": session.replace_dataset(values["keys"], rows, 4, "Load sample candidates")})

    @app.post("/api/upload")
    def upload():
        if request.mimetype != "multipart/form-data":
            raise SessionError("invalid_request", "Upload must use multipart/form-data")
        raw = request.get_data(cache=True)
        if (set(request.files) != {"file"} or set(request.form) != {"metadata"}
                or len(request.files.getlist("file")) != 1 or len(request.form.getlist("metadata")) != 1):
            raise SessionError("invalid_request", "Upload needs exactly one file and metadata object")
        uploaded = request.files["file"].stream.read(MAX_UPLOAD + 1)
        if len(uploaded) > MAX_UPLOAD or len(raw) - len(uploaded) > MAX_OVERHEAD:
            raise RequestEntityTooLarge()
        metadata = _object(_json(request.form["metadata"]), {"keys", "budget"})
        rows = _csv(uploaded)
        return response({"state": session.replace_dataset(metadata["keys"], rows, metadata["budget"], "Upload candidate CSV")})

    @app.post("/api/constraints")
    def constraints():
        values = body({"keys", "patch", "action"})
        return response({"state": session.revise(values["keys"], values["patch"], values["action"])})

    @app.post("/api/evidence")
    def evidence():
        values = body({"keys", "ids"}, {"refresh"})
        refresh = values.get("refresh", False)
        if type(refresh) is not bool:
            raise SessionError("invalid_request", "refresh must be a boolean")
        known_ids(values["keys"], values["ids"])

        def lookup(token, _deadline):
            snapshots = [session.lookup(token, identifier, refresh=refresh) for identifier in values["ids"]]
            incomplete = any(snapshot["result"]["status"] not in {"ok", "not_found", "identity_mismatch"}
                             or snapshot["result"].get("truncated", False)
                             or snapshot["result"].get("candidates_truncated", False) for snapshot in snapshots)
            outcome = _failure("partial_evidence") if incomplete else {"status": "complete"}
            return {**outcome, "result_refs": [snapshot["snapshot_id"] for snapshot in snapshots]}

        return job(values["keys"], lookup)

    @app.post("/api/compare")
    def compare():
        values = body({"keys", "a_id", "b_id"})
        ids = [values["a_id"]] if values["a_id"] == values["b_id"] else [values["a_id"], values["b_id"]]
        known_ids(values["keys"], ids)
        return job(values["keys"], lambda token, _deadline: {
            "status": "complete", "result_refs": [session.compare(token, values["a_id"], values["b_id"])["result_id"]]})

    @app.post("/api/chat")
    def chat():
        values = body({"keys", "message"})
        assistant_callback = assistant
        if assistant_callback is None:
            return error("unavailable_configuration", FAILURES["unavailable_configuration"], 503)
        def respond(token, deadline):
            return assistant_callback(session, token, values["message"], deadline=deadline)

        return job(values["keys"], respond, kind="chat", message=values["message"])

    @app.post("/api/chat/new")
    def new_chat():
        values = body({"keys"})
        return response({"state": session.new_chat(values["keys"])})

    @app.post("/api/chat/switch")
    def switch_chat():
        values = body({"keys", "chat_id"})
        return response({"state": session.switch_chat(values["keys"], values["chat_id"])})

    @app.get("/api/operation")
    def operation():
        values = query({"session_id", "operation_id"})
        return response({"operation": session.poll(values["session_id"], values["operation_id"])})

    @app.post("/api/apply")
    def apply():
        values = body({"keys", "proposal_id"})
        if type(values["proposal_id"]) is not str:
            raise SessionError("invalid_request", "proposal_id must be a string")
        return response({"state": session.apply_proposal(values["keys"], values["proposal_id"], "Apply selected proposal")})

    @app.post("/api/review")
    def review():
        values = body({"keys"})
        return response({"state": session.mark_reviewed(values["keys"])})

    @app.post("/api/export")
    def export():
        values = body({"keys", "format"})
        if values["format"] != "json":
            raise SessionError("invalid_request", "Only JSON snapshot export is currently available")
        result = response(session.capture_export(values["keys"]))
        result.headers["Content-Disposition"] = 'attachment; filename="workbench-session.json"'
        return result

    @app.get("/api/depiction")
    def depiction():
        values = query({"session_id", "candidate_id"})
        data = session.read(values["session_id"])
        row = next((row for row in (data["dataset"] or {}).get("rows", []) if row["id"] == values["candidate_id"]), None)
        if row is None:
            raise SessionError("invalid_request", "Unknown candidate ID")
        molecule = Chem.MolFromSmiles(row["smiles"])
        drawer = rdMolDraw2D.MolDraw2DSVG(360, 260)
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        return Response(drawer.GetDrawingText(), mimetype="image/svg+xml")

    def analysed(view, **arguments):
        """Views and chat use the session's exact model directory."""
        try:
            return response(session.analyze_view(view, arguments))
        except ValueError as exc:
            raise SessionError("invalid_request", str(exc)) from None

    @app.get("/api/score")
    def score():
        values = query({"smiles"})
        return analysed("predict_cyp", smiles=values["smiles"])

    @app.get("/api/diff")
    def diff():
        values = query({"a", "b"})
        return analysed("show_structures", first_smiles=values["a"], second_smiles=values["b"])

    @app.get("/api/attribution")
    def atom_attribution():
        values = query({"smiles", "enzyme"})
        return analysed("show_attribution", smiles=values["smiles"], enzyme=values["enzyme"])

    @app.get("/api/neighbourhood")
    def neighbourhood():
        values = query({"smiles", "enzyme"})
        return analysed("show_neighbourhood", smiles=values["smiles"], enzyme=values["enzyme"])

    @app.get("/api/analogs")
    def analogs():
        values = query({"smiles", "enzyme"})
        return analysed("find_analogs", smiles=values["smiles"], enzyme=values["enzyme"], measured_label=0)

    return app


def main():
    parser = argparse.ArgumentParser(description="Run Mo, the local molecular screening and evidence workbench")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--source-mode", choices=("live", "recorded"), default="live")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be from 1 to 65535")
    if args.fixture is not None and args.source_mode != "recorded":
        parser.error("--fixture requires --source-mode recorded")
    try:
        science = ScientificOperations(args.artifacts)
        source = (RecordedEvidence(args.fixture or ROOT / "examples" / "workbench-evidence.json", science.model_digest)
                  if args.source_mode == "recorded" else EvidenceClient())
        session = Session(science, source)
    except Exception:
        parser.exit(1, "Unable to load the trusted model or evidence fixture.\n")
    assistant = None
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key.strip():
        try:
            from .assistant import Assistant
            from .paid_ledger import FixedPaidLedger

            assistant = Assistant(api_key, FixedPaidLedger())
        except Exception:
            print("Mo is unavailable: provider or paid ledger configuration could not be verified. "
                  "Direct controls remain available.", file=sys.stderr)
    else:
        print("Mo is unavailable: OPENROUTER_API_KEY is not set. Direct controls remain available.",
              file=sys.stderr)
    app = create_app(session, port=args.port, assistant=assistant)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, load_dotenv=False)


if __name__ == "__main__":
    main()
