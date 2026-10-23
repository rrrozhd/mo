"""Ask one question through the same bounded session assistant used by the web app."""

import argparse
import json
import os
from pathlib import Path
import time

from ..sources.chembl import EvidenceClient
from .api import ROOT, _csv, _failure
from .assistant import Assistant, operation_attachments
from .paid_ledger import FixedPaidLedger
from .science import RecordedEvidence, ScientificOperations
from .session import Session, SessionError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--candidates", required=True, type=Path, help="CSV with id,smiles columns")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--source-mode", choices=("live", "recorded"), default="live")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.fixture is not None and args.source_mode != "recorded":
        parser.error("--fixture requires --source-mode recorded")
    science = ScientificOperations(args.artifacts)
    source = (RecordedEvidence(args.fixture or ROOT / "examples/workbench-evidence.json", science.model_digest)
              if args.source_mode == "recorded" else EvidenceClient())
    session = Session(science, source)
    state = session.bootstrap()
    keys = {"session_id": state["session_id"], "dataset_digest": state["dataset_digest"],
            "base_revision": state["revision"]}
    rows = _csv(args.candidates.read_bytes())
    state = session.replace_dataset(keys, rows, min(4, len(rows)), "Load CLI candidates")
    keys.update(session_id=state["session_id"], dataset_digest=state["dataset_digest"],
                base_revision=state["revision"])
    # The fixed ledger is shared with the browser; a CLI invocation cannot reset spend.
    assistant = Assistant(os.environ.get("OPENROUTER_API_KEY", ""), FixedPaidLedger())
    deadline = time.monotonic() + 120
    token = session.start_operation(keys, "chat", message=args.question)
    try:
        outcome = assistant(session, token, args.question, deadline=deadline)
    except SessionError as exc:
        outcome = _failure(exc.code)
    except Exception:
        outcome = _failure()
    state = session.bootstrap()
    outcome["attachments"] = operation_attachments(session.poll(state["session_id"], token), state["snapshots"])
    session.finish_operation(token, outcome)
    print(json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False))
    if outcome["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
