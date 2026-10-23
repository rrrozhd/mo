"""The dated demo capture must reproduce through the original evidence client."""

import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from src.sources import chembl as evidence
from src.screening.cli import load_candidates
from src.workbench.science import digest


def test_capture_binds_the_actual_sample_dataset():
    bundle = json.loads(Path("examples/workbench-evidence.json").read_text())
    rows = load_candidates("examples/cyp3a4-candidates.csv")
    assert bundle["dataset_digest"] == digest(sorted([[r["id"], r["smiles"]] for r in rows]))
    by_id = {row["id"]: row["smiles"] for row in rows}
    assert all(by_id[entry["candidate_id"]] == entry["canonical_query"]
               for entry in bundle["entries"])


def test_raw_http_bodies_reproduce_every_stored_observation(monkeypatch):
    bundle = json.loads(Path("examples/workbench-evidence.json").read_text())
    monkeypatch.setattr(evidence, "REQUEST_INTERVAL", 0)
    for entry in bundle["entries"]:
        pending = list(entry["raw_responses"])
        requests = []

        def replay(request, timeout):
            captured = pending.pop(0)
            url = urlsplit(request.full_url)
            assert url.path.rsplit("/", 1)[-1] == captured["resource"]
            parameters = parse_qs(url.query)
            requests.append((captured["resource"], parameters))
            assert timeout == 10
            if captured["resource"] == "activity.json":
                assert parameters["limit"] == ["10"]
                body = json.loads(captured["body_utf8"])
                assert body["page_meta"]["limit"] == bundle["record_limit"] == 10
            return io.BytesIO(captured["body_utf8"].encode("utf-8"))

        monkeypatch.setattr(evidence, "urlopen", replay)
        actual = evidence.EvidenceClient().lookup(entry["canonical_query"], 10)
        # Replay receives its own clock reading; every scientific/source field must match.
        assert actual["retrieved_at"] != entry["result"]["retrieved_at"]
        actual["retrieved_at"] = entry["result"]["retrieved_at"]
        assert actual == entry["result"]
        assert not pending and len(requests) == 3
