from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest
from rdkit import DataStructs

from src.chemistry.features import ENZYMES, fingerprints
from src.sources.chembl import EvidenceClient
from src.screening.cli import load_candidates
from src.screening.shortlist import select_shortlist


def science():
    from src.workbench import science as workbench_science
    return workbench_science


MODEL_DIGEST = "a" * 64
RETRIEVED_AT = "2026-10-21T12:00:00+00:00"


class Screen:
    def __init__(self):
        self.calls = []

    def score(self, rows):
        self.calls.append(deepcopy(rows))
        return [{"id": row["id"], "canonical_smiles": row["smiles"], "score": .5,
                 "validation_percentile": .6, "nearest_training_similarity": .7,
                 "similarity_flag": False, "confidence": "uncalibrated"} for row in rows]


def operations():
    screen = Screen()
    return science().ScientificOperations(screen=screen, model_digest=MODEL_DIGEST), screen


def candidates():
    return [{"id": " a\n", "smiles": "OCC", "label": 1},
            {"id": "é", "smiles": "CCN", "label": 0}]


def result(smiles="CCO"):
    return {"status": "error", "query_smiles": smiles,
            "retrieved_at": RETRIEVED_AT, "records": [{"activity_id": 1,
            "standard_value": "12.3", "value": "0.0123", "units": "uM",
            "standard_units": "nM", "standard_relation": ">", "text_value": "observed",
            "assay_id": "CHEMBL2", "veith_overlap": True,
            "activity_url": "https://www.ebi.ac.uk/chembl/api/data/activity/1.json"}],
            "error": "metadata timeout", "truncated": True,
            "identity_rejected_candidates": [{"compound_id": "CHEMBL3"}],
            "assay_metadata_complete": False, "source": "https://www.ebi.ac.uk/chembl/api/data"}


class LiveEvidence(EvidenceClient):
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def lookup(self, smiles, limit=10):
        self.calls.append((smiles, limit))
        return self.payload


def test_dataset_digest_ignores_labels_and_row_order_but_preserves_exact_ids():
    ops, screen = operations()
    prepared = ops.prepare(candidates())
    expected = b'[[" a\\n","CCO"],["\\u00e9","CCN"]]'
    assert prepared["dataset_digest"] == sha256(expected).hexdigest()
    assert prepared["model_digest"] == MODEL_DIGEST
    changed_labels = [{**row, "label": 1 - row["label"]} for row in reversed(candidates())]
    assert ops.prepare(changed_labels) == prepared
    assert len(screen.calls) == 1
    assert all(set(row) == {"id", "smiles"} for row in prepared["rows"])
    changed_id = [{**row, "id": row["id"].strip()} for row in candidates()]
    assert ops.prepare(changed_id)["dataset_digest"] != prepared["dataset_digest"]
    assert len(screen.calls) == 2


def test_cached_dataset_cannot_be_modified_through_returned_objects():
    ops, screen = operations()
    prepared = ops.prepare(candidates())
    expected = deepcopy(prepared)
    prepared["rows"][0]["smiles"] = "CC"
    prepared["scored"][0]["score"] = 0
    assert ops.prepare(candidates()) == expected
    assert len(screen.calls) == 1


@pytest.mark.parametrize("rows", [[], [{"id": "a", "smiles": "bad"}],
    [{"id": "a", "smiles": "CCO"}, {"id": "a", "smiles": "CCN"}],
    [{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "OCC"}]])
def test_invalid_dataset_never_scores_or_evicts_good_cache(rows):
    ops, screen = operations()
    valid = ops.prepare(candidates())
    with pytest.raises(ValueError):
        ops.prepare(rows)
    assert ops.prepare(candidates()) == valid and len(screen.calls) == 1


def test_selection_is_the_original_selector_including_ties_and_shortfall():
    ops, _ = operations()
    data = ops.prepare([{"id": "b", "smiles": "CCCCCCC"},
                        {"id": "a", "smiles": "CCCCCC"},
                        {"id": "c", "smiles": "CCO"}])
    for constraints in ({"budget": 2, "max_similarity": 1},
                        {"budget": 3, "exclude_ids": ["b"], "include_ids": ["c"]}):
        assert ops.select(data, constraints) == select_shortlist(data["scored"], **constraints)
    shortfall = ops.select(data, {"budget": 3, "exclude_ids": ["b"], "include_ids": ["c"]})
    assert not shortfall["complete"] and shortfall["selected_count"] == 2
    assert shortfall["shortfall_reason"]
    threshold = DataStructs.TanimotoSimilarity(*fingerprints(["CCCCCC", "CCCCCCC"]))
    selected = ops.select(data, {"budget": 2, "include_ids": ["a", "b"],
                                 "max_similarity": threshold})
    assert [row["id"] for row in selected["selected"]] == ["a", "b"]
    with pytest.raises(ValueError, match="Mandatory"):
        ops.select(data, {"budget": 2, "include_ids": ["a", "b"],
                          "max_similarity": threshold - .001})


@pytest.mark.parametrize("first,second,relationship", [
    ("C[C@H](O)F", "C[C@@H](O)F", "stereo_only"),
    ("F/C=C/F", "F/C=C\\F", "stereo_only"),
    ("[13CH3]CO", "CCO", "unresolved"),
    ("C[NH2+]C", "CNC", "unresolved"),
    ("CCO.Cl", "CCO", "unresolved"),
    ("CCO", "COC", "unresolved"),
])
def test_comparison_uses_graphs_preserving_isotopes_charge_and_components(first, second, relationship):
    ops, _ = operations()
    data = ops.prepare([{"id": "a", "smiles": first}, {"id": "b", "smiles": second}])
    refs = {"a": ["snapshot-1"], "b": []}
    compared = ops.compare(data, "a", "b", evidence_refs=refs)
    assert compared["structural_relationship"] == relationship
    assert compared["exact_identity_equal"] is False
    assert compared["a"]["score"] == compared["b"]["score"] == .5
    assert compared["evidence_refs"] == refs
    compared["evidence_refs"]["a"].append("other")
    assert refs["a"] == ["snapshot-1"]
    identical = ops.compare(data, "a", "a")
    assert identical["exact_identity_equal"] and identical["structural_relationship"] == "identical"
    assert identical["fingerprint_tanimoto"] == 1
    with pytest.raises(ValueError, match="Unknown"):
        ops.compare(data, "a", "missing")


def test_real_model_matches_all_sample_selections_and_comparison():
    if not Path("artifacts/models/CYP3A4.joblib").exists():
        pytest.skip("Local trained CYP3A4 artifact is unavailable")
    ops = science().ScientificOperations("artifacts")
    data = ops.prepare(load_candidates("examples/cyp3a4-candidates.csv"))
    assert data["model_digest"] == sha256(Path("artifacts/models/CYP3A4.joblib").read_bytes()).hexdigest()
    for budget, excludes, expected in ((4, [15], [10, 8, 12, 20]),
                                       (4, [15, 8], [10, 9, 12, 20]),
                                       (3, [15, 8], [10, 9, 20])):
        selected = ops.select(data, {"budget": budget, "include_ids": ["candidate-20"],
             "exclude_ids": [f"candidate-{number:02d}" for number in excludes]})
        assert [r["id"] for r in selected["selected"]] == [f"candidate-{n:02d}" for n in expected]
        if excludes == [15]:
            ranked_ids = [row["id"] for row in selected["ranked"]]
            assert ranked_ids.index("candidate-08") < ranked_ids.index("candidate-09")
            reason = next(row["reason"] for row in selected["skipped"] if row["id"] == "candidate-09")
            assert "similarity to 'candidate-08' (1)" in reason
    pair = ops.compare(data, "candidate-08", "candidate-09")
    assert pair["a"]["score"] == pair["b"]["score"] == .7637010646527569
    assert pair["fingerprint_tanimoto"] == 1 and not pair["exact_identity_equal"]
    assert pair["structural_relationship"] == "stereo_only"


def test_snapshot_calls_live_client_at_limit_ten_and_retains_complete_partial_result():
    module = science()
    ops, _ = operations()
    data = ops.prepare(candidates())
    raw = result()
    source = LiveEvidence(raw)
    live = module.evidence_snapshot(source, " a\n", data)
    assert source.calls == [("CCO", 10)]
    assert live["result"] == raw
    assert live["canonical_query"] == "CCO" and live["candidate_id"] == " a\n"
    assert live["dataset_digest"] == data["dataset_digest"]
    assert live["model_digest"] == MODEL_DIGEST and live["origin"] == "live"
    assert live["retrieved_at"] == RETRIEVED_AT
    assert "playback_at" not in live and "fixture_hash" not in live
    raw["records"].clear()
    assert len(live["result"]["records"]) == 1


@pytest.mark.parametrize("change", ["wrong_query", "nonfinite"])
def test_snapshot_refuses_result_identity_or_nonfinite_numbers(change):
    module = science()
    ops, _ = operations()
    data = ops.prepare(candidates())
    raw = result()
    if change == "wrong_query":
        raw["query_smiles"] = "CCN"
    else:
        raw["records"][0]["standard_value"] = float("nan")
    with pytest.raises(ValueError):
        module.evidence_snapshot(LiveEvidence(raw), " a\n", data)


def test_obsolete_raw_snapshot_api_cannot_relabel_recorded_results_as_live(tmp_path):
    module = science()
    source = module.RecordedEvidence(write_bundle(tmp_path, bundle()), MODEL_DIGEST)
    ops, _ = operations()
    data = ops.prepare(candidates())
    snapshot: Any = module.evidence_snapshot  # Exercise deliberately invalid runtime calls.
    with pytest.raises(TypeError):
        snapshot(source.lookup("CCO"), " a\n", data, "live")
    with pytest.raises(TypeError):
        snapshot(source, " a\n", data, origin="live")
    with pytest.raises(TypeError):
        snapshot(source, " a\n", data, fixture_hash="e" * 64)
    with pytest.raises(TypeError, match="source"):
        module.evidence_snapshot(source.lookup("CCO"), " a\n", data)


def bundle():
    raw = result()
    body = '{"activities":[]}\n'
    return {"schema_version": 1, "record_limit": 10, "dataset_digest": "c" * 64,
            "model_digest": MODEL_DIGEST, "retrieved_at": RETRIEVED_AT,
            "entries": [{"canonical_query": "CCO", "candidate_id": "original-id",
                         "result": raw, "result_sha256": science().digest(raw),
                         "raw_responses": [{"resource": "activity.json", "body_utf8": body,
                                            "sha256": sha256(body.encode()).hexdigest()}]}]}


def write_bundle(tmp_path, content):
    path = tmp_path / "recorded.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_recorded_source_matches_exact_structures_and_preserves_retrieval(tmp_path):
    module = science()
    path = write_bundle(tmp_path, bundle())
    source = module.RecordedEvidence(path, MODEL_DIGEST)
    assert source.fixture_hash == sha256(path.read_bytes()).hexdigest()
    assert source.source_dataset_digest == "c" * 64
    assert source.origin == "recorded"
    observation = source.lookup("OCC")
    assert observation == result()
    observation["records"].clear()
    assert source.lookup("CCO") == result()
    assert source.lookup("[13CH3]CO")["status"] == "not_recorded"
    ops, _ = operations()
    new_data = ops.prepare([{"id": "renamed-id", "smiles": "OCC"}])
    snapshot = module.evidence_snapshot(source, "renamed-id", new_data)
    assert snapshot["candidate_id"] == "renamed-id"
    assert snapshot["dataset_digest"] != source.source_dataset_digest
    assert snapshot["retrieved_at"] == RETRIEVED_AT and snapshot["playback_at"]
    assert snapshot["origin"] == "recorded" and snapshot["fixture_hash"] == source.fixture_hash
    other_data = ops.prepare([{"id": "unrecorded", "smiles": "[13CH3]CO"}])
    missing = module.evidence_snapshot(source, "unrecorded", other_data)
    assert missing["result"]["status"] == "not_recorded" and missing["origin"] == "recorded"
    assert missing["retrieved_at"] is None and missing["playback_at"]
    with pytest.raises(ValueError, match="limit"):
        source.lookup("CCO", 3)
    with pytest.raises(ValueError, match="model"):
        module.RecordedEvidence(path, "d" * 64)


@pytest.mark.parametrize("tamper", ["result", "raw", "query", "duplicate", "nonfinite", "limit"])
def test_recorded_source_rejects_corrupt_manifest_before_lookup(tmp_path, tamper):
    content = bundle()
    entry = content["entries"][0]
    if tamper == "result":
        entry["result"]["records"].clear()
    elif tamper == "raw":
        entry["raw_responses"][0]["body_utf8"] += " "
    elif tamper == "query":
        entry["canonical_query"] = "CCN"
    elif tamper == "duplicate":
        content["entries"].append(deepcopy(entry))
    elif tamper == "nonfinite":
        entry["result"]["records"][0]["standard_value"] = float("nan")
    else:
        content["record_limit"] = 3
    with pytest.raises(ValueError):
        science().RecordedEvidence(write_bundle(tmp_path, content), MODEL_DIGEST)


def test_recorded_bundle_rejects_more_than_ten_records_despite_matching_hash(tmp_path):
    content = bundle()
    entry = content["entries"][0]
    entry["result"]["records"] *= 11
    entry["result_sha256"] = science().digest(entry["result"])
    with pytest.raises(ValueError, match="limit"):
        science().RecordedEvidence(write_bundle(tmp_path, content), MODEL_DIGEST)


def test_recorded_bundle_rejects_raw_activity_page_limit_mismatch(tmp_path):
    content = bundle()
    raw = content["entries"][0]["raw_responses"][0]
    raw["body_utf8"] = '{"activities":[],"page_meta":{"limit":50}}'
    raw["sha256"] = sha256(raw["body_utf8"].encode()).hexdigest()
    with pytest.raises(ValueError, match="limit"):
        science().RecordedEvidence(write_bundle(tmp_path, content), MODEL_DIGEST)


def test_recorded_malformed_activity_response_survives_as_original_failure(tmp_path):
    content = bundle()
    entry = content["entries"][0]
    entry["result"]["records"] = []
    entry["result"]["error"] = "ChEMBL lookup failed: JSONDecodeError"
    entry["result_sha256"] = science().digest(entry["result"])
    raw = entry["raw_responses"][0]
    raw["body_utf8"] = '{"activities":['
    raw["sha256"] = sha256(raw["body_utf8"].encode()).hexdigest()
    source = science().RecordedEvidence(write_bundle(tmp_path, content), MODEL_DIGEST)
    assert source.lookup("CCO") == entry["result"]


def test_checked_in_capture_keeps_demo_activities_separate_and_source_overlap():
    model_digest = json.loads(Path("examples/workbench-evidence.json").read_text())["model_digest"]
    module = science()
    source = module.RecordedEvidence("examples/workbench-evidence.json", model_digest)
    dataset = module.ScientificOperations(screen=Screen(), model_digest=model_digest).prepare(
        load_candidates("examples/cyp3a4-candidates.csv"))
    rows = {row["id"]: row for row in dataset["rows"]}
    snapshots = {identifier: module.evidence_snapshot(source, identifier, dataset)
                 for identifier in ("candidate-08", "candidate-09")}
    results = {identifier: snapshot["result"] for identifier, snapshot in snapshots.items()}
    for identifier, activity in (("candidate-08", 6032270), ("candidate-09", 6048907)):
        evidence = results[identifier]
        assert evidence["status"] == "ok" and not evidence["truncated"]
        assert snapshots[identifier]["origin"] == "recorded"
        assert snapshots[identifier]["retrieved_at"] == evidence["retrieved_at"]
        record = next(row for row in evidence["records"] if row["activity_id"] == activity)
        assert record["veith_overlap"] is True
        assert evidence["retrieved_at"] and evidence["query_smiles"] == rows[identifier]["smiles"]
    assert results["candidate-08"]["records"] != results["candidate-09"]["records"]


def analysis_operations(tmp_path, monkeypatch):
    artifacts = tmp_path / "selected-artifacts"
    (artifacts / "models").mkdir(parents=True)
    for enzyme in ENZYMES:
        (artifacts / "models" / f"{enzyme}.joblib").write_bytes(enzyme.encode())
    (artifacts / "manifest.csv").write_text("fixture manifest")
    monkeypatch.setattr(science(), "CYP3A4Screen", lambda path: Screen())
    return science().ScientificOperations(artifacts), artifacts


@pytest.mark.parametrize("name", ["predict_cyp", "find_analogs", "show_attribution", "show_neighbourhood"])
def test_analysis_dispatch_binds_selected_artifacts_and_provenance(tmp_path, monkeypatch, name):
    from src.prediction import cyp as cyp_tools

    ops, artifacts = analysis_operations(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(cyp_tools, name, lambda **args: calls.append(args) or {"query_smiles": "CCO"})
    result = ops.analyze(name, {"smiles": "CCO"})
    assert calls == [{"smiles": "CCO", "artifacts": artifacts.resolve()}]
    assert result["query_smiles"] == "CCO"
    assert result["artifact_hashes"]["models/CYP3A4.joblib"] == ops.model_digest
    if name in {"predict_cyp", "show_attribution"}:
        assert all(f"models/{enzyme}.joblib" in result["artifact_hashes"] for enzyme in ENZYMES)
    else:
        assert result["artifact_hashes"]["manifest.csv"] == sha256(b"fixture manifest").hexdigest()


@pytest.mark.parametrize("file,name", [("models/CYP3A4.joblib", "predict_cyp"),
                                      ("models/CYP1A2.joblib", "predict_cyp"),
                                      ("manifest.csv", "find_analogs")])
def test_analysis_refuses_changed_artifacts_before_tool_execution(tmp_path, monkeypatch, file, name):
    from src.prediction import cyp as cyp_tools

    ops, artifacts = analysis_operations(tmp_path, monkeypatch)
    (artifacts / file).write_bytes(b"replaced")
    monkeypatch.setattr(cyp_tools, name, lambda **args: pytest.fail("changed artifacts must not be used"))
    with pytest.raises(ValueError, match="artifact.*changed"):
        ops.analyze(name, {"smiles": "CCO"})


def test_analysis_refuses_artifacts_changed_during_tool_execution(tmp_path, monkeypatch):
    from src.prediction import cyp as cyp_tools

    ops, artifacts = analysis_operations(tmp_path, monkeypatch)

    def prediction(**args):
        (artifacts / "models/CYP3A4.joblib").write_bytes(b"replaced")
        return {"query_smiles": "CCO"}

    monkeypatch.setattr(cyp_tools, "predict_cyp", prediction)
    with pytest.raises(ValueError, match="artifact.*changed"):
        ops.analyze("predict_cyp", {"smiles": "CCO"})


def test_custom_screen_analysis_fails_closed_unless_explicitly_injected(monkeypatch):
    from src.prediction import cyp as cyp_tools

    monkeypatch.setattr(cyp_tools, "predict_cyp", lambda **args: pytest.fail("no global artifact fallback"))
    ops, _ = operations()
    with pytest.raises(ValueError, match="artifact"):
        ops.analyze("predict_cyp", {"smiles": "CCO"})
    calls = []
    custom = science().ScientificOperations(
        screen=Screen(), model_digest=MODEL_DIGEST,
        analysis=lambda name, args: calls.append((name, args)) or {"fixture": True})
    assert custom.analyze("predict_cyp", {"smiles": "CCO"}) == {"fixture": True}
    assert calls == [("predict_cyp", {"smiles": "CCO"})]
    with pytest.raises(ValueError, match="analysis"):
        custom.analyze("_suite", {})
    with pytest.raises(ValueError, match="artifact"):
        custom.analyze("predict_cyp", {"smiles": "CCO", "artifacts": "other"})


def test_missing_analysis_artifact_never_falls_back_to_default_directory(tmp_path, monkeypatch):
    from src.prediction import cyp as cyp_tools

    ops, artifacts = analysis_operations(tmp_path, monkeypatch)
    (artifacts / "models/CYP2C19.joblib").unlink()
    monkeypatch.setattr(cyp_tools, "predict_cyp", lambda **args: pytest.fail("missing model must fail first"))
    with pytest.raises(ValueError, match="artifact is unavailable"):
        ops.analyze("predict_cyp", {"smiles": "CCO"})


def test_custom_screen_cannot_claim_another_model_identity(tmp_path, monkeypatch):
    _, artifacts = analysis_operations(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="differs from the screening model"):
        science().ScientificOperations(artifacts, screen=Screen(), model_digest=MODEL_DIGEST)
