import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import openai
import pandas as pd
import pytest

from src import benchmark, models
from src.chemistry import ENZYMES


class FakeSuite:
    """Stands in for the frozen bundles; synthetic scores are never scientific results."""
    descriptor_names = []

    def __init__(self, *args):
        pass

    def predict(self, vector):
        assert vector.shape == (1, 2048)
        return {"scores": {enzyme: 0.5 for enzyme in ENZYMES}}


def artifacts_fixture(tmp_path):
    artifacts = tmp_path / "artifacts"
    (artifacts / "models").mkdir(parents=True)
    rows = [{"molecule_id": hashlib.sha256(s.encode()).hexdigest(), "smiles": s,
             "scaffold": "", "split": "test", **{e: i for e in ENZYMES}}
            for i, s in enumerate(["CCO", "CCN"])]
    pd.DataFrame(rows).to_csv(artifacts / "test.csv", index=False)
    pd.DataFrame(rows).to_csv(artifacts / "manifest.csv", index=False)
    (artifacts / "manifest.json").write_text("{}")
    for enzyme in ENZYMES:
        (artifacts / "models" / f"{enzyme}.joblib").write_bytes(b"fixture")
    return artifacts


def test_prepare_freezes_cohort_and_trained_models(tmp_path):
    artifacts = artifacts_fixture(tmp_path)
    run_dir = tmp_path / "pilot"
    cohort = benchmark.prepare(artifacts, run_dir, limit=1)
    assert len(cohort) == 1
    pd.testing.assert_frame_equal(cohort, benchmark.verify_run(artifacts, run_dir))
    with pytest.raises(FileExistsError):
        benchmark.prepare(artifacts, run_dir)
    (artifacts / "models" / "CYP1A2.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        benchmark.verify_run(artifacts, run_dir)


def test_classifier_records_cover_the_cohort_once_and_resume(tmp_path, monkeypatch):
    artifacts = artifacts_fixture(tmp_path)
    run_dir = tmp_path / "pilot"
    cohort = benchmark.prepare(artifacts, run_dir)
    monkeypatch.setattr(models, "ModelSuite", FakeSuite)
    benchmark.classifier_records(artifacts, run_dir)
    records = benchmark.read_records(run_dir / "predictions.jsonl")
    assert sorted(r["molecule_id"] for r in records) == sorted(cohort["molecule_id"])
    assert all(r["arm"] == "classifier" and r["status"] == "ok" for r in records)
    benchmark.classifier_records(artifacts, run_dir)  # Resume must not duplicate records.
    assert len(benchmark.read_records(run_dir / "predictions.jsonl")) == len(records)


def test_classifier_records_refuse_changed_artifacts(tmp_path, monkeypatch):
    artifacts = artifacts_fixture(tmp_path)
    run_dir = tmp_path / "pilot"
    benchmark.prepare(artifacts, run_dir)
    monkeypatch.setattr(models, "ModelSuite", FakeSuite)
    (artifacts / "models" / "CYP3A4.joblib").write_bytes(b"retrained")
    with pytest.raises(ValueError, match="artifacts changed"):
        benchmark.classifier_records(artifacts, run_dir)


def test_run_lock_blocks_second_writer(tmp_path):
    with benchmark.run_lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            with benchmark.run_lock(tmp_path):
                pass


class FakeCompletions:
    """Stands in for OpenRouter; synthetic scores are never scientific results."""

    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        self.client.requests.append(kwargs)
        self.client.prompts.append(kwargs["messages"][-1]["content"])
        body = json.dumps({enzyme: 0.25 for enzyme in ENZYMES})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])


class FakeClient:
    def __init__(self, **kwargs):
        self.requests = []
        self.prompts = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


def llm_fixture(tmp_path, monkeypatch, named=1):
    artifacts = artifacts_fixture(tmp_path)
    run_dir = tmp_path / "pilot"
    cohort = benchmark.prepare(artifacts, run_dir)
    records = [{"molecule_id": molecule_id, "name": f"Name {index}" if index < named else None}
               for index, molecule_id in enumerate(cohort["molecule_id"])]
    (run_dir / "names.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return artifacts, run_dir, cohort


def test_name_arm_marks_unresolved_names_unavailable_and_resumes(tmp_path, monkeypatch):
    artifacts, run_dir, cohort = llm_fixture(tmp_path, monkeypatch, named=1)
    calls = benchmark.llm_records(artifacts, run_dir, "name")
    records = {r["molecule_id"]: r for r in benchmark.read_records(run_dir / "predictions.jsonl")
               if r["arm"] == "name"}
    assert calls == 1 and len(records) == len(cohort)
    statuses = sorted(r["status"] for r in records.values())
    assert statuses == ["ok", "unavailable"]
    assert benchmark.llm_records(artifacts, run_dir, "name") == 0  # Resume must not recharge.


def test_smiles_arm_sends_structures_and_stops_at_the_call_cap(tmp_path, monkeypatch):
    artifacts, run_dir, cohort = llm_fixture(tmp_path, monkeypatch, named=0)
    with pytest.raises(ValueError, match="call cap"):
        benchmark.llm_records(artifacts, run_dir, "smiles", max_calls=1)
    records = [r for r in benchmark.read_records(run_dir / "predictions.jsonl")
               if r["arm"] == "smiles"]
    assert len(records) == 1 and records[0]["status"] == "ok"
    assert set(records[0]["scores"]) == set(ENZYMES)
    assert len(cohort) > 1  # The cap, not the cohort, ended the run.


@pytest.mark.parametrize("value", [None, [], True, "0.5", {e: True for e in ENZYMES},
                                  {e: None for e in ENZYMES}, {e: "0.5" for e in ENZYMES},
                                  {e: float("nan") for e in ENZYMES}])
def test_malformed_scores_are_error_records(value):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(value)))]))))
    record = benchmark.score_one(client, "fixture/model", "smiles", "m1", "CCO")
    assert record["status"] == "error"
    assert record["model"] == "fixture/model"


@pytest.mark.parametrize("options", [{"max_calls": -1}, {"max_calls": True},
                                     {"workers": 0}, {"workers": 1.5}, {"arm": "typo"}])
def test_invalid_llm_options_fail_before_any_side_effects(tmp_path, monkeypatch, options):
    monkeypatch.setattr(benchmark, "verify_run", lambda *a: pytest.fail("must validate first"))
    with pytest.raises(ValueError):
        benchmark.llm_records(tmp_path, tmp_path, **{"arm": "smiles", **options})


def test_resume_refuses_a_different_llm_model(tmp_path, monkeypatch):
    artifacts, run_dir, _ = llm_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="call cap"):
        benchmark.llm_records(artifacts, run_dir, "smiles", model="fixture/old", max_calls=1)
    output = run_dir / "predictions.jsonl"
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="model"):
        benchmark.llm_records(artifacts, run_dir, "smiles", model="fixture/new")
    assert output.read_bytes() == previous


def test_call_cap_disables_retries_and_empty_resume_needs_no_client(tmp_path, monkeypatch):
    artifacts, run_dir, _ = llm_fixture(tmp_path, monkeypatch)
    settings = []

    def client(**kwargs):
        settings.append(kwargs)
        return FakeClient()

    monkeypatch.setattr(openai, "OpenAI", client)
    assert benchmark.llm_records(artifacts, run_dir, "smiles") == 2
    assert settings[0]["max_retries"] == 0
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("completed run needs no client"))
    assert benchmark.llm_records(artifacts, run_dir, "smiles") == 0


@pytest.mark.parametrize("change", ["prompt", "schema", "model", "name", "missing_name"])
def test_resume_rejects_changed_conditions_without_side_effects(tmp_path, monkeypatch, change):
    artifacts, run_dir, _ = llm_fixture(tmp_path, monkeypatch, named=2)
    with pytest.raises(ValueError, match="call cap"):
        benchmark.llm_records(artifacts, run_dir, "name", max_calls=1)
    model = benchmark.LLM_MODEL
    if change == "prompt":
        monkeypatch.setattr(benchmark, "LLM_PROMPT", benchmark.LLM_PROMPT + " Changed instruction.")
    elif change == "schema":
        schema = deepcopy(benchmark.SCORE_SCHEMA)
        schema["schema"]["properties"][ENZYMES[0]]["maximum"] = 0.9
        monkeypatch.setattr(benchmark, "SCORE_SCHEMA", schema)
    elif change == "model":
        model = "fixture/other"
    else:
        names = benchmark.read_records(run_dir / "names.jsonl")
        names[-1]["name"] = "Different compound" if change == "name" else None
        (run_dir / "names.jsonl").write_text("".join(json.dumps(row) + "\n" for row in names))
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(benchmark, "openrouter_key", lambda: pytest.fail("must reject before key lookup"))
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("must reject before client creation"))
    with pytest.raises(ValueError, match="changed"):
        benchmark.llm_records(artifacts, run_dir, "name", model=model)
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


@pytest.mark.parametrize("status", ["ok", "error", "unavailable"])
def test_legacy_arm_without_provenance_cannot_resume_or_migrate(tmp_path, monkeypatch, status):
    artifacts, run_dir, cohort = llm_fixture(tmp_path, monkeypatch, named=2)
    benchmark.append_record(run_dir / "predictions.jsonl", {
        "molecule_id": cohort.iloc[0]["molecule_id"], "arm": "name",
        "status": status, "model": benchmark.LLM_MODEL,
    })
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("legacy run must not make calls"))
    with pytest.raises(ValueError, match="provenance.*new run"):
        benchmark.llm_records(artifacts, run_dir, "name")
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


def test_matching_resume_reuses_frozen_conditions_and_keeps_prior_records(tmp_path, monkeypatch):
    artifacts, run_dir, cohort = llm_fixture(tmp_path, monkeypatch, named=2)
    clients = []

    def client(**kwargs):
        result = FakeClient(**kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(openai, "OpenAI", client)
    with pytest.raises(ValueError, match="call cap"):
        benchmark.llm_records(artifacts, run_dir, "name", max_calls=1)
    manifest_path = run_dir / "llm_name_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["prompt"] == benchmark.LLM_PROMPT
    assert manifest["score_schema"] == benchmark.SCORE_SCHEMA
    assert manifest["model"] == benchmark.LLM_MODEL
    assert manifest["subjects"] == {molecule_id: f"Name {index}"
                                    for index, molecule_id in enumerate(cohort["molecule_id"])}
    previous = (run_dir / "predictions.jsonl").read_bytes()
    assert benchmark.llm_records(artifacts, run_dir, "name") == 1
    assert (run_dir / "predictions.jsonl").read_bytes().startswith(previous)
    assert manifest_path.read_bytes() == manifest_bytes
    assert [prompt for client in clients for prompt in client.prompts] == [
        "Compound name: Name 0", "Compound name: Name 1"]


def test_smiles_resume_ignores_changes_to_unused_name_mapping(tmp_path, monkeypatch):
    artifacts, run_dir, _ = llm_fixture(tmp_path, monkeypatch, named=0)
    with pytest.raises(ValueError, match="call cap"):
        benchmark.llm_records(artifacts, run_dir, "smiles", max_calls=1)
    (run_dir / "names.jsonl").write_text("")
    assert benchmark.llm_records(artifacts, run_dir, "smiles") == 1


def test_calls_use_the_snapshot_even_if_live_settings_change_after_admission(tmp_path, monkeypatch):
    artifacts, run_dir, _ = llm_fixture(tmp_path, monkeypatch, named=2)
    expected_prompt, expected_schema = benchmark.LLM_PROMPT, deepcopy(benchmark.SCORE_SCHEMA)
    client = FakeClient()

    def change_live_settings(**kwargs):
        monkeypatch.setattr(benchmark, "LLM_PROMPT", "Changed after admission")
        monkeypatch.setattr(benchmark, "SCORE_SCHEMA", {})
        (run_dir / "names.jsonl").write_text("")
        return client

    monkeypatch.setattr(openai, "OpenAI", change_live_settings)
    assert benchmark.llm_records(artifacts, run_dir, "name") == 2
    assert client.prompts == ["Compound name: Name 0", "Compound name: Name 1"]
    assert all(request["messages"][0]["content"] == expected_prompt for request in client.requests)
    assert all(request["response_format"]["json_schema"] == expected_schema for request in client.requests)


@pytest.mark.parametrize("replacement", ['{"version":', '{}'])
def test_invalid_provenance_refuses_resume_without_repairing_results(tmp_path, monkeypatch, replacement):
    artifacts, run_dir, _ = llm_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="call cap"):
        benchmark.llm_records(artifacts, run_dir, "smiles", max_calls=1)
    (run_dir / "llm_smiles_manifest.json").write_text(replacement)
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("invalid provenance must not make calls"))
    with pytest.raises(ValueError, match="new run"):
        benchmark.llm_records(artifacts, run_dir, "smiles")
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


def test_provenance_initialization_does_not_replace_a_competing_snapshot(tmp_path, monkeypatch):
    artifacts, run_dir, _ = llm_fixture(tmp_path, monkeypatch)
    competing = '{"model": "different-run-conditions"}\n'

    def publish_competing_snapshot(source, destination):
        destination.write_text(competing)
        raise FileExistsError

    monkeypatch.setattr(benchmark.os, "link", publish_competing_snapshot)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: pytest.fail("competing provenance must not make calls"))
    with pytest.raises(ValueError, match="changed"):
        benchmark.llm_records(artifacts, run_dir, "smiles")
    assert (run_dir / "llm_smiles_manifest.json").read_text() == competing
    assert not (run_dir / "predictions.jsonl").exists()
    assert not list(run_dir.glob(".llm_*.tmp"))
