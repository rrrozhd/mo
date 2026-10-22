"""The release check must detect missing/corrupted scientific inputs without loading models."""

import json

import pytest

from scripts import verify_release


@pytest.fixture
def release(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return verify_release.sha256(path)

    raw_hash = write("data/cyp3a4_veith.tab", b"raw observations\n")
    manifest_hash = write("artifacts/manifest.csv", b"training identities\n")
    artifacts = {
        "manifest.csv": manifest_hash,
        "models/CYP3A4.joblib": write("artifacts/models/CYP3A4.joblib", b"never unpickle this fixture"),
        "test.csv": write("artifacts/test.csv", b"held-out identities\n"),
        "manifest.json": write("artifacts/manifest.json", {
            "manifest_sha256": manifest_hash,
            "dataset_sha256": {"cyp3a4_veith.tab": raw_hash}}),
    }
    cohort_hash = write("results/full-llm/cohort.csv", b"frozen cohort\n")
    run_hash = write("results/full-llm/cohort_manifest.json", {
        "artifact_hashes": artifacts, "cohort_sha256": cohort_hash})
    predictions_hash = write("results/full-llm/predictions.jsonl.gz", b"frozen predictions")
    summary_hash = write("results/screening-budget/summary.csv", b"summary\n")
    selected_hash = write("results/screening-budget/selected-compounds.csv", b"selected\n")
    write("results/screening-budget/provenance.json", {
        "sources": {"results/full-llm/cohort_manifest.json": run_hash,
                    "results/full-llm/cohort.csv": cohort_hash,
                    "results/full-llm/predictions.jsonl.gz": predictions_hash},
        "outputs": {"summary.csv": summary_hash, "selected-compounds.csv": selected_hash}})
    return tmp_path


def test_matching_release_passes_without_changing_files(release, capsys):
    before = {path.relative_to(release): path.read_bytes() for path in release.rglob("*") if path.is_file()}
    assert verify_release.main(["--root", str(release)]) == 0
    assert "10/10 frozen files match" in capsys.readouterr().out
    assert before == {path.relative_to(release): path.read_bytes() for path in release.rglob("*") if path.is_file()}


@pytest.mark.parametrize("name", ["artifacts/models/CYP3A4.joblib", "data/cyp3a4_veith.tab",
                                 "results/full-llm/cohort.csv", "results/full-llm/predictions.jsonl.gz"])
def test_corrupted_release_fails(release, capsys, name):
    (release / name).write_bytes(b"changed")
    assert verify_release.main(["--root", str(release)]) == 1
    assert f"MISMATCH: {name}" in capsys.readouterr().out


def test_missing_model_is_reported(release, capsys):
    (release / "artifacts/models/CYP3A4.joblib").unlink()
    assert verify_release.main(["--root", str(release)]) == 1
    assert "MISSING: artifacts/models/CYP3A4.joblib" in capsys.readouterr().out


def test_missing_manifest_fails_instead_of_skipping_verification(release, capsys):
    (release / "artifacts/manifest.json").unlink()
    assert verify_release.main(["--root", str(release)]) == 1
    assert "Cannot verify release" in capsys.readouterr().out
