import json

import pytest

from src.screening.cli import load_candidates


class FakeScreen:
    def score(self, candidates):
        return [dict(row, canonical_smiles=row["smiles"], score=score,
                     confidence="uncalibrated", validation_percentile=score,
                     nearest_training_similarity=0.7, similarity_flag=False)
                for row, score in zip(candidates, [0.9, 0.6, 0.2])]


def test_load_candidates_ignores_labels_and_rejects_ambiguous_inputs(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("id,smiles,CYP3A4\na,OCC,1\nb,c1ccccc1,0\n")
    assert load_candidates(path) == [{"id": "a", "smiles": "CCO"},
                                     {"id": "b", "smiles": "c1ccccc1"}]
    for body in ("id,smiles\na,CCO\na,CCN\n", "id,smiles\na,CCO\nb,OCC\n",
                 "id,smiles\na,not-a-molecule\n", "id,smiles\n"):
        path.write_text(body)
        with pytest.raises(ValueError):
            load_candidates(path)


def test_cli_shortlist_and_output_path_protection(tmp_path, monkeypatch, capsys):
    from src.screening import cli as screening
    path = tmp_path / "input.csv"
    original = "id,smiles\na,CCO\nb,c1ccccc1\nc,CCN\n"
    path.write_text(original)
    output = tmp_path / "result.json"
    monkeypatch.setattr(screening, "CYP3A4Screen", lambda _: FakeScreen())
    code = screening.main(["shortlist", "--candidates", str(path), "--budget", "2",
                           "--exclude", "a", "--output", str(output)])
    assert code == 0
    assert [r["id"] for r in json.loads(output.read_text())["selected"]] == ["b", "c"]
    with pytest.raises(SystemExit):
        screening.main(["shortlist", "--candidates", str(path), "--budget", "2",
                        "--output", str(path)])
    assert path.read_text() == original


@pytest.mark.parametrize("target", ["candidate", "model"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_shortlist_protects_inputs_before_scoring(tmp_path, monkeypatch, target, alias):
    from src.screening import cli as screening

    candidates = tmp_path / "input.csv"
    candidates.write_text("id,smiles\na,CCO\n")
    model = tmp_path / "models" / "CYP3A4.joblib"
    model.parent.mkdir()
    model.write_bytes(b"frozen model")
    source = candidates if target == "candidate" else model
    original = source.read_bytes()
    output = source
    if alias != "direct":
        output = tmp_path / "output.json"
        (output.symlink_to if alias == "symlink" else output.hardlink_to)(source)
    monkeypatch.setattr(screening, "CYP3A4Screen", lambda _: pytest.fail("scoring must not start"))
    with pytest.raises(SystemExit):
        screening.main(["shortlist", "--candidates", str(candidates), "--budget", "1",
                        "--artifacts", str(tmp_path), "--output", str(output)])
    assert source.read_bytes() == original


def test_cli_rejects_the_removed_agent_command(capsys):
    from src.screening import cli as screening
    with pytest.raises(SystemExit):
        screening.main(["agent", "--candidates", "x.csv", "--model", "m", "--prompt", "p"])


def test_cli_interrupted_http_returns_error_json(tmp_path, monkeypatch, capsys):
    from http.client import IncompleteRead
    from src.sources import chembl as evidence
    from src.screening import cli as screening

    class InterruptedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            raise IncompleteRead(b'{"molecules":', 20)

    monkeypatch.setattr(evidence, "urlopen", lambda *a, **kw: InterruptedResponse())
    code = screening.main(["evidence", "--smiles", "CCO"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1 and result["status"] == "error"
    assert result["records"] == [] and "IncompleteRead" in result["error"]


def test_cli_plan_keeps_partial_evidence_and_writes_readable_report(tmp_path, monkeypatch, capsys):
    from src.screening import cli as screening

    path = tmp_path / "input.csv"
    path.write_text("id,smiles\na,CCO\nb,c1ccccc1\nc,CCN\n")
    output, report = tmp_path / "batch.json", tmp_path / "batch.md"
    calls = []

    class Client:
        def lookup(self, smiles, limit):
            calls.append((smiles, limit))
            return {"status": "error" if smiles == "CCO" else "not_found",
                    "query_smiles": smiles, "error": "metadata timeout" if smiles == "CCO" else None,
                    "records": [{"standard_type": "IC50", "standard_relation": ">",
                                 "standard_value": "10000", "standard_units": "nM"}]
                               if smiles == "CCO" else [],
                    "truncated": False, "assay_metadata_complete": False}

    monkeypatch.setattr(screening, "CYP3A4Screen", lambda _: FakeScreen())
    monkeypatch.setattr(screening, "EvidenceClient", Client)
    code = screening.main(["plan", "--candidates", str(path), "--budget", "2",
                           "--include", "c", "--evidence-limit", "3",
                           "--output", str(output), "--report", str(report)])
    result = json.loads(output.read_text())
    assert code == 2 and result["status"] == "partial" and result["complete"]
    assert [r["id"] for r in result["selected"]] == ["a", "c"]
    assert calls == [("CCO", 3), ("CCN", 3)]
    assert result["evidence"]["a"]["records"][0]["standard_relation"] == ">"
    text = report.read_text()
    assert "IC50" in text and "10000" in text and "metadata timeout" in text
    assert "required inclusion" in text and "uncalibrated" in text.lower()


@pytest.mark.parametrize("collision", ["candidate", "outputs", "symlink", "hardlink", "model"])
def test_plan_rejects_output_collisions_before_scoring(tmp_path, monkeypatch, capsys, collision):
    from src.screening import cli as screening

    path = tmp_path / "input.csv"
    original = "id,smiles\na,CCO\n"
    path.write_text(original)
    output, report = tmp_path / "batch.json", tmp_path / "batch.md"
    artifacts = tmp_path / "artifacts"
    model = artifacts / "models" / "CYP3A4.joblib"
    model.parent.mkdir(parents=True)
    model.write_text("model must survive")
    if collision == "candidate":
        report = path
    elif collision == "outputs":
        report = output
    elif collision == "symlink":
        report.symlink_to(path)
    elif collision == "hardlink":
        report.hardlink_to(path)
    else:
        report = model
    monkeypatch.setattr(screening, "CYP3A4Screen", lambda _: pytest.fail("scoring must not start"))
    with pytest.raises(SystemExit):
        screening.main(["plan", "--candidates", str(path), "--budget", "1",
                        "--artifacts", str(artifacts), "--output", str(output), "--report", str(report)])
    assert "distinct paths" in capsys.readouterr().err
    assert path.read_text() == original
    assert model.read_text() == "model must survive"


def test_plan_rejects_bad_evidence_limit_before_scoring(tmp_path, monkeypatch, capsys):
    from src.screening import cli as screening

    monkeypatch.setattr(screening, "CYP3A4Screen", lambda _: pytest.fail("scoring must not start"))
    with pytest.raises(SystemExit):
        screening.main(["plan", "--candidates", "unused", "--budget", "1", "--evidence-limit", "0"])
    assert "--evidence-limit" in capsys.readouterr().err


@pytest.mark.parametrize("destination", ["--output", "--report"])
def test_plan_write_failure_preserves_retrieved_result_on_stdout(tmp_path, monkeypatch, capsys, destination):
    from pathlib import Path
    from src.screening import cli as screening

    path = tmp_path / "input.csv"
    path.write_text("id,smiles\na,CCO\n")
    target = tmp_path / "result"
    calls = []

    class Client:
        def lookup(self, smiles, limit):
            calls.append(smiles)
            return {"status": "not_found", "records": [], "truncated": False, "query_smiles": smiles}

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(screening, "CYP3A4Screen", lambda _: FakeScreen())
    monkeypatch.setattr(screening, "EvidenceClient", Client)
    monkeypatch.setattr(Path, "write_text", fail_write)
    code = screening.main(["plan", "--candidates", str(path), "--budget", "1", destination, str(target)])
    output = capsys.readouterr()
    assert code == 1 and "disk full" in output.err
    result = json.loads(output.out)
    assert result["status"] == "ok" and result["selected_count"] == 1
    assert result["evidence"]["a"]["status"] == "not_found" and calls == ["CCO"]


def test_plan_nonfinite_remote_measurement_keeps_batch_and_continues(tmp_path, monkeypatch, capsys):
    import io
    from urllib.parse import parse_qs, urlparse
    from src.sources import chembl as evidence
    from src.screening import cli as screening

    path = tmp_path / "input.csv"
    path.write_text("id,smiles\na,CCO\nb,c1ccccc1\n")
    def respond(request, **kwargs):
        url = urlparse(request.full_url)
        if url.path.endswith("molecule.json"):
            key = "molecules"
            rows = [{"molecule_chembl_id": "CHEMBL1", "molecule_structures": {"canonical_smiles": "CCO"}}]
            if parse_qs(url.query)["molecule_structures__canonical_smiles__connectivity"] != ["CCO"]:
                rows = []
        elif url.path.endswith("activity.json"):
            key = "activities"
            rows = [{"activity_id": 1, "molecule_chembl_id": "CHEMBL1", "canonical_smiles": "CCO",
                     "assay_chembl_id": "CHEMBL10", "target_chembl_id": "CHEMBL340",
                     "target_organism": "Homo sapiens", "standard_value": "numeric-token"}]
        else:
            key = "assays"
            rows = [{"assay_chembl_id": "CHEMBL10", "target_chembl_id": "CHEMBL340", "src_id": 1}]
        payload = {key: rows, "page_meta": {"total_count": len(rows), "next": None}}
        return io.BytesIO(json.dumps(payload).replace('"numeric-token"', '1e309').encode())
    monkeypatch.setattr(screening, "CYP3A4Screen", lambda _: FakeScreen())
    monkeypatch.setattr(evidence, "urlopen", respond)
    monkeypatch.setattr(evidence.time, "sleep", lambda _: None)
    code = screening.main(["plan", "--candidates", str(path), "--budget", "2"])
    result = json.loads(capsys.readouterr().out)
    assert code == 2 and result["complete"]
    assert [r["id"] for r in result["selected"]] == ["a", "b"]
    assert "non-finite" in result["evidence"]["a"]["error"]
    assert result["evidence"]["b"]["status"] == "not_found"
