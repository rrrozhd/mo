import csv
import io
import json
from datetime import datetime, timezone
from email.utils import format_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs

import pytest

from src.sources import pubchem


def response(smiles="CCO", title="Ethanol", cid=702):
    return io.BytesIO(json.dumps({"PropertyTable": {"Properties": [
        {"CID": cid, "SMILES": smiles, "Title": title}
    ]}}).encode())


def http_error(code, retry_after=None, body=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    body = body if body is not None else {"Fault": {"Code": "PUGREST.NotFound"}}
    return HTTPError(pubchem.SOURCE, code, "mock response", headers,
                     io.BytesIO(json.dumps(body).encode()))


@pytest.fixture
def transport(monkeypatch):
    clock = [0.0]
    calls, queue = [], []

    def sleep(seconds):
        clock[0] += seconds

    def open_request(request, timeout):
        calls.append((request, timeout, clock[0]))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(pubchem.time, "sleep", sleep)
    monkeypatch.setattr(pubchem.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pubchem, "urlopen", open_request)
    return queue, calls


def test_form_encoding_preserves_charge_and_double_bond_stereo(transport):
    queue, calls = transport
    smiles = "[NH3+]C/C=C\\C"
    queue.append(response(smiles, "record title", 123))
    record = pubchem.PubChemClient().lookup("m1", smiles)
    request, timeout, _ = calls[0]
    assert request.get_method() == "POST"
    assert request.full_url == pubchem.SOURCE
    assert request.get_header("Content-type") == "application/x-www-form-urlencoded"
    assert parse_qs(request.data.decode()) == {"smiles": [smiles]}
    assert b"%2B" in request.data and b"%2F" in request.data and b"%5C" in request.data
    assert timeout == 20
    assert record["status"] == "ok"
    assert record["name"] == "record title"
    assert record["cid"] == 123
    assert record["returned_smiles"] == smiles
    assert record["source"] == pubchem.SOURCE


def test_equivalent_smiles_is_accepted_and_requests_are_rate_limited(transport):
    queue, calls = transport
    queue.extend(response("OCC") for _ in range(5))
    client = pubchem.PubChemClient()
    for index in range(5):
        assert client.lookup(str(index), "CCO")["status"] == "ok"
    assert all(b[2] - a[2] >= 0.25 for a, b in zip(calls, calls[1:]))


def test_interrupted_response_becomes_an_error_record(transport):
    from http.client import IncompleteRead

    class Interrupted(io.BytesIO):
        def read(self, *args):
            raise IncompleteRead(b'{"PropertyTable":', 30)

    queue, calls = transport
    queue.append(Interrupted())
    result = pubchem.PubChemClient().lookup("m1", "CCO")
    assert result["status"] == "error" and "IncompleteRead" in (result["error"] or "")
    assert len(calls) == 1


def test_name_cache_does_not_clobber_a_cohort_named_like_its_scratch(tmp_path):
    from types import SimpleNamespace

    cohort = tmp_path / "names.jsonl.tmp"
    original = "molecule_id,smiles\nm1,CCO\n"
    cohort.write_text(original)
    client = SimpleNamespace(lookup=lambda identifier, smiles: {
        "molecule_id": identifier, "smiles": smiles, "status": "error", "error": "fixture"})
    output = tmp_path / "names.jsonl"
    pubchem.lookup_cohort(cohort, output, client)
    assert cohort.read_text() == original
    assert json.loads(output.read_text())["molecule_id"] == "m1"


@pytest.mark.parametrize("query,returned", [
    ("N[C@H](C)C(=O)O", "N[C@@H](C)C(=O)O"),
    ("[13CH3]CO", "CCO"),
    ("CC(=O)[O-]", "CC(=O)O"),
    ("CCO", "CCN"),
])
def test_identity_changes_are_unavailable(query, returned, transport):
    queue, _ = transport
    queue.append(response(returned, "wrong molecule title"))
    result = pubchem.PubChemClient().lookup("m1", query)
    assert result["status"] == "unavailable"
    assert result["name"] is None
    assert result["returned_smiles"] == returned
    assert result["cid"] == 702
    assert "differs" in result["error"]


def test_multiple_records_are_rejected_without_picking_a_name(transport):
    queue, _ = transport
    queue.append(io.BytesIO(json.dumps({"PropertyTable": {"Properties": [
        {"CID": 2, "SMILES": "CCO", "Title": "a"},
        {"CID": 1, "SMILES": "CCO", "Title": "b"},
    ]}}).encode()))
    result = pubchem.PubChemClient().lookup("m1", "CCO")
    assert result["status"] == "unavailable"
    assert result["name"] is None and result["cid"] is None
    assert "ambiguous" in result["error"]


@pytest.mark.parametrize("title", [None, "", "   "])
def test_matching_record_without_title_is_unavailable(title, transport):
    queue, _ = transport
    queue.append(response(title=title))
    result = pubchem.PubChemClient().lookup("m1", "CCO")
    assert result["status"] == "unavailable"
    assert "no Title" in result["error"]


def test_429_and_503_retry_with_retry_after_and_bounded_attempts(transport):
    queue, calls = transport
    queue.extend([http_error(429, "2"), http_error(503, "3"), response()])
    result = pubchem.PubChemClient().lookup("m1", "CCO")
    assert result["status"] == "ok"
    assert [call[2] for call in calls] == [0, 2, 5]
    queue.extend(http_error(503) for _ in range(3))
    result = pubchem.PubChemClient().lookup("m2", "CCN")
    assert result["status"] == "error"
    assert "HTTP 503" in result["error"]
    assert len(calls) == 6


def test_long_retry_after_defers_to_next_run_without_retrying_early(transport):
    queue, calls = transport
    queue.append(http_error(429, "3600"))
    result = pubchem.PubChemClient().lookup("m1", "CCO")
    assert result["status"] == "error"
    assert len(calls) == 1


def test_retry_after_http_date_and_malformed_header(monkeypatch):
    monkeypatch.setattr(pubchem.time, "time", lambda: 1000)
    header = format_datetime(datetime.fromtimestamp(1007, timezone.utc), usegmt=True)
    assert pubchem._retry_delay(header, 0) == 7
    assert pubchem._retry_delay("not a date", 1) == 2


@pytest.mark.parametrize("failure,status", [
    (http_error(404), "unavailable"),
    (http_error(404, body={"Fault": []}), "error"),
    (http_error(400), "error"),
    (http_error(500), "error"),
    (URLError("temporary network failure"), "error"),
    (TimeoutError("timeout"), "error"),
])
def test_only_genuine_pubchem_not_found_is_permanent(failure, status, transport):
    queue, calls = transport
    queue.append(failure)
    result = pubchem.PubChemClient().lookup("m1", "CCO")
    assert result["status"] == status
    assert result["error"]
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [
    b"<html>maintenance</html>", b"{}", b"[]",
    json.dumps({"PropertyTable": {"Properties": []}}).encode(),
    json.dumps({"PropertyTable": {"Properties": [{"CID": 1, "Title": "ethanol",
                                                 "SMILES": "invalid"}]}}).encode(),
    json.dumps({"PropertyTable": {"Properties": [{"CID": 1, "Title": "ethanol",
                                                 "ConnectivitySMILES": "CCO"}]}}).encode(),
])
def test_malformed_or_non_stereochemical_response_is_error(payload, transport):
    queue, _ = transport
    queue.append(io.BytesIO(payload))
    assert pubchem.PubChemClient().lookup("m1", "CCO")["status"] == "error"


def write_cohort(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "smiles"])
        writer.writerows(rows)


def test_resume_reuses_success_and_not_found_but_retries_errors(tmp_path, transport):
    queue, calls = transport
    cohort, output = tmp_path / "cohort.csv", tmp_path / "names.jsonl"
    write_cohort(cohort, [("m1", "CCO"), ("m2", "CCN"), ("m3", "c1ccccc1")])
    queue.extend([response(), URLError("offline"), http_error(404)])
    records = pubchem.lookup_cohort(cohort, output)
    assert [row["status"] for row in records] == ["ok", "error", "unavailable"]
    queue.append(response("CCN", "Ethylamine", 6341))
    resumed = pubchem.lookup_cohort(cohort, output)
    assert [row["status"] for row in resumed] == ["ok", "ok", "unavailable"]
    assert len(calls) == 4
    assert calls[-1][0].data == b"smiles=CCN"
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(saved) == 3
    assert {row["molecule_id"] for row in saved} == {"m1", "m2", "m3"}


def test_resume_invalidates_cache_when_molecule_changes(tmp_path, transport):
    queue, calls = transport
    cohort, output = tmp_path / "cohort.csv", tmp_path / "names.jsonl"
    write_cohort(cohort, [("m1", "CCO")])
    queue.append(response())
    pubchem.lookup_cohort(cohort, output)
    write_cohort(cohort, [("m1", "CCN")])
    queue.append(response("CCN", "Ethylamine", 6341))
    assert pubchem.lookup_cohort(cohort, output)[0]["name"] == "Ethylamine"
    assert len(calls) == 2
    assert len(output.read_text().splitlines()) == 1


def test_duplicate_cohort_ids_fail_before_network_or_output_changes(tmp_path, transport):
    _, calls = transport
    cohort, output = tmp_path / "cohort.csv", tmp_path / "names.jsonl"
    write_cohort(cohort, [("m1", "CCO"), ("m1", "CCN")])
    with pytest.raises(ValueError, match="unique"):
        pubchem.lookup_cohort(cohort, output)
    assert not calls and not output.exists()
