import io
import json
from email.message import Message
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from src.sources import chembl as evidence


def page(key, rows, total=None, next_page=None):
    return {key: rows, "page_meta": {"total_count": len(rows) if total is None else total,
                                    "next": next_page}}


def compound(smiles="CCO", identifier="CHEMBL1"):
    return {"molecule_chembl_id": identifier, "pref_name": "Example",
            "molecule_structures": {"canonical_smiles": smiles}}


def activity(identifier=1, assay="CHEMBL10", **overrides):
    row = {"activity_id": identifier, "molecule_chembl_id": "CHEMBL1",
           "canonical_smiles": "CCO", "assay_chembl_id": assay,
           "assay_description": "Inhibition using probe A", "assay_type": "A",
           "target_chembl_id": "CHEMBL340", "target_organism": "Homo sapiens",
           "document_chembl_id": "CHEMBL100", "src_id": 1,
           "standard_type": "IC50", "standard_relation": ">",
           "standard_value": "10000.0", "standard_units": "nM",
           "type": "IC50", "relation": ">", "value": "10", "units": "uM",
           "data_validity_comment": None, "potential_duplicate": 0}
    return row | overrides


def assay(identifier="CHEMBL10", **overrides):
    return {"assay_chembl_id": identifier, "target_chembl_id": "CHEMBL340",
            "src_id": 1, "src_assay_id": "example", "description": "Probe A, 5 min",
            "assay_type": "A", "confidence_score": 9} | overrides


@pytest.fixture
def transport(monkeypatch):
    queue, calls, clock = [], [], [0.0]

    def open_request(request, timeout):
        calls.append((request, timeout, clock[0]))
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, io.BytesIO):
            return result
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(evidence, "urlopen", open_request)
    monkeypatch.setattr(evidence.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(evidence.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return queue, calls


def test_matching_identity_preserves_separate_measurements_and_provenance(transport):
    queue, calls = transport
    queue.extend([page("molecules", [compound("OCC")]),
                  page("activities", [activity(), activity(2, "CHEMBL11", standard_type="Ki",
                                                          standard_value="35", standard_relation="=")]),
                  page("assays", [assay(), assay("CHEMBL11", src_id=7, src_assay_id="1851_3")])])
    result = evidence.EvidenceClient().lookup("OCC")
    assert result["status"] == "ok"
    assert result["query_smiles"] == "CCO"
    assert result["matched_compound_ids"] == ["CHEMBL1"]
    assert result["activity_count"] == 2 and not result["truncated"]
    first, second = result["records"]
    assert (first["standard_type"], first["standard_relation"], first["standard_value"],
            first["standard_units"]) == ("IC50", ">", "10000.0", "nM")
    assert first["value"] == "10" and first["units"] == "uM"
    assert second["standard_type"] == "Ki" and second["standard_value"] == "35"
    assert first["veith_overlap"] is False and second["veith_overlap"] is True
    assert first["assay_description"] == "Probe A, 5 min"
    assert first["document_url"].endswith("/document/CHEMBL100")
    assert first["target_url"].endswith("/target/CHEMBL340")
    assert len(calls) == 3
    params = parse_qs(urlparse(calls[1][0].full_url).query)
    assert params["target_chembl_id"] == ["CHEMBL340"]
    assert params["target_organism"] == ["Homo sapiens"]
    assert params["molecule_chembl_id__in"] == ["CHEMBL1"]
    assert all(b[2] - a[2] >= 0.25 for a, b in zip(calls, calls[1:]))


def test_deadline_caps_each_http_timeout_and_stops_before_later_request(transport):
    queue, calls = transport
    queue.extend([page("molecules", [compound()]), page("activities", [activity()])])
    result = evidence.EvidenceClient().lookup("CCO", deadline=.4,
                                               clock=evidence.time.monotonic)
    assert len(calls) == 2
    assert calls[0][1] == pytest.approx(.4)
    assert calls[1][1] == pytest.approx(.15)
    assert result["status"] == "error" and "deadline" in result["error"]
    assert result["records"][0]["standard_relation"] == ">"


def test_expired_lookup_sends_no_http_request(transport):
    _, calls = transport
    result = evidence.EvidenceClient().lookup("CCO", deadline=0,
                                               clock=evidence.time.monotonic)
    assert not calls and result["status"] == "error"
    assert "deadline" in result["error"]


@pytest.mark.parametrize("query,returned", [
    ("N[C@H](C)C(=O)O", "N[C@@H](C)C(=O)O"),
    ("N[C@H](C)C(=O)O", "NC(C)C(=O)O"),
    ("CCO.[Na+]", "CCO"), ("[13CH3]CO", "CCO"),
    ("CC(=O)[O-]", "CC(=O)O"), ("C/C=C/C", "C/C=C\\C"),
])
def test_stereo_isotope_salt_and_charge_changes_never_attach_activity(query, returned, transport):
    queue, calls = transport
    queue.append(page("molecules", [compound(returned)]))
    result = evidence.EvidenceClient().lookup(query)
    assert result["status"] == "identity_mismatch"
    assert result["records"] == [] and result["matched_compound_ids"] == []
    assert result["identity_rejected_candidates"][0]["compound_id"] == "CHEMBL1"
    assert result["activity_count"] is None
    assert len(calls) == 1


def test_identity_mismatches_are_retained_when_an_exact_match_exists(transport):
    queue, calls = transport
    queue.extend([page("molecules", [compound("CCN", "CHEMBL2"), compound()]),
                  page("activities", [])])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "not_found"
    assert result["matched_compound_ids"] == ["CHEMBL1"]
    assert result["activity_count"] == 0
    assert result["identity_rejected_candidates"][0]["compound_id"] == "CHEMBL2"
    assert "inactive" not in result
    assert len(calls) == 2


def test_not_found_is_limited_to_search_and_candidate_pagination_is_explicit(transport):
    queue, calls = transport
    queue.append(page("molecules", []))
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "not_found" and result["activity_count"] is None
    queue.append(page("molecules", [compound("CCN")], total=25, next_page="/more"))
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "identity_mismatch"
    assert result["candidate_count"] == 25 and result["candidates_truncated"]
    assert result["truncated"]
    assert len(calls) == 2


def test_activity_pagination_never_fetches_more_than_requested(transport):
    queue, calls = transport
    queue.extend([page("molecules", [compound()]),
                  page("activities", [activity()], total=263, next_page="/more"),
                  page("assays", [assay()])])
    result = evidence.EvidenceClient().lookup("CCO", limit=1)
    assert result["activity_count"] == 263 and result["truncated"]
    assert len(result["records"]) == 1 and len(calls) == 3


def test_metadata_failure_preserves_measurement_but_overlap_remains_unknown(transport):
    queue, _ = transport
    queue.extend([page("molecules", [compound()]), page("activities", [activity()]),
                  URLError("service down")])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "error" and "service down" in result["error"]
    assert result["records"][0]["veith_overlap"] is None
    assert result["records"][0]["standard_relation"] == ">"
    assert result["assay_metadata_complete"] is False


@pytest.mark.parametrize("failure_stage", ["first_request", "assay_metadata"])
def test_incomplete_http_read_returns_error_and_preserves_prior_measurements(failure_stage, transport):
    queue, calls = transport
    read_sizes = []

    class TruncatedResponse(io.BytesIO):
        def read(self, size: int | None = -1):
            read_sizes.append(size)
            raise IncompleteRead(b'{"partial":', 20)

    response = TruncatedResponse()
    if failure_stage == "assay_metadata":
        queue.extend([page("molecules", [compound()]), page("activities", [activity()])])
    queue.append(response)
    result = evidence.EvidenceClient().lookup("CCO")

    assert result["status"] == "error" and "IncompleteRead" in result["error"]
    assert result["assay_metadata_complete"] is False
    assert response.closed
    assert read_sizes == [evidence.MAX_RESPONSE_BYTES + 1]
    if failure_stage == "first_request":
        assert len(calls) == 1
        assert result["records"] == [] and result["activity_count"] is None
    else:
        assert len(calls) == 3 and result["activity_count"] == 1
        assert result["matched_compound_ids"] == ["CHEMBL1"]
        assert len(result["records"]) == 1
        assert result["records"][0]["standard_value"] == "10000.0"
        assert result["records"][0]["standard_relation"] == ">"
        assert result["records"][0]["veith_overlap"] is None


def test_text_only_assay_outcomes_survive_without_invented_numeric_values(transport):
    queue, _ = transport
    queue.extend([page("molecules", [compound()]),
                  page("activities", [activity(standard_relation=None, standard_value=None,
                      standard_units=None, standard_text_value="Not active", relation=None,
                      value=None, units=None, text_value="Not Active",
                      activity_comment="Inhibition < 50% @ 10 uM")]),
                  page("assays", [assay(src_id=None, src_assay_id=None)])])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "ok"
    record = result["records"][0]
    assert record["standard_value"] is None and record["value"] is None
    assert record["standard_text_value"] == "Not active"
    assert record["activity_comment"] == "Inhibition < 50% @ 10 uM"
    assert record["veith_overlap"] is None


def test_missing_assay_metadata_does_not_claim_no_training_overlap(transport):
    queue, _ = transport
    queue.extend([page("molecules", [compound()]), page("activities", [activity()]),
                  page("assays", [])])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "error" and not result["assay_metadata_complete"]
    assert result["records"][0]["veith_overlap"] is None


@pytest.mark.parametrize("overrides", [
    {"target_organism": "Mus musculus"}, {"target_chembl_id": "CHEMBL999"},
    {"molecule_chembl_id": "CHEMBL2"}, {"canonical_smiles": "CCN"},
])
def test_wrong_activity_target_or_identity_fails_closed(overrides, transport):
    queue, calls = transport
    queue.extend([page("molecules", [compound()]), page("activities", [activity(**overrides)])])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "error" and not result["records"]
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [URLError("offline"), TimeoutError("timeout"),
    HTTPError("https://example.test", 429, "rate limited", Message(), None),
    {}, {"molecules": [], "page_meta": {"total_count": "bad"}}, []])
def test_transport_and_malformed_responses_are_errors_not_missing_evidence(failure, transport):
    queue, calls = transport
    queue.append(failure)
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "error" and result["error"]
    assert len(calls) == 1


@pytest.mark.parametrize("smiles,limit", [("bad", 10), ("CCO", 0), ("CCO", 51), ("CCO", True)])
def test_bad_inputs_fail_before_network(smiles, limit, transport):
    _, calls = transport
    with pytest.raises(ValueError):
        evidence.EvidenceClient().lookup(smiles, limit=limit)
    assert calls == []


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_nonfinite_remote_numbers_are_lookup_errors(token, transport):
    queue, calls = transport
    payload = page("activities", [activity(standard_value="numeric-token")])
    raw = json.dumps(payload).replace('"numeric-token"', token).encode()
    queue.extend([page("molecules", [compound()]), io.BytesIO(raw), page("assays", [assay()])])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "error" and "non-finite" in result["error"]
    assert result["records"] == [] and len(calls) == 2
    json.dumps(result, allow_nan=False)


def test_nonfinite_metadata_preserves_previously_retrieved_measurements(transport):
    queue, _ = transport
    payload = page("assays", [assay(confidence_score="numeric-token")])
    queue.extend([page("molecules", [compound()]), page("activities", [activity()]),
                  io.BytesIO(json.dumps(payload).replace('"numeric-token"', '1e309').encode())])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "error" and "non-finite" in result["error"]
    assert result["records"][0]["standard_value"] == "10000.0"
    assert result["records"][0]["veith_overlap"] is None
    json.dumps(result, allow_nan=False)


def test_finite_remote_number_keeps_its_value(transport):
    queue, _ = transport
    queue.extend([page("molecules", [compound()]),
                  page("activities", [activity(standard_value=1e3)]),
                  page("assays", [assay()])])
    result = evidence.EvidenceClient().lookup("CCO")
    assert result["status"] == "ok"
    assert result["records"][0]["standard_value"] == 1000.0
    json.dumps(result, allow_nan=False)
