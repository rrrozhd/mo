import io
import json
from datetime import datetime
from email.message import Message
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse

import pytest
from rdkit import DataStructs

from src import chembl
from src.chemistry import canonical_smiles, fingerprints


def candidate(smiles, identifier, similarity, name=None):
    return {"molecule_chembl_id": identifier, "pref_name": name,
            "molecule_structures": {"canonical_smiles": smiles}, "similarity": similarity}


def page(rows, total=None, next_page=None):
    return {"molecules": rows, "page_meta": {
        "total_count": len(rows) if total is None else total, "next": next_page}}


@pytest.fixture
def transport(monkeypatch):
    queue, calls = [], []

    def open_request(request, timeout):
        calls.append((request, timeout))
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, io.BytesIO):
            return result
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(chembl, "urlopen", open_request)
    return queue, calls


def test_search_excludes_query_deduplicates_sorts_and_keeps_metrics_distinct(transport):
    queue, calls = transport
    queue.append(page([
        candidate("CCN", "CHEMBL20", "83.5"),
        candidate("CCC", "CHEMBL4", "90", "Propane"),
        candidate("NCC", "CHEMBL2", "83.5"),
        candidate("OCC", "CHEMBL1", "100", "Query"),
    ]))
    result = chembl.EvidenceClient().search_analogs("OCC")
    assert result["status"] == "ok" and result["error"] is None
    assert result["query_smiles"] == "CCO" and result["source"] == chembl.API
    assert result["threshold_percent"] == 70 and result["candidate_count"] == 4
    assert not result["truncated"]
    assert datetime.fromisoformat(result["retrieved_at"]).utcoffset() is not None
    assert [row["compound_id"] for row in result["analogs"]] == ["CHEMBL4", "CHEMBL2"]
    first, second = result["analogs"]
    assert first["name"] == "Propane" and second["name"] is None
    assert first["chembl_similarity_percent"] == 90
    expected = DataStructs.TanimotoSimilarity(*fingerprints(["CCO", "CCC"]))
    assert first["morgan_similarity"] == expected
    assert first["compound_url"].endswith("/compound/CHEMBL4")
    assert set(first) == {"compound_id", "name", "smiles", "chembl_similarity_percent",
                          "morgan_similarity", "compound_url"}
    assert len(calls) == 1
    params = parse_qs(urlparse(calls[0][0].full_url).query)
    assert params["limit"] == [str(chembl.MAX_CANDIDATES)]
    assert params["order_by"] == ["-similarity"]


def test_query_path_is_encoded_and_stereo_is_not_erased(transport):
    queue, calls = transport
    query, stereo = "C/C=C/C", "C/C=C\\C"
    queue.append(page([candidate(query, "CHEMBL1", 100), candidate(stereo, "CHEMBL2", 100)]))
    result = chembl.EvidenceClient().search_analogs(query, threshold=100)
    assert result["analogs"][0]["smiles"] == canonical_smiles(stereo)
    assert f"/similarity/{quote(canonical_smiles(query), safe='')}/100.json" in calls[0][0].full_url


@pytest.mark.parametrize("rows,total,next_page,max_results,expected_count,truncated", [
    ([], 0, None, 3, 0, False),
    ([candidate("CCO", "CHEMBL1", 100)], 1, None, 3, 0, False),
    ([candidate("CCC", "CHEMBL2", 90)], 9, "/more", 3, 1, True),
    ([candidate("CCC", "CHEMBL2", 90), candidate("CCN", "CHEMBL3", 80)], 2, None, 1, 1, True),
])
def test_not_found_and_truncation_are_scoped_to_one_page(
        transport, rows, total, next_page, max_results, expected_count, truncated):
    queue, calls = transport
    queue.append(page(rows, total, next_page))
    result = chembl.EvidenceClient().search_analogs("CCO", max_results=max_results)
    assert result["status"] == ("ok" if expected_count else "not_found")
    assert len(result["analogs"]) == expected_count and result["truncated"] is truncated
    assert result["candidate_count"] == total and len(calls) == 1


@pytest.mark.parametrize("kwargs", [
    {"max_results": 0}, {"max_results": 4}, {"max_results": True}, {"max_results": 1.0},
    {"threshold": 39}, {"threshold": 101}, {"threshold": False}, {"threshold": 70.0},
    {"deadline": float("nan")}, {"deadline": True},
])
def test_invalid_limits_fail_before_network(transport, kwargs):
    _, calls = transport
    with pytest.raises(ValueError):
        chembl.EvidenceClient().search_analogs("CCO", **kwargs)
    assert not calls


def test_invalid_structure_fails_before_network(transport):
    _, calls = transport
    with pytest.raises(ValueError):
        chembl.EvidenceClient().search_analogs("not a molecule")
    assert not calls


@pytest.mark.parametrize("failure", [
    URLError("offline"), HTTPError("https://example.test", 503, "Unavailable", Message(), None),
    {}, page([candidate("CCN", "bad-id", 80)]),
    page([candidate("CCN", "CHEMBL1", "NaN")]),
    page([candidate("CCN", "CHEMBL1", True)]),
    page([candidate("CCN", "CHEMBL1", 101)]),
    page([candidate("bad smiles", "CHEMBL1", 80)]),
    page([candidate("CCN", "CHEMBL1", 80)], total=0),
    page([candidate("CCN", f"CHEMBL{i}", 80) for i in range(1, 22)]),
    io.BytesIO(b"x" * (chembl.MAX_RESPONSE_BYTES + 1)),
])
def test_remote_failures_never_become_no_results_or_partial_unvalidated_rows(transport, failure):
    queue, calls = transport
    queue.append(failure)
    result = chembl.EvidenceClient().search_analogs("CCO")
    assert result["status"] == "error" and result["error"]
    assert result["analogs"] == [] and len(calls) == 1


def test_deadline_prevents_request_and_caps_http_timeout(transport):
    queue, calls = transport
    result = chembl.EvidenceClient().search_analogs("CCO", deadline=10, clock=lambda: 10)
    assert result["status"] == "error" and "deadline" in result["error"] and not calls
    queue.append(page([]))
    chembl.EvidenceClient().search_analogs("CCO", deadline=12, clock=lambda: 10)
    assert calls[0][1] == 2
