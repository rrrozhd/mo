import io
import json
from datetime import datetime
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest

from src.sources import chembl


def document(identifier="CHEMBL1148425", **overrides):
    return dict(document_chembl_id=identifier, title="An assay-linked paper", authors="Smith A.",
                year=2004, journal="Example Journal", doi="10.1016/j.example.2004.01.002",
                pubmed_id=14698201, abstract="Measured enzyme inhibition.", doc_type="PUBLICATION") | overrides


def page(rows, **metadata):
    return {"documents": rows, "page_meta": {"total_count": len(rows), "next": None} | metadata}


@pytest.fixture
def transport(monkeypatch):
    queue, calls = [], []

    def open_request(request, timeout):
        calls.append((request, timeout))
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(chembl, "urlopen", open_request)
    return queue, calls


def test_one_document_request_preserves_publication_and_dataset_distinction(transport):
    queue, calls = transport
    queue.append(page([document("CHEMBL1201862", title="PubChem BioAssay data set",
                              authors=None, year=None, journal=None, doi=None, pubmed_id=None,
                              abstract="", doc_type="DATASET"), document()]))
    result = chembl.EvidenceClient().documents(["CHEMBL1201862", "CHEMBL1148425"])
    assert result["status"] == "ok" and result["error"] is None
    assert result["missing_ids"] == [] and result["source"] == chembl.API
    assert datetime.fromisoformat(result["retrieved_at"]).utcoffset() is not None
    assert result["document_ids"] == ["CHEMBL1148425", "CHEMBL1201862"]
    paper, dataset = result["papers"]
    assert paper == dict(document_id="CHEMBL1148425", title="An assay-linked paper", authors="Smith A.",
        year=2004, journal="Example Journal", doi="10.1016/j.example.2004.01.002", pubmed_id=14698201,
        abstract="Measured enzyme inhibition.", document_type="PUBLICATION",
        document_url=f"{chembl.EXPLORER}/document/CHEMBL1148425",
        doi_url="https://doi.org/10.1016/j.example.2004.01.002",
        pubmed_url="https://pubmed.ncbi.nlm.nih.gov/14698201/")
    assert dataset["document_type"] == "DATASET" and dataset["title"] == "PubChem BioAssay data set"
    assert dataset["abstract"] is None and dataset["authors"] is None
    assert dataset["doi_url"] is None and dataset["pubmed_url"] is None
    assert len(calls) == 1
    params = parse_qs(urlparse(calls[0][0].full_url).query)
    assert params["document_chembl_id__in"] == ["CHEMBL1148425,CHEMBL1201862"]
    assert params["limit"] == ["2"] and params["order_by"] == ["document_chembl_id"]


@pytest.mark.parametrize("rows,status,missing", [
    ([document()], "ok", ["CHEMBL1201862"]),
    ([], "not_found", ["CHEMBL1148425", "CHEMBL1201862"]),
])
def test_missing_documents_are_explicit_without_discarding_found_metadata(transport, rows, status, missing):
    queue, _ = transport
    queue.append(page(rows))
    result = chembl.EvidenceClient().documents(["CHEMBL1148425", "CHEMBL1201862"])
    assert result["status"] == status and result["missing_ids"] == missing
    assert len(result["papers"]) == len(rows)


@pytest.mark.parametrize("identifiers", [
    [], ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4"], ["CHEMBL1", "CHEMBL1"],
    ["bad"], "CHEMBL1", [1], ["CHEMBL1/../../molecule"],
])
def test_invalid_document_ids_fail_before_http(transport, identifiers):
    _, calls = transport
    with pytest.raises(ValueError):
        chembl.EvidenceClient().documents(identifiers)
    assert not calls


@pytest.mark.parametrize("payload", [
    URLError("offline"), {}, page([document("CHEMBL999")]),
    page([document(), document()]), page([document()], total_count=2, next="/more"),
    page([document(abstract=[])]), page([document(year=True)]), page([document(pubmed_id=-1)]),
    page([document(doc_type={})]),
])
def test_unrelated_duplicate_and_malformed_metadata_fail_closed(transport, payload):
    queue, _ = transport
    queue.append(payload)
    result = chembl.EvidenceClient().documents(["CHEMBL1148425", "CHEMBL1201862"])
    assert result["status"] == "error" and result["error"]
    assert result["papers"] == [] and result["missing_ids"] is None


def test_unrecognized_doi_remains_metadata_without_becoming_link(transport):
    queue, _ = transport
    queue.append(page([document(doi="https://untrusted.test/paper")]))
    result = chembl.EvidenceClient().documents(["CHEMBL1148425"])
    assert result["status"] == "ok"
    assert result["papers"][0]["doi"] == "https://untrusted.test/paper"
    assert result["papers"][0]["doi_url"] is None


def test_deadline_validation_and_enforcement(transport):
    _, calls = transport
    with pytest.raises(ValueError):
        chembl.EvidenceClient().documents(["CHEMBL1"], deadline=float("nan"))
    result = chembl.EvidenceClient().documents(["CHEMBL1"], deadline=0, clock=lambda: 1)
    assert result["status"] == "error" and "deadline" in result["error"] and not calls
