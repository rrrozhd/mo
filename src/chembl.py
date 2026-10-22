"""Bounded ChEMBL analogue search and exact-structure human CYP3A4 evidence.

Connectivity search supplies candidates only. RDKit canonical isomeric SMILES
must match without stripping salts, charge, isotopes, or stereochemistry. Values
remain separate assay observations; this module does not infer inactivity,
clinical risk, substrate status, induction, or independent model validation.

API: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
"""

import json
import re
import time
from datetime import datetime, timezone
from http.client import HTTPException
from math import isfinite
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from rdkit import DataStructs

from .chemistry import canonical_smiles, fingerprints

API = "https://www.ebi.ac.uk/chembl/api/data"
EXPLORER = "https://www.ebi.ac.uk/chembl/explore"
TARGET = "CHEMBL340"
MAX_CANDIDATES = 20
MAX_RECORDS = 50
MAX_RESPONSE_BYTES = 2_000_000
REQUEST_INTERVAL = 0.25


def _finite_float(token):
    value = float(token)
    if not isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"CHEMBL[0-9]+", value):
        raise ValueError("invalid ChEMBL identifier")
    return value


def _link(kind, identifier):
    return None if identifier is None else f"{EXPLORER}/{kind}/{_identifier(identifier)}"


def _page(payload, key, limit):
    rows, meta = payload[key], payload["page_meta"]
    total = meta["total_count"]
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or type(total) is not int or total < len(rows) or len(rows) > limit
            or (total > 0 and not rows)):
        raise ValueError(f"invalid {key} page")
    return rows, total, total > len(rows) or bool(meta.get("next"))


class EvidenceClient:
    """Sequential client: at most three requests per lookup, with no retries.

One page each of candidates, activities, and assay metadata is requested.
Analogue searches request one similarity page without activity measurements.
Pagination is reported rather than followed. A client can be reused to retain
the request spacing across molecules; no results are cached across lookups.
"""

    def __init__(self, timeout=10):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 30:
            raise ValueError("timeout must be greater than zero and at most 30 seconds")
        self.timeout = timeout
        self.last_request = None

    def _get(self, resource, params, *, deadline=None, clock=None):
        clock = clock or time.monotonic
        remaining = float("inf") if deadline is None else deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Evidence operation deadline reached")
        if self.last_request is not None:
            delay = max(0.0, REQUEST_INTERVAL - (clock() - self.last_request))
            if delay >= remaining:
                raise TimeoutError("Evidence operation deadline reached before request spacing")
            time.sleep(delay)
        remaining = self.timeout if deadline is None else min(self.timeout, deadline - clock())
        if remaining <= 0:
            raise TimeoutError("Evidence operation deadline reached")
        self.last_request = clock()
        request = Request(f"{API}/{resource}.json?{urlencode(params)}",
                          headers={"Accept": "application/json"})
        with urlopen(request, timeout=remaining) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds the evidence lookup size limit")
        payload = json.loads(data, parse_float=_finite_float, parse_constant=_finite_float)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload

    def search_analogs(self, smiles: str, max_results: int = 3, threshold: int = 70,
                       *, deadline=None, clock=None) -> dict:
        """Search one ChEMBL similarity page without inferring measured activity.

The service count includes the query and duplicate structures. Returned analogs
exclude the exact canonical isomeric query, retain salts and stereochemistry,
and are ordered by ChEMBL similarity then ID. ChEMBL's similarity percentage is
distinct from our locally calculated Morgan radius-2, 2048-bit Tanimoto score.
Pagination and the output cap are reported rather than followed.
"""
        query = canonical_smiles(smiles)
        if type(max_results) is not int or not 1 <= max_results <= 3:
            raise ValueError("max_results must be an integer from 1 to 3")
        if type(threshold) is not int or not 40 <= threshold <= 100:
            raise ValueError("threshold must be an integer from 40 to 100")
        if deadline is not None and (type(deadline) not in (int, float) or not isfinite(deadline)):
            raise ValueError("deadline must be a finite monotonic time")
        result: dict[str, Any] = dict(status="error", query_smiles=query, source=API,
                      retrieved_at=datetime.now(timezone.utc).isoformat(),
                      threshold_percent=threshold, candidate_count=None, truncated=False,
                      analogs=[], error=None)
        try:
            payload = self._get(f"similarity/{quote(query, safe='')}/{threshold}", {
                "limit": MAX_CANDIDATES, "order_by": "-similarity",
                "only": "molecule_chembl_id,molecule_structures,pref_name,similarity"},
                deadline=deadline, clock=clock)
            candidates, total, truncated = _page(payload, "molecules", MAX_CANDIDATES)
            result.update(candidate_count=total, truncated=truncated)
            query_fp = fingerprints([query])[0]
            analogs = []
            for row in candidates:
                identifier = _identifier(row["molecule_chembl_id"])
                structure = canonical_smiles(row["molecule_structures"]["canonical_smiles"])
                similarity = float(row["similarity"])
                if isinstance(row["similarity"], bool) or not isfinite(similarity) or not 0 <= similarity <= 100:
                    raise ValueError("invalid ChEMBL similarity percentage")
                name = row.get("pref_name")
                if name is not None and not isinstance(name, str):
                    raise ValueError("invalid compound name")
                if structure != query:
                    analogs.append(dict(compound_id=identifier, name=name, smiles=structure,
                        chembl_similarity_percent=similarity,
                        morgan_similarity=DataStructs.TanimotoSimilarity(
                            query_fp, fingerprints([structure])[0]),
                        compound_url=_link("compound", identifier)))
            analogs.sort(key=lambda row: (-row["chembl_similarity_percent"], row["compound_id"]))
            unique = {}
            for row in analogs:
                unique.setdefault(row["smiles"], row)
            result.update(analogs=list(unique.values())[:max_results],
                          truncated=truncated or len(unique) > max_results,
                          status="ok" if unique else "not_found")
        except HTTPError as exc:
            result["error"] = f"ChEMBL HTTP {exc.code}: {exc.reason}"
            exc.close()
        except (HTTPException, URLError, OSError, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            result["error"] = f"ChEMBL analogue search failed: {type(exc).__name__}: {exc}"
        return result

    def documents(self, document_ids: list[str], *, deadline=None, clock=None) -> dict:
        """Retrieve up to three exact source documents in one request.

Documents may be publications or datasets. Abstracts and bibliographic metadata
describe their source; they do not establish independent assay validation.
missing_ids is populated only after a successful, complete API response.
"""
        if not isinstance(document_ids, list) or not 1 <= len(document_ids) <= 3:
            raise ValueError("document_ids must contain one to three unique ChEMBL IDs")
        identifiers = sorted(_identifier(value) for value in document_ids)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("document_ids must be unique")
        if deadline is not None and (type(deadline) not in (int, float) or not isfinite(deadline)):
            raise ValueError("deadline must be a finite monotonic time")
        result: dict[str, Any] = dict(status="error", source=API,
                      retrieved_at=datetime.now(timezone.utc).isoformat(),
                      document_ids=identifiers, papers=[], missing_ids=None, error=None)
        try:
            payload = self._get("document", {
                "document_chembl_id__in": ",".join(identifiers), "limit": len(identifiers),
                "order_by": "document_chembl_id"}, deadline=deadline, clock=clock)
            rows, _, truncated = _page(payload, "documents", len(identifiers))
            returned = [_identifier(row["document_chembl_id"]) for row in rows]
            if truncated or len(set(returned)) != len(returned) or not set(returned) <= set(identifiers):
                raise ValueError("incomplete, duplicate, or unexpected document metadata")
            papers = []
            for row in sorted(rows, key=lambda row: row["document_chembl_id"]):
                paper = {key: row.get(key) for key in ("title", "authors", "journal", "doi", "abstract")}
                paper["document_type"] = row.get("doc_type")
                if any(value is not None and not isinstance(value, str) for value in paper.values()):
                    raise ValueError("invalid document text metadata")
                paper = {key: value or None for key, value in paper.items()}
                for key in ("year", "pubmed_id"):
                    value = row.get(key)
                    if value is not None and (type(value) is not int or value <= 0):
                        raise ValueError(f"invalid document {key}")
                    paper[key] = value
                identifier, doi, pubmed = row["document_chembl_id"], paper["doi"], paper["pubmed_id"]
                paper.update(document_id=identifier, document_url=_link("document", identifier),
                    doi_url=f"https://doi.org/{quote(doi, safe='/')}"
                        if doi and re.fullmatch(r"10\.[0-9]{4,9}/\S+", doi) else None,
                    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed}/" if pubmed else None)
                papers.append(paper)
            result.update(status="ok" if papers else "not_found", papers=papers,
                          missing_ids=sorted(set(identifiers) - set(returned)))
        except HTTPError as exc:
            result["error"] = f"ChEMBL HTTP {exc.code}: {exc.reason}"
            exc.close()
        except (HTTPException, URLError, OSError, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            result["error"] = f"ChEMBL document lookup failed: {type(exc).__name__}: {exc}"
        return result

    def lookup(self, smiles: str, limit: int = 10, *, deadline=None, clock=None) -> dict:
        """Return exact-match measurements and coverage, never a safety verdict.

Invalid input raises ValueError before HTTP. Remote failures use status=error,
with any already retrieved records retained. activity_count is the service's
total for matched IDs, not the returned count; None means it was not established.
veith_overlap flags the known PubChem 1851_3 mirror only. False is not evidence
that a measurement is otherwise independent of the training dataset.
"""
        query = canonical_smiles(smiles)
        if type(limit) is not int or not 1 <= limit <= MAX_RECORDS:
            raise ValueError(f"limit must be an integer from 1 to {MAX_RECORDS}")
        if deadline is not None and (type(deadline) not in (int, float) or not isfinite(deadline)):
            raise ValueError("deadline must be a finite monotonic time")
        result: dict[str, Any] = dict(status="error", query_smiles=query, matched_compound_ids=[],
                      identity_rejected_candidates=[], candidate_count=None,
                      candidates_truncated=False, activity_count=None, truncated=False,
                      assay_metadata_complete=False, records=[], error=None, source=API,
                      retrieved_at=datetime.now(timezone.utc).isoformat())
        try:
            payload = self._get("molecule", {
                "molecule_structures__canonical_smiles__connectivity": query,
                "limit": MAX_CANDIDATES, "order_by": "molecule_chembl_id",
                "only": "molecule_chembl_id,molecule_structures,pref_name"}, deadline=deadline, clock=clock)
            candidates, total, truncated = _page(payload, "molecules", MAX_CANDIDATES)
            result.update(candidate_count=total, candidates_truncated=truncated, truncated=truncated)
            for candidate in candidates:
                identifier = _identifier(candidate["molecule_chembl_id"])
                returned = candidate["molecule_structures"]["canonical_smiles"]
                actual = canonical_smiles(returned)
                if actual == query:
                    if identifier not in result["matched_compound_ids"]:
                        result["matched_compound_ids"].append(identifier)
                else:
                    result["identity_rejected_candidates"].append(dict(
                        compound_id=identifier, returned_smiles=actual,
                        reason="canonical isomeric structure differs", compound_url=_link("compound", identifier)))
            if not result["matched_compound_ids"]:
                result["status"] = "identity_mismatch" if candidates else "not_found"
                return result

            payload = self._get("activity", {
                "molecule_chembl_id__in": ",".join(result["matched_compound_ids"]),
                "target_chembl_id": TARGET, "target_organism": "Homo sapiens",
                "limit": limit, "order_by": "activity_id"}, deadline=deadline, clock=clock)
            activities, total, truncated = _page(payload, "activities", limit)
            result.update(activity_count=total, truncated=result["truncated"] or truncated)
            # Validate the entire page before attaching any potentially unrelated activity.
            result["records"] = [self._record(row, query, result["matched_compound_ids"])
                                 for row in activities]
            if not activities:
                result.update(status="not_found", assay_metadata_complete=True)
                return result

            assay_ids = sorted({row["assay_chembl_id"] for row in activities})
            payload = self._get("assay", {"assay_chembl_id__in": ",".join(assay_ids),
                                          "limit": len(assay_ids), "order_by": "assay_chembl_id"},
                                deadline=deadline, clock=clock)
            assays, _, incomplete = _page(payload, "assays", len(assay_ids))
            metadata = {row["assay_chembl_id"]: row for row in assays}
            if incomplete or set(metadata) != set(assay_ids) or len(metadata) != len(assays):
                raise ValueError("incomplete or unexpected assay metadata")
            if any(row.get("target_chembl_id") != TARGET for row in assays):
                raise ValueError("assay metadata has an unexpected target")
            for record in result["records"]:
                assay = metadata[record["assay_id"]]
                source_id, source_assay = assay.get("src_id"), assay.get("src_assay_id")
                overlap = None if source_id is None else source_id == 7 and source_assay == "1851_3"
                if source_id == 7 and source_assay is None:
                    overlap = None
                record.update(assay_description=assay.get("description") or record["assay_description"],
                              assay_source_id=source_id, source_assay_id=source_assay,
                              assay_confidence_score=assay.get("confidence_score"),
                              veith_overlap=overlap)
            result.update(status="ok", assay_metadata_complete=True)
        except HTTPError as exc:
            result["error"] = f"ChEMBL HTTP {exc.code}: {exc.reason}"
            exc.close()
        except (HTTPException, URLError, OSError, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            result["error"] = f"ChEMBL lookup failed: {type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _record(row, query, matched_ids):
        if (row.get("target_chembl_id") != TARGET or row.get("target_organism") != "Homo sapiens"
                or row.get("molecule_chembl_id") not in matched_ids
                or canonical_smiles(row["canonical_smiles"]) != query):
            raise ValueError("activity target or structure does not match the query")
        activity_id = row["activity_id"]
        if type(activity_id) is not int or activity_id <= 0:
            raise ValueError("invalid activity identifier")
        assay_id = _identifier(row["assay_chembl_id"])
        record = {key: row.get(key) for key in (
            "standard_type", "standard_relation", "standard_value", "standard_units",
            "standard_upper_value", "standard_text_value", "standard_flag",
            "type", "relation", "value", "units", "upper_value", "text_value",
            "activity_comment", "assay_description", "assay_type", "data_validity_comment",
            "potential_duplicate", "document_year", "document_journal")}
        record.update(activity_id=activity_id, compound_id=row["molecule_chembl_id"],
                      assay_id=assay_id, target_id=TARGET, target_organism="Homo sapiens",
                      document_id=row.get("document_chembl_id"), source="ChEMBL",
                      source_id=row.get("src_id"), assay_source_id=None, source_assay_id=None,
                      veith_overlap=None, assay_confidence_score=None,
                      activity_url=f"{API}/activity/{activity_id}.json",
                      compound_url=_link("compound", row["molecule_chembl_id"]),
                      assay_url=_link("assay", assay_id), target_url=_link("target", TARGET),
                      document_url=_link("document", row.get("document_chembl_id")))
        return record
