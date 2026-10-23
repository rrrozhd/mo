"""A reviewable CYP3A4 assay batch, with predictions and measurements kept apart."""

from ..sources.chembl import MAX_RECORDS, EvidenceClient
from .shortlist import select_shortlist


def plan_batch(scored, budget, include_ids=None, exclude_ids=None, max_similarity=.8,
               evidence_limit=10, evidence_client=None):
    """Select once, then retrieve bounded evidence for selected structures only.

    Measurements do not change the ranking or prove a proposed assay is redundant.
    The caller can review the observations and revise include/exclude constraints.
    No evidence query is made until all selection constraints have been validated.
    """
    if type(evidence_limit) is not int or not 1 <= evidence_limit <= MAX_RECORDS:
        raise ValueError(f"evidence_limit must be an integer from 1 to {MAX_RECORDS}")
    selection = select_shortlist(scored, budget, include_ids, exclude_ids, max_similarity)
    required = set(include_ids or [])
    selection["selected"] = [
        {**row, "selection_reason": "required inclusion" if row["id"] in required
         else "score-ranked under diversity constraint"}
        for row in selection["selected"]
    ]
    client = evidence_client if evidence_client is not None else EvidenceClient()
    evidence = {row["id"]: client.lookup(row["canonical_smiles"], evidence_limit)
                for row in selection["selected"]}
    partial = not selection["complete"] or any(
        result["status"] == "error" or result["truncated"] for result in evidence.values()
    )
    return {
        **selection, "status": "partial" if partial else "ok", "target": "CYP3A4",
        "objective": "Prioritize compounds for CYP3A4 inhibition assays",
        "constraints": {"budget": budget, "include_ids": list(include_ids or []),
                        "exclude_ids": list(exclude_ids or []), "max_similarity": max_similarity},
        "evidence_limit": evidence_limit, "evidence": evidence,
    }
