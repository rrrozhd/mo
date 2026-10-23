from copy import deepcopy

import pytest

from src.screening.shortlist import select_shortlist


def scored():
    return [dict(id=name, canonical_smiles=smiles, score=score,
                 validation_percentile=score, nearest_training_similarity=.7,
                 similarity_flag=False, confidence="uncalibrated")
            for name, smiles, score in [("a", "CCCCCC", .9), ("b", "CCCCCCC", .8),
                                        ("c", "c1ccccc1", .7), ("d", "CCO", .2)]]


class Evidence:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    def lookup(self, smiles, limit):
        self.calls.append((smiles, limit))
        return deepcopy(self.results.get(smiles, {
            "status": "not_found", "query_smiles": smiles, "records": [],
            "error": None, "truncated": False,
        }))


def test_plan_preserves_selection_and_queries_only_selected_once():
    from src.screening.batch import plan_batch

    rows, client = scored(), Evidence()
    before = deepcopy(rows)
    baseline = select_shortlist(rows, 2, include_ids=["d"], exclude_ids=["a"], max_similarity=.5)
    report = plan_batch(rows, 2, include_ids=["d"], exclude_ids=["a"],
                        max_similarity=.5, evidence_limit=3, evidence_client=client)
    assert report["status"] == "ok" and report["complete"]
    assert report["target"] == "CYP3A4"
    assert [{k: v for k, v in row.items() if k != "selection_reason"}
            for row in report["selected"]] == baseline["selected"]
    assert report["skipped"] == baseline["skipped"]
    assert client.calls == [(row["canonical_smiles"], 3) for row in baseline["selected"]]
    assert set(report["evidence"]) == {"b", "d"}
    assert report["selected"][1]["selection_reason"] == "required inclusion"
    assert report["constraints"]["include_ids"] == ["d"]
    assert rows == before


def test_partial_remote_failure_keeps_measurements_batch_and_later_lookups():
    from src.screening.batch import plan_batch

    partial = {"status": "error", "query_smiles": "CCCCCC", "error": "metadata timeout",
               "truncated": False, "records": [{"standard_type": "IC50",
               "standard_relation": ">", "standard_value": "10000", "standard_units": "nM",
               "veith_overlap": None}]}
    client = Evidence({"CCCCCC": partial})
    report = plan_batch(scored(), 2, max_similarity=1, evidence_client=client)
    assert report["complete"] and report["status"] == "partial"
    assert report["evidence"]["a"] == partial
    assert report["evidence"]["b"]["status"] == "not_found"
    assert len(client.calls) == 2


def test_truncation_and_shortfall_are_explicit_without_relaxing_constraints():
    from src.screening.batch import plan_batch

    client = Evidence({"CCCCCC": {"status": "ok", "query_smiles": "CCCCCC",
                       "records": [{"standard_text_value": "Not active"}], "truncated": True}})
    report = plan_batch(scored(), 4, exclude_ids=["d"], max_similarity=.5, evidence_client=client)
    assert report["status"] == "partial" and not report["complete"]
    assert report["selected_count"] == 2 and report["shortfall_reason"]
    assert report["evidence"]["a"]["truncated"]
    assert [row["id"] for row in report["selected"]] == ["a", "c"]


@pytest.mark.parametrize("kwargs", [{"evidence_limit": 0}, {"evidence_limit": 51},
    {"evidence_limit": True}, {"budget": 0}, {"include_ids": ["missing"]},
    {"include_ids": ["a"], "exclude_ids": ["a"]},
    {"include_ids": ["a", "b"], "max_similarity": .5}])
def test_invalid_plan_makes_no_evidence_requests(kwargs):
    from src.screening.batch import plan_batch

    client = Evidence()
    with pytest.raises(ValueError):
        plan_batch(scored(), **{"budget": 2, **kwargs}, evidence_client=client)
    assert client.calls == []
