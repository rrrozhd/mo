from copy import deepcopy
import re


def batch():
    return {
        "target": "CYP3A4", "objective": "Find inhibition assay candidates",
        "status": "partial", "requested_count": 2, "selected_count": 1,
        "complete": False, "shortfall_reason": "Only one diverse candidate",
        "method": "score_ordered_greedy",
        "constraints": {"budget": 2, "include_ids": ["a"], "exclude_ids": ["b"],
                        "max_similarity": 0.8},
        "selected": [{"id": "a", "canonical_smiles": "CCO", "score": 0.91,
                      "validation_percentile": 0.97, "nearest_training_similarity": 0.7,
                      "similarity_flag": False, "confidence": "uncalibrated",
                      "selection_reason": "required inclusion"}],
        "ranked": [], "skipped": [{"id": "b", "reason": "excluded"}],
        "evidence": {"a": {
            "status": "error", "error": "metadata timeout", "truncated": True,
            "candidates_truncated": False, "assay_metadata_complete": False,
            "retrieved_at": "2026-10-21T12:00:00Z", "activity_count": 3,
            "source": "https://www.ebi.ac.uk/chembl/api/data",
            "records": [
                {"activity_id": 1, "standard_type": "IC50", "standard_relation": ">",
                 "standard_value": "10000", "standard_units": "nM",
                 "assay_description": "Probe A at 5 minutes", "veith_overlap": True,
                 "activity_url": "https://example.org/activity/1",
                 "assay_url": "https://example.org/assay/10"},
                {"activity_id": 2, "standard_type": "Activity",
                 "standard_text_value": "Inconclusive", "text_value": "Not interpretable",
                 "activity_comment": "Optical interference", "veith_overlap": None,
                 "assay_description": "Probe B", "document_url": "https://example.org/paper"},
            ]}},
    }


def test_report_separates_predictions_observations_and_incomplete_evidence():
    from src.batch_report import render_batch_report

    report = batch()
    original = deepcopy(report)
    rendered = render_batch_report(report)
    assert "CYP3A4" in rendered and "Find inhibition assay candidates" in rendered
    assert "Selected: 1 of 2" in rendered and "Only one diverse candidate" in rendered
    assert "uncalibrated" in rendered and "required inclusion" in rendered
    assert "0.9100" in rendered and "0.9700" in rendered and "0.7000" in rendered
    assert "IC50" in rendered and "10000" in rendered and "nM" in rendered
    assert "Inconclusive" in rendered and "Not interpretable" in rendered
    assert "Optical interference" in rendered and "Probe A at 5 minutes" in rendered
    assert "metadata timeout" in rendered and "truncated: yes" in rendered
    assert "Assay metadata complete: no" in rendered and "Evidence status: error" in rendered
    assert "known training-source overlap" in rendered and "overlap: unknown" in rendered
    assert "2026" in rendered and "https://example.org/activity/1" in rendered
    assert "https://example.org/assay/10" in rendered and "https://example.org/paper" in rendered
    assert "excluded" in rendered and "0.8" in rendered
    assert report == original


def test_report_escapes_external_text_and_allows_only_safe_http_links():
    from src.batch_report import render_batch_report

    report = batch()
    identifier = "a|b\n# forged <script> [click](javascript:bad)"
    report["selected"][0]["id"] = identifier
    evidence = report["evidence"].pop("a")
    report["evidence"][identifier] = evidence
    record = evidence["records"][0]
    record["assay_description"] = "line 1\r\n## injected | <img src=x> `code`"
    record["activity_url"] = "javascript:alert(1)"
    record["assay_url"] = "https://example.org/a(b)?q=<x>"
    record["document_url"] = "https://example.org/x\n# injected"
    rendered = render_batch_report(report)
    assert r"a\|b" in rendered and r"\\n" in rendered
    assert "\n# forged" not in rendered and "\n## injected" not in rendered
    assert re.search(r"(?<!\\)<(?:script|img)", rendered) is None
    assert "javascript:alert" not in rendered
    assert "https://example.org/a%28b%29?q=%3Cx%3E" in rendered
    assert "https://example.org/x" not in rendered
    assert "\\`code\\`" in rendered


def test_missing_evidence_and_empty_selection_do_not_imply_inactivity():
    from src.batch_report import render_batch_report

    report = batch()
    report["evidence"] = {}
    rendered = render_batch_report(report)
    assert "Evidence status: missing" in rendered
    assert "No measured observations returned" in rendered
    assert "does not mean inactive" in rendered
    report["selected"] = []
    report["selected_count"] = 0
    rendered = render_batch_report(report)
    assert "Selected: 0 of 2" in rendered
    assert "No compounds selected" in rendered


def test_distinct_candidate_ids_remain_distinct_in_the_report():
    from src.batch_report import render_batch_report

    report = batch()
    identifiers = ["a b", "a\nb", "a  b", "a\tb", "candidate&copy;1", "candidate©1"]
    report["selected"] = [{**report["selected"][0], "id": value} for value in identifiers]
    report["evidence"] = {}
    report["constraints"]["include_ids"] = identifiers
    report["skipped"] = [{"id": value, "reason": "excluded"} for value in identifiers]
    rendered = render_batch_report(report)
    headings = [line for line in rendered.splitlines() if line.startswith("### ")]
    assert len(headings) == len(set(headings)) == len(identifiers)
    assert r"\&copy;" in rendered
    assert "IDs use JSON string notation" in rendered


def test_literal_entities_in_measurements_are_preserved_as_text():
    from src.batch_report import render_batch_report

    report = batch()
    report["evidence"]["a"]["records"][0]["assay_description"] = "A &copy; label; B &lt; threshold"
    rendered = render_batch_report(report)
    assert r"A \&copy; label; B \&lt; threshold" in rendered
