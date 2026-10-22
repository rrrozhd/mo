"""Render a screening batch without conflating predictions and assay evidence."""

import json
import re
from urllib.parse import quote, urlsplit


def _text(value):
    text = "not reported" if value is None else " ".join(str(value).split())
    return re.sub(r"([\\`*_{}\[\]()#+.!|<>~&-])", r"\\\1", text)


def _candidate_id(value):
    # JSON escapes distinguish whitespace/control characters and literal escapes.
    return _text(json.dumps(value, ensure_ascii=True).replace(" ", r"\u0020"))


def _yes_no(value):
    return "unknown" if value is None else "yes" if value else "no"


def _link(label, value):
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return ""
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
    except ValueError:
        return ""
    return f"[{label}]({quote(value, safe=':/?=&%#@+,;~')})"


def _measurement(record, prefix):
    parts = [record.get(prefix + key) for key in ("type", "relation", "value", "units")]
    result = " ".join(_text(part) for part in parts if part is not None and part != "")
    upper = record.get(prefix + "upper_value")
    return (result or "not reported") + (f"; upper value: {_text(upper)}" if upper is not None else "")


def render_batch_report(report):
    """Return Markdown for a composed batch report; do not mutate its evidence."""
    constraints = report["constraints"]
    lines = [f"# {_text(report['target'])} screening batch", "",
             f"Objective: {_text(report['objective'])}", "",
             f"Report status: {_text(report['status'])}. "
             f"Selection complete: {_yes_no(report['complete'])}.", "",
             f"Selected: {report['selected_count']} of {report['requested_count']}", "",
             "Predictions are uncalibrated ranking scores, not measured assay results "
             "or probabilities. Validation percentiles are ranks, not confidence.", "",
             "Similarity flags are heuristic. Assay observations remain separate; "
             "measurements are not averaged. Missing evidence does not mean inactive.", "",
             "Selection uses model scores and the requested constraints. Measurements "
             "are attached afterward for review and do not change the selection.", "",
             f"Constraints: budget {constraints['budget']}; maximum pairwise similarity "
             f"{constraints['max_similarity']}; required IDs: "
             f"{', '.join(_candidate_id(x) for x in constraints['include_ids']) or 'none'}; "
             f"excluded IDs: {', '.join(_candidate_id(x) for x in constraints['exclude_ids']) or 'none'}.", "",
             "IDs use JSON string notation; spaces are shown as \\u0020 to preserve their identity.", "",
             f"Selection method: {_text(report['method'])}", ""]
    if report.get("shortfall_reason"):
        lines.extend([f"Selection shortfall: {_text(report['shortfall_reason'])}", ""])
    lines.extend(["## Selected compounds", ""])
    if report["selected"]:
        lines.extend(["| ID | Score | Validation percentile | Training similarity | Similarity flag | Selection reason |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for row in report["selected"]:
            lines.append(f"| {_candidate_id(row['id'])} | {row['score']:.4f} | "
                         f"{row['validation_percentile']:.4f} | "
                         f"{row['nearest_training_similarity']:.4f} | "
                         f"{_yes_no(row['similarity_flag'])} | {_text(row['selection_reason'])} |")
    else:
        lines.append("No compounds selected.")
    lines.extend(["", "## Measured assay evidence", "",
                  "A source-overlap value of no excludes only the recognized Veith source; "
                  "it does not establish independent evidence.", ""])
    for row in report["selected"]:
        evidence = report.get("evidence", {}).get(row["id"], {})
        lines.extend([f"### {_candidate_id(row['id'])}", "",
                      f"Structure: {_text(row['canonical_smiles'])}", "",
                      f"Evidence status: {_text(evidence.get('status', 'missing'))}", "",
                      f"Retrieved: {_text(evidence.get('retrieved_at'))}", "",
                      f"Results truncated: {_yes_no(evidence.get('truncated'))}; "
                      f"candidate search truncated: {_yes_no(evidence.get('candidates_truncated'))}.", "",
                      f"Assay metadata complete: {_yes_no(evidence.get('assay_metadata_complete'))}", ""])
        source = _link("Evidence source", evidence.get("source"))
        if source:
            lines.extend([source, ""])
        if evidence.get("error"):
            lines.extend([f"Evidence error: {_text(evidence['error'])}", ""])
        records = evidence.get("records", [])
        total = evidence.get("activity_count")
        lines.extend([f"Returned observations: {len(records)}; total reported: {_text(total)}.", ""])
        if not records:
            lines.extend(["No measured observations returned.", ""])
        for index, record in enumerate(records, 1):
            lines.extend([f"**Observation {index}** (activity {_text(record.get('activity_id'))})", "",
                          f"Standardized result: {_measurement(record, 'standard_')}", "",
                          f"Original result: {_measurement(record, '')}", ""])
            for label, key in (("Standardized text", "standard_text_value"),
                               ("Original text", "text_value"), ("Activity comment", "activity_comment"),
                               ("Data validity", "data_validity_comment")):
                if record.get(key) is not None:
                    lines.extend([f"{label}: {_text(record[key])}", ""])
            overlap = _yes_no(record.get("veith_overlap"))
            if record.get("veith_overlap") is True:
                overlap += " (known training-source overlap)"
            lines.extend([f"Assay: {_text(record.get('assay_description'))}", "",
                          f"Source overlap: {overlap}", ""])
            links = [_link(label, record.get(key)) for label, key in
                     (("Activity", "activity_url"), ("Assay", "assay_url"),
                      ("Document", "document_url"), ("Compound", "compound_url"), ("Target", "target_url"))]
            if any(links):
                lines.extend(["Sources: " + ", ".join(link for link in links if link), ""])
    lines.extend(["## Skipped candidates", ""])
    lines.extend(f"- {_candidate_id(row['id'])}: {_text(row['reason'])}" for row in report["skipped"])
    if not report["skipped"]:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"
