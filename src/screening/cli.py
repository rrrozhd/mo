"""CYP3A4 evidence lookup, deterministic shortlisting, and assay batch reports."""

import argparse
import csv
import json
import sys
from pathlib import Path

from .batch import plan_batch
from ..chemistry.features import canonical_smiles
from ..sources.chembl import MAX_RECORDS, EvidenceClient
from .shortlist import CYP3A4Screen, select_shortlist


def validate_candidates(rows):
    """Keep only user identities and structures; discard any supplied assay labels."""
    result, ids, structures = [], set(), set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError("Candidate IDs must be nonempty, unique strings")
        smiles = canonical_smiles(row.get("smiles"))
        if smiles in structures:
            raise ValueError("Duplicate canonical structure; supply each molecule only once")
        result.append({"id": identifier, "smiles": smiles})
        ids.add(identifier)
        structures.add(smiles)
    if not result:
        raise ValueError("Provide at least one candidate")
    return result


def load_candidates(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "smiles"}.issubset(reader.fieldnames or []):
            raise ValueError("Candidate CSV requires id and smiles columns")
        return validate_candidates(reader)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evidence = commands.add_parser("evidence", help="Retrieve exact-structure assay evidence")
    evidence.add_argument("--smiles", required=True)
    evidence.add_argument("--limit", type=int, default=10)
    shortlist = commands.add_parser("shortlist", help="Run deterministic CYP3A4 selection")
    plan = commands.add_parser("plan", help="Select an assay batch and retrieve evidence for review")
    plan.add_argument("--evidence-limit", type=int, choices=range(1, MAX_RECORDS + 1), default=10,
                      metavar="1-50", help="Maximum measurements returned per selected compound")
    plan.add_argument("--report", type=Path, help="Also write a readable Markdown report")
    for command in (shortlist, plan):
        command.add_argument("--budget", required=True, type=int)
        command.add_argument("--include", nargs="*", default=[])
        command.add_argument("--exclude", nargs="*", default=[])
        command.add_argument("--max-similarity", type=float, default=0.8)
        command.add_argument("--candidates", required=True, type=Path)
        command.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    for command in (evidence, shortlist, plan):
        command.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command in {"shortlist", "plan"}:
        outputs = [path for path in (args.output, getattr(args, "report", None)) if path is not None]
        inputs = [args.candidates, args.artifacts / "models" / "CYP3A4.joblib"]
        for index, path in enumerate(outputs):
            for other in outputs[index + 1:] + inputs:
                if path.resolve() == other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
                    parser.error("inputs, JSON output, and Markdown report require distinct paths")
    try:
        if args.command == "evidence":
            result = EvidenceClient().lookup(args.smiles, args.limit)
        else:
            candidates = load_candidates(args.candidates)
            screen = CYP3A4Screen(args.artifacts)
            scored = screen.score(candidates)
            if args.command == "plan":
                result = plan_batch(scored, args.budget, args.include, args.exclude, args.max_similarity,
                                    args.evidence_limit, EvidenceClient())
            else:
                result = select_shortlist(scored, args.budget,
                                          args.include, args.exclude, args.max_similarity)
    except (ValueError, OSError) as exc:
        result = {"status": "error", "error": str(exc)}
    payload = json.dumps(result, indent=2, allow_nan=False)
    print(payload)
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        if args.command == "plan" and args.report:
            from .report import render_batch_report

            text = ("# Batch planning failed\n\nSee the JSON error output for details.\n"
                    if result.get("status") == "error" else render_batch_report(result))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 1
    if result.get("status") == "error":
        return 1
    return 2 if result.get("status") == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
