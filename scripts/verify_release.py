"""Check the frozen scientific release using retained hashes; no dependencies or network."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("results/full-llm")
SCREENING = Path("results/screening-budget")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_files(root):
    """Join existing provenance without replacing it with newly computed expectations."""
    run = json.loads((root / RUN / "cohort_manifest.json").read_text())
    training = json.loads((root / "artifacts/manifest.json").read_text())
    screening = json.loads((root / SCREENING / "provenance.json").read_text())
    expected = {}

    def add(name, digest):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError(f"Manifest path must stay inside the release: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for {name}")
        if path in expected and expected[path] != digest:
            raise ValueError(f"Retained manifests disagree about {name}")
        expected[path] = digest

    for name in ("cohort_manifest.json", "cohort.csv", "predictions.jsonl.gz"):
        path = RUN / name
        add(path, screening["sources"][path.as_posix()])
    add(RUN / "cohort.csv", run["cohort_sha256"])
    for name, digest in run["artifact_hashes"].items():
        add(Path("artifacts") / name, digest)
    add("artifacts/manifest.csv", training["manifest_sha256"])
    for name, digest in training["dataset_sha256"].items():
        add(Path("data") / name, digest)
    for name in ("summary.csv", "selected-compounds.csv"):
        add(SCREENING / name, screening["outputs"][name])
    return expected


def verify(root):
    root = Path(root).resolve()
    results = []
    for relative, expected in sorted(expected_files(root).items()):
        path = root / relative
        if not path.resolve().is_relative_to(root):
            status = "OUTSIDE RELEASE"
        elif not path.is_file():
            status = "MISSING"
        else:
            status = "OK" if sha256(path) == expected else "MISMATCH"
        results.append((relative.as_posix(), status))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Release directory (defaults to this checkout)")
    args = parser.parse_args(argv)
    try:
        results = verify(args.root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"Cannot verify release: {exc}")
        return 1
    for name, status in results:
        print(f"{status}: {name}")
    failed = sum(status != "OK" for _, status in results)
    print(f"{len(results) - failed}/{len(results)} frozen files match retained SHA-256 hashes.")
    print("This checks released bytes, not fresh training, scientific validity, or live services.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
