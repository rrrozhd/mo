"""Cache PubChem record titles for a frozen cohort, requiring exact identity.

Titles are PubChem summary-page titles, not guaranteed common names. We request
the documented stereo/isotope-aware SMILES property and compare RDKit canonical
isomeric SMILES; PubChem standardization must not silently change the molecule.
API reference: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

import argparse
import csv
import json
import time
from collections import Counter
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .chemistry import canonical_smiles


SOURCE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/Title,SMILES/JSON"
REQUEST_INTERVAL = 0.25  # One sequential client, at most four requests/second.
TIMEOUT = 20
MAX_ATTEMPTS = 3
MAX_RETRY_DELAY = 30


def _retry_delay(header, attempt):
    """Honor seconds or HTTP-date Retry-After; callers defer long waits to resume."""
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(header).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return float(2**attempt)


class PubChemClient:
    """A synchronous rate-limited client; use one instance per lookup run."""

    def __init__(self):
        self.last_request = None

    def lookup(self, molecule_id, smiles):
        expected = canonical_smiles(smiles)
        record = dict(molecule_id=molecule_id, smiles=smiles, name=None, cid=None,
                      returned_smiles=None, status="error", error=None, source=SOURCE)
        request = Request(SOURCE, data=urlencode({"smiles": smiles}).encode("utf-8"),
                          headers={"Content-Type": "application/x-www-form-urlencoded",
                                   "Accept": "application/json"}, method="POST")
        for attempt in range(MAX_ATTEMPTS):
            if self.last_request is not None:
                time.sleep(max(0.0, REQUEST_INTERVAL - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                with urlopen(request, timeout=TIMEOUT) as response:
                    payload = json.load(response)
            except HTTPError as exc:
                if exc.code in (429, 503) and attempt < MAX_ATTEMPTS - 1:
                    delay = _retry_delay(exc.headers.get("Retry-After"), attempt)
                    if delay <= MAX_RETRY_DELAY:
                        exc.close()
                        time.sleep(delay)
                        continue
                record["error"] = f"HTTP {exc.code}: {exc.reason}"
                if exc.code == 404:
                    try:
                        fault = json.load(exc)
                    except (HTTPException, OSError, ValueError, UnicodeDecodeError):
                        fault = {}
                    detail = fault.get("Fault") if isinstance(fault, dict) else None
                    if isinstance(detail, dict) and detail.get("Code") == "PUGREST.NotFound":
                        record.update(status="unavailable", error="PubChem reports no matching record")
                exc.close()
                return record
            except (HTTPException, URLError, OSError, ValueError, UnicodeDecodeError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                return record
            return _select_title(record, expected, payload)
        return record  # Every final attempt returns a success or an error above.


def _select_title(record, expected, payload):
    try:
        properties = payload["PropertyTable"]["Properties"]
        if not isinstance(properties, list) or not properties:
            raise ValueError("missing compound properties")
        if len(properties) != 1:
            record.update(status="unavailable", error="ambiguous PubChem response: multiple compound records")
            return record
        row = properties[0]
        cid, returned_smiles = row["CID"], row["SMILES"]
        if type(cid) is not int or cid <= 0 or not isinstance(returned_smiles, str):
            raise ValueError("invalid CID or SMILES property")
        record.update(cid=cid, returned_smiles=returned_smiles)
        if canonical_smiles(returned_smiles) != expected:
            record.update(status="unavailable", error="returned structure differs from cohort molecule after canonicalization")
            return record
        title = row.get("Title")
        if title is None or title == "":
            record.update(status="unavailable", error="matching PubChem record has no Title")
        elif not isinstance(title, str):
            raise ValueError("invalid Title property")
        elif not title.strip():
            record.update(status="unavailable", error="matching PubChem record has no Title")
        else:
            record.update(status="ok", name=title.strip())
    except (KeyError, TypeError, ValueError) as exc:
        record["error"] = f"invalid PubChem response: {exc}"
    return record


def _read_cohort(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"molecule_id", "smiles"}.issubset(reader.fieldnames or []):
            raise ValueError("cohort CSV requires molecule_id and smiles columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row["molecule_id"] or row["molecule_id"] in seen:
            raise ValueError("cohort molecule_id values must be nonempty and unique")
        canonical_smiles(row["smiles"])
        seen.add(row["molecule_id"])
    return rows


def _reusable(record, row):
    if record.get("source") != SOURCE or record.get("status") not in ("ok", "unavailable"):
        return False
    try:
        same = canonical_smiles(record["smiles"]) == canonical_smiles(row["smiles"])
        if record["status"] == "ok":
            same = same and bool(record.get("name")) and type(record.get("cid")) is int
            same = same and canonical_smiles(record["returned_smiles"]) == canonical_smiles(row["smiles"])
        return same
    except (KeyError, TypeError, ValueError):
        return False


def lookup_cohort(cohort, output, client=None):
    """Resume permanent outcomes; retry errors and leave exactly one row per ID.

    Existing successful/unavailable records are compacted atomically first, then
    new records are appended and flushed individually so progress survives stops.
    """
    cohort, output = Path(cohort), Path(output)
    if cohort.resolve() == output.resolve():
        raise ValueError("cohort and output must be different files")
    rows = _read_cohort(cohort)
    cache = {}
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    cache[record["molecule_id"]] = record
    retained = {row["molecule_id"]: cache[row["molecule_id"]] for row in rows
                if row["molecule_id"] in cache and _reusable(cache[row["molecule_id"]], row)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for record in retained.values():
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    client = client or PubChemClient()
    records = dict(retained)
    with output.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["molecule_id"] in retained:
                continue
            record = client.lookup(row["molecule_id"], row["smiles"])
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            records[row["molecule_id"]] = record
    return [records[row["molecule_id"]] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path, help="frozen cohort CSV")
    parser.add_argument("--output", required=True, type=Path, help="resumable title cache JSONL")
    args = parser.parse_args()
    records = lookup_cohort(args.cohort, args.output)
    print(json.dumps(dict(Counter(row["status"] for row in records)), sort_keys=True))
    return 1 if any(row["status"] == "error" for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
