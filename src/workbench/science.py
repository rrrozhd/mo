"""Scientific operations for the local workbench; no session or model-provider state."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from uuid import uuid4

from rdkit import Chem, DataStructs

from ..chemistry.features import ENZYMES, canonical_smiles, fingerprints
from ..sources.chembl import EvidenceClient
from ..screening.cli import validate_candidates
from ..screening.shortlist import CYP3A4Screen, select_shortlist

RECORD_LIMIT = 10
MODEL_ANALYSES = {"predict_cyp", "show_attribution"}
CORPUS_ANALYSES = {"find_analogs", "show_neighbourhood"}
ANALYSES = MODEL_ANALYSES | CORPUS_ANALYSES | {"compare_structures", "show_structures"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 digest")
    return value


def _candidate(dataset, identifier):
    for row in dataset["scored"]:
        if row["id"] == identifier:
            return row
    raise ValueError(f"Unknown candidate ID: {identifier!r}")


def _without_stereo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    # isomericSmiles=False would also erase isotope identity.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


class ScientificOperations:
    """Reuse the frozen scorer and selector; cache by canonical dataset/model identity."""

    def __init__(self, artifacts=None, *, screen=None, model_digest=None, analysis=None):
        self.artifacts = Path(artifacts).resolve() if artifacts is not None else None
        if screen is None:
            self.artifacts = self.artifacts or Path("artifacts").resolve()
            path = self.artifacts / "models" / "CYP3A4.joblib"
            model_digest = sha256(path.read_bytes()).hexdigest()
            screen = CYP3A4Screen(self.artifacts)
            if sha256(path.read_bytes()).hexdigest() != model_digest:
                raise ValueError("Model artifact changed while loading")
        self.model_digest = _require_digest(model_digest)
        self.screen = screen
        self._datasets = {}
        self._analysis = analysis
        self._artifact_hashes = {}
        if self.artifacts is not None:
            paths = {f"models/{enzyme}.joblib" for enzyme in ENZYMES} | {"models/CYP3A4.joblib", "manifest.csv"}
            self._artifact_hashes = {relative: sha256((self.artifacts / relative).read_bytes()).hexdigest()
                                     for relative in paths if (self.artifacts / relative).is_file()}
            if self._artifact_hashes.get("models/CYP3A4.joblib") != self.model_digest:
                raise ValueError("CYP3A4 model artifact differs from the screening model")

    def _verify_analysis_artifacts(self, name):
        if self.artifacts is None:
            raise ValueError("Chemistry analysis requires a bound artifact directory")
        paths = {"models/CYP3A4.joblib"}
        if name in MODEL_ANALYSES:
            paths.update(f"models/{enzyme}.joblib" for enzyme in ENZYMES)
        if name in CORPUS_ANALYSES:
            paths.add("manifest.csv")
        hashes = {}
        for relative in sorted(paths):
            try:
                hashes[relative] = sha256((self.artifacts / relative).read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError(f"Required analysis artifact is unavailable: {relative}") from exc
            if hashes[relative] != self._artifact_hashes.get(relative):
                raise ValueError(f"Analysis artifact changed since the session started: {relative}")
        return hashes

    def analyze(self, name, args):
        """Dispatch trusted local tools using this session's model and corpus only."""
        if name not in ANALYSES:
            raise ValueError("Unknown chemistry analysis")
        if type(args) is not dict or "artifacts" in args:
            raise ValueError("Analysis arguments cannot select another artifact directory")
        if self._analysis is not None:
            result = self._analysis(name, deepcopy(args))
        else:
            from ..prediction import cyp as cyp_tools

            hashes = self._verify_analysis_artifacts(name)
            kwargs = {"artifacts": self.artifacts} if name in MODEL_ANALYSES | CORPUS_ANALYSES else {}
            result = getattr(cyp_tools, name)(**args, **kwargs)
            self._verify_analysis_artifacts(name)
            result = {**result, "artifact_hashes": hashes}
        canonical_json(result)
        return deepcopy(result)

    def prepare(self, rows):
        validated = sorted(validate_candidates(rows), key=lambda row: row["id"])
        dataset_digest = digest([[row["id"], row["smiles"]] for row in validated])
        key = (dataset_digest, self.model_digest)
        if key not in self._datasets:
            dataset = {"dataset_digest": dataset_digest, "model_digest": self.model_digest,
                       "rows": validated, "scored": self.screen.score(validated)}
            canonical_json(dataset)
            self._datasets[key] = deepcopy(dataset)
        return deepcopy(self._datasets[key])

    def select(self, dataset, constraints):
        return select_shortlist(dataset["scored"], **constraints)

    def compare(self, dataset, a_id, b_id, evidence_refs=None):
        a, b = _candidate(dataset, a_id), _candidate(dataset, b_id)
        first, second = a["canonical_smiles"], b["canonical_smiles"]
        equal = first == second
        relationship = ("identical" if equal else "stereo_only"
                        if _without_stereo(first) == _without_stereo(second) else "unresolved")
        return {"a": deepcopy(a), "b": deepcopy(b), "exact_identity_equal": equal,
                "fingerprint_tanimoto": DataStructs.TanimotoSimilarity(*fingerprints([first, second])),
                "structural_relationship": relationship,
                "evidence_refs": {identifier: deepcopy((evidence_refs or {}).get(identifier, []))
                                  for identifier in (a_id, b_id)}}


def evidence_snapshot(source, candidate, dataset, *, smiles=None, deadline=None, clock=None):
    """Look up an exact structure at the session limit and bind actual provenance.

    Call once for a newly attached lookup/playback. Cached reuse returns that stored
    snapshot instead of calling this function, preserving ID, origin and timestamps.
    """
    if candidate is None:
        query = canonical_smiles(smiles)
    else:
        query = _candidate(dataset, candidate)["canonical_smiles"]
        if smiles is not None and canonical_smiles(smiles) != query:
            raise ValueError("Evidence structure differs from the candidate")
    if isinstance(source, RecordedEvidence):
        if source.model_digest != dataset["model_digest"]:
            raise ValueError("Recorded evidence model differs from the current dataset")
        origin = "recorded"
    elif isinstance(source, EvidenceClient):
        origin = "live"
    else:
        raise TypeError("Evidence source must be EvidenceClient or RecordedEvidence")
    # Fixture sources override lookup without performing HTTP. The concrete live
    # client also receives the operation's clock and admission deadline.
    result = (source.lookup(query, RECORD_LIMIT, deadline=deadline, clock=clock)
              if type(source) is EvidenceClient else source.lookup(query, RECORD_LIMIT))
    if result.get("query_smiles") != query:
        raise ValueError("Evidence query does not match the exact candidate structure")
    if not isinstance(result["records"], list) or len(result["records"]) > RECORD_LIMIT:
        raise ValueError("Evidence records exceed the session limit")
    snapshot = {"candidate_id": candidate, "canonical_query": query,
                "dataset_digest": dataset["dataset_digest"], "model_digest": dataset["model_digest"],
                "record_limit": RECORD_LIMIT, "origin": origin,
                "retrieved_at": result.get("retrieved_at"), "result": deepcopy(result)}
    if isinstance(source, RecordedEvidence):
        snapshot.update(fixture_hash=source.fixture_hash,
                        playback_at=datetime.now(timezone.utc).isoformat())
    snapshot["cache_key"] = digest([dataset["dataset_digest"], dataset["model_digest"],
                                    query, origin, RECORD_LIMIT])
    snapshot["content_digest"] = digest(snapshot)
    snapshot["snapshot_id"] = uuid4().hex
    return snapshot


class RecordedEvidence:
    """Read a trusted local capture; unknown exact structures never fall back to HTTP.

    The capture's dataset identity is provenance, not a restriction on future IDs.
    A new dataset may reuse an exact structure; its snapshot binds the new identity.
    Hashes detect damaged captures, not malicious replacement of a trusted fixture.
    """

    origin = "recorded"

    def __init__(self, path, model_digest, record_limit=RECORD_LIMIT):
        body = Path(path).read_bytes()
        self.fixture_hash = sha256(body).hexdigest()
        try:
            bundle = json.loads(body)
            canonical_json(bundle)  # Reject NaN, infinity and overflowing JSON numbers.
            if type(bundle["schema_version"]) is not int or bundle["schema_version"] != 1:
                raise ValueError("Unsupported recorded evidence schema")
            if (type(record_limit) is not int or record_limit != RECORD_LIMIT
                    or type(bundle["record_limit"]) is not int
                    or bundle["record_limit"] != record_limit):
                raise ValueError("Recorded evidence limit must match the session limit")
            self.model_digest = _require_digest(bundle["model_digest"])
            if self.model_digest != _require_digest(model_digest):
                raise ValueError("Recorded evidence model digest differs from the loaded model")
            self.source_dataset_digest = _require_digest(bundle["dataset_digest"])
            self.record_limit = record_limit
            self._results = {}
            if not isinstance(bundle["entries"], list):
                raise ValueError("Recorded evidence entries must be a list")
            for entry in bundle["entries"]:
                query, result = entry["canonical_query"], entry["result"]
                if (canonical_smiles(query) != query or result["query_smiles"] != query
                        or query in self._results):
                    raise ValueError("Recorded evidence has a duplicate or mismatched query")
                if not isinstance(entry["candidate_id"], str) or not entry["candidate_id"].strip():
                    raise ValueError("Recorded evidence candidate ID is missing")
                if not result.get("retrieved_at"):
                    raise ValueError("Recorded evidence original retrieval time is missing")
                if not isinstance(result["records"], list) or len(result["records"]) > record_limit:
                    raise ValueError("Recorded evidence records exceed the session limit")
                if digest(result) != _require_digest(entry["result_sha256"]):
                    raise ValueError("Recorded evidence result hash differs")
                if not isinstance(entry["raw_responses"], list):
                    raise ValueError("Recorded evidence raw responses must be a list")
                for response in entry["raw_responses"]:
                    if response["resource"] not in {"molecule.json", "activity.json", "assay.json"}:
                        raise ValueError("Unknown recorded response resource")
                    actual = sha256(response["body_utf8"].encode("utf-8")).hexdigest()
                    if actual != _require_digest(response["sha256"]):
                        raise ValueError("Recorded evidence raw response hash differs")
                    if response["resource"] == "activity.json":
                        try:
                            page = json.loads(response["body_utf8"])
                        except json.JSONDecodeError:
                            if result["status"] != "error" or result["records"]:
                                raise ValueError("Malformed activity response cannot support observations")
                            continue
                        if isinstance(page, dict) and isinstance(page.get("page_meta"), dict):
                            limit = page["page_meta"].get("limit")
                            if limit is not None and (type(limit) is not int or limit != record_limit):
                                raise ValueError("Recorded activity page limit differs from the session limit")
                self._results[query] = result
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed recorded evidence bundle") from exc

    def lookup(self, smiles, limit=RECORD_LIMIT):
        if type(limit) is not int or limit != self.record_limit:
            raise ValueError("Recorded evidence limit must match the session limit")
        query = canonical_smiles(smiles)
        if query in self._results:
            return deepcopy(self._results[query])
        return {"status": "not_recorded", "query_smiles": query, "retrieved_at": None,
                "matched_compound_ids": [], "identity_rejected_candidates": [],
                "candidate_count": None, "candidates_truncated": False,
                "activity_count": None, "truncated": False, "assay_metadata_complete": False,
                "records": [], "error": None, "source": None}
