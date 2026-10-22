# Evidence for the reported numbers

`runs/` is not tracked, so the files behind the [submission](../docs/submission/writeup.md)
and [three-arm pilot](../docs/history/three-arm-pilot.md) are copied here.
Each is raw output, not a transcription.

| File | What it is |
|---|---|
| `predictions.jsonl` | One record per molecule per arm. Each carries its arm, status, five scores, and the model that produced it. Failures stay as error records; they are never converted to zeros. |
| `metrics.csv` | Per-arm, per-enzyme AUC and average precision with scaffold-bootstrap intervals, plus the evaluable counts and class balance behind each row. |
| `contrasts.csv` | Paired differences between arms. Arms share each resample, so these are the comparisons to read — marginal intervals in `metrics.csv` overlap where the paired interval does not. |
| `summary.json` | Cohort coverage per arm (planned, attempted, succeeded, error, unavailable, missing), label support, and the stated limitations of the run. |
| `cohort_manifest.json` | Hashes of the trained artifacts and the frozen cohort, so a rerun that changes either is refused rather than silently mixed. |

## Runs

- `screening-budget/` — retrospective CYP3A4 hit counts and scaffold coverage at
  budgets 24, 48 and 96, comparing classifier and LLM rankings with and without
  the workbench's diversity cap. Includes a reproducible figure and selected IDs.
  `full-llm/cohort.csv` now contains the exact frozen cohort needed to reproduce
  this comparison offline without model bundles or retraining.
- `pilot-haiku/` — 120-molecule uniform sample, three arms: `classifier`, `smiles`, `name`.
- `novelty/` — the same full-set contrast recomputed inside bands of nearest-training
  similarity. `novelty_contrasts.csv` is the table that qualifies the headline: the
  classifier's advantage is concentrated where the held-out molecule has a close
  training analog, even though no scaffold is shared across partitions.
- `full-llm/` — the entire 3,274-molecule held-out partition, two arms: `classifier`, `smiles`.
  Its `predictions.jsonl` is gzipped; everything else is plain text.

Reproduce either from the commands in `README.md`. The cohort is frozen before any
arm runs, and each arm resumes rather than recomputing, so a rerun cannot quietly
change the population a number was computed on.
