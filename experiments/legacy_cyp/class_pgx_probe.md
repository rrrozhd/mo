# Feasibility probe 4 — class-level pharmacogenomics (2026-10-21)

Question: can the existing CYP stack be generalized from "one molecule, five
inhibition scores" to "one drug class, genotype-aware ranking"? Beta blockers
as the test class.

## Verified numbers (probes run today, not estimates)

- ChEMBL beta-1 adrenergic receptor = `CHEMBL213` (human, SINGLE PROTEIN).
  **7,162 activities; 2,032 with a pChEMBL value.** Beta-2 `CHEMBL210`: 13,208.
  A class cohort of registered + experimental molecules is real and pullable.
- TDC carries the missing endpoint: `cyp2d6_substrate_carbonmangels` (667, 29% pos),
  `cyp3a4_substrate_carbonmangels` (670, 53%), `cyp2c9_substrate_carbonmangels`
  (669, 21%). The five Veith sets predict **inhibition**; these predict
  **substrate status**, which is the axis genotype acts on.

## The blocking finding: leakage, hidden by stereochemistry

Ten marketed beta blockers, matched two ways against the training data.

| match method | Veith (inhibition) | CarbonMangels (substrate) |
|---|---|---|
| canonical **isomeric** SMILES (what `src/chemistry.py` uses) | 0 / 10 | not run |
| InChIKey **skeleton** block (stereo-insensitive) | **4 / 10** | **7 / 10** |

Veith skeleton hits: propranolol, nadolol, labetalol, acebutolol — all at
Morgan Tanimoto **1.00**, i.e. the same compound stored with different
stereo annotation. CarbonMangels hits: metoprolol, propranolol, carvedilol,
bisoprolol, timolol, labetalol, acebutolol, also at 1.00.

Two consequences:

1. **Any retrospective check on known beta blockers is contaminated** unless they
   are held out explicitly. Hold out by skeleton key, not by canonical SMILES.
2. **`src/chemistry.py` identity is stereo-sensitive**, so racemic and enantiopure
   representations of one drug count as two molecules. The documented salt/tautomer
   caveat in the README has a stereo sibling, and it bites every
   cross-dataset join. The 16,366-molecule union may double-count stereovariants.

Genotype-robust controls that are *absent* from both sets and so usable as held-out
negatives: atenolol (nearest Tanimoto 0.62), sotalol (0.49), nadolol (0.55 in
CarbonMangels).

## What this does NOT show

Nothing was trained. No substrate model exists yet, so there is no evidence the
667-molecule sets support useful accuracy; published ADMET baselines for these
sets sit near 0.7-0.8 AUC, well below the 0.85-0.93 of the Veith inhibition models.
ChEMBL activity heterogeneity (IC50/Ki/EC50, mixed assays) was not assessed.
