# Experiments

These files preserve earlier research and alternative approaches. The supported
workflow is documented in the root [README](../README.md).

- `legacy_cyp/`: the original random/scaffold split comparison, model ladder,
  retained CSV outputs and pharmacogenomics probe notes. These scripts train on
  execution and write results into the working directory. If revisiting them,
  use a fresh scratch directory so the retained CSV evidence is not overwritten.
  PyTDC's default data directory is also relative to that working directory.
- `legacy_ord/`: procedure-quantity audit and reaction inspection. Each reads the
  Parquet file supplied as its first argument; it does not depend on its old folder.
  The optional ORD dependencies are not part of the supported CYP workflow.
- `kiba/incomplete.ipynb`: unfinished KIBA exploration, preserved without changing
  cells or outputs. It has shell syntax errors, use-before-definition, missing
  embeddings and inconsistent split names; it is not a runnable pipeline. Its
  original working directory was `notebooks/`, whose `data/` cache stays in place.
- `jev_router.py`: isolated routing evaluation. The workbench uses one assistant
  and does not call this router before admitting a request.

No data downloads, model training or notebook execution were performed for the
directory migration.
