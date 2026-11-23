# AI handoff — 2026-11-23

Goal: one journal research project combining multiscale B0 label quality with
Independent Learning, MTL, Voting, Bagging, Stacking, Boosting and (conditional)
disagreement-aware adaptive ensemble (DAME). GRU independent/MTL are candidate
encoders. Related Work is planned instead of a standalone SLR.

Current execution layer: provisional binary CSV contract, conservative chronological
session holdout, validation-only independent logistic benchmark, optional untrained
PyTorch GRU architecture. No BovDB files or real experimental outputs exist here.

Open scientific blockers:
- Confirm actual B0 labels (binary vs three-class), pivot and confirmation timing.
- Confirm decision timestamp relative to candle close and causal feature availability.
- Obtain licensed/readable snapshot, ground truth protocol, and data provenance.
- Agree on folds, inference unit, primary metric, test lock, and regimes with advisors.
- Evaluate negative transfer BEFORE designing/training DAME.

Next implementation: a read-only BovDB adapter that produces a reviewed snapshot
plus independent label audit, then a multi-fold walk-forward runner. Run pytest on
every change; do not push results invented from synthetic fixtures.

## Infrastructure update — 2026-11-23

- The intended test host is `ginfolab-maquinario` (Linux), accessed as `ginfolab` over Tailscale. The project clone is `/home/ginfolab/EnsembleLearning`.
- The Linux clone was initialized at commit `2971f72`; this local checkout is at merge commit `bcebe16`. The clone has not yet been synchronized with the newer implementation. Its untracked `AGENTS.md`, `docs/ambiente-linux.md`, and `environment-linux.freeze.txt` must be preserved when reconciling it.
- Two local SQLite files were copied into the Linux clone's `data/`: `bovdb-labels-B4-v1.sqlite` and `bovdb-v3.1-internal.2.sqlite`. SHA-256 values matched the source copies and SQLite `PRAGMA quick_check` returned `ok` for both. This verifies transfer and basic integrity, not provenance, label semantics, suitability, or scientific validity. The earlier statement that no BovDB files exist here no longer applies to these local and remote data directories; an audited snapshot is still pending.
- The server has an NVIDIA GeForce RTX 5060 Ti (16 GB), driver 595.99.02 and Python 3.12. Its project `.venv` contains PyTorch 2.11.0+cu128, NumPy, pandas, SciPy and scikit-learn. CUDA availability and a GPU matrix multiplication were verified; `pip check` passed. `environment-linux.freeze.txt` captures installed versions. The CUDA compiler `nvcc` was not installed.
- `docs/ambiente-linux.md` is an untracked setup note present locally and on the server. No real-data training or research results were produced during setup. Existing experimental code and tests are described in `docs/implementation.md`; the server has not yet been validated against the newer checkout.
- Next: reconcile the Linux clone with the current branch without overwriting its untracked files or databases; verify dependencies and tests, then audit the source and label availability before preparing any training snapshot. Do not run the reserved test set during debugging.
