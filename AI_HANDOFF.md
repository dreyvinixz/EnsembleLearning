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
