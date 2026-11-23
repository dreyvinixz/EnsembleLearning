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

## BovDB source repository — 2026-11-23

- Authorized read access verified through GitHub CLI to the private repository `GInfoFinanceLab/bovdb-pipeline`, default branch `main`, observed commit `0a79b64292c8856349d5aaa38f68644f7010a211`. Access was read-only; no clone or changes were made to that repository.
- `rotulos/README.md` describes B4 as separate SQLite output containing B0 binary trends in `trend_5m`, `trend_10m`, and `trend_15m`, a derived `trend_mean`, and `label_run` provenance. It states SIDEWAYS features use centered windows and are ex-post, and that the mean is derived from the three labels. This is the repository's description; the local database's provenance and correspondence to this commit remain unverified.
- Next inspect the actual B0/B4 generator and database `label_run`, timestamps, label confirmation and source hash before constructing any causal dataset. Do not use SIDEWAYS or `trend_mean` as causal predictors or treat derived consensus as independent ground truth. The binary benchmark contract is compatible with the README's described trend classes, but still needs verification against the actual database and generator.

## BovDB provenance and label audit — 2026-11-23

- Read-only audit is documented in `docs/auditoria-bovdb-inicial.md`. B4 `label_run.source_sha256` matches the actual source SQLite hash; both contain 1,255,202 source/label rows, 41 tickers and 119 dates. Counts do not establish key-level correspondence.
- The generator commit recorded in the database (`7c224cd807041ba0384b908a97a2696bae754254`) has author/committer time 2026-10-26 13:50:18 UTC, later than the database's recorded generation time 13:33:15.993570 UTC. Provenance is therefore unresolved; do not treat the commit as proof of the exact executed tree. The B4 builder and B3 labeling implementation blobs match between that commit and the inspected main commit, but other dependencies were not fully checked.
- Inspected code uses future macro candles and retrospective pivots. B4 has no proven confirmation timestamp; `segment_end_time` is a pivot timestamp, not confirmation. Never fabricate `label_known_at` from that field or horizon. SIDEWAYS is centered/ex-post. Null counts for 5/10/15-minute labels are 11,270 / 15,402 / 20,588 respectively. Further work must audit causal timing, keys, and provenance before training.

## B0 label availability rule — 2026-11-23

- `docs/disponibilidade-rotulos.md` defines a conservative per-label policy: use the entire input session as the information interval and release non-null labels only after all session candles are available and processing completes. This is an upper-bound policy, not an observed first confirmation timestamp. Apply separately per asset/session, and require all three task labels for the current joint contract.
- The recorded B4 artifact was generated on 2026-10-26, not on each historical trading day. A historical walk-forward must explicitly state its simulated source/processing latency and vintage assumptions; original receipt, candle close semantics, timezone, completeness and actual job completion per session are not recorded. No exact `label_known_at` values were fabricated and no training snapshot was created.
- Read-only database query found 3,330 asset-sessions, with only 292 ending at timestamp 18:00; early final observations do not prove complete sessions. WINM26 on 2026-04-29 ranges from 09:02 to 18:00. An adapter and boundary tests remain necessary before training; session-close policy also depends on resolving the recorded generator-version inconsistency and validating that session-local computation matches the actual run.

## Study period decision — 2026-11-23

- The research owner selected the available six-month period, 2026-01-02 through 2026-06-30, for the study. The aggregate source has 119 dates, 41 tickers and 1,255,202 one-minute candles; this does not imply complete coverage for each ticker. Asset inclusion, chronological folds and the held-out test period remain to be defined. Apply the conservative session-level label availability policy; no historical confirmation times have been established.

## Mini-index universe and split — 2026-11-23

- User chose only mini-index contracts and the two most liquid each month. `docs/protocolo-mini-indice.md` records an initial causal operationalization: prior-month liquidity ranking among contracts eligible at the beginning of the target month, fixed before the month, with no retroactive replacement. Confirm the source `volume` units and expiry/calendar. January has no previous month in the supplied snapshot; obtain prior information or exclude January from predictive comparisons that require this selection. Monthly full-period rankings in the document are descriptive, not causal selections.
- Initial coverage thresholds: at least 80% of eligible sessions per contract-month and 90% jointly non-null 5/10/15 labels among matched source candles, with reporting of gaps and session completeness. These are proposed thresholds, not empirically validated; expiration and no-trade periods must not be treated as missing data automatically.
- Proposed chronological blocks: Jan–Apr training (January conditional), May validation, June final test, with session-level separation, conservative label availability and development-only walk-forward. No model evaluation or final test executed.
