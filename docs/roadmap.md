# Research implementation roadmap

## Scientific objective

Train supervised models to classify retrospective B0 UPTREND/DOWNTREND labels
at 5, 10 and 15 minutes from causal candles and indicators. B0 labels are
targets and later evaluation references; they are never model inputs. This
classification objective does not by itself predict exact future returns or
establish trading profitability. See `data-contract.md`, `protocol.md`,
`disponibilidade-rotulos.md` and `auditoria-gerador-b0.md` for data and timing
constraints.

## Delivery stages

| Stage | Scope | State |
|---|---|---|
| #10 | Expanding walk-forward, classical baseline, shared metrics, immutable tracking and individual predictions | Implemented; exploratory Logistic Regression run on Linux; June untouched |
| #11 | MLP/CNN/LSTM/GRU, independent per-scale and shared three-head MTL, causal sequences, early stopping, checkpoints | Integrated in PR #11; exact population/y_true pairing, full purged-block refit, CI green, and 16 post-refit GPU runs audited |
| #12 | Audit existing runs, paired statistics, Voting, seed ensemble, temporal block bagging and OOF stacking | Implemented and executed on Linux development data; PR/CI review pending; all comparisons exploratory |
| Later | B0 label-quality analysis, regime/label disagreement, calibration, paired statistics; assess adaptive DAME only if evidence supports it | Not implemented as a complete analysis |
| Final evaluation | Freeze protocol after development and May validation; train permitted final models and evaluate June once | Reserved; must not be used for debugging, tuning or selection |

## Current split and model matrix

January initializes training. Folds 01–03 produce OOF predictions for February,
March and April. Fold 04 trains through April and evaluates May as external
development validation. The current neural runner supports 4 architectures ×
2 strategies, one seed by default. Independent fits one model per horizon;
MTL fits one shared model with three heads. The full matrix is 64 fits for one
seed across four folds: 12 independent plus 4 MTL fits per architecture.

The sequence contract is a causal 60-clock-minute window, at least 48 observed
candles, explicit missing-candle mask, no crossing sessions/contracts, and
first decision at 10:00 local. The snapshot's simulated post-session label
release is conservative but not historically observed. Any change to candle
availability, label intervals, feature generation or splits requires targeted
tests and an updated source manifest.

## Required gates

1. Pass CPU CI and targeted boundary/leakage tests for each code change.
2. On the Linux RTX host, verify CUDA, record hardware/software, benchmark
   FP32/AMP, TF32, worker/pinned transfers, batch sizes and optional compilation
   on development-only data. Report startup and steady-state time separately.
3. Compare predictions and metrics against the FP32 reference; document numeric
   tolerances before treating an optimization as the standard configuration.
4. Record every run's code/data hashes, parameters, seed, device, runtime,
   memory, checkpoints, history, predictions and metrics without overwriting.
5. Keep June inaccessible until the protocol, model selection and ensemble
   choices are frozen. Then execute its evaluation once and retain its full
   manifest and predictions.

## Known non-goals for the current neural stage

No claim of DAME novelty, label truth, exact future-price prediction, or PnL is
made. Trading metrics require a separate pre-specified execution and cost model.
Do not invent missing candles, label-known timestamps or training results.

A matriz pós-refit e o bootstrap pareado estão em `reports/stage12/refit-2026-11-23/`; a auditoria antiga sem refit permanece em `reports/stage12/preliminary-2026-11-23/` somente como histórico. Os 16 novos runs usam o mesmo snapshot e têm as mesmas 170.100 chaves e classes que o baseline. Resultados de Voting/Seed Ensemble, Block Bagging e Stacking estão em `reports/stage12/averaging-2026-11-23/`, `reports/stage12/block-bagging-2026-11-23/` e `reports/stage12/stacking-2026-11-23/`; os artefatos versionados contêm agregados e manifestos, sem previsões privadas completas.

Os ensembles usam probabilidades pós-refit: pesos uniformes em Voting/Seed Ensemble; metamodelo cronológico treinado apenas em folds OOF anteriores para Stacking; Logistic Regression com blocos móveis de cinco sessões para Block Bagging. Maio é somente validação e junho não foi acessado. ICs pareados usam 5.000 bootstraps por sessão sem correção de multiplicidade; resultados são exploratórios. Ver `docs/protocol.md` e os relatórios por método.
