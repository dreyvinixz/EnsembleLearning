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
| #12 | Audit existing runs, paired statistics, Voting, seed ensemble, temporal block bagging and OOF stacking | Merged to main (`528b74f`); development experiments executed, exploratory |
| #12.1 | Neural diagnostics, complementarity, feature/contract ablations, statistical correction and external-benchmark plan | Active on `research/stage12.1-diagnostics`; figures and descriptive complementarity complete; feature ablation implemented but not run; Linux access unavailable |
| #13 | Reproduce/adapt selected open-source benchmarks after protocol and license review | Specification only in `docs/stage13-external-benchmarks.md`; no model implemented |
| Final evaluation | Freeze protocol after development and May validation; train permitted final models and evaluate June once | Reserved; must not be used for debugging, tuning or selection |

## Reexecução para o universo de um contrato mensal

O roster desta revisão é WING26 (jan), WINJ26 (fev–mar), WINM26 (abr–mai) e
WINQ26 (jun). Junho permanece fora do novo snapshot; o CSV foi cortado em
31/05/2026. Janeiro foi fixado pelo responsável e não possui evidência local de
cobertura de dezembro para validar o corte de 25%.

Próximos passos: reexecutar baselines clássicos, Independent e MTL no novo CSV
com os mesmos folds/âncoras e seeds; depois, reexecutar os métodos Stage 12
(Voting/seed averaging, block bagging, stacking OOF) sob a mesma população e
orçamento comparável. Parear previsões com o legado por contrato, pregão,
instante e escala e apresentar separadamente cobertura e suporte. O comparativo
atual registra apenas a variação de população (63.517 para 44.925 âncoras
conjuntas até maio), não desempenho de modelos. MCC não seleciona contratos.

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

The current 21 features are grouped into returns (5), candle/price shape (4),
volatility (4), close-versus-SMA trend (4) and volume/activity (4). The Stage
12.1 ablation runner compares full, group-only and leave-one-group-out sets
while asserting identical anchor eligibility. Actual model ablation runs remain
pending; protocol and budget are in `docs/experimental-budget-stage12.1.md`.

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

Figuras de MCC e acurácia na mesma população OOF março–abril e validação de maio estão em `figures/stage12-mcc-2026-11-23.png` e `figures/stage12-accuracy-2026-11-23.png`, com PDFs vetoriais e dados/manifesto em `reports/stage12/paired-metrics-2026-11-23/`.
