# AI handoff — 2026-11-23

## Integração de arquiteturas — 2026-11-23

- PR #8, branch `feat/integrate-architectures`, parte da main após merge do PR #7. Integra seletivamente regressão logística, random forest, hist gradient boosting, XGBoost e arquiteturas MLP/CNN/LSTM/GRU independentes e multitarefa. O CLI seleciona modelos tabulares e preserva o split por datas. XGBoost foi incluído nas dependências.
- GitHub Actions Linux executou pytest/Ruff e job PyTorch CPU com testes de forward e gradientes, usando dados sintéticos. Não houve treinamento real nem avaliação do teste final. Arquiteturas neurais ainda não possuem gerador causal de sequências nem treinador. Próximo passo: implementá-los e verificar disponibilidade de dados antes de qualquer treino.

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

## QuantBase ClickHouse December inspection — 2026-11-23

- User authorized read-only inspection of `quantbase-clickhouse-dev` container. ClickHouse queries used `--readonly 2`. December 2025 exists in `candles_1d`, but searches for symbols beginning WIN returned zero rows for December in `candles_1d`, `candles_1s`, and `asset_historical_registry`.
- `candles_1s` has WIN history outside December: WING26, WINJ26 and WINZ25 in October and five November dates; WING26 and WINJ26 in January. This instance cannot currently establish the requested December WIN ranking from those tables. Other stores, snapshots or symbol conventions remain possible; no writes or exports performed.

## Public B3 source for December — 2026-11-23

- Found B3 daily BDI documents with WIN volume and contract counts by expiration, for example https://arquivos.b3.com.br/bdi/download/bdi/2025-12-02/BDI_03-1_20251202.pdf. These can support reconstruction of the December ranking without the missing local database. No complete monthly aggregation or selected pair is yet verified. Archive source links, hashes, sessions and extraction method; filter December-expired contracts before selecting January's eligible pair.

## B3 December ranking extraction — 2026-11-23

- Extracted official B3 BDI data for 20 December 2025 sessions, with URLs, page numbers and PDF SHA-256, into ignored local `data/b3-win-dezembro-2025.csv` (60 rows; CSV SHA-256 b1114cfe81b7f0b523f7295852496a38401d92a203aa5525bf68969933047a37). Read `data/b3-win-dezembro-2025-LEIA-ME.md`; reproduce with `scripts/extrair-bdi-dezembro-2025.py` without overwriting existing output.
- Aggregated published BDI volume: WING26 3,938,357,648,152; WINJ26 4,222,235,450; WINM26 1,033,256,290. Top two among these three: WING26 and WINJ26. PDF extraction was spot-checked; other eligible maturities have not been comprehensively checked, '-' cells were normalized as zero, and BDI volume units versus BovDB require verification. Selection remains provisional pending those checks. December data is for January universe selection, not training.

## B3 WIN universe and volume comparison — 2026-11-23

- Full BDI extraction covers nine detected WIN maturities from 2026 onward across 20 December sessions; normalized 180-row `data/b3_win/b3-win-dezembro-2025-todos.csv`, SHA-256 eb544c656d220e017cd565f860f3f1fe0b339a95fca4395f87c2fd0ec9227159. Six maturities beyond WING26/WINJ26/WINM26 have no numeric volume published (`-` stays null). All 60 previous rows match after accounting for null versus zero. The leading published values are WING26 then WINJ26. Code: `scripts/auditar-bdi-todos-vencimentos.py`; evidence and limitations: `docs/auditoria-liquidez-b3-bovdb.md`.
- Local BovDB checkout `8b9ff1a8a59cea5e9437a16bcfef1f6adbeb0620` computes minute volume as sum(price * quantity); B3 WIN multiplier is 0.20 BRL per point. Theoretical conversion is x0.20, but a four-case comparison for Jan 2 and Jan 5 shows adjusted BovDB/BDI daily ratios 0.8318–0.9863 and quantities differ. Cause unverified; cannot claim equality or combine values without reconciliation. Normalized comparison: `data/b3_win/comparacao-bdi-bovdb-janeiro-2026.csv`.
- Data were reorganized under `data/db/` and `data/b3_win/` since prior handoff entries; preserve these paths. No datasets or source databases were changed by this audit.

## January contracts selected — 2026-11-23

- User explicitly decided to proceed with WING26 and WINJ26 for January 2026. This supersedes earlier provisional-selection wording for January only. Monthly selection for later months remains to be determined under the prior-month policy.
- Read-only B4 cross-scale key intersection in January: WING26 21 dates, 11,126 common rows, 11,046 with three non-null labels (99.28%); WINJ26 21 dates, 8,677 common rows, 8,598 with three non-null labels (99.09%). These are label-table intersection denominators, not a verified complete reconciliation to price1 or proof of session completeness. No training snapshot or model run was created. Preserve user-selected contracts while auditing remaining coverage and causal availability.

## Monthly WIN contract coverage figure — 2026-11-23

- User requested the start and end of both monthly contracts across all six months. Created `figures/contratos-win-seis-meses-2026.png` and PDF, plus per-session CSV and `scripts/figura-selecao-mensal-win.py`, based on read-only `price1` observations. Each dot is an observed session, not evidence of complete intraday coverage or proof of expiration.
- January WING26/WINJ26 is user-selected. February WING26/WINJ26; March and April WINJ26/WINM26; May and June WINM26/WINQ26 are preliminary illustrations based on prior-month volume and excluding expired contracts. Only January was explicitly selected; remaining assignments, eligibility and liquidity metric require validation. No models were trained.

## Contract count and data gaps — 2026-11-23

- User asked whether visible gaps justify three contracts. Read-only `price1` date enumeration shows three absent May sessions across the entire source table: May 13, 21 and 25. BovDB `release.py` states these dates have no retained source. A third symbol in the same database cannot fill a date when no `price1` rows exist for any instrument. This is a source-coverage issue; some other gaps are weekends/holidays or contract end dates and require calendar classification.
- Recommendation: maintain two eligible contracts selected using prior information; separately audit exchange sessions and per-contract intraday completeness, report missing dates, seek authorized independent source if available, and never synthesize missing candles. If a rolling replacement after expiry is desired, define it prospectively and treat it as a distinct methodology; the user has not selected that approach.

## Approved contract replacement and calendar — 2026-11-23

- User approved maintaining up to two WIN contracts, with a prospective replacement after expiry. `docs/protocolo-mini-indice.md` specifies the first trading session after expiration, preserving the other contract and choosing the next eligible candidate from a ranking fixed using prior information. This supersedes the previous no-replacement rule. Concrete replacement contracts, dates and eligibility must still be calculated and validated; do not present the earlier no-replacement chart as implementing this decision.
- Added `figures/calendario-b3-bovdb-jan-jun-2026.png`, PDF and CSV with script `scripts/figura-calendario-b3.py`; previous figures preserved. B3 calendar verifies closed weekdays and special session on Feb 18. The figure marks May 13, 21, 25 as no `price1` records, weekends, B3 closures, and days with at least one WIN candle; it does not prove intraday completeness or coverage of both selected contracts.

## Regular trading hours — user decision 2026-11-23

- User selected Monday through Friday, regular B3 sessions only, 09:00–18:00 exchange-local time; exclude weekends, B3 holidays and special-hour sessions, including 2026-02-18. Missing regular sessions (May 13, 21, 25) are missing data, not holidays. Existing coverage figures preserve observed data and do not represent the final filter.
- Confirm timezone and whether `price1.time` identifies candle start or end before implementing the exact 18:00 endpoint and feature-availability rules. No database or training snapshot was modified, and the filter has not yet been applied.

## Candle timing and intraday coverage — 2026-11-23

- BovDB v3.1 data dictionary defines `price1.time` as the start of each one-minute interval in B3 local time; inspected parser truncates seconds and groups trades by minute. For 09:00–18:00 use full minute starts 09:00 through 17:59; `18:00` is the beginning of the next minute, and full OHLC is not available at 18:00. Historical B4 label generation included bars timestamped 18:00; restricting predictors does not remove label dependencies. Exact executed-tree provenance remains qualified.
- `scripts/auditar-cobertura-intradiaria.py` generated read-only local `data/b3_win/auditoria-cobertura-win-2026.csv`, 377 observed contract-days, 540 expected clock minutes per regular session, gaps and hours; excludes special/closed days. It contains no rows for entirely absent contract-days, and missing minutes do not by themselves prove missing source data. `docs/auditoria-cobertura-intradiaria.md` records methods and limitations.
- January WING26: 21 sessions, median 538 minutes; WINJ26: 21 sessions, median 408. A 90% (486-minute) diagnostic would leave 20/21 WING26 sessions and only 5/21 WINJ26; this threshold is NOT approved. WINQ26 in February has only 3–14 minutes per observed session. Select contracts using previously available liquidity and coverage; audit retrospective coverage separately and apply causal intraday no-trade/abstention rules without future lookahead. No training snapshot created.

## Three contracts and causal recent coverage — 2026-11-23

- User chose an experimental universe of up to three WIN contracts, replacing the previous cap of two. Keep prior-information selection/replacement and regular-session rules. WING26 and WINJ26 remain chosen for January; WINM26 is third by reported December BDI volume but December historical intraday coverage under the new rule is unverified. No three-contract roster or coverage threshold for historical selection is established.
- Added `src/ensemble_learning/coverage.py` and `tests/test_coverage.py`: timezone-aware minute-aligned 60-minute same-session window, 80% (48) closed bars by default, optional 90% sensitivity (54), optional longest-gap cap (value not chosen). Future/incomplete bars do not count; no previous-session history is borrowed. This is a standalone check, not wired into training or the data adapter; no real-data predictions or TEST evaluation performed.
- Global Python lacked scikit-learn, so the initial full pytest collection failed. An isolated local `.venv` is being prepared for complete validation; report actual final results.
- Validation completed in isolated Windows `.venv`: `pytest -q` passed 13 tests with 1 skipped (optional dependency), and Ruff passed for the new coverage module/tests. The earlier global-environment collection failure was resolved in this isolated environment.

## January third contract approved — 2026-11-23

- User explicitly approved WINM26 as the third January 2026 contract, alongside WING26 and WINJ26. This overrides the earlier pending-candidate status. Approval does not waive causal coverage checks or establish a maximum gap; later monthly rosters and replacement candidates remain to be validated. No training performed.

## Three-contract calendar figure — 2026-11-23

- Created `figures/contratos-win-tres-calendario-2026.png`, PDF and per-day CSV using `scripts/figura-tres-contratos-calendario.py`; earlier figures remain untouched. Colors identify weekend, B3 closure, excluded special session, missing price1 date, absent contract candle and post-expiry periods. Dots represent at least one regular-hours candle, not meeting 80% recent coverage.
- January WING26/WINJ26/WINM26 is user-approved. Subsequent months depict illustrative candidates, not final liquidity/coverage selections; approved rollover is not yet calculated, and post-expiry areas are explicitly marked. The figure is an availability view, not a result of the model or the causal coverage gate.

## Figure with causal 80% coverage — 2026-11-23

- User corrected the preceding request: the figure must use the 80% recent-window criterion, not merely the presence of a candle. Generated `figures/contratos-win-80porcento-2026.png`, PDF, and `figures/contratos-win-80porcento-dados.csv` with `scripts/figura-tres-contratos-80.py`; previous figures preserved.
- A dot means at least one decision minute from 10:00 through 18:00 has 48 or more completed 1-minute bars in its preceding 60 minutes, within 09:00–18:00. CSV records the number of qualifying decision windows for each ordinary contract-day. No maximum gap is applied because no threshold was selected. A dot does not imply the whole day qualifies. Calendar exclusions and source gaps remain marked. January contracts approved; later rosters preliminary and no replacement yet calculated. Spot-checked against `recent_coverage` for sampled sessions over all 481 decision minutes.
- WINM26 has no qualifying windows in January. The figure is retrospective descriptive coverage, not a model run or evidence of prediction performance. No train/test data selection or model evaluation was performed.

## 80% coverage figure contrast — 2026-11-23

- Increased contrast of `figures/contratos-win-80porcento-2026.png` and PDF and added distinct hatch patterns for B3 closures, special sessions, missing data, no qualifying window, and post-expiration; updated the figure script and visually inspected the generated PNG. Coverage calculation and interpretation are unchanged.

## WIN roster with 50% prior-session coverage — 2026-11-23

- User explicitly selected minimum coverage of 50% of eligible prior-month regular sessions, counting eligible days with no candle. Operational definition: a qualifying prior day has at least one causal 60-minute window with >=48 closed candles; candidates must have known positive prior-month BovDB volume and no null volume among observed candles. Rank within source by volume, then trades and symbol; at most three; replacement after expiry only from the ranking frozen before the month. No max-gap threshold is yet set.
- `scripts/evidencia-selecao-win.py` and `scripts/selecionar-win-cobertura.py` generated ignored evidence, candidate ranking, daily roster and manifest in `data/b3_win/`; `docs/selecao-win-cobertura-50.md` contains outputs and caveats. January is the user's manual WING26/WINJ26/WINM26 exception. Diagnostic roster: February WING26/WINJ26 then WINJ26 after expiry; March WINJ26 only; April WINJ26/WINM26 then WINM26; May WINM26/WINQ26; June WINM26/WINQ26 then WINQ26.
- The denominator conservatively uses all regular prior-month sessions through expiry; actual first listing dates were not verified. Three source-missing May sessions count against June coverage. Historical vintage and availability remain assumed, not proven; one qualifying window does not guarantee full-session suitability. No training dataset, model evaluation or TEST performance was generated. Older figures show candidate rosters and do not implement the new schedule. 363 daily slots were validated for prior-month evidence, 50% and expiration.

## January roster corrected — 2026-11-23

- User explicitly removed WINM26 from January after observing it has no qualifying 80% windows; January roster is WING26 and WINJ26 only. The decision overrides the earlier approval of a third January contract.
- Generated versioned v2 criteria, daily schedule and manifest under `data/b3_win/selecao-win-50-v2-*` via `scripts/selecionar-win-cobertura-v2.py`, preserving v1. Verified only January slot 3 changed and all hashes match. Updated `docs/selecao-win-cobertura-50.md` and protocol. This is a retrospective user decision using January coverage; it is not proof of ex-ante selection, although the causal per-window gate would reject every WINM26 January prediction regardless. December historical coverage remains unverified. No training dataset or model run.

## Figura da programação v2 e janelas de 80% — 2026-11-23

- Gerada `figures/contratos-win-selecao50-janela80-2026.png` e PDF, mais CSV e `scripts/figura-selecao50-janela80.py`. A figura combina a programação v2 por pregão com a auditoria anterior de cobertura; WING26 e WINJ26 são os únicos de janeiro, e períodos após vencimento aparecem diferenciados. As figuras anteriores permanecem preservadas.
- Cada ponto indica pelo menos uma janela de 60 minutos com 48 candles fechados, não o pregão inteiro. A seleção de janeiro é manual e retrospectiva; a de fevereiro a junho é diagnóstica, com listagem histórica e disponibilidade ainda por validar. Conferido que todos os contratos da programação aparecem no CSV da figura e WINM26 não aparece em janeiro.


## Seleção mensal recalculada com corte de 25% — 2026-11-23

- Usuário autorizou incluir WINM26 em março e adotar 25% de pregões com pelo menos uma janela causal de 60 minutos com 48 candles fechados, usando o mês anterior. 25% é mais permissivo que o corte anterior de 50%. Fevereiro tem 17 pregões elegíveis; WINM26 passou em 7 (41,2%), acima do mínimo inteiro de 5. Foi o segundo candidato aprovado por volume BovDB para março.
- Recalculada a seleção inteira de fevereiro a junho em arquivos v3, preservando todos os resultados v2 de 50%: `data/b3_win/selecao-win-25-v3-criterios.csv`, `...-por-pregao.csv`, `...-manifest.json`; script `scripts/selecionar-win-cobertura-v3.py`. Janeiro segue manual com WING26/WINJ26. A única composição mensal alterada foi março, que agora tem WINJ26/WINM26; demais rosters mantiveram-se. Conferido que todas as sessões de março no cronograma trazem ambos os contratos.
- Nova figura e dados: `figures/contratos-win-selecao25-janela80-2026.png`, PDF e CSV, via `scripts/figura-selecao25-janela80.py`; figura inspecionada visualmente. O Matplotlib não está no `.venv` do projeto, então a figura foi renderizada com o Python global que possui a dependência; nenhuma dependência do projeto foi alterada. Documentação: `docs/selecao-win-cobertura-25.md` e atualização aditiva de `docs/protocolo-mini-indice.md`.
- Limitações mantidas: denominador inclui datas sem candles e não confirma o início de listagem; histórico/vintage e liquidez BovDB precisam de validação; 25% mensal não garante qualidade no instante da previsão; nenhuma base de treino/modelo foi criado/executado.

## Base inicial de mini-índice em Parquet — 2026-11-23

- Gerada base local `data/processed/win_2026_v1/` da seleção v3 de 25%, com `market_candles.parquet` (105.319 candles), `b0_targets_retrospective.parquet` (rótulos alinhados, nulos preservados), `session_coverage.csv` (216 contrato-pregões) e manifesto com hashes SHA-256. Fontes SQLite foram abertas como `mode=ro&immutable=1`; não foram alteradas.
- Universo efetivo: WING26/WINJ26 em jan-fev; WINJ26/WINM26 em mar-abr (WINJ até o vencimento); WINM26/WINQ26 em mai-jun (WINM até o vencimento). Apenas candles iniciados entre 09:00 e 17:59, em dias regulares da programação; nenhum candle sintético. Seis contrato-pregões ficaram sem fonte: WINM26 e WINQ26 em 13, 21 e 25/05. 210/216 sessões têm ao menos uma janela causal de 80%; 89.347 instantes de decisão passam no gate.
- 105.078/104.753/104.609 alvos B0 não nulos para 5/10/15m; 104.387 linhas têm as três escalas não nulas. Labels são retrospectivos. B4 não registra tempo de conhecimento nem intervalos informacionais individuais; não foi criado `label_known_at`. Por isso o artefato é base analítica/feature development, ainda não um snapshot pronto para treinamento causal. `trend_mean`, SIDEWAYS e `segment_end_time` foram excluídos.
- Sem indicadores técnicos ou imputação nesta etapa. Relatório de conteúdo/uso: `docs/base-win-parquet.md`; reprodução: `scripts/build_win_parquet.py` (somente leitura das fontes; saída não sobrescrita). Validação estrutural confirmou chaves únicas, correspondência de chaves entre candles e alvos, contratos/período conforme cronograma e exclusão de candles 18:00+. Não foram executados testes de modelos nem treinamento.

## Auditoria do gerador B0 e liberação dos alvos — 2026-11-23

- Inspecionados `rotulos/scripts/build_b4_experimental.py` e `generate_chart_b3_regime.py` do checkout BovDB local, sem modificá-lo. O B4 gera cada ativo/data separadamente, usa apenas candles dessa sessão entre 09:00 e 18:00 inclusive, agrega 5/10/15m, identifica sequências de pivôs e atribui cada classe até o pivô seguinte; depois do último, o alvo é nulo. `segment_end_time` é a fronteira de pivô, não sua confirmação.
- Os blobs dos dois arquivos no commit B4 registrado `7c224cd807041ba0384b908a97a2696bae754254` coincidem com o checkout atual `8b9ff1a8a59cea5e9437a16bcfef1f6adbeb0620`. Reexecução somente leitura do algoritmo atual em 216 contrato-pregões selecionados reproduziu 105.495 linhas por escala e todos os rótulos, fronteiras e tipos, sem chaves ausentes ou divergências. A hora B4 gerada em 13:33 UTC antecede a hora do commit às 13:50 UTC em 17 minutos; a identidade dos dois arquivos não resolve a proveniência do restante do ambiente nem vintages da fonte.
- Janela conservadora adotada: desde 09:00 até a conclusão/disponibilidade de todos os candles utilizados na sessão. O builder inclui o candle com início às 18:00; esse minuto só fecha às 18:01. Liberar qualquer uma das três escalas após a fonte declarar a sessão finalizada e o job B0 concluir; não usar alvos da sessão no treino intrassessão. Aplicar split e purge por sessão inteira para todos os contratos e tarefas. Máximo de distância entre a linha rotulada e `segment_end_time`: 77/129/203 min (5/10/15m), que não mede atraso de confirmação.
- O B4 não contém `label_known_at`, watermark de ingestão nem tempo de conclusão por sessão; o artefato foi gerado em lote em outubro de 2026. A liberação pós-sessão é uma hipótese de simulação, não uma observação histórica. Não foram inseridos horários inventados na base Parquet nem executado treino/teste. Evidência em `docs/auditoria-gerador-b0.md`; o relatório de disponibilidade foi atualizado.

## Etapa #10 — walk-forward e tracking concluídos — 2026-11-23

- Branch `feat/walk-forward-tracking`, HEAD `daecb0af7ee8f3e015abe31ce80786d2de3769ad`. O host Linux `ginfolab-maquinario` foi sincronizado; CUDA 12.8/PyTorch 2.11 estavam disponíveis (`torch.cuda.is_available() == True`), mas o baseline Logistic Regression usa CPU.
- O primeiro ensaio foi interrompido antes de publicar artefatos porque o bootstrap copiava repetidamente grandes tabelas de linhas. Implementado bootstrap equivalente somando contagens de confusão por sessão; teste compara o IC com reamostragem explícita. O snapshot tinha offset ISO 8601 básico (`-0300`), válido, mas recusado pelo validador; regex agora aceita formatos básico e estendido, mantendo obrigatoriedade do fuso explícito.
- Testes no Linux: `29 passed`, quatro avisos de depreciação NumPy preexistentes no pipeline de preparação; Ruff passou. Execução `logreg`, seed 42 concluiu em 8,90 s e gerou `/home/ginfolab/EnsembleLearning/results/EXP_F776BE93EB` (3,61 MB), 174.189 previsões, 12 checkpoints (4 folds × 3 horizontes), métricas, configuração e manifesto. SHA-256 da fonte: `410ccbf212204b94d4705b83ec6813974e0bd2460a26deaa4aec84fed1826219`.
- Fold01/02/03 produzem previsões OOF de fevereiro/ março/abril; fold04 é validação externa de maio, agora separada das métricas agregadas OOF. Previsões terminam em 2026-05-29; assert de integridade confirmou três escalas, quatro folds, chave de amostra sem duplicatas e `test_june_accessed=false`. Junho não foi lido pelo runner nem avaliado.
- Métricas calculadas incluem MCC provisório principal, Accuracy, Balanced Accuracy, medidas por classe e Macro-F1, ROC-AUC, Average Precision, Brier, Log Loss, ECE 10 bins, matriz de confusão e IC diagnóstico de MCC por bootstrap de sessões. Exemplos de MCC (OOF fev–abr / validação maio): 5m 0,525/0,526; 10m 0,476/0,493; 15m 0,464/0,465. São resultados exploratórios do baseline; não demonstram melhoria nem justificam seleção final.
- Pendências antes de comparação confirmatória: calibração, métricas por regime, selective risk/coverage/AURC, diferenças pareadas e ajuste por múltiplas comparações, geração automática das figuras e decisão congelada de métricas. Early stopping e busca de hiperparâmetros não foram necessários para Logistic Regression; serão definidos por modelo nas próximas etapas.
- O artefato fica somente no Linux e em `results/` ignorado; nenhum dado privado, checkpoint ou previsão foi adicionado ao Git. Um ensaio anterior com métricas OOF e maio agrupadas foi preservado imutavelmente como `EXP_BF7ED064C5`, mas substituído pela execução `EXP_F776BE93EB`, que reporta esses conjuntos separadamente. O diretório temporário incompleto da tentativa lenta foi removido após conferir o caminho.
- Próximo: abrir/revisar PR da Etapa #10; depois iniciar Etapa #11 usando este contrato de previsão/artefatos. Não executar junho até congelar o protocolo final após Etapas #10–#12.

## Otimização e auditoria da Etapa #11 — 2026-11-23

- Treinamento experimental com dados continua pausado. Branch local: `feat/neural-training-stage11`, com a correção do CI no commit mais recente. No início desta investigação, a branch remota ainda apontava para `77e47fd`; o clone Linux continha a implementação em alterações não commitadas sobre essa base. O arquivo `environment-linux.freeze.txt` não rastreado foi preservado.
- Implementado: CUDA obrigatório por padrão na API/CLI (`--device cpu` explícito para testes), AMP configurável (`fp32`, `bf16`, `fp16` com GradScaler), AdamW fused automático em CUDA, TF32 opcional, DataLoader com `num_workers`, pinning automático em CUDA e cópias non-blocking, `torch.compile` opcional com pré-checagem de compilador. Batch size permanece configurável. O padrão científico é FP32, algoritmos determinísticos, zero workers, TF32 desligado e sem compile.
- Corrigido o manifesto de split para registrar `fit_start`/`fit_end` depois da purga interna; teste cronológico agora verifica o subconjunto real usado para ajustar pesos, não o limite externo do fold.
- Criados/atualizados `docs/roadmap.md`, `docs/protocol.md` e `docs/implementation.md`. `docs/roadmap.md` estava ausente antes desta tarefa. Auditoria: B0 é alvo retrospectivo por escala, não retorno exatamente em t+h; features são apenas `x_`; Independent treina um modelo por horizonte; MTL usa três saídas. Fold01–03 são OOF fev–abr, fold04 valida maio, junho permanece bloqueado. Executor final de junho, ensembles, boosting integrado, comparação estatística pareada e DAME seguem pendentes.
- Teste Linux direcionado após alterações: Ruff dos arquivos tocados passou; suíte total final: 48 testes passaram, 8 avisos de depreciação preexistentes. Ruff no repositório inteiro ainda retorna 16 achados em scripts históricos não tocados; não foram corrigidos aqui.
- CI no commit remoto anterior `77e47fd` falhou nos 8 casos parametrizados do treino neural somente porque o teste comparava `train_end` externo com o início da validação interna. Treino e artefatos completaram; a falha foi a asserção temporal. O teste e o sumário foram corrigidos para registrar/validar `fit_end` após a purga interna. A branch remota aguarda nova execução após push do commit local.
- RTX 5060 Ti confirmou CUDA. Smoke sintético do treinador concluiu FP32 e BF16+TF32+2 workers, ambos com 264 probabilidades finitas e `test_june_accessed=false`; 1,916 s e 2,144 s. FP16+GradScaler também concluiu com 264 probabilidades finitas em 2,433 s. São testes pequenos, não benchmark de throughput; não evidenciam aceleração e BF16/FP16 foram mais lentos neste fixture. AdamW fused foi utilizado.
- `torch.compile` falhou inicialmente porque o host não tem `cc` nem `gcc` e Triton não consegue compilar o backend. A opção agora falha com mensagem clara antes de iniciar a execução. Manter desligada até instalar e validar toolchain C no host.
- Adicionado `scripts/benchmark_neural_gpu.py`, que gera CSV/manifesto únicos sob `results/gpu-benchmarks/` usando somente tensores sintéticos. Matriz repetida na RTX 5060 Ti cobriu quatro arquiteturas, batches 128/256/512, FP32/BF16/FP16, TF32 e AdamW fused/unfused; outputs completos no Linux: `gpu-bench-20261123T165833Z-3183db0c`. Nos casos medidos, fused ficou cerca de 1,3–1,5× acima do unfused; BF16/FP16 tiveram throughput inferior ao FP32, TF32 ficou quase igual. Batch 512 aumentou samples/s, mas não foi adotado por alterar o regime de otimização.
- O benchmark do loader original mostrou CPU como gargalo: zero workers ~93,7k amostras/s, contra ~74,7k (2) e ~74,9k (4). `_WindowDataset` foi alterado para materializar janelas vetorizadas e reutilizá-las; teste cobre valores e máscara de lacunas. Nova medição: zero workers ~355k/s, 2 workers ~122k/s, 4 ~107k/s, artefato `gpu-bench-20261123T170153Z-35065d01`. O custo aumenta RAM em N×60×(features+mask)×4 bytes; o sumário grava bytes materializados por fold.
- Próximo: verificar os limites de RAM no snapshot real de desenvolvimento, comparar previsões/métricas de pipeline FP32 com e sem otimizações autorizadas somente em dados até maio, e escolher perfil após protocolo/tolerâncias. Não executar o holdout de junho. Os microbenchmarks sintéticos não são resultados de treinamento BovDB nem justificam ganho preditivo.

## Pareamento da Etapa 11 — investigação

- CI do commit aaf37ef concluído com sucesso no run 45231694334, incluindo o job neural.
- O runner clássico usa validate_frame e recebe apenas linhas rotuladas; o neural usa validate_sequence_frame e mantém candles sem rótulo para compor janelas. A regra de 60 minutos e 48 candles exige o contexto completo em ambos; filtrar o snapshot clássico antes de formar janelas contaria incorretamente os candles.
- Pendentes: adaptar runner clássico ao contexto completo, registrar motivos de exclusão, implementar comparação pela interseção de chaves e checagem de y_true, testes e CI; documentar early stopping e eventual refit. Não iniciar treino real ou Etapa 12 antes da validação do pareamento.

## Pareamento Etapa 11 — implementação em andamento

- O runner clássico agora recebe o contexto completo, usa janelas causais comuns 60/48 e registra coverage.csv; rótulos ainda indisponíveis são anulados sem descartar candles de contexto. Neural também preserva contexto na purga interna e registra cobertura. A comparação por interseção valida chaves únicas e igualdade de y_true; ensemble-pair grava métricas e previsões pareadas. Refit após early stopping ainda não implementado; diferença documentada.
- Em cópia isolada no Linux, suíte completa passou: 50 testes, 12 avisos e Ruff aprovado; inclui 8 combinações neurais e pareamento sintético clássico versus neural. Os resultados antigos do clássico usaram outra população e não devem ser comparados como pareados.
- Checkout Linux original preservado. Snapshot de contexto completo ainda não foi localizado; criar a extração limitada a janeiro–maio antes do primeiro treino real. CI remoto para estas alterações, PR e treino real ainda pendentes.

## Pareamento e primeira execução real — 2026-11-23

- PR #10 aberto em rascunho; CI do commit 6450d0e passou em pytest e neural. Extração limitada até maio no Linux: 88.097 candles, quatro contratos, 183 contrato-pregões; snapshot completo hash SHA-256 a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91. Disponibilidade dos rótulos pós-sessão permanece simulada.
- Primeiro treino real: EXP_STAGE11_MLP_INDEPENDENT_42, GPU RTX 5060 Ti, FP32, seed 42, até 5 épocas e paciência 2, batch 256, 12 ajustes nos 4 folds, duração registrada 38,59 s. Nenhum dado de junho foi consultado pelo runner. Baseline EXP_STAGE11_PAIR_CLASSICAL_42 e neural geraram 170.100 previsões cada; interseção contém 170.100, chaves únicas e classes iguais. Comparação em PAIR_STAGE11_LOGREG_MLP_42.
- Lote Linux iniciado no worktree isolado ensemble-stage11-run: demais 15 combinações de MLP/CNN/LSTM/GRU, Independent/MTL e seeds 42/43, com o mesmo orçamento inicial. Está em execução; não presumir conclusão. A seleção científica final e treino de ensembles continuam pendentes.

- Revisão adicional: fit_start/fit_end agora derivam das âncoras efetivamente usadas, inclusive quando a purga remove todos os rótulos de uma sessão que continua como contexto. Teste de fronteira confirma que label_known_at igual ao primeiro instante da validação é excluído; 14 testes direcionados e Ruff passaram no Linux. Os experimentos do lote em curso usam o commit 6450d0e anterior a esse ajuste de relato; conferir os limites antes de reutilizar seus manifestos.

## Pareamento completo e matriz neural no Linux — 2026-11-23

- Estado da branch `feat/neural-training-stage11`: HEAD `f0ef0e4`; CI do PR #10 passou no run `45234698468` (jobs `pytest` e `neural`, incluindo Ruff no job principal). O PR ainda não foi integrado.
- Concluído no Linux o lote exploratório de 16 execuções: MLP, CNN1D, LSTM e GRU × Independent/MTL × seeds 42/43. Todas usaram CUDA na RTX 5060 Ti, FP32, batch 256, até 5 épocas e paciência 2; cada configuração gerou previsões em quatro folds. Independent ajustou 12 modelos por execução e MTL quatro. Duração agregada registrada: 500,25 s. O commit da implementação usado nos treinos foi `6450d0e`; os commits seguintes corrigiram relato dos limites de ajuste e validação de fuso, sem alterar treino ou população. Os resultados permanecem exploratórios.
- Os 16 experimentos usaram o snapshot Jan–Mai SHA-256 `a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91`. Cada experimento produziu 170.100 linhas; as 170.100 foram pareadas com o baseline clássico, sem exclusões da interseção, chaves duplicadas ou divergência de `y_true`. O audit preservou 2.721.600 linhas pareadas no total e confirmou `test_june_accessed=false` em todos os manifestos. Período das previsões: 2026-02-02 a 2026-05-29; junho não foi lido nem avaliado.
- Artefatos permanecem fora do Git em `~/EnsembleLearning/results/`. Auditoria agregada: `STAGE11_PAIR_AUDIT_20261123/manifest.json`, `experiment_pairing_summary.csv` e `paired_metrics_by_fold_scale.csv`; cada comparação também tem `PAIR_STAGE11_LOGREG_<arquitetura>_<estratégia>_<seed>/`. Métricas são por fold/escala; ainda não foi feita inferência estatística nem selecionado modelo com base em junho.
- O baseline clássico usa todo o bloco de treino e a rede usa uma separação interna para early stopping. Refit neural no bloco integral após selecionar época ainda não está implementado; comparar métricas deve conservar essa diferença como limitação. A hipótese de liberação dos rótulos B0 após a sessão segue simulada.
- Próximo passo: deixar o PR #10 pronto para revisão externa; não fazer merge sem revisão. Etapa #12 e avaliação final de junho seguem pendentes.
