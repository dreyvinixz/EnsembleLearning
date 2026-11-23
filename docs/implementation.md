# Estado da implementação experimental

## Objetivo executável atual

O problema implementado é classificação supervisionada dos rótulos retrospectivos
B0 binários em 5, 10 e 15 minutos. Entradas são exclusivamente colunas `x_` de
candles/indicadores disponíveis até a decisão; os rótulos e metadados de rótulo
são alvos/controle temporal, nunca features. Independent treina um modelo por
escala; MTL compartilha encoder e produz três saídas. Isso estima as classes B0,
não o retorno de preço em t+5/10/15 nem rentabilidade de trading.

## Etapa #11 — treinador neural

`ensemble-neural-walk-forward` executa MLP, CNN 1D, LSTM ou GRU em Independent
ou MTL. As janelas são 60 minutos de relógio, com pelo menos 48 candles
observados, máscara explícita para ausências, sessões/contratos separados e
primeira decisão às 10:00 local. O snapshot conserva candles sem rótulo para
contexto. Os folds 01–03 geram OOF de fevereiro–abril; fold 04 é validação
externa de maio. Janeiro inicia o treino. Junho é rejeitado pelo runner e não
foi avaliado.

O treino exige CUDA por padrão (`--device cuda`; use `--device cpu` explicitamente
para testes). Também aceita `--precision bf16|fp16`, `--allow-tf32`,
`--num-workers N` e `--compile`. CUDA usa batches pinned e cópias non-blocking.
As opções FP32, TF32, workers e compilação devem ser comparadas em benchmark
sem junho; precisão/TF32/compilação podem alterar valores ou custo inicial. O
modo padrão mantém FP32, algoritmos determinísticos, cuDNN benchmark desligado,
zero workers e sem compilação. `--compile` precisa de compilador C para Triton;
no host Linux atual, a tentativa falhou por ausência de `cc`/`gcc`, então deve
permanecer desligado. O manifesto registra essas escolhas. Testes CPU
não são evidência de treino GPU.

O AdamW fused é ativado automaticamente em CUDA. Smoke tests sintéticos no RTX
5060 Ti concluíram em FP32, BF16+TF32+2 workers e FP16 com GradScaler, gerando
264 probabilidades válidas por execução e sem consultar junho. No conjunto
pequeno de teste, FP32 levou 1,916 s, BF16+TF32+workers 2,144 s e FP16 2,433 s;
esses tempos incluem quatro folds, inicialização e escrita de artefatos e não
representam benchmark de throughput. O benchmark sintético maior favoreceu
AdamW fused e montagem vetorizada de janelas; AMP não acelerou essas redes e
TF32 quase não alterou o throughput. Ainda não há comparação end-to-end com o
snapshot real. Workers, AMP, TF32 e batch size devem ser escolhidos após essa
validação e comparação de métricas contra FP32.

`python scripts/benchmark_neural_gpu.py` faz benchmark sintético, salva CSVs e
manifesto em um diretório único sob `results/gpu-benchmarks/`, e nunca lê os
dados do estudo. Na RTX 5060 Ti, três repetições por configuração mostraram
AdamW fused ~1,3–1,5× o unfused nas redes/batches medidos; BF16/FP16 ficaram
abaixo do FP32 (aprox. 0,60–0,79×); TF32 ficou perto do FP32 (0,98–1,03×).
Batch 512 elevou throughput sintético, mas altera o regime de otimização e não
foi adotado como configuração científica. Depois de vetorizada a montagem das
janelas, o benchmark sintético do `_WindowDataset` mediu mediana ~355 mil
amostras/s com zero workers, ~122 mil com dois e ~107 mil com quatro. Antes da
vetorização, zero workers atingia ~94 mil. A melhoria de ~3,8× é da montagem e
transferência no microbenchmark; a materialização reserva RAM proporcional a
N × 60 × (features+mask) × 4 bytes, informada por fold em `training_summary`.
O conjunto continua sintético; esses números não são desempenho do treinamento
BovDB.

Cada execução salva probabilidades individuais, métricas por fold/sessão/ativo,
histórico por época, checkpoints e manifesto com hash do CSV, commit, ambiente,
GPU/memória, durações e configuração. A partição interna de early stopping é
cronológica e purgada pela disponibilidade declarada. `fit_start`/`fit_end`
descrevem o subconjunto realmente usado para ajustar pesos.

O clássico e o neural usam o mesmo snapshot completo e exigem os três alvos,
60 minutos de relógio e 48 candles para cada âncora. `coverage.csv` registra
candles de contexto sem os três alvos, decisões antes das 10h, cobertura
insuficiente e âncoras aceitas por fold e partição. Após a purga, candles sem
rótulo continuam disponíveis como contexto, mas não são exemplos de ajuste.
`ensemble-pair <predictions_a.parquet> <predictions_b.parquet> <novo-diretorio>`
compara a população exata de instrumento, sessão, instante, escala e fold;
rejeita chaves ausentes em qualquer lado, duplicatas, timestamps incompatíveis
e diferenças em `y_true`. O gate comum é uma janela causal de 60 minutos, com
pelo menos 48 candles observados, decisão a partir das 10h e os três alvos B0
presentes. `coverage.csv` registra contexto, alvos ausentes, decisões anteriores
às 10h, cobertura insuficiente e amostras aceitas.

O neural seleciona a quantidade de épocas na partição interna cronológica e
purgeada. Por padrão, então reinicializa o modelo e ajusta por esse número de
épocas em todo o bloco de treino purgado do fold; validação interna e OOF não
entram nesse refit. Checkpoints de seleção e refit são separados e a contagem,
intervalo de datas, tempo e opção de refit ficam nos artefatos. `--no-refit-after-early-stopping`
mantém o comportamento antigo apenas para diagnósticos explicitamente
identificados. As comparações finais devem usar o refit padrão e verificar a
igualdade completa da população antes de calcular métricas pareadas.

### Limites científicos e trabalho restante

- A liberação pós-sessão de B0 é uma hipótese conservadora simulada; não é hora
  histórica observada. Origem/vintage e semântica do rótulo permanecem descritas
  em `auditoria-gerador-b0.md` e `disponibilidade-rotulos.md`.
- O refit e a validação estrita de população foram implementados nesta correção;
  16 execuções Linux e previsões pareadas anteriores existem, mas foram feitas
  sem refit. Devem ser preservadas como exploratórias; novas execuções reais e
  auditoria dos artefatos com este código são necessárias antes de comparar
  métodos ou iniciar ensembles.
- O runner de teste final de junho, seleção final após maio, calibração treinada
  sem vazamento, ensembles, boosting comparativo, PCGrad e DAME não estão
  implementados nesta etapa. Não executar junho durante desenvolvimento.
- A matriz inclui 4 folds e uma seed por configuração, salvo execução explícita
  adicional. Comparações confirmatórias e orçamento equitativo ainda precisam
  ser congelados no protocolo.

## Etapa #10 — Walk-forward e rastreamento de experimentos (2026-11-23)

O comando `ensemble-walk-forward` executa um modelo clássico independente em
folds expansivos. `fold01`–`fold03` geram previsões OOF para fevereiro–abril;
`fold04` treina até abril e avalia maio como validação externa. Junho é rejeitado
na entrada e não é consultado. A liberação pós-sessão dos rótulos continua sendo
a hipótese documentada em `disponibilidade-rotulos.md`, não um timestamp observado.

Cada execução cria um diretório imutável `results/EXP_<id>/` com configuração,
manifesto e hash SHA-256 da fonte, métricas por horizonte/fold/sessão/contrato,
previsões Parquet com probabilidades e identificador de amostra, tempos de ajuste
e inferência e checkpoints. As métricas incluem MCC, Accuracy, Balanced Accuracy,
precision/recall/F1 por classe, Macro-F1, ROC-AUC, Average Precision, Brier,
Log Loss, ECE em 10 bins e matriz de confusão. O MCC é a métrica principal
provisória. Intervalos percentis de 95% por bootstrap de pregões são diagnósticos;
comparações pareadas entre modelos e correção de múltiplas comparações ainda
precisam ser implementadas antes de inferência confirmatória.

Execução de referência no Linux `ginfolab-maquinario`:

```bash
ensemble-walk-forward \
  --csv data/processed/win_2026_development_jan_may_v1.csv \
  --model logreg --seed 42 --results results
```

Resultado exploratório salvo em `/home/ginfolab/EnsembleLearning/results/EXP_F776BE93EB`;
174.189 previsões OOF/validação, máximo em 2026-05-29, 3 escalas, 4 folds,
execução em 8,90 s. A fonte tem SHA-256
`410ccbf212204b94d4705b83ec6813974e0bd2460a26deaa4aec84fed1826219`.
Os resultados OOF de fevereiro–abril e validação de maio são separados nos CSVs.
São baseline de desenvolvimento, não resultados de teste final nem comparação
de métodos. Artefatos ficam ignorados pelo Git; o dado privado não foi versionado.

Limitações remanescentes desta etapa: sem ajuste de hiperparâmetros/early stopping,
calibração pós-treino, métricas por regime, risco seletivo, intervalos pareados,
correção de multiplicidade ou renderização automática de figuras. A cobertura
temporal foi controlada por folds; disponibilidade real de rótulos permanece
simulada pós-sessão. O roadmap atualizado está em `docs/roadmap.md`.

## Contrato legado inicial
- Validador CSV explícito para rótulos **binários 0/1** (uma escolha de execução, **não** afirmação sobre o contrato real do B0).
- Timestamp de decisão, disponibilidade mais recente dos atributos e intervalos informacionais completos por rótulo.
- Holdout por sessões inteiras, partição igual para todos os ativos e purge conservador usando o instante mais tardio de confirmação entre 5/10/15m.
- Independent Logistic Regression para cada horizonte; imputação e normalização treinadas apenas no conjunto TRAIN.
- Acurácia, balanced accuracy, F1 binário, MCC, log loss e Brier; hash SHA-256 da fonte e relatório JSON.
- Arquiteturas PyTorch independentes e multitarefa GRU definidas, **sem** treinador, sem resultados e sem alegação de causalidade.

## CSV de entrada — somente snapshots auditados
Colunas obrigatórias: instrument, session_date, decision_time, features_available_at,
x_<atributo>, y_5m, y_10m, y_15m e, para cada h em {5,10,15}:
label_info_start_<h>m, label_info_end_<h>m, label_known_at_<h>m.

Exemplo de timestamps: 2026-04-29T13:00:00+00:00. session_date é a data da
sessão da bolsa (YYYY-MM-DD), que pode diferir do calendário UTC. Atributos x_
precisam ser numéricos; são permitidos valores ausentes parciais, mas não infinitos.
features_available_at é o instante de disponibilidade **mais tardio** entre todas
as features da observação, calculado a partir de um manifesto auditado da origem.
Todos os labels são confirmados, 0 ou 1. Candles sem rótulo confirmado devem ser
excluídos do snapshot e sua cobertura reportada separadamente.

A versão original do B0 deve ser auditada antes de qualquer conversão de classes.
Se a fonte real usar {-1,0,+1}, definição de SIDEWAYS ou labels prospectivos,
não renomear silenciosamente: revisar o contrato, versionar a transformação e seus
efeitos no universo de amostras. Valores de label_known_at são comprovados pela
implementação do gerador, não simplesmente t + horizonte. Uma data específica
foi usada em testes sintéticos apenas para testar o software.

## Primeira execução (validação, sem tocar no teste)
    python -m pip install -e ".[dev]"
    pytest -q
    ensemble-b0 --csv data/approved_snapshot.csv --output results/baseline_v1.json

O holdout usa datas fixas: janeiro–abril para treino, maio para validação e junho para
teste final reservado. Os rótulos pós-sessão são uma hipótese de simulação, não horários
históricos observados. O CSV usa timestamps ISO 8601 com offset `-03:00`.

O comando só funciona depois que o arquivo de entrada for gerado com proveniência,
regras e classes verificadas. Os resultados do TEST não são computados por padrão.
Para uma avaliação final acordada e única, documentar o protocolo congelado e
executar com --evaluate-test --protocol-id <identificador>, apontando para outro
arquivo de saída. Os argumentos não garantem congelamento automaticamente;
o controle científico depende de revisão e disciplina experimental.

Este contrato CSV foi a primeira especificação de referência e pode não refletir
os formatos/parquet versionados mais recentes. Antes de gerar novos dados, seguir
`docs/data-contract.md`, `docs/protocol.md` e o manifesto do snapshot aprovado.

## Pendências gerais do estudo

Modelos tabulares disponíveis via `--model`: `logreg`, `random_forest`,
`hist_gradient_boosting` e `xgboost`. MLP, CNN, LSTM e GRU independentes e
multitarefa estão definidos como arquiteturas PyTorch; ainda não há gerador de
sequências ou treinador neural. A execução de testes usa dados sintéticos.
1. Extrator versionado BovDB/QuantBase de leitura apenas e manifesto de proveniência.
2. Congelar o B0 efetivo, intervalo informacional e ground truth independente.
3. Determinar divisão walk-forward além deste primeiro holdout e orçamento por método.
4. Gerador de sequências estritamente causais e treino MLP/GRU independente/MTL.
5. Seed ensemble, block bagging, Stacking OOF, análise de discordância e PCGrad.
6. Investigar DAME somente após evidência e revisão de originalidade.
7. Calibração, inferência pareada por pregão, ablações e robustez externa.

## Resultados Stage 12 após refit — 2026-11-23

A matriz nova de 16 execuções e a auditoria pareada estão documentadas em `reports/stage12/refit-2026-11-23/`. Todas as execuções produziram 170.100 previsões, com igualdade completa de população e classes em relação ao baseline. A avaliação reporta MCC médio sobre duas seeds; não há vencedor global. OOF contém 44.702 observações por escala em 59 sessões, e maio 11.998 por escala em 17 sessões. Foram usados 5.000 bootstraps pareados de sessão, sem correção de multiplicidade; portanto, conclusões continuam exploratórias. Junho não foi acessado.

## Etapa #12 — ensembles executados — 2026-11-23

Foram implementados e executados Voting, Seed Ensemble, Block Bagging e
Stacking em previsões pós-refit. Voting calcula média uniforme entre oito
Seed Ensembles; cada Seed Ensemble combina seeds 42/43 da mesma arquitetura e
estratégia. Block Bagging usa Logistic Regression, 10 membros e blocos de
cinco sessões. Stacking ajusta Logistic Regression por escala sobre oito
probabilidades-base usando apenas folds anteriores: fold01→fold02,
fold01–02→fold03 e fold01–03→maio. Maio não foi usado para ajuste/seleção;
junho não foi consultado.

Voting e Block Bagging cobrem 170.100 previsões, pareadas com o baseline;
Stacking cobre 132.192 previsões de março a maio, pois fold01 não tem OOF
anterior para treinar o metamodelo. Block Bagging V2 gerou 1.701.000 previsões
de membros; tempo de parede 98,31 s e soma de ajustes 64,93 s. Os relatórios
agregados, CSVs e manifestos estão em `reports/stage12/`.

Diferenças de MCC de Voting e Stacking contra Logistic Regression têm ICs
pareados por sessão que incluem zero nas células reportadas. No Block Bagging,
dois de seis ICs de maio excluem zero em direções opostas; há 12 contrastes sem
correção de multiplicidade. Não há evidência suficiente de vantagem geral.
Resultados são exploratórios e não medem PnL. Previsões completas de amostras
e membros permanecem no Linux, fora do Git.

## Figuras comparativas Stage 12 — 2026-11-23

Foram geradas figuras de MCC e acurácia para Logistic Regression, Voting,
Block Bagging e Stacking. Para preservar o pareamento e incluir o Stacking,
OOF está restrito a março–abril (folds 02–03); maio é mostrado separadamente.
Cada ponto tem IC percentil de 95% via 5.000 reamostragens de sessões. OOF
contém 42 sessões e 32.066 amostras por escala; maio contém 17 sessões e
11.998 por escala. As chaves e `y_true` são verificadas exatamente entre os
quatro métodos em cada período/escala. As barras são ICs marginais; não
substituem os contrastes pareados nem corrigem multiplicidade.

PNG/PDF, dados agregados, manifesto e instruções de reprodução estão em
`figures/stage12-{mcc,accuracy}-2026-11-23.*` e
`reports/stage12/paired-metrics-2026-11-23/`. Junho não foi acessado.
