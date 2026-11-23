# Auditoria preliminar da matriz Stage 11

## Escopo e proveniência

Os Parquet e manifestos foram lidos diretamente no host Linux e reprocessados
por `scripts/audit_stage11_results.py`. O snapshot Jan–Mai tem SHA-256
`a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91`; o treino
foi executado pelo commit `6450d0e` na RTX 5060 Ti. Junho não foi acessado.

Foram auditadas 16 execuções (MLP, CNN1D, LSTM, GRU × Independent, MTL × seeds
42 e 43) e o baseline Logistic Regression. Cada execução neural contém 170.100
previsões, cobre os quatro folds e usa o mesmo conjunto integral do baseline:
chaves de instrumento, sessão, instante, escala e fold idênticas, sem chaves
repetidas e com `y_true` igual em todas as observações. O auditor falha se a
interseção não for completa. A população tem 44.702 previsões por escala nos
folds OOF de fevereiro–abril e 11.998 por escala na validação de maio, em 59 e
17 pregões, respectivamente.

## Leitura descritiva do MCC

Valores abaixo são médias dos MCCs das duas seeds, com limiar fixo 0,5. São
descrições exploratórias dos treinos curtos existentes, não uma seleção final.

| Período | Escala | Melhor neural pela média | MCC neural | Logistic Regression | MCC LR |
| --- | --- | --- | ---: | --- | ---: |
| OOF fev–abr | 5m | LSTM Independent | 0,5233 | LR | 0,5259 |
| OOF fev–abr | 10m | LSTM MTL | 0,4766 | LR | 0,4763 |
| OOF fev–abr | 15m | LSTM MTL | 0,4627 | LR | 0,4619 |
| Validação maio | 5m | GRU MTL | 0,5340 | LR | 0,5264 |
| Validação maio | 10m | LSTM MTL | 0,4938 | LR | 0,4960 |
| Validação maio | 15m | LSTM Independent | 0,4645 | LR | 0,4661 |

Não há uma estratégia neural que lidere todas as escalas e períodos. MTL supera
Independent em alguns contrastes e perde em outros; CNN1D ficou abaixo dos demais
modelos nesta configuração de cinco épocas. Esses achados não demonstram
superioridade estatística nem desempenho negociável.

O arquivo `paired_mcc_session_bootstrap.csv` contém 192 contrastes com IC
percentil de 95%, 5.000 reamostragens pareadas de pregões e os dois modelos
avaliados nas mesmas datas. Os intervalos são não ajustados para multiplicidade
e permanecem exploratórios; as 59 sessões OOF e 17 sessões de maio limitam a
precisão. A coluna `bootstrap_probability_difference_positive` não é um
valor-p.

## Limitações e próximo uso

O treino neural selecionou época em sessões internas e ajustou pesos nesse
subconjunto; Logistic Regression utilizou todo o bloco permitido. O refit neural
com a época escolhida ainda não foi feito. Portanto, diferenças atuais misturam
efeito de arquitetura com diferença de volume de treino. Antes de selecionar
modelos ou formar ensembles, implementar o refit, repetir a matriz e recalcular
estes relatórios. Junho continua reservado.

## Artefatos

- `metrics_by_run_fold_scale.csv`: métricas recalculadas para cada modelo,
  estratégia, seed, fold, escala e períodos agregados OOF/maio.
- `paired_mcc_session_bootstrap.csv`: diferenças de MCC e ICs por pregão para
  MTL versus Independent, seed 43 versus 42 e cada rede versus Logistic Regression.
- `manifest.json`: hash do snapshot, commits, população, métricas e método de
  reamostragem.
- Os Parquet de previsões e checkpoints continuam apenas no Linux, em
  `~/EnsembleLearning/results/`; nenhum dado bruto foi copiado para o Git.
