# Auditoria Stage 11 após refit neural

## Execução e integridade

Foram auditadas 16 execuções novas: MLP, CNN1D, LSTM e GRU; Independent e MTL;
seeds 42 e 43. Cada execução usou CUDA na NVIDIA RTX 5060 Ti, FP32, batch 256,
limite de cinco épocas e paciência dois. O refit foi habilitado em todas. O
snapshot Jan–Mai tem SHA-256
`a3c9ea7ff2d9fe9c103a06a5ff211e41ebccb1ce081fe46c8f55e6e70bc2ca91` e o
commit do código foi `6fa1ea6555d6cb7078e29663ad8fa8d184be28f3`.

As 16 execuções produziram 170.100 linhas cada, totalizando 2.721.600 linhas de
previsão armazenadas. Todas têm chaves únicas, as mesmas datas (02/02 a 29/05),
quatro folds e três escalas. A comparação estrita com Logistic Regression
confirmou igualdade completa de `instrument`, `session_date`, `decision_time`,
`scale`, `fold` e `y_true` em todas as execuções. Nenhum dado de junho foi
acessado. OOF fev–abr contém 44.702 amostras por escala em 59 sessões; maio tem
11.998 por escala em 17 sessões.

O total dos tempos registrados nos 128 ajustes de fold foi 362,1 s, incluindo
150,1 s em refit. São durações dos ajustes reportadas pelo treinador, não o
tempo de parede da preparação e da auditoria.

## Resultados descritivos de MCC

Média de MCC entre as duas seeds para cada arquitetura e estratégia, limiar
0,5. A tabela mostra a maior média neural e o baseline LR. Não foi usado maio
para ajustar os modelos nem junho para seleção.

| Período | Escala | Melhor média neural | MCC neural | MCC LR |
| --- | --- | --- | ---: | ---: |
| OOF fev–abr | 5m | LSTM Independent | 0,5257 | 0,5259 |
| OOF fev–abr | 10m | GRU MTL | 0,4739 | 0,4763 |
| OOF fev–abr | 15m | GRU MTL | 0,4576 | 0,4619 |
| Validação maio | 5m | GRU MTL | 0,5310 | 0,5264 |
| Validação maio | 10m | GRU Independent | 0,4981 | 0,4960 |
| Validação maio | 15m | LSTM Independent | 0,4666 | 0,4661 |

Os resultados variam entre períodos e escalas; não há vencedor global. O
bootstrap de sessões calcula diferenças pareadas com 5.000 reamostragens, mas
os 192 contrastes são exploratórios e não têm correção de multiplicidade. Os
intervalos não devem ser lidos como confirmação ou usados para selecionar
modelos. Duas seeds e cinco épocas constituem um orçamento exploratório curto.
Classificação B0 não demonstra previsão de retorno negociável nem rentabilidade.

## Reprodutibilidade

- `metrics_by_run_fold_scale.csv`: métricas recalculadas por execução, fold,
  escala e agregados OOF/maio.
- `paired_mcc_session_bootstrap.csv`: diferenças e ICs pareados por sessão.
- `manifest.json`: hash, commits, universo, método e configuração auditados.
- Predições, checkpoints, históricos e manifestos individuais permanecem no
  Linux, sob `results/EXP_STAGE11_REFIT_*/`; não foram copiados ao Git.

Esses resultados substituem a auditoria preliminar de execuções sem refit para
as comparações seguintes. Maio continua validação de desenvolvimento e junho
permanece como teste final reservado.
