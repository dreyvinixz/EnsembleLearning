# Block Bagging temporal — Stage 12

## Desenho e integridade

O controle usa Logistic Regression com 10 membros por fold e escala. Cada
membro recebe amostragem com reposição de blocos móveis de cinco sessões
cronológicas; os membros são combinados pela média uniforme de probabilidades.
O seed base é 42. Nenhum hiperparâmetro foi escolhido com maio ou junho.

A execução V2 (`EXP_STAGE12_BLOCKBAG_LOGREG_B5_M10_V2_20261123`) produziu
170.100 previsões agregadas e 1.701.000 previsões de membros em 98,31 s de
tempo de parede; a soma dos tempos individuais de ajuste foi 64,93 s. A
população e `y_true` coincidem integralmente com o baseline Logistic Regression.
OOF fev–abr tem 44.702 observações por escala em 59 sessões; maio tem 11.998
por escala em 17 sessões. Junho não foi acessado.

## MCC e diferenças pareadas contra Logistic Regression

| Período | Escala | MCC Bagging | Diferença | IC 95% |
| --- | --- | ---: | ---: | ---: |
| OOF fev–abr | 5m | 0,5259 | 0,0000 | [-0,0019; 0,0018] |
| OOF fev–abr | 10m | 0,4740 | -0,0023 | [-0,0049; 0,0005] |
| OOF fev–abr | 15m | 0,4620 | 0,0001 | [-0,0025; 0,0028] |
| Validação maio | 5m | 0,5286 | 0,0021 | [0,0001; 0,0043] |
| Validação maio | 10m | 0,4935 | -0,0026 | [-0,0049; -0,0002] |
| Validação maio | 15m | 0,4649 | -0,0012 | [-0,0086; 0,0048] |

Dois dos seis intervalos não ajustados em maio excluem zero, em sentidos
opostos entre escalas. Com 12 contrastes exploratórios e sem correção de
multiplicidade, não se conclui vantagem do bagging.

## Artefatos e limites

`manifest.json`, `fit_records.csv`, `metrics_by_period_scale.csv` e
`paired_mcc_bootstrap.csv` documentam configuração e resultados. Previsões
completas de membros permanecem no Linux. O bloco é definido sobre sessões
elegíveis observadas no fold; comprimentos e lacunas refletem essa população.
É um baseline de diversidade amostral, não evidência de melhora preditiva nem
de desempenho financeiro.
