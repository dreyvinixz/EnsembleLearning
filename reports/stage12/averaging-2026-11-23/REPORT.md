# Voting e Seed Ensemble — Stage 12

## Desenho e integridade

Foram avaliados oito Seed Ensembles, um por arquitetura/estratégia (MLP, CNN1D,
LSTM e GRU; Independent e MTL), cada um combinando igualmente as seeds 42 e 43.
O Voting combina igualmente os oito Seed Ensembles. Pesos foram fixados antes
da validação de maio; nenhum ajuste usou maio ou junho. As previsões vieram das
execuções Stage 11 com refit no snapshot Jan–Mai (`a3c9ea7f…bc2ca91`).

Cada ensemble cobre 170.100 chaves, com classes reais idênticas ao baseline:
44.702 amostras por escala nos folds OOF de fevereiro–abril (59 sessões) e
11.998 por escala em maio (17 sessões). Foram calculados MCC, métricas de
classificação, probabilidades e ECE; ICs pareados usam 5.000 bootstraps por
sessão. Junho não foi acessado.

## Voting: MCC e diferença pareada contra Logistic Regression

| Período | Escala | MCC Voting | Diferença | IC 95% |
| --- | --- | ---: | ---: | ---: |
| OOF fev–abr | 5m | 0,5226 | -0,0033 | [-0,0102; 0,0034] |
| OOF fev–abr | 10m | 0,4764 | 0,0001 | [-0,0082; 0,0082] |
| OOF fev–abr | 15m | 0,4657 | 0,0038 | [-0,0053; 0,0132] |
| Validação maio | 5m | 0,5269 | 0,0004 | [-0,0095; 0,0117] |
| Validação maio | 10m | 0,5002 | 0,0041 | [-0,0084; 0,0193] |
| Validação maio | 15m | 0,4578 | -0,0083 | [-0,0257; 0,0084] |

Todos os intervalos incluem zero. O Voting uniforme não mostra vantagem clara
contra Logistic Regression neste experimento exploratório.

## Seed Ensembles

As duas seeds foram combinadas com pesos iguais. Os contrastes contra os
modelos individuais estão em `paired_mcc_vs_members_bootstrap.csv`; a maior
parte dos intervalos inclui zero (sem ajuste por multiplicidade). Há casos de
ganho e de perda, portanto a média entre seeds não demonstra melhora geral.

## Artefatos e limites

`manifest.json` registra hashes das execuções-fonte, snapshot, população,
métricas e configuração; CSVs preservam resultados agregados e contrastes.
Previsões completas e checkpoints permanecem no host Linux e não são versionados.
As 5.000 reamostragens não corrigem a multiplicidade das comparações. O
experimento não avalia rentabilidade, seleção final nem teste de junho.
