# Stacking cronológico OOF — Stage 12

## Desenho e integridade

O metamodelo é Logistic Regression por escala, ajustado sobre as probabilidades
de oito Seed Ensembles (quatro arquiteturas × Independent/MTL). O protocolo
cronológico é: fold 01 prevê fold 02; folds 01–02 preveem fold 03; folds 01–03
preveem maio (fold 04). Não há previsões de stacking para fold 01 porque não há
OOF anterior para treinar o metamodelo. Maio nunca entra no ajuste do
metamodelo, e junho não foi acessado.

Foram produzidas 132.192 previsões em março, abril e maio. O snapshot Jan–Mai
tem SHA-256 `a3c9ea7f…bc2ca91`; a população de cada fold é pareada com os
modelos-base disponíveis. O IC de diferença do MCC usa 5.000 bootstraps por
sessão e não tem correção por multiplicidade.

## MCC e diferenças pareadas

| Período | Escala | MCC Stacking | Diferença vs LR | IC 95% vs LR | Diferença vs Voting |
| --- | --- | ---: | ---: | ---: | ---: |
| OOF mar–abr | 5m | 0,5320 | 0,0039 | [-0,0032; 0,0112] | 0,0052 |
| OOF mar–abr | 10m | 0,4812 | 0,0029 | [-0,0050; 0,0106] | -0,0018 |
| OOF mar–abr | 15m | 0,4715 | 0,0001 | [-0,0114; 0,0119] | 0,0013 |
| Validação maio | 5m | 0,5268 | 0,0003 | [-0,0113; 0,0135] | -0,0001 |
| Validação maio | 10m | 0,4992 | 0,0032 | [-0,0090; 0,0154] | -0,0010 |
| Validação maio | 15m | 0,4603 | -0,0059 | [-0,0225; 0,0103] | 0,0025 |

Os ICs contra LR e Voting incluem zero em todos os casos. O stacking
cronológico não demonstra superioridade sobre os controles avaliados.

## Artefatos e limites

`manifest.json` registra agenda cronológica, hashes, features do metamodelo,
coeficientes, limites temporais, quantidade de previsões e ausência de acesso a
junho. Métricas e ICs estão nos CSVs adjacentes. Os resultados são exploratórios;
sem ajuste de multiplicidade, validação externa adicional ou avaliação
econômica. A liberação dos rótulos após sessão continua sendo uma simulação
conservadora documentada, não um timestamp histórico observado.
