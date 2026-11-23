# Modelagem: dois eixos ortogonais

**Eixo A — Modelos/arquiteturas:** Logistic Regression, Random Forest,
HistGradientBoosting, SVM (pendente), MLP, 1D-CNN, LSTM e GRU. Apenas os
três primeiros possuem baseline de validação executável; as redes neurais
possuem definições de forward, sem treinamento. A GRU não é o único modelo.

**Eixo B — Estratégias de aprendizado:** Independent Learning, Multi-Task
Learning (MTL), Voting, Seed Ensemble, Block Bagging, Stacking e hipótese DAME.
Boosting pode atuar como baseline clássico adicional.

Os modelos clássicos operam sobre atributos tabulares; MLP/CNN/LSTM/GRU
recebem janelas históricas que deverão ser construídas sem informação futura.
As redes podem ser independentes ou compartilhar encoder e ter cabeças
por horizonte. Bagging e Stacking podem combinar arquiteturas homogêneas
ou heterogêneas e não são sinônimos de MTL.

Matrizes: experiments/architecture-matrix.csv (modelos) e
experiments/experiment-matrix.csv (estratégias). O desenho final deve
priorizar contrastes informativos, não o produto cartesiano completo.
Documentar orçamento, sementes, parâmetros, folds e custo.

Rótulos retrospectivos, inclusive discordâncias entre 5/10/15m, não podem
entrar como features de inferência no instante t. DAME, a hipótese de gate
adaptativo, dependerá de discordância entre previsões disponíveis em t,
estudo de transferência negativa, ablações e revisão de originalidade.
