# Multiscale Financial Trend Labeling and Disagreement-Aware Ensemble Learning

O projeto combina análise da qualidade dos rótulos B0 de 5, 10 e 15 minutos
com **múltiplas arquiteturas e estratégias de aprendizado**. A GRU é apenas
uma arquitetura entre MLP, 1D-CNN e LSTM; também há modelos clássicos.

## Eixo 1 — Modelos
- Clássicos: Logistic Regression, Random Forest e HistGradientBoosting com
  baseline de validação executável; SVM candidato, ainda não implementado.
- Neurais: MLP, 1D-CNN, LSTM e GRU, cada uma nas variantes Independent e MTL.
  Apenas arquitetura/forward definidos; treinadores ainda não existem.

## Eixo 2 — Estratégias
Independent Learning, MTL, Voting, Seed Ensemble, Block Bagging, Stacking,
Boosting e possível DAME. Estes métodos não devem ser confundidos com
arquiteturas; ver docs/methods.md e experiments/architecture-matrix.csv.

## Estado real
Validador de contrato CSV provisório binário, holdout por sessões com purging,
três baselines tabulares executáveis e definições neurais não treinadas.
**Sem dados BovDB conectados, sem treino neural, sem ensembles treinados,
sem validação empírica ou DAME implementado.**

## Executar (após snapshot auditado)
    python -m pip install -e ".[dev]"
    pytest -q
    ensemble-b0 --csv data/approved_snapshot.csv --validation-sessions 20 --test-sessions 20 --model logreg --output results/logreg_v1.json

Para as demais famílias tabulares: selecionar random_forest ou
hist_gradient_boosting em --model. As redes neurais exigem o extra deep.
Veja docs/data-contract.md, docs/protocol.md, docs/implementation.md,
docs/roadmap.md, AGENTS.md e AI_HANDOFF.md. TEST permanece reservado.
