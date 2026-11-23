# Fase 0 — Implementação reproduzível (PR inicial)

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
simulada pós-sessão. `docs/roadmap.md` não existe neste checkout; o plano ativo
está em `docs/protocol.md` e no `AI_HANDOFF.md`.

## O que já existe em código
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

## Pendente

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
