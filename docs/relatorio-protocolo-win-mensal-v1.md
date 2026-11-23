# Protocolo WIN mensal v1 — população e rastreabilidade

## Roster fixado

| Mês | Contrato único |
|---|---|
| Janeiro | WING26 |
| Fevereiro | WINJ26 |
| Março | WINJ26 |
| Abril | WINM26 |
| Maio | WINM26 |
| Junho | WINQ26 |

Não há substituição durante o mês. Fevereiro–junho usam cobertura e volume do
mês anterior para elegibilidade e ranking. Janeiro WING26 foi fixado pelo
responsável; como a base local não possui cobertura de dezembro, o gate de 25%
para janeiro não pode ser conferido. A ressalva não muda o roster solicitado.
O ranking usa `price1.volume` dentro do BovDB; equivalência de unidades com o
volume B3 BDI não está resolvida. A disponibilidade/vintage antes da decisão
mensal também não está demonstrada.

O protocolo legado, seus arquivos, seleções e resultados foram preservados. O
novo cronograma, ranking dos candidatos e manifesto ficam em
`reports/win_single_contract_v1/`. A figura original
`figures/contratos-win-selecao25-janela80-2026.png` também foi preservada. A
versão destacando o contrato mensal está em
`figures/contratos-win-contrato-mensal-v1.png` e `.pdf`; a figura mostra a
cobertura observada de WINQ26 em junho. Isso não inclui junho no snapshot de
treino.

## Snapshot e auditoria

A base foi extraída em leitura imutável até 31/05/2026. A base Parquet possui
51.058 candles e o mesmo número de alvos alinhados; o CSV de sequência possui
51.058 linhas de contexto, 21 features `x_` e alvos retrospectivos B0 de 5, 10
e 15 minutos. Junho não foi consultado para construir a base ou o CSV.

| Mês | Contrato | Pregões regulares | Pregões com janela 60m/48 | Sem candle na fonte | Amostras conjuntas válidas |
|---|---:|---:|---:|---:|---:|
| Jan | WING26 | 21 | 21 | 0 | 9.826 |
| Fev | WINJ26 | 17 | 17 | 0 | 7.858 |
| Mar | WINJ26 | 22 | 22 | 0 | 10.091 |
| Abr | WINM26 | 20 | 20 | 0 | 9.029 |
| Mai | WINM26 | 20 | 17 | 3 | 8.121 |
| **Total** |  | **100** | **97** | **3** | **44.925** |

As amostras conjuntas exigem os três alvos não nulos e ao menos 48 candles
observados na janela causal de 60 minutos. Na população conjunta, as classes
foram:

| Alvo | Classe 0 | Classe 1 | Total |
|---|---:|---:|---:|
| B0 5m | 22.207 | 22.718 | 44.925 |
| B0 10m | 22.499 | 22.426 | 44.925 |
| B0 15m | 22.384 | 22.541 | 44.925 |

Cobertura por contrato e escala, contagens completas de não nulos/classes, hash
dos dois SQLite locais, Parquets, CSV, manifestos e figura estão em
`reports/win_single_contract_v1/snapshot_audit.json`,
`sequence_snapshot_manifest.json` e no manifesto ao lado dos dados locais
ignorados pelo Git. Os três pregões sem candles foram 13, 21 e
25 de maio; não foram preenchidos.

O gate da população mensal e o gate de cada âncora são distintos: aprovação do
contrato não remove sessões ou amostras silenciosamente; cada janela ainda
precisa passar pela regra 60/48. Features e janelas reiniciam em cada contrato
e sessão, inclusive na rolagem WINJ26 → WINM26.

## Comparação com a população anterior

Aplicando o mesmo gate causal 60m/48 aos CSVs históricos de desenvolvimento
preservado e novo, havia 63.517 âncoras com os três alvos no antigo protocolo e
44.925 no novo: redução de 18.592 (29,3%). Por mês, antigo → novo:

| Mês | Antigo | Novo |
|---|---:|---:|
| Jan | 12.970 | 9.826 |
| Fev | 11.623 | 7.858 |
| Mar | 16.291 | 10.091 |
| Abr | 12.514 | 9.029 |
| Mai | 10.119 | 8.121 |

Essa é uma mudança de população/cobertura, não uma comparação de desempenho.
As duas entradas e seus hashes estão em `protocol_comparison.json`; o CSV antigo
permanece intacto. Espera-se menor suporte de treino e menor mistura simultânea
de contratos. Isso pode alterar variância, transferências entre escalas e
comparabilidade de Independent, MTL e ensembles; não implica melhora nem piora.

## Próxima reexecução

1. Reexecutar os baselines clássicos no novo CSV, com mesmos folds, seeds e
   população de âncoras.
2. Reexecutar Independent e MTL para as três escalas, mantendo orçamento e
   configuração comparáveis e registrando runtime/checkpoints.
3. Reexecutar os métodos Stage 12 já disponíveis (Voting/seed averaging, block
   bagging e stacking OOF) somente com previsões cronológicas fora da amostra.
4. Parear resultados antigo/novo por contrato, pregão, instante e escala;
   apresentar também cobertura e suporte de cada população.
5. Não usar MCC para selecionar contratos. Junho permanece reservado para a
   avaliação final única, após congelar as decisões de desenvolvimento.

Nenhum modelo foi treinado nesta revisão. Os horários históricos de confirmação
dos alvos B0 seguem ausentes; `label_known_at` pós-sessão é hipótese simulada,
não disponibilidade histórica observada. Seleção de contratos não demonstra
PnL nem causalidade econômica.
