# Seleção WIN usando cobertura anterior de 25%

## Regra e interpretação

Para fevereiro a junho, a seleção mensal usa apenas a evidência do mês anterior. O contrato deve estar vigente no início do mês-alvo, ter volume BovDB positivo sem volume nulo observado e possuir ao menos uma janela causal de 60 minutos com 48 candles fechados em pelo menos **25% dos pregões regulares elegíveis** do mês anterior. Com 17 pregões no denominador, por exemplo, são necessários 5 dias (arredondamento para cima); o WINM26 teve 7/17 em fevereiro e, portanto, passa para março. O limite de 25% é mais permissivo que o critério anterior de 50%, não mais conservador.

O denominador inclui sessões elegíveis sem candle e datas sem dados no snapshot; não são automaticamente feriados. A janela não possui limite máximo de lacuna, e passar a seleção mensal não elimina a checagem de cobertura causal em cada instante de previsão. O volume é usado para ordenar apenas candidatos que já passaram nos critérios; selecionam-se até três, sem preencher vagas com inelegíveis. A substituição após vencimento segue o ranking congelado com o mês anterior.

Janeiro permanece WING26 e WINJ26 por decisão manual. Fevereiro a junho são uma reconstrução diagnóstica com dados históricos; início de listagem, vintage e disponibilidade em tempo real não estão demonstrados. O denominador inclui pregões anteriores à primeira negociação observada do contrato até vencimento, o que pode ser conservador. A regra não comprova integridade da fonte nem ausência de viés retrospectivo.

## Programação resultante

| Período | Contratos | Evidência/observação |
| --- | --- | --- |
| Janeiro | WING26, WINJ26 | Escolha manual; WINM26 mantido fora por decisão retrospectiva anterior. |
| 02–13 fevereiro | WING26, WINJ26 | WING26 vence em 18/02; WINJ26 permanece. |
| 19–27 fevereiro | WINJ26 | Sessão especial de 18/02 excluída; sem outro substituto elegível. |
| Março | WINJ26, WINM26 | WINM26: 7/17 dias com janela (41,2%), corte mínimo: 5/17; segundo entre candidatos elegíveis por volume BovDB. |
| 01–15 abril | WINJ26, WINM26 | WINJ26 vence em 15/04. |
| 16–30 abril | WINM26 | Mantido após vencimento de WINJ26. |
| Maio | WINM26, WINQ26 | WINQ26 passou com 10/20 pregões anteriores. |
| 01–17 junho | WINM26, WINQ26 | WINM26 vence em 17/06. |
| 18–30 junho | WINQ26 | Sem substituto elegível no ranking congelado. |

A mudança de 50% para 25% não alterou os outros meses na seleção recalculada; março passou de WINJ26 sozinho a WINJ26 e WINM26. A programação continua limitada a até três contratos. Os pontos da figura representam ao menos uma janela intradiária aprovada e não a sessão completa.

## Artefatos e reprodução

- `data/b3_win/selecao-win-25-v3-criterios.csv`: candidatos, denominadores, cobertura e ranking.
- `data/b3_win/selecao-win-25-v3-por-pregao.csv`: programação por pregão com três vagas.
- `data/b3_win/selecao-win-25-v3-manifest.json`: regra, hipóteses e hashes.
- `scripts/selecionar-win-cobertura-v3.py`: reproduz a seleção sem sobrescrever a versão anterior.
- `figures/contratos-win-selecao25-janela80-2026.png` e `.pdf`: calendário da seleção e cobertura observada.
- `figures/contratos-win-selecao25-janela80-dados.csv` e `scripts/figura-selecao25-janela80.py`: dados e reprodução da figura.

Nenhum banco de treino foi criado nem modelo foi treinado.
