# Auditoria integral dos vencimentos e comparação de volume

## Dezembro de 2025

Os boletins BDI dos 20 pregões foram reprocessados para todos os vencimentos WIN encontrados a partir de 2026. O arquivo `data/b3_win/b3-win-dezembro-2025-todos.csv` registra 180 linhas (nove vencimentos × 20 pregões), página, URL e SHA-256 do PDF. Campos com `-` no boletim ficam vazios, não são comprovados como zero de negociação. A versão anterior com três contratos permanece preservada em `data/b3_win/`.

| Vencimento | Pregões presentes | Volume publicado somado | Negócios | Contratos negociados |
| --- | ---: | ---: | ---: | ---: |
| WING26 | 20 | 3.938.357.648.152 | 37.738.612 | 121.494.806 |
| WINJ26 | 20 | 4.222.235.450 | 20.062 | 127.695 |
| WINM26 | 20 | 1.033.256.290 | 430 | 30.657 |
| WINQ26, WINV26, WINZ26, WINM27, WINQ27, WINV27 | 20 cada | sem volume numérico publicado | — | — |

A seleção inicial para janeiro é **WING26 e WINJ26**, segundo o volume publicado nos boletins e o universo de vencimentos detectados. `WINZ25` vence em dezembro e não é elegível em janeiro. O calendário da B3 define vencimento de WIN nos meses pares na quarta-feira mais próxima do dia 15. O CSV completo tem SHA-256 `eb544c656d220e017cd565f860f3f1fe0b339a95fca4395f87c2fd0ec9227159`; seu extrator é `scripts/auditar-bdi-todos-vencimentos.py`. Os três contratos presentes no primeiro CSV foram conciliados linha a linha, incluindo hashes dos PDFs e valores numéricos; a extração de PDF foi automática com verificação estrutural e checagem pontual, não conferência visual de todas as 180 linhas. PDFs não estão arquivados; URLs e hashes permitem recuperá-los e verificar alterações.

## Unidade e diferenças observadas

No código local `bovdb-pipeline/src/bovdb_pipeline/intraday.py`, checkout observado `8b9ff1a8a59cea5e9437a16bcfef1f6adbeb0620`, a construção de `price1.volume` soma `price * quantity` e `amount_stock` soma a quantidade. A [especificação B3](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/mercado-de-acoes/futuro-de-ibovespa.htm) define o valor do minicontrato WIN como pontos × R$ 0,20. Assim, multiplicar por 0,20 é a conversão teórica do notional para WIN quando preço e quantidade têm as mesmas convenções. Isso não prova a versão efetivamente usada para construir cada linha nem a equivalência de cobertura ou de regras dos dados.

| Dia | Contrato | BovDB `sum(volume) × 0,20` | Volume BDI | BovDB quantidade | Contratos BDI |
| --- | --- | ---: | ---: | ---: | ---: |
| 02/01 | WING26 | 442.672.701.548 | 451.645.971.700 | 13.536.252 | 13.811.842 |
| 02/01 | WINJ26 | 221.522.163 | 266.302.394 | 6.623 | 7.972 |
| 05/01 | WING26 | 533.925.707.962 | 541.321.596.681 | 16.323.279 | 16.548.286 |
| 05/01 | WINJ26 | 313.495.486 | 370.577.864 | 9.381 | 11.084 |

Os valores ajustados ficam entre 83,18% e 98,63% dos valores BDI nesses quatro casos. Quantidade e negócios também diferem; a causa ainda não foi demonstrada. A janela `price1` é limitada a 09:00–18:00 e os dados do BDI são diários; diferenças de horário, cobertura, filtros ou tratamento de negociações são hipóteses a investigar, não conclusões. Comparação disponível em `data/b3_win/comparacao-bdi-bovdb-janeiro-2026.csv`; fontes BDI por linha. A regra de ranking por `volume` deve ser congelada apenas após conciliar métrica, horários e universo; não juntar diretamente totais de fontes distintas como se fossem iguais.

Fonte calendário e operação: [B3](https://www.b3.com.br/pt_br/noticias/como-sera-o-funcionamento-da-b3-no-natal-e-no-ano-novo.htm). Fontes exemplos para reconciliação: [02/01/2026](https://arquivos.b3.com.br/bdi/download/bdi/2026-01-02/BDI_03-4_20260102.pdf), [05/01/2026](https://arquivos.b3.com.br/bdi/download/bdi/2026-01-05/BDI_03-4_20260105.pdf).
