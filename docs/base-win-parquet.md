# Base de dados WIN em Parquet

## Conteúdo

O snapshot `data/processed/win_2026_v1/` reúne as linhas observadas para os contratos da programação v3 (cobertura mensal mínima de 25%), entre 2 de janeiro e 30 de junho de 2026. Inclui somente sessões regulares aprovadas, segunda a sexta, e candle starts de 09:00 a 17:59 em horário local B3.

- `market_candles.parquet`: 105.319 candles com ativo, id, data, início do minuto, OHLC, preço médio, volume, negócios e quantidade. Não cria candles para minutos sem negócio, não imputa valores e ainda não inclui indicadores.
- `b0_targets_retrospective.parquet`: os mesmos 105.319 índices de candle com alvos B0 de 5, 10 e 15 minutos, por chave ativo/data/hora. Mantém rótulos nulos; não inclui `trend_mean`, variáveis SIDEWAYS nem endpoints de segmentos.
- `session_coverage.csv`: 216 combinações contrato-pregão programadas, inclusive as seis sem candles (WINM26 e WINQ26 em 13, 21 e 25/05). 210 sessões têm ao menos uma janela causal de 60 minutos com 48 candles fechados. A soma das janelas por decisão elegíveis é 89.347; não se aplicou limite máximo de lacuna.
- `manifest.json`: período, regra de seleção, hashes SHA-256 das duas origens e do cronograma, estatísticas e hashes dos arquivos produzidos.

## Uso correto e limites

Os alvos B0 são retrospectivos e servem como **targets**. A auditoria do gerador confirma um limite conservador de sessão inteira: os alvos podem depender de candles até a finalização da entrada da sessão, inclusive o candle iniciado às 18:00 (fechado após 18:01). O B4 não registra `label_known_at` real, watermark da fonte ou conclusão do job; nenhum horário foi inventado. O snapshot serve para desenvolvimento e walk-forward sob uma hipótese explícita de liberação pós-sessão, com purge por sessões completas; não é um conjunto pronto para o treinador. Não usar os alvos como atributos.

`candle_start_local` é horário sem timezone embutido, no fuso `America/Sao_Paulo`; indica o início do minuto. O OHLCV desse minuto só é conhecido após o fechamento e recebimento do candle. Uma previsão causal deve usar candles estritamente encerrados antes do instante de decisão. Os resumos de cobertura por sessão não significam que todos os candles daquela sessão satisfazem a janela de 80%; `decision_minutes_with_80pct_history` quantifica os instantes que passam.

A ausência de candles pode representar ausência de negócios ou uma lacuna da fonte; não é preenchida nem classificada como falha. A seleção histórica depende dos dados de ranking anteriores já registrados, mas o início de listagem e o vintage/disponibilidade do snapshot não foram comprovados. Não houve treino nem avaliação de modelos.

## Reprodução

`python scripts/build_win_parquet.py` lê os bancos SQLite em modo somente leitura e se recusa a sobrescrever a saída. Para gerar uma versão adicional, use `--output data/processed/<novo_nome>`. A seleção usada está fixada em `data/b3_win/selecao-win-25-v3-por-pregao.csv`.
