# Auditoria do gerador B0 e política de liberação

Data da auditoria: 2026-11-23. Inspeção somente de leitura dos scripts BovDB e dos SQLite locais; os bancos fonte não foram alterados.

## Identidade e reprodução

O B4 local registra `label_version=B4-v1`, origem `bovdb-v3.1-internal.2.sqlite`, hash `32cadf06387ae172edc804804fe762ef26f9f0928424ee83ce9e37a848389eab`, geração em `2026-10-26T13:33:15.993570+00:00` e commit `7c224cd807041ba0384b908a97a2696bae754254`. A hora de geração antecede em cerca de 17 minutos a hora registrada desse commit. Portanto, o registro não prova sozinho qual árvore foi executada.

A auditoria comparou os blobs do commit registrado com o checkout BovDB atual (`8b9ff1a8a59cea5e9437a16bcfef1f6adbeb0620`): `build_b4_experimental.py` e `generate_chart_b3_regime.py` são idênticos. Em seguida, o B0 foi reexecutado em modo somente leitura nos **216 contrato-pregões selecionados**. Nas escalas de 5, 10 e 15 minutos foram comparadas 105.495 linhas por escala, incluindo rótulo, `segment_end_time` e `segment_end_type`: **zero diferenças e zero chaves ausentes**. Isso confirma que o código auditado reproduz o B4 nas sessões do experimento; não elimina a incerteza sobre a disponibilidade histórica da fonte ou do processo de geração.

## Janela de informação observada no código

1. `build_b4_experimental.py` separa e processa cada ativo e data individualmente. A consulta inclui timestamps de 09:00 a 18:00, inclusive; `build_session` mantém o mesmo intervalo fechado. O B0 não consulta pregões seguintes nem usa janelas entre sessões.
2. Para cada escala, `resample_candles` agrega os candles de 1 minuto em barras de 5, 10 ou 15 minutos. `dropna()` remove barras inteiramente vazias, mas não verifica se cada barra contém todos os minutos esperados e não preenche candles ausentes.
3. `detectar_topos_fundos` encontra sequências de barras ascendentes/descendentes, escolhe o maior/menor fechamento e refina o pivô com fechamentos de 1 minuto. Também examina a barra macro seguinte e pode mover o pivô se ela contiver um extremo mais alto/baixo. A quantidade de barras necessárias varia com o comprimento da sequência e o pivô.
4. `pivots_to_binary_trend_labels` atribui a classe até o próximo pivô, inclusive; depois do último pivô, deixa o alvo nulo. Assim, `segment_end_time` identifica o pivô que delimita o segmento, não o momento em que esse pivô ficou confirmado. O deslocamento entre o timestamp rotulado e essa fronteira variou nos alvos selecionados: máximo de 77, 129 e 203 minutos para 5, 10 e 15 minutos, respectivamente. Esses números **não** são atrasos de confirmação.

A fronteira de um rótulo pode depender de um pivô encontrado depois do candle rotulado, da barra macro posterior usada para refiná-lo e da sequência de barras que encerra o movimento. O B4 não grava a sequência de barras lida, o intervalo individual de informação ou quando o resultado do lote ficou disponível. Não é correto definir a confirmação como `t + 5/10/15 min` nem usar `segment_end_time` como `label_known_at`.

## Consequência dos candles das 18:00

Na programação de 216 contrato-pregões, 210 têm dados no snapshot; seis não têm candles. O B0 leu 105.495 candles de sessão, incluindo **176 candles com início às 18:00**. A base de atributos criada para o estudo exclui esses candles, pois o intervalo regular definido termina em 18:00 e o candle das 18:00 ainda não fechou nesse instante. Mesmo assim, os rótulos B0 até 17:59 foram produzidos usando a sessão do B4, que pode incluir o candle das 18:00. Como `price1.time` marca o início do minuto, esse candle só pode estar completo depois de 18:01, além do recebimento da fonte. Em 34 sessões com algum dado não havia candle das 18:00; isso não prova que a fonte estivesse completa antes do encerramento do pregão.

## Regra conservadora estabelecida

Aplicar a mesma regra às três escalas: tratar cada rótulo não nulo da sessão como tendo intervalo informacional conservador **desde a abertura da sessão (09:00) até o fechamento e a disponibilização de todos os candles usados pelo B0 naquela sessão**. Para cobrir uniformemente a entrada permitida às 18:00, o limite de relógio não é anterior a 18:01 local. A liberação ocorre somente depois de a fonte marcar a sessão como finalizada e o processamento B0 terminar. Rótulos nulos continuam sem alvo.

Consequências para o experimento:

- Nenhum rótulo da sessão pode participar do treino ou de uma atualização durante a própria sessão.
- Dividir por sessões inteiras; manter todos os contratos e escalas da mesma data no mesmo bloco e aplicar purge pela sessão completa. Não estimar uma janela menor a partir de `segment_end_time`.
- Na inferência, atributos usam apenas candles fechados antes da decisão. O uso dos alvos da sessão em treino posterior pressupõe que a sessão-fonte e a execução B0 já tenham terminado.
- `label_known_at` observado continua desconhecido. O banco não tem horários de recebimento, watermark/finalização ou conclusão do job. Para uma simulação histórica, a liberação pós-sessão é uma **hipótese operacional**, a declarar e congelar no protocolo; não pode ser apresentada como disponibilidade realmente registrada em 2026. Não gravamos timestamps artificiais no Parquet.

## Artefatos e reprodução

- Gerador: checkout BovDB local em `C:\mysystems\services\bovdb\bovdb-pipeline`, arquivos `rotulos/scripts/build_b4_experimental.py` e `rotulos/scripts/generate_chart_b3_regime.py`.
- Reprodução somente leitura: `python scripts/audit_b0_generator.py` (aceita `--pipeline <caminho>`). Compara os rótulos e fronteiras B4 nas sessões da seleção v3; nesta máquina requer as dependências usadas pelos scripts BovDB, incluindo pandas, NumPy e Plotly.
- Banco de treino não foi alterado; a política está documentada em `docs/disponibilidade-rotulos.md` e no `AI_HANDOFF.md`.

## Limitações restantes

A reprodução das saídas B4 valida a igualdade do algoritmo para os dados locais, mas o B4 foi gerado em outubro de 2026 e não há vintages históricos nem logs de ingestão e execução por sessão. A política de liberação é adequada como limite causal conservador para uma simulação de treino offline; ela não demonstra que esses dados ou rótulos existiam em tempo real no dia histórico correspondente. Não foram treinados modelos nem consultado o bloco de teste para avaliar desempenho.
