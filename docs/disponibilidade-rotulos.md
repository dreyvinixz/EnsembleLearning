# Disponibilidade temporal dos rótulos B0/B4

## Resultado da análise

Para cada candle de um ativo e sessão, nas três escalas, o gerador B4 analisado carrega os candles daquela sessão entre 09:00 e 18:00 e calcula os pivôs sobre o conjunto completo. Uma regra conservadora para experimentos históricos é considerar **toda a sessão do ativo como intervalo informacional de cada rótulo não nulo**. O rótulo só é elegível depois que os candles completos da sessão estiverem disponíveis e o processamento tiver terminado. Isso é uma regra de liberação conservadora, não o primeiro instante real de confirmação de cada pivô.

A regra depende de confirmar que os arquivos efetivamente executados correspondem ao gerador inspecionado e de que a fonte e os candles daquele dia estavam disponíveis e sem revisões posteriores no instante simulado. A proveniência exata do código executado ainda tem a ressalva descrita em `auditoria-bovdb-inicial.md`. O banco histórico existente registra geração em 2026-10-26; sem registros históricos de recebimento e processamento, não se pode afirmar que esse mesmo artefato ou seus rótulos estavam realmente disponíveis em cada data de 2026. Uma simulação histórica deve identificar explicitamente essa hipótese de disponibilização; não apresentá-la como horário observado.

## Regra operacional por rótulo

Para ativo `a`, sessão `d`, instante de decisão `t` e horizonte `h`:

1. Se `trend_hm.trend_label` for nulo, não há alvo confirmado a liberar; preserve a exclusão e reporte a cobertura.
2. O início informacional conservador é o início da sessão de entrada de `a` em `d`; o fim é a conclusão e disponibilização do último candle usado na sessão de entrada, incluindo eventual candle com timestamp 18:00. A auditoria confirmou que `price1.time` marca o início do minuto em `America/Sao_Paulo`: o candle 18:00 só fecha às 18:01. A presença desse candle não prova, sozinha, completude da sessão ou disponibilidade naquele horário; aguardar a finalização registrada da fonte.
3. `label_known_at_hm` deve ser o horário **registrado ou simulado de conclusão do processamento**, não anterior à disponibilização dos candles completos. A mesma liberação conservadora vale para as três escalas da sessão. Em execução real, registre recebimento, versão da fonte e conclusão do job; em backtest, congele e documente a hipótese de latência antes de executar.
4. Na decisão ou atualização em `T`, use no treinamento apenas rótulos com `label_known_at_hm < T`; MTL e o contrato atual exigem disponibilidade das três tarefas. Previsões de validação/teste são feitas sem acesso aos respectivos rótulos e pontuadas apenas depois da liberação. Para treino em sessão posterior, comprove que processamento e dados da sessão anterior foram concluídos antes do início dessa sessão.
5. Separe sessões completas entre treino, validação e teste para todos os ativos, e aplique o purge aos intervalos informacionais reais ou conservadores. Ajuste pré-processamento, hiperparâmetros e stacking somente em dados disponíveis em cada etapa. Features centradas SIDEWAYS e `segment_end_time` não são entradas causais.

Exemplo conceitual: para `WINM26` em 2026-04-29, o banco registra candles de 09:02 até 18:00. Não se pode afirmar, a partir desses valores, que o candle 18:00 estava completo às 18:00 nem que o processamento já havia terminado. Só depois de confirmar essas condições e registrar a conclusão do processamento os alvos desse dia podem ser liberados para treinamento ou pontuação. Os 5/10/15 minutos são escalas de construção, não prazos máximos de confirmação.

## Evidência e limitações

Foram observadas 3.330 combinações ativo-sessão em `price1`; 292 têm último timestamp 18:00, e os demais terminam antes. Isso não prova que as demais sessões estavam completas: ausência de dados, horário específico ou interrupções precisam ser auditados. Não usar o último candle observado como prova de fechamento. Os rótulos finais já existentes foram gerados em lote, e o esquema B4 não registra tempos individuais de confirmação. O campo `segment_end_time` é tempo do pivô, não sua confirmação.

O benchmark atual valida timestamps explícitos e purga linhas cujo horário de disponibilidade mais tardio entre as tarefas alcança o próximo bloco; ele não cria nem comprova esses timestamps. Precisamos de um adaptador auditado que use a regra acima, manifeste as hipóteses e preserve nulos; teste de limites temporais, fuso, candles incompletos, sessões e regressão com o gerador precede qualquer treino. Uma confirmação intradiária mais cedo exigiria instrumentar o gerador e comprovar estabilidade perante dados futuros; a primeira concordância com o rótulo final não é prova de confirmação.

## Auditoria do gerador B0 — 2026-11-23

A auditoria reproduziu exatamente classes e fronteiras do B4 para as 216 sessões da seleção atual: 105.495 linhas por escala (5/10/15m), sem divergências. Os dois arquivos de código que calculam e exportam B0 no commit registrado pelo B4 têm blobs idênticos aos do checkout BovDB atual; permanece a inconsistência de que o timestamp de geração registrado antecede o timestamp do commit em 17 minutos.

A lógica é sessão-local, mas cada rótulo usa uma sequência de pivôs construída sobre os candles da sessão inteira. O gerador lê timestamps de 09:00 a 18:00 inclusive; 176 candles selecionados começam às 18:00 e só fecham às 18:01. Regra adotada: intervalo informacional conservador da abertura até a finalização dos candles da sessão (incluindo o eventual candle das 18:00); liberar alvos somente depois da finalização da fonte e da conclusão do job B0. Isso impede treino com alvos da própria sessão e requer partições/purge por sessão. `segment_end_time` não é confirmação; os offsets máximos até a fronteira foram 77/129/203 minutos para 5/10/15m, não prazos de liberação.

Horários históricos reais de recebimento e conclusão não existem no B4. A regra pós-sessão pode ser usada numa simulação se registrada como hipótese; não gravamos `label_known_at` nem afirmamos disponibilidade observada. Relatório e reprodução: `docs/auditoria-gerador-b0.md` e `scripts/audit_b0_generator.py`. Nenhum treino foi feito.
