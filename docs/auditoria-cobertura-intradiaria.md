# Horário dos candles e cobertura intradiária WIN

## Significado do horário

O dicionário do BovDB v3.1 define `price1.time` como o **início** do intervalo de um minuto no horário local B3. O parser em `src/bovdb_pipeline/trades.py` mantém hora e minuto da negociação e zera os segundos; `intraday.py` agrupa esses negócios por minuto. Assim, `09:00:00` representa `[09:00:00, 09:01:00)`, e seus dados finais não estão disponíveis antes do fechamento do minuto. A versão local inspecionada está no checkout `8b9ff1a8a59cea5e9437a16bcfef1f6adbeb0620`; isso confirma a especificação/código examinado, sem provar por si só os detalhes da execução histórica de cada arquivo.

Para o intervalo de decisão aprovado 09:00–18:00, os 540 minutos completos têm início entre 09:00 e 17:59. Um candle marcado 18:00 começa às 18:00, então não deve fornecer o seu OHLC completo como informação disponível até 18:00. O banco B4 existente foi gerado com sessão incluindo timestamps até 18:00; restringir os **atributos** não muda retroativamente a construção dos rótulos. A dependência dos rótulos, incluindo o processamento da sessão completa, deve continuar no intervalo informacional conservador.

## Auditoria gerada

`scripts/auditar-cobertura-intradiaria.py` lê o SQLite em modo somente leitura e grava `data/b3_win/auditoria-cobertura-win-2026.csv`: minutos de 09:00 a antes de 18:00 com e sem candle, maior sequência de minutos sem candle incluindo bordas, primeiro/último minuto observado, horas com pelo menos um candle e presença do candle 18:00 excluído. Exclui fins de semana, feriados B3 e o pregão especial de 18/02 conforme decisão do usuário. Contém somente combinações contrato-dia com ao menos uma linha no banco; ausência de uma linha no CSV não significa que a sessão estava completa. Há 377 combinações observadas; cada uma tem denominador de 540 minutos de relógio.

Em janeiro, WING26 aparece em 21 sessões com mediana de 538 minutos, mínimo de 334; 20/21 têm pelo menos 486 minutos (90%). WINJ26 aparece em 21 sessões com mediana de 408, mínimo de 286; apenas 5/21 têm pelo menos 486 minutos. Em 27/01 WING26 tem 334 minutos e lacuna máxima de 206; WINJ26 tem 286 e lacuna máxima de 207. A causa dessas lacunas não foi estabelecida. Como exemplo de contrato muito esparso, WINQ26 tem 3 a 14 minutos registrados por sessão nas oito datas observadas de fevereiro. Presença em uma data não garante cobertura adequada.

## Política proposta para decisão e avaliação

1. Para escolha de contratos, medir liquidez e cobertura em dados anteriores ao momento da seleção, com ranking e regra de substituição congelados. Não usar a quantidade final de candles do dia como sinal de que ele será completo no início do próprio dia.
2. Para executar em parte do dia, definir janelas mínimas e consultar apenas minutos já fechados naquele instante; se houver lacuna ou histórico insuficiente, abster-se naquele instante. Não completar OHLC artificialmente. A detecção de lacunas não pode usar o próximo candle para antecipar o futuro.
3. Como auditoria retrospectiva, reportar cobertura por sessão, horário e contrato e propor exclusão de sessões ou trechos de baixa cobertura **com regra congelada previamente**. Um limiar de 90%/486 minutos é apenas diagnóstico de sensibilidade, não critério aprovado: eliminaria 16/21 sessões de WINJ26 em janeiro. Os limites de cobertura, tamanho máximo de lacuna, janelas e tratamento de dias inteiros ausentes precisam ser acordados antes de gerar o snapshot.
4. Não assumir que todo minuto sem linha corresponde a dado perdido: o pipeline não cria candles para minutos sem negócios. Diagnosticar contra fonte de negócios independente e calendário de negociação; registrar eventuais suspensões e latência. Um número de linhas sozinho não garante integridade da sessão.
5. Comparar métodos na mesma população e relatar cobertura/exclusões por contrato e período. Não descartar sessões do teste com base no desempenho dos modelos. Limitar conclusões a dados efetivamente disponíveis.

O CSV de auditoria é derivado local e ignorado pelo Git. SHA-256 `05447948b3853c954403e23acbab74e9e27ab2fd098b20f947939a15d913f696`. Contagens e limites foram verificados; não houve alteração do banco nem geração de snapshot de treinamento.
