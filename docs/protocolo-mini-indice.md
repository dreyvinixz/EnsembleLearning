# Protocolo inicial — mini-índice e divisões temporais

## Universo e seleção

Usar exclusivamente contratos futuros de mini-índice WIN, identificados e confirmados no cadastro; não incluir IND nem confundir o contrato com uma série contínua. Por mês, selecionar até os dois contratos elegíveis de maior liquidez. Para evitar usar o futuro na seleção do mês M, ordenar pelo volume negociado acumulado em `price1` no mês anterior, por contrato, após verificar unidade e significado do campo `volume` na fonte. Confirmar elegibilidade, vencimento e calendário usando cadastro/calendário conhecido no início de M; excluir contratos vencidos antes do mês. Desempate: número de negócios no mês anterior, seguido de ticker em ordem lexicográfica. Não calcular o ranking usando o mês que será previsto. A lista é congelada antes de M; ausência, vencimento ou cobertura insuficiente não autoriza substituição retroativa.

Janeiro de 2026 não tem dezembro de 2025 neste snapshot. Definir a seleção inicial com informação documentada disponível antes do primeiro pregão ou obter dados anteriores; caso contrário, não incluir janeiro nas comparações preditivas que dependem dessa seleção. Para contratos sem histórico no mês anterior, a ausência não significa volume zero comprovado; registrar como indisponível. Se menos de dois forem elegíveis com dados suficientes, reportar quantos existem e não fabricar um segundo. A regra proposta é uma operacionalização causal da escolha dos dois mais líquidos, sujeita à confirmação das unidades e do calendário; diferenças em relação ao ranking retrospectivo mensal devem ser relatadas.

## Cobertura mínima proposta

- Medir por contrato e mês os pregões elegíveis **durante sua vida útil**, descontando feriados e suspensões comprovadas; não exigir presença depois do vencimento. Inicialmente exigir candles em pelo menos 80% desses pregões. Quando não houver calendário e datas de elegibilidade confiáveis, marcar cobertura como indeterminada, não aprovada.
- Em cada pregão elegível, verificar horários previstos, duplicatas, integridade dos candles e possíveis lacunas. A ausência de candle pode refletir ausência de negócios: não preencher artificialmente nem confundir a frequência de operações com falha da fonte. Definir janela mínima por sessão após confirmar horários e convenção de timestamp; reportar o número de minutos observados e de pregões rejeitados.
- Nas linhas com candles e chaves conciliadas, exigir inicialmente pelo menos 90% com rótulos não nulos nas três escalas conjuntamente, além de informar a cobertura individual. Não imputar rótulos; excluir apenas pela regra pré-definida e comparar métodos na mesma população.
- Os limiares 80% e 90% são critérios iniciais de qualidade, a congelar antes da avaliação final. Aplicar verificações prospectivamente e relatar contratos não aprovados sem selecionar modelos ou amostras olhando desempenho futuro. Medir sensibilidade a critérios diferentes somente no desenvolvimento, preservando o teste final.

## Partições propostas

| Bloco | Período | Uso |
| --- | --- | --- |
| Treino | 02/01–30/04/2026 | Ajuste, sujeito à seleção inicial de janeiro e dados elegíveis |
| Validação | 01/05–31/05/2026 | Seleção de métodos e hiperparâmetros |
| Teste final | 01/06–30/06/2026 | Avaliação única após congelar protocolo |

Agrupar por pregão para todos os contratos e escalas. Usar a sessão inteira como intervalo informacional conservador, com liberação após disponibilidade dos candles e processamento; detalhes e hipóteses em `disponibilidade-rotulos.md`. Pré-processamento e calibração devem ser ajustados somente com dados anteriores. Stacking requer previsões fora da amostra em folds cronológicos internos; os folds, janelas e limites precisam ser fixados antes de executar. O teste de junho não orienta seleção de métodos, limiares de qualidade ou hiperparâmetros. Este único holdout não substitui walk-forward; especificar folds de desenvolvimento e relatar limitações dos seis meses.

## Conferência descritiva dos contratos

Uma consulta somente de leitura ao snapshot ordenou retrospectivamente cada mês por soma do campo `volume` (o significado financeiro ainda precisa ser validado). Os dois primeiros foram: janeiro WING26/WINJ26; fevereiro WING26/WINJ26; março WINJ26/WINM26; abril WINM26/WINJ26; maio WINM26/WINQ26; junho WINM26/WINQ26. **Esta lista descreve o mês completo e não é a seleção causal do experimento.** Alguns contratos vencem dentro do mês; confirmar calendário e validade por pregão antes de montar o universo. Não foram calculadas métricas de modelos nem consultados resultados do teste.

## ClickHouse QuantBase: busca por dezembro de 2025

Em 2026-11-23, consultas somente de leitura no container `quantbase-clickhouse-dev` não retornaram nenhum símbolo iniciado por `WIN` em dezembro de 2025 nas tabelas `quantbase.candles_1s` e `quantbase.candles_1d`, nem registros `WIN` no mesmo mês em `asset_historical_registry`. `candles_1d` contém dados de dezembro para outros símbolos. Na tabela intradiária, uma consulta adicional por WING26, WINJ26 e WINZ25 mostrou dados em outubro e nos primeiros cinco pregões de novembro, seguidos por dados de WING26 e WINJ26 em janeiro de 2026; não retornou linhas desses contratos em dezembro. Esses resultados descrevem esta instância e tabelas consultadas, não provam inexistência em outros bancos, arquivos, backups ou fontes.

Portanto, o ranking de liquidez de dezembro ainda não pode ser calculado a partir dessas consultas. É necessário localizar outra origem ou revisar a regra inicial de janeiro sem usar dados futuros. Não foram feitas alterações no ClickHouse nem exportação dos dados.

## Fonte pública para a seleção de janeiro

A B3 publica boletins diários de informações (BDI) com contratos WIN por vencimento e colunas de negócios, contratos negociados e volume; exemplo verificável de 02/12/2025: https://arquivos.b3.com.br/bdi/download/bdi/2025-12-02/BDI_03-1_20251202.pdf. A fonte oficial de especificação e vencimentos é https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/mercado-de-acoes/futuro-de-ibovespa.htm. Essa evidência mostra que é possível reconstruir o ranking de dezembro, mas o ranking mensal completo ainda não foi coletado nem validado. Agregar todas as sessões e partes relevantes dos BDIs, documentar métrica, disponibilidade, arquivos e hashes; filtrar contratos vencidos antes de janeiro antes de escolher os dois elegíveis. Não inferir o segundo contrato por amostra de um único pregão.

## Ranking inicial com dados oficiais B3

Foram extraídos os 20 pregões identificados no mês, com dados normalizados e referência/hash de cada PDF em `data/b3-win-dezembro-2025.csv` (ignorado pelo Git). A conferência do calendário inclui o [comunicado oficial da B3](https://www.b3.com.br/pt_br/noticias/como-sera-o-funcionamento-da-b3-no-natal-e-no-ano-novo.htm). Foram consolidados WING26, WINJ26 e WINM26, elegíveis em janeiro. Totais de volume conforme coluna BDI: WING26 3.938.357.648.152; WINJ26 4.222.235.450; WINM26 1.033.256.290. Portanto, **WING26 e WINJ26 são a seleção inicial entre os três consolidados**. Os dados incluem página, URL e hash do PDF; valores `-` viram zero na agregação e os documentos não foram armazenados localmente. Extração automatizada com conferência pontual; falta validar de forma integral outros vencimentos elegíveis e a equivalência de unidade com BovDB. Não tratar a seleção como aprovada definitivamente antes dessa verificação. Script: `scripts/extrair-bdi-dezembro-2025.py`.

## Auditoria adicional de todos os vencimentos

Os nove vencimentos WIN a partir de 2026 detectados nos 20 boletins foram extraídos em `data/b3_win/b3-win-dezembro-2025-todos.csv`. WING26 e WINJ26 permanecem os primeiros com volume publicado. Os demais seis além de WING26, WINJ26 e WINM26 apresentaram apenas `-` para volume; a auditoria preserva esses valores como ausentes. O relatório completo e a comparação com o BovDB estão em `docs/auditoria-liquidez-b3-bovdb.md`. A métrica e a conciliação entre fontes ainda precisam de definição antes do congelamento experimental.

## Decisão e conferência inicial — janeiro de 2026

O responsável pela pesquisa decidiu prosseguir com **WING26 e WINJ26 em janeiro de 2026**. Esta decisão substitui o caráter provisório da escolha inicial descrito acima; a regra mensal para fevereiro em diante permanece a ser aplicada e verificada. A escolha não elimina as ressalvas sobre conciliação de volume e disponibilidade causal.

Consulta somente de leitura ao B4 para janeiro encontrou 21 datas em cada contrato. Após junção por `id_ticker`, `date` e `time` entre as três tabelas de rótulos, WING26 possui 11.046 linhas com os três rótulos não nulos em 11.126 linhas comuns (99,28%); WINJ26 possui 8.598 em 8.677 (99,09%). Esse denominador é a interseção das tabelas de rótulos, não uma conciliação completa com todos os candles `price1`. Não prova completude de cada sessão nem aprovação do critério de 80% dos pregões elegíveis. Ainda é necessário confirmar calendário, diferenças de chaves e horários, disponibilidade e regras de nulos antes de criar o snapshot de treino.

## Regra aprovada: substituição após vencimento

O responsável aprovou manter até dois contratos WIN e substituir o contrato que vencer pelo próximo elegível no ranking pré-definido com informações disponíveis antes da substituição. A substituição ocorre a partir do primeiro pregão posterior ao vencimento, preservando o outro contrato. Para evitar seleção retrospectiva, a implementação deverá fixar antes do mês a ordenação de candidatos e o calendário de vencimentos; se não houver candidato elegível com dados anteriores suficientes, manter menos de dois e documentar. A seleção inicial de janeiro continua WING26 e WINJ26. Esta decisão substitui a regra anterior que proibia toda substituição durante o mês; não autoriza troca baseada em desempenho ou dados futuros.

Os candidatos e datas concretas ainda serão calculados e validados; a figura anterior de contratos mostra a seleção inicial mensal sem substituições e permanece como registro. Não preencher pregões ou candles inexistentes. A figura adicional `figures/calendario-b3-bovdb-jan-jun-2026.png` distingue calendário e datas ausentes na tabela `price1` como um todo, não determina a cobertura por contrato. Calendário oficial: https://www.b3.com.br/pt_br/noticias/calendario-de-negociacao-da-b3-confira-o-funcionamento-da-bolsa-em-2026.htm. A sessão de 18/02 possui horário especial, que exige tratamento próprio no cálculo de completude intradiária.

## Horário e sessões regulares — decisão do estudo

Usar somente sessões regulares da B3 de segunda a sexta-feira, no intervalo **09h às 18h**, no horário local da bolsa. Excluir feriados, fins de semana e sessões de horário especial, incluindo 18/02/2026. Pregões regulares sem dados disponíveis, como 13, 21 e 25/05 neste snapshot, devem ser registrados como ausentes, não confundidos com feriados nem preenchidos artificialmente. As figuras históricas mostram a disponibilidade original e não representam o filtro de elegibilidade do estudo.

Antes de implementar o filtro, confirmar o fuso e se o timestamp de `price1` representa início ou fim do candle; definir explicitamente se o candle marcado 18:00 é elegível com base nessa convenção e na disponibilidade no instante de decisão. Não presumir que o candle das 18:00 esteja fechado às 18:00. O filtro ainda não foi aplicado aos bancos ou a um snapshot.

## Atualização: três contratos e cobertura causal

O responsável decidiu ampliar o universo experimental para **até três contratos WIN simultâneos**, substituindo a regra anterior de até dois. A escolha e substituições continuam usando somente liquidez, cobertura e elegibilidade conhecidas anteriormente; não completar a quantidade com contratos sem dados suficientes. WING26 e WINJ26 já foram escolhidos para janeiro; WINM26 é o terceiro no ranking BDI de dezembro, mas sua cobertura anterior e elegibilidade pelo novo critério precisam ser verificadas antes de confirmá-lo. Janeiro a junho não têm ainda uma lista final de três contratos nem regras de corte de cobertura histórica congeladas.

Para as previsões, a configuração inicial da pesquisa é janela móvel de **60 minutos completos**, com pelo menos **48 candles (80%)**, calculada apenas com candles já encerrados e dentro da mesma sessão. Comparar **54 de 60 (90%)** como sensibilidade, sem usar o teste final para escolher. Os primeiros 59 minutos após a abertura não fornecem uma janela completa. O critério adicional de maior lacuna não tem valor definido; não afirmar que a regra de 80% sozinha elimina lacunas longas. Escolher e congelar o limite com dados de desenvolvimento antes do teste.

A função `ensemble_learning.coverage.recent_coverage` implementa a verificação isolada, com limite de lacuna opcional. Ela não seleciona contratos, não consulta dados automaticamente e ainda não está ligada ao pipeline de treino ou a snapshots. Evidências de cobertura dos três candidatos, calendário, decisões de substituição e descarte ainda precisam ser calculadas de forma causal e verificadas. Não usar informações do restante do dia para decidir uma previsão naquele instante.

## Terceiro contrato de janeiro aprovado

O responsável aprovou **WINM26** como terceiro contrato de janeiro de 2026, junto a WING26 e WINJ26. Essa escolha substitui sua condição anterior de candidato. A aprovação do contrato não implica que todas as suas sessões ou janelas tenham cobertura suficiente: aplicar a regra causal de cobertura e registrar abstenções; os critérios de seleção dos meses posteriores e o limite máximo de lacuna continuam pendentes.

## Seleção mensal por liquidez e cobertura anterior

O responsável escolheu exigir cobertura em **50% dos pregões elegíveis do mês anterior**. A execução inicial, incluindo janeiro como escolha manual e substituição apenas entre candidatos pré-qualificados, está documentada em `docs/selecao-win-cobertura-50.md`. A programação pode conter menos de três contratos; os gráficos de candidatos anteriores não representam esta programação. As limitações de histórico de listagem, integridade da fonte e disponibilidade histórica continuam abertas.

## Janeiro: ajuste aprovado pelo responsável

O responsável decidiu manter **WING26 e WINJ26** em janeiro e retirar WINM26, que não apresentou janelas de 80% no mês observado. A programação corrente está em `data/b3_win/selecao-win-50-v2-por-pregao.csv`; os arquivos anteriores foram preservados. Essa exclusão foi decidida retrospectivamente, após consultar janeiro, e não demonstra que a escolha era conhecida antes do mês; nenhum instante do WINM26 passaria no filtro causal de 80% independentemente da seleção manual.


## Atualização: limite mensal de cobertura em 25%

Em 2026-11-23, o responsável decidiu incluir WINM26 em março e reduzir de 50% para **25%** o mínimo de pregões elegíveis do mês anterior com ao menos uma janela causal de 60 minutos contendo 48 candles fechados. Portanto, 25% é um critério mais permissivo que o anterior. Para março, fevereiro teve 17 pregões no denominador e WINM26 passou em 7 (41,2%); o corte exige 5. Entre os contratos vigentes e aprovados, WINM26 ficou em segundo no ranking por volume BovDB. A seleção foi recalculada para todos os meses de fevereiro a junho; março é a única composição mensal alterada.

Janeiro continua WING26/WINJ26 por escolha manual. A nova programação versionada está em `data/b3_win/selecao-win-25-v3-por-pregao.csv`; detalhes e limitações estão em `docs/selecao-win-cobertura-25.md`. Os arquivos e a figura da versão 50% foram preservados. A seleção mensal não substitui o filtro causal de cobertura em cada instante, e a reconstrução histórica ainda não prova disponibilidade/vintage em tempo real ou início de listagem.
