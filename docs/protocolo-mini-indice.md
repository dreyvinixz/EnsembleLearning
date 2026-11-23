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
