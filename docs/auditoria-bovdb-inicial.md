# Auditoria inicial de proveniência e rótulos B4

Data: 2026-11-23. Inspeção somente de leitura dos dois SQLite locais e do repositório privado `GInfoFinanceLab/bovdb-pipeline`. Não foram executados treinamento nem reconstrução de rótulos.

## Identidade e proveniência

O SHA-256 do banco `bovdb-v3.1-internal.2.sqlite` é `32cadf06387ae172edc804804fe762ef26f9f0928424ee83ce9e37a848389eab`, igual a `label_run.source_sha256` do banco B4. Isso vincula os bytes atuais da origem ao hash registrado, mas não prova que o artefato de saída foi reproduzido pelo código versionado. O SHA-256 do B4 é `2d8c18c8d1a46be788ee2810a9caba043c2413102e4ee4f5c9927646037ce4c8`. Ambos passaram anteriormente em `quick_check` na origem e no servidor.

`label_run` registra B4-v1, 1.255.202 candles, 41 tickers, 119 datas entre 2026-01-02 e 2026-06-30, sem filtros de ticker/data e sessão 09:00–18:00. Uma consulta independente em `price1` confirmou esses totais e datas; contagens iguais isoladamente não verificam correspondência linha a linha.

O registro indica commit gerador `7c224cd807041ba0384b908a97a2696bae754254` e geração em `2026-10-26T13:33:15.993570+00:00`; o GitHub informa data de autor e committer desse commit às `2026-10-26T13:50:18Z`, cerca de 17 minutos depois. Essa inconsistência temporal precisa ser esclarecida; não assumir que o commit identifica exatamente os arquivos executados ou que a árvore estava limpa. Os blobs de `build_b4_experimental.py` e `generate_chart_b3_regime.py` são idênticos nesse commit e na `main` consultada (`0a79b64292c8856349d5aaa38f68644f7010a211`). Outros arquivos e dependências não foram comparados integralmente.

## Construção e disponibilidade

O código inspecionado forma candles de 5/10/15 minutos, percorre sequências direcionais, verifica o próximo candle macro e aplica alternância de pivôs. Atribui retrospectivamente rótulos binários aos intervalos delimitados pelos pivôs: 0 para DOWNTREND e 1 para UPTREND; após o último pivô o rótulo fica nulo. O cálculo usa dados posteriores ao instante rotulado e pode depender de sequências de comprimento variável. O campo `segment_end_time` é o instante do pivô final do segmento, não uma prova do instante em que o pivô se tornou conhecido. Não há coluna de tempo de confirmação no esquema B4; calcular `label_known_at` como fim do segmento ou t + horizonte não está justificado.

As colunas SIDEWAYS usam janelas centradas e são ex-post; não usá-las como atributos causais. `trend_mean` é uma classe derivada das três escalas, não uma referência independente. Rótulos retrospectivos podem ser alvos, desde que a avaliação e o treinamento respeitem seu intervalo informacional e disponibilidade comprovada.

## Cobertura observada

| Tabela | Linhas | Classe 0 | Classe 1 | Nulos |
| --- | ---: | ---: | ---: | ---: |
| trend_5m | 1.255.202 | 641.113 | 602.819 | 11.270 |
| trend_10m | 1.255.202 | 637.708 | 602.092 | 15.402 |
| trend_15m | 1.255.202 | 632.931 | 601.683 | 20.588 |

A tabela `trend_mean` tem 25.125 nulos e 1.230.077 linhas com três escalas válidas. Estas contagens não demonstram qualidade, alinhamento por chave ou validade causal.

## Próximas verificações

Esclarecer a inconsistência dos timestamps com a proveniência original; comparar o conteúdo e dependências executados com o commit, se houver evidência; auditar chaves, cobertura por sessão, fuso e origem dos timestamps. Definir e testar reconstrução conservadora ou instrumentação do instante de confirmação e intervalos informacionais antes de gerar snapshot auditado ou treinar. Verificar o protocolo de referência independente e decidir com a equipe a política para nulos e janela de teste.
