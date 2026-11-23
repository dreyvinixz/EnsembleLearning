# Roteiro de pesquisa

Todos os itens abaixo estão pendentes. Cada etapa deve registrar dados, código, configuração, saídas, limitações e decisão antes da seguinte; decisões científicas dependem da equipe.

## 0. Inventário e reprodutibilidade

Identificar código e resultados anteriores, origem e versão dos bancos, licença de uso, ambiente e recursos disponíveis. Registrar hashes e período coberto sem copiar bancos grandes. Verificar se experimentos prévios são realmente comparáveis; documentar diferenças. Saída: manifesto e plano de reprodução.

## A. Rótulos e dados

Documentar o algoritmo B0 efetivo, classes, janelas e disponibilidade de cada escala. Auditar nulos, duplicatas, cobertura, distribuição por ativo e sessão, estabilidade e discordância. Definir uma referência independente das previsões e avaliar suas próprias limitações. Congelar regras de revisão ou filtragem usando apenas desenvolvimento. Não chamar uma agregação dos mesmos rótulos de referência independente. Saída: relatório e contrato aprovado.

## B. Baselines e mecanismos

Reproduzir baseline existente, se disponível; treinar independentes e MTL com protocolo comum. Investigar transferência negativa por tarefa e complementaridade; comparar MTL com PCGrad como controle quando viável. Adicionar controles de ensembles segundo a matriz e orçamento registrado.

## C. Método proposto

Somente após B, decidir se a evidência justifica formular DAME. Definir entradas causais, combinação, treinamento temporal e ablações antes de consultar o teste final. O nome não constitui prova de novidade ou superioridade.

## D. Avaliação e artigo

Executar walk-forward, métricas e inferência pré-definidas; comparar células pareadas e estimar interação. Publicar resultados negativos e limitações, inclusive se o método não melhorar. Reunir tabelas, proveniência e instruções de reprodução antes de redigir conclusões.

## Decisões abertas

- Banco e snapshot específicos, ativos, períodos, fuso, sessões e definições reais de B0.
- Horizonte informacional efetivo de cada rótulo, atributos disponíveis e tratamento de SIDEWAYS.
- Referência independente, estratégia revisada/filtrada, regras de exclusão e população comum.
- Datas e número de folds, embargo justificado, orçamento, sementes e critérios de seleção.
- Métrica principal, família de hipóteses, unidade de reamostragem e poder estatístico.
- Definição e disponibilidade de regimes, protocolo de calibração e eventual avaliação econômica.
