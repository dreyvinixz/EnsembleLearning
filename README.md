# Multiscale Financial Trend Labeling and Disagreement-Aware Ensemble Learning

Repositório de pesquisa para investigar como a qualidade e a discordância dos rótulos de tendência em 5, 10 e 15 minutos se relacionam com a generalização de modelos independentes, multitarefa e ensembles. O título e o nome DAME são provisórios.

**Estado:** estrutura e protocolo inicial; não há dados, modelos implementados, experimentos executados ou resultados validados neste repositório. Hipóteses e escolhas propostas ainda precisam ser acordadas e congeladas antes do teste final.

## Navegação

- [Plano e etapas](docs/roadmap.md): sequência, critérios de passagem e pendências.
- [Protocolo experimental](docs/protocol.md): hipóteses, comparação, avaliação temporal, métricas e inferência.
- [Dados e rótulos](docs/data-contract.md): proveniência, disponibilidade temporal e referência independente.
- [Métodos](docs/methods.md): controles, ensembles, DAME e ablações.
- [Matriz de experimentos](experiments/experiment-matrix.csv): escopo proposto e status.
- [Artigo](paper/README.md): organização do manuscrito e evidências necessárias.
- [Dados locais](data/README.md) e [resultados](results/README.md): regras de armazenamento.

## Ordem de trabalho

1. Inventariar dados e pesquisas existentes sem copiar ou alterar as fontes; registrar versões e permissões.
2. Verificar definição e disponibilidade temporal dos rótulos e atributos; consolidar a auditoria.
3. Congelar referência, partições, métricas e desenho de avaliação; reproduzir baselines existentes.
4. Investigar discordância e transferência entre tarefas; desenvolver o DAME somente se as evidências justificarem.
5. Executar comparações, ablações e análise estatística; preservar o teste final até o protocolo estar congelado.

Não há comando de treinamento por enquanto. As pastas são pontos de organização, não indicam implementação ou validação concluída.
