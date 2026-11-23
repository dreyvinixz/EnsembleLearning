# Multiscale Financial Trend Labeling and Disagreement-Aware Ensemble Learning

Repositório de pesquisa para investigar como a qualidade e a discordância dos rótulos de tendência em 5, 10 e 15 minutos se relacionam com a generalização de modelos independentes, multitarefa e ensembles. O título e o nome DAME são provisórios.

**Estado:** fase 0 com validador de dados, holdout cronológico e baseline independente executável; modelos GRU apenas definidos (não treinados). Sem dataset BovDB conectado, sem experimentos reais executados e sem resultados validados. A definição do B0 ainda precisa ser auditada.

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

## Começar com o primeiro baseline
```bash
python -m pip install -e ".[dev]"
pytest -q
ensemble-b0 --csv data/approved_snapshot.csv --validation-sessions 20 --test-sessions 20 --output results/baseline_v1.json
```

O comando só pode ser executado com um **CSV aprovado no contrato**, contendo timestamps informacionais e rótulos binários explicitamente definidos. Nenhum CSV real foi incluído. O teste final é reservado por padrão; a implementação não substitui o congelamento formal do protocolo.

Veja [guia de execução e esquema do CSV](docs/implementation.md). As arquiteturas GRU independente e MTL estão em código, mas ainda não possuem loop de treino.
