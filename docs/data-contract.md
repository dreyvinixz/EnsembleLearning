# Contrato de dados e rótulos

Este contrato é uma especificação inicial, não uma descrição verificada do banco. Preencher um manifesto por snapshot: origem e permissão, caminho local sem credenciais, hash, versão/esquema, extração, período, ativos, fuso, calendário, sessão, frequência e integridade. Manter fontes em modo somente leitura e fora do Git. Não presumir que bases ou estudos anteriores sejam equivalentes.

## Unidade e disponibilidade

A unidade candidata é ativo × instante de decisão, com candles de um minuto e tarefas de 5/10/15 minutos. Confirmar se o candle de t está fechado antes da decisão; registrar para cada atributo a última observação e o instante em que se torna disponível, incluindo indicadores e regimes. Ajustes, imputação, normalização e seleção de atributos devem ser ajustados somente no treinamento de cada fold.

Para cada rótulo registrar classe, definição e versão do algoritmo B0, timestamp de decisão, intervalo completo de observações usado para construí-lo, timestamp em que fica conhecido e política para fim de sessão, dados ausentes e empates. Não presumir que o intervalo termina exatamente em t+5/10/15: verificar a implementação real, inclusive janelas centradas e qualquer pós-processamento. Informações retrospectivas, inclusive SIDEWAYS se usar janela centrada, não podem ser preditores em t.

## Qualidade e referência

Auditar cobertura, duplicatas, nulos, classes por ativo/sessão/regime, transições e estabilidade; quantificar concordância e discordância entre escalas alinhadas, incluindo desbalanceamento. Diferenças podem refletir horizontes distintos e não são, por si, erro de rotulagem. Uma referência deve ser construída independentemente das previsões avaliadas e sua origem, regras, horizonte, viés e incerteza registrados; independência dos modelos também não garante verdade absoluta. Não usar a maioria dos mesmos rótulos como ground truth independente.

Versões revisadas ou filtradas exigem regras pré-definidas no desenvolvimento, versionamento e avaliação explícita da cobertura. Comparar métodos na mesma população e relatar a população comum; se filtros mudarem amostras, apresentar também cobertura e análise separada, sem atribuir automaticamente diferenças ao método. Rótulos retrospectivos podem ser analisados após sua disponibilidade, mas não fornecidos como entradas de inferência no instante de decisão.
