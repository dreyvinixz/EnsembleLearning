# Protocolo experimental inicial

## Pergunta e hipóteses

Como qualidade, consistência e discordância dos rótulos multiescala influenciam generalização e comparação entre estratégias? H1: a diferença de desempenho entre métodos varia com a estratégia de rotulagem (interação, sem direção presumida). H2: MTL pode apresentar transferência positiva ou negativa por horizonte. H3: combinação adaptativa pode melhorar uma métrica pré-especificada em relação a controles simples. São hipóteses, não resultados; definir estimandos, métrica principal e limiares antes do teste.

## Desenho temporal

Usar folds walk-forward cronológicos com treino, validação e teste posteriores. Registrar datas, ativos, sessões e limites de cada fold. Purge removerá exemplos de treinamento cujo intervalo completo de construção do rótulo intersecte o intervalo informacional do conjunto de avaliação; aplicar a mesma verificação às fronteiras e aos folds internos. Considerar dependências de atributos, execução e janelas entre sessões; definir embargo quando necessário e justificar duração. A duração deve derivar dos intervalos efetivos, não apenas dos nomes 5/10/15. Não misturar observações de um mesmo instante entre partições de forma que haja vazamento entre ativos.

Toda escolha de features, hiperparâmetros, filtro, calibração, limiar e regime ocorre dentro do desenvolvimento temporal. Uma avaliação final intocada é consultada somente depois do congelamento. Registrar sementes, orçamento computacional, número de modelos e buscas; igualdade de dados e orçamento comparável não implica custo idêntico: reportar ambos e diferenças de capacidade.

## Stacking e adaptativo

Gerar previsões out-of-fold estritamente cronológicas para treinar metamodelos e pesos; em cada previsão, modelos e pré-processamento usam apenas informações anteriores, rótulos já disponíveis e intervalos não sobrepostos. Calibração e gate devem respeitar a mesma separação. Em avaliação, o gate recebe somente probabilidades correntes de modelos disponíveis e atributos/regimes observáveis em t; não recebe discordância retrospectiva dos rótulos, nem o alvo corrente.

## Análise

Relatar desempenho por horizonte, classe, ativo e regime com suporte e cobertura, além de métrica principal e medidas complementares de classificação e calibração escolhidas antes da avaliação. Usar baselines simples, matriz de confusão, perdas probabilísticas e custo computacional quando aplicáveis. Comparar métodos e versões de rótulos em folds e unidades comuns; estimar interação método × rotulagem com contrastes pareados e intervalos de confiança. Observações intradiárias são dependentes: definir antes unidade de inferência e reamostragem em blocos de tempo/sessões, preservando pareamento e dependência entre ativos. Não tratar candles como réplicas independentes; poucos folds limitam conclusões. Pré-especificar comparações principais, ajuste de multiplicidade e tratamento de resultados ausentes.

Se houver avaliação econômica, definir previamente execução, custos, slippage, restrições e disponibilidade; classificação superior não demonstra lucratividade. Não interpretar associação como causalidade sem desenho adequado.
