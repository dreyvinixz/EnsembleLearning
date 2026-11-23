# Protocolo experimental inicial

## Correspondência entre objetivo e implementação atual

O alvo implementado é a classe retrospectiva B0 observada em cada instante de
decisão, separadamente em 5/10/15 minutos. A tarefa não é classificar o retorno
realizado exatamente 5/10/15 minutos depois. As três colunas B0 são targets e
referências para avaliação posterior; `feature_columns()` aceita apenas `x_`,
portanto rótulos, `trend_mean` e metadados temporais não entram na inferência.
Independent treina um classificador por horizonte; MTL usa representação
compartilhada e três saídas. Predições contêm probabilidades por amostra.

O código cobre MLP, CNN 1D, LSTM e GRU e o desenho Jan–Abr de desenvolvimento,
maio de validação e junho reservado. Janeiro–abril são implementados como quatro
folds expansivos: fevereiro, março e abril são OOF; maio é validação externa.
O treinamento neural usa janelas causais de 60 minutos, mínimo 48 candles,
slots ausentes com máscara, sem cruzar sessões, e decisão inicial às 10:00.
Junho é rejeitado pelo treinador atual; ainda falta um executor controlado de
holdout final.

Clássico e neural usam o mesmo gate de âncoras: instrumento/sessão/instante com
janela causal de 60 minutos, pelo menos 48 candles observados, decisão a partir
das 10:00 e os três rótulos B0 presentes. A comparação pareada exige igualdade
completa de instrument, session_date, decision_time, scale e fold e igualdade
de `y_true`; qualquer chave ausente em um dos métodos invalida a comparação.
Cobertura por partição registra linhas de contexto, rótulos ausentes, decisões
antes das 10h, cobertura insuficiente e amostras aceitas.

Para o treinamento neural, early stopping seleciona o número de épocas em uma
partição interna cronológica e purgeada. O modelo é então reinicializado e
reajustado pelo número selecionado em todo o bloco de treino permitido do fold,
excluindo validação e OOF; este é o comportamento padrão. A decisão de épocas
vem somente da validação interna. Comparações anteriores sem refit são
exploratórias e precisam ser refeitas antes de seleção de modelos.

Não estão cobertos pela implementação atual: treino final em junho, seleção
congelada após maio, Voting/Seed Ensemble/Block Bagging/Stacking, boosting no
walk-forward completo, calibração pós-treino, regime, predição seletiva,
comparações estatísticas pareadas ou DAME. Brier/Log Loss/ECE são registrados,
mas calibração ainda é diagnóstica. Os números de folds descrevem partições,
não resultados; cada arquitetura/estratégia usa uma seed por padrão.

O código usa datas explícitas de disponibilidade de rótulos para purge, mas o
snapshot atual simula a liberação B0 após a sessão. Portanto, as separações
cronológicas são condicionais a essa hipótese conservadora, que não equivale a
timestamps históricos observados. Features e timestamp de fechamento precisam
continuar vinculados ao manifesto auditado.

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

## Atualização Stage 12 — matriz pós-refit

MCC permanece a métrica principal com limiar fixo 0,5; Macro-F1, ROC-AUC, Average Precision, Brier e Log Loss são complementares. A matriz pós-refit foi comparada em população completa e comum. Os intervalos de bootstrap por sessão e todas as comparações de modelo seguem exploratórios: 16 runs, duas seeds, quatro folds e nenhum ajuste de multiplicidade. Resultados de maio são validação de desenvolvimento; não usar para treinar modelos de combinação quando se reportar avaliação externa.
