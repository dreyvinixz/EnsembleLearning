# Contrato de dados e rótulos

## Revisão do universo WIN mensal v1

A versão vigente do seletor de estudo fixa exatamente um contrato para todos os
pregões regulares do mês, sem substituição intramensal. A elegibilidade exige
vigência até o último pregão regular do mês-alvo, pelo menos 25% dos pregões
regulares do mês anterior com uma janela causal de 60 minutos contendo 48
candles fechados, volume anterior positivo sem linhas de volume nulo e ranking
por volume anterior (desempates por negócios e ticker). MCC e qualquer alvo B0
são proibidos na seleção. A programação e os candidatos são versionados em
`reports/win_single_contract_v1/`; não substituir os dados, manifestos ou
resultados do protocolo anterior.

O ranking utiliza o volume de `price1` dentro do BovDB; a equivalência de
unidade com o volume dos boletins B3 não está resolvida. O responsável fixou WING26 para janeiro; a cobertura WIN de dezembro não está
disponível no snapshot local e o gate de 25% não pode ser verificado para esse
mês. A seleção de fevereiro a junho usa os critérios anteriores, mas ainda é
uma reconstrução retrospectiva: ela não prova que listagem, volume ou vintage
estavam disponíveis na data de decisão. A figura mostra cobertura observada de
WINQ26 em junho, mas junho permanece fora da extração da nova base e da avaliação
nesta revisão.

Este contrato é uma especificação inicial, não uma descrição verificada do banco. Preencher um manifesto por snapshot: origem e permissão, caminho local sem credenciais, hash, versão/esquema, extração, período, ativos, fuso, calendário, sessão, frequência e integridade. Manter fontes em modo somente leitura e fora do Git. Não presumir que bases ou estudos anteriores sejam equivalentes.

## Unidade e disponibilidade

A unidade candidata é ativo × instante de decisão, com candles de um minuto e tarefas de 5/10/15 minutos. Confirmar se o candle de t está fechado antes da decisão; registrar para cada atributo a última observação e o instante em que se torna disponível, incluindo indicadores e regimes. Ajustes, imputação, normalização e seleção de atributos devem ser ajustados somente no treinamento de cada fold.

Para cada rótulo registrar classe, definição e versão do algoritmo B0, timestamp de decisão, intervalo completo de observações usado para construí-lo, timestamp em que fica conhecido e política para fim de sessão, dados ausentes e empates. Não presumir que o intervalo termina exatamente em t+5/10/15: verificar a implementação real, inclusive janelas centradas e qualquer pós-processamento. Informações retrospectivas, inclusive SIDEWAYS se usar janela centrada, não podem ser preditores em t.

## Qualidade e referência

Auditar cobertura, duplicatas, nulos, classes por ativo/sessão/regime, transições e estabilidade; quantificar concordância e discordância entre escalas alinhadas, incluindo desbalanceamento. Diferenças podem refletir horizontes distintos e não são, por si, erro de rotulagem. Uma referência deve ser construída independentemente das previsões avaliadas e sua origem, regras, horizonte, viés e incerteza registrados; independência dos modelos também não garante verdade absoluta. Não usar a maioria dos mesmos rótulos como ground truth independente.

Versões revisadas ou filtradas exigem regras pré-definidas no desenvolvimento, versionamento e avaliação explícita da cobertura. Comparar métodos na mesma população e relatar a população comum; se filtros mudarem amostras, apresentar também cobertura e análise separada, sem atribuir automaticamente diferenças ao método. Rótulos retrospectivos podem ser analisados após sua disponibilidade, mas não fornecidos como entradas de inferência no instante de decisão.
