# Métodos planejados

Todas as famílias são propostas; implementação, arquitetura e parâmetros ainda não foram escolhidos. A matriz distingue núcleo e complementos para controlar custo.

- **Independent Learning:** modelo por horizonte; baseline para medir desempenho especializado.
- **MTL:** representação compartilhada e saídas por horizonte; comparar por tarefa com modelos independentes. Medir conflitos de gradientes e incluir PCGrad como controle conhecido quando aplicável.
- **Voting/agregação:** combinação simples de previsões compatíveis; definir votos ou médias, ponderação e calibração antecipadamente.
- **Seed ensemble:** mesma arquitetura e dados com sementes diferentes, para controlar variância de inicialização; registrar custo e número de membros.
- **Block bagging:** membros treinados com blocos temporais amostrados apenas do treino, preservando estrutura de sessão e sem ultrapassar limites de fold. Definir tamanho de bloco e comparabilidade de orçamento.
- **Stacking:** metamodelo ajustado em previsões cronológicas fora da amostra; compará-lo à média simples e impedir vazamento entre níveis.
- **Boosting:** baseline complementar competitivo, com implementação e orçamento a definir; não confundir sua família de modelo com um controle idêntico de arquitetura.
- **DAME:** hipótese de combinação adaptativa usando discordância entre previsões, confiabilidade e regime observáveis. Definição matemática, treinamento e novidade ainda não estabelecidos. Não implementar antes de caracterizar o fenômeno.

Ablações previstas para DAME: combinação fixa, sem discordância, sem confiabilidade, sem regime e gate completo, com iguais entradas causais disponíveis e avaliação pareada. Definir como medir incerteza e confiabilidade sem acesso ao alvo corrente. Distinguir discordância entre rótulos retrospectivos da discordância entre previsões em tempo de inferência. A investigação de novidade exige revisão bibliográfica posterior.
