# Seleção WIN usando cobertura anterior de 50%

## Regra aplicada e limites

Para os meses de fevereiro a junho de 2026, a seleção usa somente o mês anterior: o contrato deve estar vigente no início do mês, ter volume positivo sem valores nulos nos candles observados e possuir **pelo menos uma janela com 48 candles fechados em 60 minutos em pelo menos 50% dos pregões regulares elegíveis do mês anterior**. O denominador inclui pregões sem candle e datas sem dados no snapshot; não são automaticamente feriados. A janela não tem limite máximo de lacuna definido. O resultado não garante cobertura para todas as decisões do dia selecionado.

O volume é a soma de `price1.volume` nos candles das 09:00 às 17:59 do mês anterior, para ordenação dentro da mesma fonte; negócios e símbolo são desempates. São excluídos fins de semana, feriados e pregão especial de 18/02. Até três são escolhidos, sem promover quem não atingiu o critério. A ordem de substitutos é congelada com o ranking do mês anterior; após vencimento, se não houver substituto qualificado, a vaga fica vazia. No início de cada mês um novo ranking é aplicado, sem olhar dados desse mês. Janeiro mantém WING26 e WINJ26 por decisão do responsável; WINM26 foi removido posteriormente após observação de que não havia janelas válidas. É uma decisão retrospectiva, não uma seleção comprovadamente disponível antes de janeiro; a cobertura de dezembro permanece não demonstrada.

O denominador conta todos os pregões regulares do mês anterior até o vencimento; início histórico de listagem não foi comprovado. Se um contrato passou a ser negociável depois do começo do mês, esse denominador pode ser conservador demais e deverá ser revisto com evidência disponível à época. O snapshot histórico foi gerado posteriormente e latência, revisões e disponibilidade real histórica não estão demonstradas; isto é uma simulação, não prova de execução ao vivo. Não há preenchimento artificial. A seleção é um resultado inicial reproduzível, sujeito a validação do calendário, da completude da fonte e dos critérios antes do congelamento do experimento.

## Seleção resultante

| Período | Contratos que passam ou foram aprovados | Observação |
| --- | --- | --- |
| Janeiro | WING26, WINJ26 | Escolha manual; WINM26 removido após observação retrospectiva; cobertura anterior não demonstrada. |
| 02–13 fevereiro | WING26, WINJ26 | WINM26 teve 0/21 pregões anteriores com janela de 80%. |
| 19–27 fevereiro | WINJ26 | WING26 venceu em 18/02; sessão especial de 18/02 é excluída; nenhum substituto do ranking prévio passou. |
| Março | WINJ26 | WINM26 teve 7/17, abaixo de 50%; WINQ26 teve 0/17. |
| 01–15 abril | WINJ26, WINM26 | Ambos tiveram 22/22 pregões anteriores com janela. |
| 16–30 abril | WINM26 | WINJ26 venceu em 15/04; nenhum substituto pré-qualificado. |
| Maio | WINM26, WINQ26 | WINQ26 teve 10/20 pregões anteriores; WINV26 teve 0/20. |
| 01–17 junho | WINM26, WINQ26 | Ambos tiveram 17/20; os três dias sem fonte em maio integram o denominador. |
| 18–30 junho | WINQ26 | WINM26 venceu em 17/06; nenhum substituto pré-qualificado. |

Os números são pregões com pelo menos uma janela aprovada / pregões regulares elegíveis, não a fração dos 540 minutos do dia. Os dois contratos de janeiro são decisões explícitas do usuário, não garantias de janelas aprovadas em cada dia. WINM26 não teve janela de 80% em janeiro e foi removido retrospectivamente; nenhum dos seus instantes passaria no filtro causal de cobertura.

## Artefatos e reprodução

- `data/b3_win/evidencia-selecao-mensal-2026.csv`: evidência por candidato e mês anterior, incluindo volumes nulos e quantidade de janelas.
- `data/b3_win/selecao-win-50-v2-criterios.csv`: candidatos, denominadores e resultado dos critérios.
- `data/b3_win/selecao-win-50-v2-por-pregao.csv`: três vagas por pregão, com vagas sem candidato explicitamente vazias.
- `data/b3_win/selecao-win-50-v2-manifest.json`: hashes, regra e hipóteses.
- `scripts/evidencia-selecao-win.py` e `scripts/selecionar-win-cobertura-v2.py`: reproduzem a análise sem sobrescrever saídas existentes.

Nenhum banco de treino foi criado nem modelos foram treinados. A cobertura 80% será verificada novamente em cada instante de previsão; a seleção mensal não substitui essa checagem.
