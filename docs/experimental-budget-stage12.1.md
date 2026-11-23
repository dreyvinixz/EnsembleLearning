# Orçamento experimental da Etapa 12.1

Este plano usa somente janeiro–maio de 2026. Junho continua reservado e não pode
ser aberto para diagnóstico, seleção ou depuração. A métrica principal provisória
é MCC em limiar 0,5; acurácia, Macro-F1, ROC-AUC, Average Precision, Brier e
Log Loss são complementares.

## Execução progressiva na RTX 5060 Ti

| Fase | Execuções | Critério para avançar |
|---|---|---|
| 0 — integridade | Revalidar hashes, manifests, chaves pareadas e ausência de junho; nenhum treino | Todos os hashes e períodos coincidem com o snapshot Jan–Mai aprovado |
| 1 — benchmark de orçamento | GRU MTL e LSTM Independent, seeds 42/43, épocas 5/10/20/40, mesma configuração FP32; selecionar épocas somente pela validação interna de cada fold | Comparar loss/MCC de treino e validação, épocas escolhidas, tempo, VRAM e refit; interromper se não houver convergência ou houver divergência numérica |
| 2 — estabilidade | Repetir apenas configurações promissoras com seeds adicionais 44/45 | Relatar média/desvio entre seeds e bootstrap pareado por sessão; não selecionar pela média de maio sem declarar essa escolha como validação |
| 3 — hipóteses dirigidas | Ablation de features no Logistic Regression e no melhor baseline neural definido dentro do desenvolvimento; MTL versus Independent; PCGrad apenas se o diagnóstico confirmar conflito de tarefas | População de avaliação idêntica, seleção interna cronológica, estimandos e correção de multiplicidade definidos antes das comparações |
| 4 — expansão | Mais arquiteturas, hiperparâmetros ou seeds somente para uma hipótese que sobreviva às fases anteriores | Nova versão do protocolo, orçamento calculado e manifesto de aprovação antes de executar |

O histórico disponível tem teto de cinco épocas e duas seeds: 46 de 64 decisões
run-fold escolheram época 4 ou 5. Isso justifica testar um teto maior, mas não
autoriza afirmar subtreinamento. Não reutilizar os históricos antigos como se
fossem evidência sobre treinos de 10–40 épocas.

## Estimativa de custo

O benchmark deve registrar tempo de parede, segundos de ajuste, pico de VRAM,
épocas efetivas, tempo de compilação/inicialização e throughput, separados por
arquitetura e estratégia. Como referência observada, os 16 treinos anteriores
somaram 362,1 segundos de `fit_seconds` (150,1 s de refit), mas o tempo total de
parede foi superior e o limite era cinco épocas. Não extrapolar linearmente sem
medir: épocas selecionadas, arquitetura, tamanho dos folds e refit alteram o
custo. Executar uma configuração de cada vez e interromper se a memória exceder
o limite registrado no manifesto.

## Contrastes e inferência

Pré-especificar como confirmatórios apenas os contrastes necessários para cada
hipótese; outros resultados são exploratórios. Usar bootstrap pareado por
sessão, mantendo juntas as observações dos contratos na mesma data e o mesmo
vetor de previsões entre métodos. Reportar IC de 95% e diferença de MCC. O
script `paired_session_max_t.py` produz intervalos simultâneos max-T para MCC e
acurácia dentro de cada família período × escala. A família confirmatória deve
ser congelada antes da avaliação; Holm pode ser usado se o protocolo
pré-especificar testes e p-valores válidos. Com 17 sessões em maio, os ICs são
diagnósticos com resolução limitada; não tratar os candles como réplicas
independentes.

Para ablações, comparar `full` com `without_<grupo>` na mesma âncora, fold e
sessão. `only_<grupo>` responde suficiência descritiva, não contribuição
incremental. Colinearidade entre atributos e interações entre grupos impedem
interpretar a queda ao remover um grupo como importância causal.

## Portão holdout

Nenhum comando neste plano acessa junho. A avaliação final só ocorre depois de
congelar código, features, hiperparâmetros, seeds, ensembles, métrica e
contrastes, e documentar o protocolo final. Esse portão requer revisão separada.
