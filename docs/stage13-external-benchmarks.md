# Especificação preliminar da Etapa #13 — benchmarks externos

Pesquisa documental registrada em 2026-11-23. Nenhum código foi clonado,
executado ou incorporado. Revalidar branches, commits, dependências e licenças
antes de iniciar qualquer reprodução.

## Avaliação de compatibilidade

| Trabalho | Código público e licença observada | Dados/tarefa original | Compatibilidade com WIN/B0 |
|---|---|---|---|
| CRAFT (Zhang et al., 2025) | [Repositório dos autores](https://github.com/CRAFTinTSF/CRAFT), branch `main`; página de arquivos não apresenta licença. `requirements.txt` observado contém somente o typo `pytroch`. | Forecasting de séries temporais com *cross-future behavior*; o artigo apresenta dataset/código. | Baixa a moderada. É forecasting temporal geral, não classificação binária dos rótulos B0. O conceito de comportamento que antecede e afeta o futuro exige auditoria causal específica: nenhuma matriz CFB pode usar observações posteriores à decisão. Primeiro corrigir/validar ambiente e identificar protocolo/dados reais do artigo. Sem licença explícita, não redistribuir/adaptar código até esclarecer permissão. |
| LOBCAST (Prata et al., 2024) | [Repositório oficial](https://github.com/matteoprata/LOBCAST), branch `mini-LOBCAST`; README documenta modelos, execução e FI-2010. Nenhum arquivo LICENSE apareceu na listagem consultada. | Predição de tendências a partir de LOB; `mini-LOBCAST` prioriza FI-2010. Branch `v0-LOBCAST` descrita como contendo procedimentos multibase adicionais. | Moderada como protocolo/modelos, baixa como reprodução literal. Usa dados LOB/eventos e targets próprios, distintos de candles WIN e B0. Reutilizar arquitetura exige novo adapter, alvo B0 congelado, split walk-forward/purging do EnsembleLearning e auditoria de features disponíveis. Não comparar números do artigo diretamente. Licença precisa ser esclarecida. |
| TLOB (Berti & Kasneci, 2025) | [Repositório oficial dos autores](https://github.com/LeonardoBerti00/TLOB), licença MIT observada; [paper arXiv:2502.15757](https://arxiv.org/abs/2502.15757). Um fork identifica-se como oficial, mas usar e fixar o upstream dos autores. | Arquitetura de atenção dual temporal/espacial para LOB; FI-2010, NASDAQ/LOBSTER e Bitcoin; alvo e protocolo próprios, incluindo nova rotulagem. | Baixa para reprodução fiel, moderada para adaptação de arquitetura. A estrutura espacial por níveis do book não existe no snapshot OHLCV; não simular níveis ausentes. Uma adaptação a séries de candles seria um experimento novo, com input mapping explícito, não reprodução de TLOB. Preservar licença MIT, avisos e dependências. |

## Regras de desenho para uma comparação válida

1. Separar **reprodução de referência** (dados, targets e métricas originais) de
   **adaptação ao WIN/B0**. Resultados de datasets/tarefas diferentes nunca
   entram na tabela comparativa de MCC do BovDB.
2. Para adaptação WIN/B0, usar exclusivamente a população e o gate causal já
   aprovados: janela de 60 minutos, pelo menos 48 candles, mesmo contrato e
   sessão, rótulos B0 como targets e disponibilidade conservadora pós-sessão.
3. Calcular normalização apenas no treino de cada fold; auditar qualquer
   feature auxiliar e qualquer transformação de dados do artigo quanto a
   futuro, vazamento entre sessões e uso de rótulo.
4. Rodar primeiro teste de forma/forward pass em dados sintéticos. Depois, um
   smoke test pequeno limitado a Jan–Mai. Só então autorizar uma comparação
   walk-forward completa com orçamento e seeds pré-registrados.
5. Fixar commit/tag do upstream, hash do código e dados, licença, ambiente,
   hiperparâmetros, dispositivo e tempo. Isolar dependências em ambiente
   separado; não copiar arquivos de código para `src/` antes da revisão de
   licença e diferenças de implementação.

## Ordem recomendada

1. **TLOB:** melhor candidato para adaptação arquitetural, pois tem implementação
   pública MIT e protocolo explícito; entretanto a falta do book impede
   reprodução literal e a arquitetura adaptada deve ser descrita como tal.
2. **LOBCAST:** melhor candidato para comparar procedimentos e modelos de LOB;
   determinar primeiro a licença e inspecionar a diferença entre branches.
3. **CRAFT:** adiar até existir uma hipótese clara que use apenas covariáveis
   realmente observáveis no instante de decisão. O repositório está pouco
   documentado e o arquivo de dependências observado parece inválido.

## Portões antes da implementação

- Revisão manual de licença e política de dados dos datasets originais.
- Pin de commit e reprodução mínima das métricas originais quando os dados
  forem legalmente acessíveis.
- Mapeamento formal de entradas, targets, disponibilidade temporal e
  equivalência (ou não) de tarefa.
- Aprovação de uma experiência adaptada antes de consultar maio para escolha;
  junho permanece inacessível até congelamento total do protocolo final.

## Fontes consultadas

- Zhang et al., [CRAFT (arXiv)](https://arxiv.org/abs/2505.13896) e [repositório](https://github.com/CRAFTinTSF/CRAFT).
- Prata et al., [LOB-based deep learning models (Springer)](https://link.springer.com/article/10.1007/s10462-024-10715-4) e [LOBCAST](https://github.com/matteoprata/LOBCAST).
- Berti & Kasneci, [TLOB (arXiv)](https://arxiv.org/abs/2502.15757) e [repositório oficial](https://github.com/LeonardoBerti00/TLOB).
