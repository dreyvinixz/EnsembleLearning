# AI handoff — 2026-11-23

Projeto de artigo BovDB B0 integrado. Arquiteturas: Logistic Regression,
Random Forest, HistGradientBoosting; SVM futuro; MLP, 1D-CNN, LSTM e GRU.
GRU não é a abordagem exclusiva. Estratégias separadas: Independent,
MTL, Voting, Bagging, Stacking, Boosting, DAME hipotético.

Há três baselines clássicos com validação executável, holdout temporal,
contrato CSV binário provisório e arquiteturas neurais *sem treinamento*.
Nenhum experimento real BovDB, treinamento sequencial ou ensemble foi rodado.

Próxima fase: auditar o B0 original (classes/instante de confirmação) e
criar adaptador somente leitura para os dados autorizados; posteriormente
janelas temporais causais, treinamento MLP/CNN/LSTM/GRU e ensembles.
Preservar TEST e não alegar novidade ou rentabilidade do DAME.
