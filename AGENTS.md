# Agent guide — EnsembleLearning

This research is not GRU-only. Keep predictive architecture (LR/RF/HGB,
MLP/CNN/LSTM/GRU, SVM future) separate from learning strategy (Independent,
MTL, Voting, Bagging, Stacking, DAME hypothesis).

Read docs/data-contract.md, docs/protocol.md, docs/methods.md and
docs/implementation.md before modifying data or model code.
Do not treat retrospective labels or regimes as causal inputs.
Do not infer label availability from horizon name; do not use TEST for
debugging. Do not copy BovDB/QuantBase secrets or source databases.
Require data contract approval, tests, and provenance before real training.
No claims of DAME originality, scientific improvement or financial profit
without independent evidence. Forward pass is not trained/validated model.
