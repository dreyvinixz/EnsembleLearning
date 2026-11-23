# Agent guide — EnsembleLearning

Research repository. Preserve the distinction between retrospective label reconstruction,
causal prediction, and trading PnL. Never infer B0 label availability from a horizon name.
Do not ingest private data, secrets, external databases, or write to QuantBase/BovDB
without an explicit authorized data contract and read-only adapter.

Read docs/data-contract.md, docs/protocol.md, docs/roadmap.md and
docs/implementation.md before changing model/data code. Changes involving labels,
splits, feature generation or model selection require targeted tests and a discussion
of information availability. Do not evaluate the reserved TEST just to debug.

Do not claim experimental improvement or DAME novelty without evidence.
Prefer a small coherent change plus tests and source manifest over a speculative
end-to-end pipeline. Report what ran and what remains unimplemented.

## Handoff maintenance

Read `AI_HANDOFF.md` at the start of work. Update it after meaningful changes or
investigations with the outcome, evidence, validation, limitations, current
checkout state, and next steps. Preserve prior context; distinguish verified
facts from plans and unverified assumptions. Do not include credentials or
private data.
