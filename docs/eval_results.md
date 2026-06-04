# Ablation Results (A/B Evaluation)

Does each ML module actually improve debates? We ran the same topics under
increasing module sets and scored every turn with the LLM judge
(`scripts/eval_ab.py`).

**Setup:** 3 topics × 2 rounds × 3 agents · judge = `gemma3:4b` · higher = better.

| Config | Persuasion | Evidence | Attack |
|--------|-----------:|---------:|-------:|
| LLM-only | 0.711 | 0.772 | 0.522 |
| +RAG | 0.722 | **0.811** | 0.528 |
| +RAG+GNN | 0.711 | 0.800 | 0.500 |
| full (RAG+GNN+RL) | 0.717 | 0.744 | 0.533 |

## Verdict

- **RAG is the only module that earns its place.** Adding evidence retrieval
  raised the evidence score by **+0.039 (~5%)** and nudged persuasion up. Real
  citations give arguments something to stand on.
- **GNN adds nothing measurable.** `+RAG+GNN` is slightly *below* `+RAG`.
- **RL adds nothing measurable.** `full` does not beat `+RAG+GNN`; evidence drops.
- **full vs LLM-only:** persuasion +0.006 (noise), evidence −0.028.

**Conclusion: keep RAG; treat GNN and RL as `experimental`.** In this configuration
they show no demonstrated improvement. This is confounded with weak training (see
caveats) — it means "the modules as trained here add nothing", which cannot be
cleanly separated from "the modules are inherently unhelpful". They are kept as
integration demonstrations, not oversold.

## Caveats (why this is "no evidence of benefit", not "proven useless")
- Small sample (3 topics × 2 rounds); differences are within judge noise.
- A single small judge model (`gemma3:4b`) scores coarsely.
- The GNN's headline training metric (Delta Acc 0.956) is **misleading**:
  validation accuracy is frozen from epoch 0, i.e. it reflects class imbalance
  (most conversation nodes are non-delta) — the model largely predicts the
  majority class, not genuine persuasion. Consistent with GNN adding nothing here.
- RL reward rose during training (0.53 → 1.9), but its reward is derived from the
  GNN, so it learns the GNN's preferences rather than real debate outcomes.

## Takeaway
This is the point of the architecture: the ablation switches + this harness let
the project **measure and honestly report** which parts help. RAG earns its keep;
GNN/RL are kept as labeled experiments, not oversold.
