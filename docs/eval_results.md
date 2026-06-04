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

> **What "Persuasion" here means (and what it does *not*).** This column is scored by
> the **LLM judge reading the generated argument text** (`evaluate_response_effects`,
> prompt: *"how likely it is to move a reasonable opponent"*). It is **independent of
> the GNN.** The GNN has its own, separate `persuasion_prediction` — a would-be signal
> fed into the analysis/RL reward — and *that* is the part the ablation finds empty.
> So the debates are genuinely persuasive, but that quality is the **LLM's**: what is
> hollow is the **GNN's claimed contribution** to persuasion, not persuasion itself,
> and not the CMV delta labels it was trained on (real data; the model just failed to
> learn from it under class imbalance).

## Verdict

First, the honest statistical frame, applied **evenly**: at 3 topics × 2 rounds
scored by a single small judge (`gemma3:4b`), every difference in the table sits
inside judge noise. **No module reaches significance — RAG included.** What the
ablation gives us is not certification but a *diagnosis*, and the three modules are
not equally well-supported:

- **RAG — best-supported, but not "proven."** It shows a directional gain on the
  exact metric it should affect (evidence, **+0.039**, ~5%) and has a clean causal
  mechanism: real citations give arguments something to stand on. Among the three it
  is the only module with both a directional signal *and* a believable mechanism, so
  it is the one worth keeping — but at this sample size that is "consistent with
  helping," not "demonstrated." Marking it a confirmed win on the same evidence we
  call GNN/RL inconclusive would be a double standard.
- **GNN — no signal, and a known-broken one.** `+RAG+GNN` sits slightly *below*
  `+RAG`. More importantly its non-result is *uninformative*: training collapsed to
  the majority class (Delta Acc 0.956 is frozen from epoch 0 — class imbalance, not
  learned persuasion), so the GNN never produced a trustworthy persuasion signal to
  begin with. We can't conclude much from a measurement whose instrument is broken.
- **RL — structurally couldn't win here.** `full` does not beat `+RAG+GNN`, and
  evidence drops (0.744 vs 0.800). This is not bad luck: RL's reward is *derived from
  the GNN*, so the policy faithfully optimizes a mis-measured target. Stacking a
  policy on top of a broken estimator is doomed by construction — the ablation just
  records the inevitable.
- **full vs LLM-only:** persuasion +0.006 (noise), evidence −0.028.

**Conclusion.** At its current power this harness cannot certify *any* module, RAG
included. What it *can* show is the diagnosis above: the one module with a directional
signal and a clean reward/mechanism (RAG) is the one that looks useful, and the
GNN → RL chain fails for an identifiable reason — no clean reward signal — rather than
at random. Keep RAG as the best-supported (not proven) module; keep GNN and RL as
labeled, `experimental` diagnostic demonstrations. The point of the project is to
*measure and diagnose* this honestly, not to oversell any of it.

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
This is the point of the architecture. The ablation switches + this harness let the
project **measure honestly** and, where a module fails, **diagnose why** — here, that
naively stacking a GNN-derived reward under an RL policy can't win when the GNN signal
is itself broken. The result is reported the same way for good and bad news: nothing
is significant at this scale, RAG is the best-supported module without being proven,
and GNN/RL are kept as labeled experiments whose failure is itself the finding — not
oversold.
