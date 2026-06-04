# Positioning & Framing

How to talk about this project — in the README, an SOP, or an interview — so that
its honest result reads as competence, not as a broken integration.

## The core principle

Do not frame this as *"a system that integrates RAG, GNN, and RL."* Frame it as
*"a study that asks whether they help"* — whose answer happens to be mostly negative.

Once the headline is a **question**, two modules not working stops being a liability
and becomes the finding. The negative result is the deliverable, not a footnote.

## One-line hooks

- **Repo / title:** *"A multi-agent LLM debate system, built as an instrument to
  test — not assume — whether RAG, GNN and RL improve persuasion. Honest ablation
  included."*
- **SOP topic sentence:** *"When does adding learned social signals (GNN/RL) to an
  LLM debate pipeline actually help? I built the system to find out — and measured
  that it mostly doesn't, then diagnosed why."*

## The narrative arc (five beats — order matters)

1. **Motivation** — Can social signals learned from real CMV conversation graphs make
   an LLM more persuasive?
2. **Build** — To *answer* rather than *assume*, the system is built to be measurable:
   one LLM seam, ablation switches (`USE_RAG/USE_GNN/USE_RL`), a reproduce harness.
3. **Result** — RAG helps directionally; GNN and RL show no measurable gain.
4. **Diagnosis** (the load-bearing beat) — The failure is not random. The GNN
   collapsed to the majority class under label imbalance, and the RL reward was
   *derived from that broken signal*, so the policy faithfully optimized a mis-measured
   target. Without a clean reward signal, stacking more models cannot help.
5. **Judgment learned** — Applying ML to behavioral data is bottlenecked less by model
   choice than by signal quality and measurement design.

Most people stop at beat 3 (or quietly skip it). Beat 4 is what lifts the project from
"student build" to graduate-level.

## SOP paragraph (drop-in)

> I built a multi-agent LLM debate system over the ChangeMyView corpus to test whether
> graph- and RL-learned persuasion signals could improve argument quality. Rather than
> assume they helped, I designed the system around an ablation harness that measures
> each module's marginal contribution. The result was negative: retrieval helped
> directionally, but the GNN collapsed to the majority class under label imbalance, and
> because the RL reward was derived from that signal, the policy optimized a
> mis-measured target. The project taught me that applying ML to behavioral data is
> bottlenecked less by model choice than by signal quality and measurement design — the
> methodological problem I want to work on.

## Three rules (do / don't)

| Principle | Do | Don't |
|---|---|---|
| Honesty is the selling point, not a disclaimer | Push the negative result to the front as the argument | Shrink it to a corner footnote |
| Frame as a question, not a product | Lead with the question you answered | Lead with "integrates X/Y/Z" |
| Sell judgment, not modules | Emphasize that you *diagnosed* the failure | Imply the modules nearly worked |

## The trap to avoid

Do not over-rhetoricize this into *"I discovered a new failure mode of RL
integration."* That a policy can't win on a reward derived from a broken estimator is
textbook; anyone in the field will see through the inflation and it costs you the
credibility you earned by being honest. The correct register is: *"this is a known
trap; I walked into it, recognized it, and explained it."* What you are demonstrating
is judgment, not new knowledge.
