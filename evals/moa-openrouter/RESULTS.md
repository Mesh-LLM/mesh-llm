# MoA evidence: what actually helps?

Results from the live OpenRouter studies in
`crates/mesh-mixture-of-agents/tests/eval_openrouter.rs`. All tool-selection
numbers use the preregistered 40-task fixture (`tests/fixtures/ablation_tasks.json`,
4 strata × 10), 10 draws, paired hierarchical bootstrap
(`analyze_ablation.py`).

Method for every ablation: **one pinned actor, identical sampling / token
budget / prompt scaffold across arms — only the references vary.**

- **A** actor alone
- **B** actor + real references
- **C** actor + shuffled references (advice generated for a *different* task)

Primary metric: net uplift = P(rescue) − P(harm), equal-weight mean over tasks.
Arm C separates *advice content* from *extra tokens + a think-carefully prompt*.

## Headline: most of the "harm" was our packing, not references

| actor | reference packing | B pass | net uplift | 95% CI |
|---|---|---|---|---|
| strong (qwen3-32b) | original | 359/400 | −0.102 | [−0.170, −0.045] |
| strong (qwen3-32b) | Hermes-style | 385/400 | −0.037 | [−0.090, −0.003] |
| weak (qwen3-8b) | original | 365/400 | −0.013 | [−0.090, +0.070] |
| weak (qwen3-8b) | Hermes-style | **377/400** | **+0.017** | [−0.053, +0.100] |

Two monotonic effects:

1. **Fixing the packing helps in both actor conditions** (+0.065 strong,
   +0.030 weak).
2. **References are worth more to a weaker actor** (+0.054 weak-vs-strong at
   matched packing).

The only *statistically significant* cell in the matrix is the original-packing
strong-actor harm — i.e. the bug. After the fix, nothing is significant:
strong is marginal (upper bound −0.003), weak is a positive point estimate with
a CI spanning zero.

### What the packing bug was

Our references were packed with the agent's full system prompt, the tool-call
transcript, and a preamble instructing them to *"respond with your best answer
or tool call"* — while holding no tool schemas. So advisors (a) role-played the
actor instead of advising it, (b) anchored on the trajectory already taken
(destroying the error-independence aggregation depends on), and (c) emitted
tool-shaped prose that pulled the actor off its own better choice.

`context::pack_for_reference` follows Hermes: conversation user/assistant prose
only, no system prompt, no tool transcript, advisor framing, 600-token cap.

## Where references help: actor headroom

Weak actor + Hermes packing, per stratum:

| stratum | A alone | B real | C shuffled |
|---|---|---|---|
| inspect | 100/100 | 93/100 | 100/100 |
| search | 90/100 | **100/100** | 92/100 |
| execute | 80/100 | **84/100** | 62/100 |
| no_tool | 100/100 | 100/100 | 100/100 |

References **help exactly where the actor has headroom** (search +10, execute
+4) and **hurt where it was already perfect** (inspect −7). That is a gating
signal, not a global verdict.

Note `execute` arm C: irrelevant advice costs −18 points. Relevant advice on the
same stratum is +4. Advice content matters enormously; the risk is not "extra
tokens", it is *wrong-context* advice.

## Paired B vs C (advice content, both arms carry equal extra context)

| configuration | B−C | 95% CI | |
|---|---|---|---|
| weak + Hermes | +0.057 | [−0.018, +0.138] | spans 0 |
| weak + original | −0.010 | [−0.080, +0.062] | spans 0 |
| strong + Hermes | −0.015 | [−0.035, +0.000] | spans 0 |
| strong + original | −0.075 | [−0.125, −0.033] | **significant** |

With correct packing the content effect flips sign with actor strength: real
advice beats shuffled for a weak actor, and is indistinguishable for a strong
one.

## Interventions that did NOT help tool selection

| intervention | result |
|---|---|
| pre-hoc *structured* proposals (diverse vs homogeneous vs solo) | flat, 37–39/40 all arms |
| post-hoc deterministic correction (schema-validate + re-prompt) | +0.000 — never fired; the weak actor already emits structurally valid calls ~95% of the time |
| post-hoc semantic correction (different-family critic reviews the concrete call) | slightly negative — the revision still runs through the weak actor, so the capability gap persists |

Residual tool-selection failures are **semantic** (wrong tool for the job), not
structural. Validation and criticism cannot close a capability gap.

## Reasoning / answer turns (committee)

15 realistic agent-session reasoning turns, fixed aggregator, judged pairwise by
an out-of-pool different-family judge (gpt-4o-mini), position-swapped.

| comparison | win / tie / loss | sign test |
|---|---|---|
| committee (B) vs solo (A) | 6 / 2 / 2 | p=0.29 |
| layered (C) vs solo (A) | 5 / 2 / 3 | p=0.73 |
| layered (C) vs committee (B) | 2 / 2 / 6 | p=0.29 |

Directionally positive for a single-round committee, **not significant at
n=10**. Together's extra refinement round (`layers`) *loses* to single-round
synthesis while costing another full round of peer calls.

Caveats: every committee win was also the longer answer (length not cleanly
ruled out); 20/30 trials were dropped because the aggregator returned empty
content (reasoning-budget exhaustion — the `content: null` failure Hermes'
troubleshooting doc also documents).

## Reproducing

```bash
export OPENROUTER_API_KEY=...

# tool-selection ablation (2x2: actor strength x packing)
MOA_REFERENCE_PACKING=hermes MOA_ABLATION_ACTOR=qwen/qwen3-8b \
MOA_ABLATION_OUT=/tmp/x.jsonl \
cargo test -p mesh-mixture-of-agents --test eval_openrouter \
  ablation_scaled_study -- --ignored --nocapture

python3 evals/moa-openrouter/analyze_ablation.py /tmp/x.jsonl
```

Other studies: `matched_peer_structured_study`,
`correction_rescues_weak_tool_caller`, `committee_beats_solo_on_reasoning`.

## Status

Directional. 40 tasks × 10 draws is enough to detect the packing bug (a large
effect) but **not** to separate the remaining ±0.05 effects. A merge-blocking
claim needs the preregistered protocol: ~40+ held-out stratified tasks, k≥10,
the production-selected actor, and end-to-end agent-task success rather than
first-tool label match.
