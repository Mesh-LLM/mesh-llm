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

## Reasoning / answer turns (committee) — the one place MoA clearly wins

40 preregistered agent-session reasoning turns (4 strata × 10) × 3 draws = 120
trials. Fixed aggregator (`qwen3-32b`); peers `qwen3-14b`,
`mistral-small-24b`, `minimax-m2.5`. Judged pairwise by an **out-of-pool,
different-family** judge (`gpt-4o-mini`), **position-swapped** — a win counts
only if it survives both orderings, otherwise it is a tie.

- **A** aggregator alone
- **B** committee: aggregator synthesizes 3 peer drafts (single round)
- **C** layered: peers first refine seeing each other's drafts, then synthesize
  (Together's `layers`)

| comparison | win / tie / loss | mean | 95% CI | sign test |
|---|---|---|---|---|
| **committee (B) vs solo (A)** | **86 / 16 / 18** | +0.567 | [+0.392, +0.733] | **p = 8.2e-12** |
| **layered (C) vs solo (A)** | **90 / 11 / 19** | +0.592 | [+0.408, +0.758] | **p = 3.1e-12** |
| layered (C) vs committee (B) | 57 / 30 / 33 | +0.200 | [+0.008, +0.392] | p = 0.015 |

Consistent across every stratum (B vs A): planning 25/2/3, explain 22/2/6,
code_review 20/6/4, reason_over_output 19/6/5.

### Length control

The verbosity confound runs the *opposite* way here, which strengthens the
result:

| | mean chars |
|---|---|
| A solo | 3136 |
| B committee | 2548 |
| C layered | 2196 |

The committee produces **shorter** answers than solo and still wins. Restricted
to the 61 trials where B was shorter than A, B wins **40–14** (p = 5.4e-4). So
the preference is not length-driven.

### This reverses the pilot — and why

An earlier 15-prompt pilot found B vs A at 6/2/2 (p=0.29, "not significant")
and layered *losing* to single-round 2/2/6. Both conclusions were wrong,
because 20 of 30 pilot trials were silently dropped: `response_text` read only
`/message/content`, so a reasoning model that spends its budget in `reasoning`
and returns `content: null` looked like an empty answer. That dropped exactly
the trials where the aggregator struggled — a biased sample. With the fallback
fixed, **0 of 120 trials skipped**.

The earlier claim "Together's layering is negative value, don't build it" is
**retracted**: layered beats solo about as strongly as single-round does, and
edges single-round itself (p=0.015, CI lower bound +0.008 — the weakest of the
three results, and it costs an extra round of peer calls).

### Caveats

- Prompts are authored for this repo's domain, not a standard benchmark; these
  numbers are **not** comparable to AlpacaEval-style scores.
- One aggregator, one peer set, one judge. A single judge model is the main
  residual risk; self-preference is unlikely (judge is OpenAI-family, pool is
  Qwen/Mistral/MiniMax) but unmeasured.
- Judged answer quality, not task success in a real agent loop.

## The mesh case: can a pool of small models beat its best member?

The question that decides whether mesh MoA is worth running on consumer
hardware. Same 40 prompts × 3 draws, same judge and controls — but the whole
pool is 8B-class and **diverse by family**, the shape a few laptops actually
have:

- aggregator `qwen/qwen3-8b`
- peers `meta-llama/llama-3.1-8b-instruct`, `ibm-granite/granite-4.1-8b`,
  `mistralai/ministral-8b-2512`

| comparison | win / tie / loss | mean | 95% CI | sign test |
|---|---|---|---|---|
| committee (1 round) vs solo | 26 / 75 / 19 | +0.058 | [−0.092, +0.200] | p = 0.37 **ns** |
| **layered (2 rounds) vs solo** | **42 / 66 / 12** | +0.250 | [+0.100, +0.400] | **p = 5.2e-05** |
| **layered vs committee** | **39 / 73 / 8** | +0.258 | [+0.133, +0.392] | **p = 5.5e-06** |

**Yes — but only with the refinement round.** For a small pool, single-round
synthesis is indistinguishable from the aggregator working alone; the
cross-peer refinement round is what produces the gain. Compare the strong
aggregator above, where single-round already wins big and the extra round adds
comparatively little (p=0.015).

Reading: with weaker members the aggregator has little to work with until the
peers have *seen each other* and improved their drafts. That is the mechanism
Together's `layers` provides, and it matters most exactly where mesh operates.

This retracts (again) the pilot claim that layering is negative value. It is
essential for consumer-hardware pools and merely marginal for strong ones —
hence `RefinementPolicy::Auto` gates on pool shape rather than always/never.

Caveat: the small pool's ties dominate (75/120 for single-round), so the
effect is real but smaller than the strong-aggregator case; and answers here
are *longer* than solo (3654 vs 4064 chars for B, 3839 for C), so unlike the
strong-pool result the length control is not clean — the C-vs-A win restricted
to shorter-C trials is 21–12, p=0.16.

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

**Tool selection — directional, no clear win for multi-model.** 40 tasks × 10
draws detects the packing bug (a large effect) but cannot separate the
remaining ±0.05 effects. Every intervention tried (prose advice, structured
proposals, deterministic correction, semantic correction) was null-to-harmful
against simply routing to a capable model. Correctly-packed references are
roughly break-even, positive for a weak actor, mildly negative for a strong
one — hence gating on actor headroom rather than always/never.

**Reasoning/answer turns — a clear, robust win.** 120 trials, p < 1e-11,
consistent across all four strata, and the length confound runs against the
result rather than explaining it. This is where multi-model MoA earns its cost.

The task split follows from the evidence: **route tool turns to the best
tool-caller; convene the committee on reasoning/answer turns.**

Still outstanding for a merge-blocking claim: end-to-end agent-task success
(not first-tool label match, not judged answer quality), the production-selected
actor rather than a pinned one, and a second judge model to bound
single-judge risk.
