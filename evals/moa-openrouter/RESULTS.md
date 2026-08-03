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

**Note on the judge.** These numbers were collected with the pre-fix judge
wording that was later found to reward length (see "Withdrawn" below). This
section's result survives that finding, because the bias ran *against* the
winner here: the shorter arm won anyway, and won on the shorter-only subset.
The small-pool and e2e sections did not have that protection and were re-run.

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

**These are the length-controlled numbers** (n=80). See the judge-bias section
below for why the earlier, larger figures are withdrawn.

| comparison | win / tie / loss | sign test |
|---|---|---|
| committee (1 round) vs solo | 6 / 73 / 1 | p = 0.125 **ns** |
| **layered (2 rounds) vs solo** | **11 / 68 / 1** | **p = 0.0063** |
| layered vs committee | 3 / 77 / 0 | p = 0.25 **ns** |

**Yes — but only with the refinement round.** Single-round synthesis is
indistinguishable from the aggregator working alone; layering is what produces
the gain, winning 11–1 on decided trials.

Reading: with weaker members the aggregator has little to work with until the
peers have *seen each other* and improved their drafts. That is the mechanism
Together's `layers` provides, and it matters most exactly where mesh operates.

Honest scale: ties dominate (68/80). On most prompts a small mesh and a single
small model are indistinguishable; the mesh wins a minority and almost never
loses. That is a real but modest effect, not the large one the first pass
reported.

### Withdrawn: the length-biased numbers

The first run of this study scored **42/66/12, p=5.2e-05** for layered-vs-solo,
and **39/73/8** for layered-vs-committee. Both are withdrawn.

The judge was asked which response was "more accurate, complete, and useful".
"Complete" reads as "longer", and the judge duly scored length. Measured on the
e2e run with the same judge:

| | n | win | loss | winrate |
|---|---|---|---|---|
| MoA answer **longer** than solo | 25 | 13 | 0 | 100% |
| MoA answer **shorter** than solo | 55 | 4 | 24 | 14% |

point-biserial r(length delta, verdict) = **+0.681**.

Re-run with a judge told to score correctness and relevance only, and that
length is explicitly not quality, r fell to +0.132 and most former "wins"
became ties. The direction survived; the magnitude did not.

This is the same control the strong-pool section applies — it was simply never
carried into the small-pool and e2e harnesses.

## Eval-vs-production fidelity

A measured gain only counts if the shipped path reproduces the measured
configuration. Three gaps were found and closed after the numbers above were
collected — all in the same class as the reference-packing bug, where code that
looked equivalent was not:

| | measured in eval | shipped (before) | now |
|---|---|---|---|
| refinement input per draft | untruncated (~3.8k chars) | 1200 chars (~30%) | 4000 chars |
| reducer payload per answer | untruncated (~3.8k chars) | 500 chars (~13%) | 4000 chars (text) |
| refinement prompt | aggregator wording + `[Response N]` | different wording + `[Answer N]` | matches eval |

The truncation gaps were the serious ones: the reducer was seeing ~13% of each
refined answer, discarding most of exactly what the refinement round produces.
Tool turns deliberately keep the tight 500-char bound — there the signal is the
proposal itself and long prose crowds out the schemas.

Sampling already matched (`SamplingParams::worker()`, thinking off, 1024
tokens).

**Implication for reading the numbers above:** they were produced by the eval
harness, and production now matches that configuration — but the small-pool
result has not been *re-measured* through the shipped code path since these
fixes. The engine is transport-agnostic and the packing is now identical, so
the gain should carry; that is an expectation, not an observation.

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

**Reasoning/answer turns, strong pool — a clear win.** 120 trials, p < 1e-11,
consistent across all four strata, and the length confound runs *against* the
result rather than explaining it (the winning arm was the shorter one).

**Reasoning/answer turns, small pool — a real but modest win, and only with
refinement.** Length-controlled: layered beats solo 11–1 on decided trials
(p = 0.0063), single-round does not (6–1, ns). Ties dominate at 68/80 — on most
prompts a small mesh and a single small model are indistinguishable.

**End-to-end through `handle_turn` — parity, not yet a win.** Latest:
9 / 59 / 12 (p = 0.66) against the pool's best member; the prior run was
5 / 65 / 10 (p = 0.30). Both are parity, and the difference between them is
noise at this sample size.

Six eval-vs-production divergences were found by measuring the shipped path and
fixed: grace finalizing the turn before refinement could run, two truncation
bounds that discarded most of each answer, two prompts that contradicted the
measured configuration, and named-vs-anonymous reducer inputs. A seventh
hypothesis (removing the worker preamble from the reducer) was measured,
*rejected*, and reverted.

The gap that remains is unexplained:

| | win / tie / loss | sign test |
|---|---|---|
| harness (`refine` + `synthesize` helpers) | 11 / 68 / 1 | p = 0.0063 |
| shipped (`moa::handle_turn`) | 9 / 59 / 12 | p = 0.66 |

Same models, same prompts, same judge, same packing. The mechanism works when
driven directly; something in the shipped orchestration still costs the gain.

Two prompt-level explanations were tested and **rejected**:

| change | decided-trial winrate | MoA output |
|---|---|---|
| baseline (v7) | 5/15 = 33% | 3606 |
| anonymize reducer inputs (v8) | 9/21 = 43% | 3679 |
| also drop preamble + "Reason for synthesis" (v9) | 8/23 = 35% | 3314 |

Dropping the worker preamble shortened output and lost ground on both
occasions it was tried (v6 3534 chars, v9 3314) versus keeping it (v7 3606,
v8 3679), so it was reverted twice. Reading: the preamble ("the best parts of
each will be combined; give your most accurate and complete answer") does
useful work on the reducer even though it is nominally addressed to a worker.
**Matching the harness exactly is not automatically right** — the harness sent
no system prompt at all, production sends one, and the preamble evidently
compensates.

Anonymization (v8) is retained: it is what Hermes does and what the study
measured, and it did not hurt. But 43% vs 33% on ~20 decided trials is not a
result; both runs are parity.

Still unruled-out: the arbiter short-circuiting synthesis when refined drafts
converge (74/80 turns did reach the reducer, so partial at most), and
differences in what `normalize_worker_output` does to prose before refinement
consumes it. Neither has been tested.

After two rejected hypotheses in a row, the honest read is that the remaining
gap is not another prompt-wording difference. It needs a diff of the actual
prompt bytes sent by each path on the same input, not more guesses.

Caution on the numbers above: r(length, verdict) was +0.465 in the latest e2e
run versus +0.132 in the small-pool study, so length bias is not fully
suppressed even with the corrected judge. Treat single-run e2e deltas as
directional only.

The task split follows from the evidence: **route tool turns to the best
tool-caller; convene the committee on reasoning/answer turns.**

Outstanding before this is a merge-blocking claim:

- close the remaining harness-vs-production gap (parity → the harness's 11–1)
- end-to-end agent-task success, not judged answer quality
- a second judge model to bound single-judge risk
- 2-node mesh validation (everything here is measured through the engine, not
  gossip)
