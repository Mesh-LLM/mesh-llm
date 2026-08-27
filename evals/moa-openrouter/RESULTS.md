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

## Admission control: does a weak node help a strong pool? (No.)

The A/B/C test the goal hinges on — should a modest node be admitted into a
committee that already has a stronger member? Same 40 prompts, same judge,
layered arm, through the committee harness (no admission control, so C is the
counterfactual "what if we admitted it"):

| arm | pool | layered vs solo | decided winrate | losses |
|---|---|---|---|---|
| B | 32B ×2 | 48W / 23T / 2L, p=2e-12 | 96% | 2 |
| C | 32B ×2 + 8B | 50W / 25T / 5L, p=2e-10 | 91% | 5 |

Fisher exact B-vs-C: p=0.44 — not statistically separable at n=80, but the
direction is one-way: **admitting the weak 8B node never helped and modestly
raised losses (5 vs 2).** Both still beat solo — a weak node does not collapse
the pool — but it adds latency and cost for no upside and a small tail risk.

This is the measured basis for tier-based admission control
(`apply_admission_control`): when a big-tier worker is present, drop small-tier
ones. The conservative choice, and the evidence says conservative is right here.

Caveat: this rejects the *possibility* that a genuinely complementary weak model
could help a specific prompt. The data says that possibility is not worth the
average-case cost at these scales; a per-turn admission signal (measured
marginal contribution) could revisit it later, but tier is the safe default now.

### But only when a committee survives the exclusion

Arm C dropped the 8B from a pool that *still had two 32B*. The other case — one
strong + one weak, where dropping the 8B collapses the pool to a solo 32B — is
different, and admission must NOT drop there:

| pool | vs solo 32B | length r |
|---|---|---|
| 32B + 8B, layered | 47W / 27T / 5L, p=1.3e-9 | −0.01 (clean) |
| 32B + 8B, single-round | 40W / 31T / 8L, p=3.3e-6 | +0.02 |

A mixed committee beats a solo strong model decisively. So the rule is not
"drop small whenever a big is present" — it is **drop small only when ≥2 big
remain**. Dropping to protect quality is right when a real committee is left; it
is wrong when it would throw away MoA entirely.

This is exactly the core mesh case: a modest node joining a single strong node
*does* help (47/5), and admission control now keeps it. Adding a modest node to
an *already-strong committee* does not help (arm C), and admission drops it.
Both are handled by the same "≥2 big remain" gate.

## The problem with N=2 was scale, not count

The N=2-8B null (below) suggested "two peers isn't enough". A mid-scale re-run
refutes that reading: the count was fine, the 8B *members* were too weak.

Same 40 prompts, same judge, same shipped `handle_turn`, layered arm:

| pool | layered vs best member | p (sign) | length r | MoA vs solo length |
|---|---|---|---|---|
| N=2, 8B (Qwen + Meta) | 2W / 75T / 3L | 1.00 | n/a (5 decided) | — |
| N=3, 8B (+ IBM) | 3W / 76T / 1L | 0.63 | — | — |
| N=4, 8B (+ Mistral) | 11W / 68T / 1L | 0.006 | — | — |
| **N=2, mid (32B + 24B)** | **49W / 24T / 6L** | **2e-9** | **−0.04** | MoA shorter (2516 vs 3064) |
| N=4, strong (32B agg) | 90W / 11T / 19L | 3e-12 | +0.30 | MoA shorter (2196 vs 3136) |

The N=2-mid result is the cleanest in the whole study: no length confound
(r=−0.04), MoA answers *shorter* than solo, and it still won 93% of the trials
where it was shorter. Two mid-size models beat one of equal strength decisively.

So diversity/capability of the *members*, not the raw count, is what matters —
at 8B you need ~4 models before it pays; at 24–32B, 2 already win.

**Resolved: diversity is not the active ingredient — ensembling is.**
Compute-matched Self-MoA (two samples of the *same* qwen3-32b, same aggregator,
same drafts+refine+synthesize) was run against Mixed (qwen3-32b + mistral-24b):

| arm | layered vs solo | p | length r |
|---|---|---|---|
| Mixed (2 different models) | 49W / 24T / 6L | 1.8e-9 | −0.04 |
| Self (same model ×2) | 48W / 23T / 2L | 2.3e-12 | −0.04 |

Mixed vs Self: Fisher exact p = 0.27 — **statistically indistinguishable**.
Both crush the single best member, both with no length confound and MoA answers
*shorter* than solo (winning ~93% of shorter-MoA trials in each arm).

So different-family membership is **not required**. The active mechanism is
test-time ensembling — several sampled drafts (workers run at temperature 0.8,
so repeated draws genuinely differ), a cross-peer refinement round, and
synthesis — and it works whether the drafts come from different models or
repeated sampling of one. This matches the Self-MoA paper (arXiv:2502.00674):
proposal quality and sampling, not heterogeneity, carry the gain.

Practical consequence for a mesh: the value does not depend on curating a
diverse pool. Any ≥2 reasonably-capable participants — distinct models *or*
repeated instances of one — beat picking a single model, once they are strong
enough individually (mid-scale here; 8B needs ~4).

## The 8B ladder: how many small models beat one?

Does stacking 8B models beat a single 8B? Same prompts/judge/harness, layered arm:

| pool | vs one 8B | p |
|---|---|---|
| 2× 8B, different (Qwen + Meta) | 2W / 75T / 3L | 1.00 |
| 2× 8B, **same** (Qwen ×2) | 2W / 78T / 0L | 0.50 |
| 3× 8B, different | 3W / 76T / 1L | 0.63 |
| 4× 8B, different | 11W / 68T / 1L | 0.006 |

Two results worth stating plainly:

- **You need ~4 at 8B.** Two or three 8B models don't beat one; four do. There
  is a floor below which stacking small models buys nothing.
- **Same-vs-different makes no difference at N=2** (2W/3L vs 2W/0L, both null),
  exactly as the mid-scale Self-vs-Mixed test showed. Count and member
  strength drive the result; family identity does not.

The one gap in this ladder: 4× 8B *same-model* (four qwen3-8b instances) was not
run, so "does the 8B N=4 win need distinct models or just four drafts?" is
open. Mid-scale Self-MoA predicts four drafts alone would suffice, but that is
an extrapolation.

## How many peers does it take? (N=2 vs N=4)

Same 40 prompts, same judge, same harness — only the pool size differs. N=2 is
`qwen3-8b` + `llama-3.1-8b` with qwen3-8b aggregating (Together's
`advanced-moa.py` shape, where the aggregator is also a reference).

| pool | 1 round vs solo | 2 rounds vs solo | decided trials |
|---|---|---|---|
| **N=2** (Qwen, Meta) | 2W / 75T / 3L, p=1.0 | 2W / 75T / 3L, p=1.0 | 5 of 80 |
| **N=4** (+ IBM, Mistral) | 6W / 73T / 1L, p=0.125 | **11W / 68T / 1L, p=0.0063** | 12 of 80 |

Fisher exact on decided win/loss, N=2 vs N=4: **p = 0.053**.

**Two peers was not enough.** At 8B scale, adding one different model produced
no measurable gain — 2 wins against 3 losses, indistinguishable from noise. The
same harness with four models across four families wins 11–1.

Two things move together as peers are added: the number of *decided* trials
rises (5 → 12; more peers produce more differentiated output rather than
near-identical answers the judge calls a tie), and the win share among those
rises (40% → 92%).

Caveats: only 5 decided trials at N=2, so this is weak evidence of absence, not
evidence of no effect. And it is a claim about *small* models — Hermes reports
a two-model preset (`claude-opus-4.8` aggregating a `gpt-5.5` reference) at
0.8202 vs 0.7607 for the stronger model alone, so frontier pairs may behave
differently. Untested here.

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

**End-to-end through `handle_turn` — was a significant LOSS; the refinement
fix removes it but does not reach the harness win.** Measured 2026-08-27 at
`lab/moa-fleet-sim` `e4b611e8d`, both arms through the shipped entrypoint,
40 tasks x 2 draws, same pool and judge:

| arm | win / tie / loss | decided | sign p |
|---|---|---|---|
| single-round (the behaviour that shipped) | 5 / 60 / 15 | 20 | **0.041** |
| refinement on (`Auto` after the fix) | 7 / 66 / 5 | 12 | 0.77 |

The cause found: `RefinementPolicy::Auto` excluded all-small pools, so the
shipped path ran the arm this file measured as null (6/73/1) while the harness
ran the arm that won (11/68/1). The exclusion was justified by the withdrawn
6x-8B single-aggregation cell.

Earlier runs recorded 9 / 59 / 12 (p = 0.66) and 5 / 65 / 10 (p = 0.30) and were
read as parity; the direction was consistent with this loss all along.

Direct arm comparison is Fisher p = 0.13 — directional, not separated, at
~12-20 decided trials per arm.

A second pool was run to test the policy itself: **2x `qwen/qwen3.5-9b`** (the
Buzz mid rung, and a same-model pair is what self-fill builds), both arms, same
tasks and judge:

| pool | arm | win / tie / loss | sign p |
|---|---|---|---|
| 2x 9B same | single-round | 5 / 74 / 1 | 0.22 |
| **2x 9B same** | **refine** | **11 / 69 / 0** | **0.00098** |

**Both rows above are superseded — they were underpowered.** Re-run at 5 draws
(~200 trials/arm instead of 80), with a second judge scoring the identical
answer texts:

| arm | n | judge1 W/T/L | p | judge2 W/T/L | both-judges W/L |
|---|---|---|---|---|---|
| refine | 197 | 29 / 160 / 8 | 0.00075 | 116 / 66 / 15 | 26 / 8 (p=0.0029) |
| single-round | 200 | 23 / 174 / 3 | 0.00009 | 90 / 101 / 9 | 15 / 1 (p=0.00052) |

Three conclusions changed:

1. **Single-round is NOT null on this pool.** 5/74/1 (p=0.22) at 80 trials
   becomes 23/174/3 (p=0.00009) at 200. The earlier "single-round is null,
   layered is the entire win" reading was a sample-size artifact.
2. **Refine is not measurably better than single-round.** Fisher on decided
   win/loss: p=0.50 (judge1), p=0.67 (judge2). With ~10x the decided trials the
   arms *converged* rather than separated, so the extra ~N calls per turn buy
   no detectable quality here.
3. **The length confound explains both wins.** MoA output is longer on both
   arms. Every win sits in the longer-output subset (refine 19W/0L, n=131;
   single 18W/0L, n=134); in the MoA-shorter subset both arms are
   indistinguishable from solo (refine 10W/8L p=0.81; single 5W/3L p=0.73).
   r(length delta, verdict) = +0.387 / +0.405. The second judge shows the
   *same* bias, so two-judge agreement here is correlated error, not
   independence — it rules out one-judge idiosyncrasy and nothing else.

Consequence: do not lift the all-small collapse guard on this evidence, and the
refinement fix must rest on the 8B harm-reduction result (15 losses -> 5), not
on a 9B quality lift. The blocking methodological need is now a metric an LLM
judge cannot satisfy by writing more prose.

Full data: `RESEARCH/MOA_REFINEMENT_E2E_2026_08_27.md` (first run) and
`RESEARCH/MOA_9B_CONFIRMATION_2026_08_27.md` (this one) in the Buzz workspace.
Scored by `evals/moa-openrouter/analyze_e2e.py`.

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

## Width sprint: many small models (2026-08-05)

Aggregator = `qwen/qwen3-8b`, peers 8B-class, judge `gpt-4o-mini`,
position-swapped + length-noted, 2 draws × 40 prompts, shipped committee path.
Three arms per trial: single-aggregation (Hermes-shape, draft→synthesize),
layered (Together-shape, draft→synth→refine→synth), and the refine-vs-single
delta.

| pool | single-agg vs solo | layered vs solo | refine vs single-agg |
|---|---|---|---|
| 2× 8B diverse | 2W/77T/1L, p=1.0 | 3W/74T/3L, p=1.0 | null |
| 4× 8B diverse | 5W/73T/0L, p=0.06 | 6W/70T/2L, p=0.29 | 0W/77T/1L |
| **6× 8B diverse** | **12W/65T/2L, p=0.013** | 9W/69T/1L, p=0.021 | 2W/76T/1L |
| 6× qwen3-8b SAME | 4W/75T/1L, p=0.38 | 1W/79T/0L, p=1.0 | 1W/79T/0L |

> **WITHDRAWN — see "Withdrawal: the 6x8B small-pool win did not replicate" at
> the end of this file.** The 6× 8B cell below did not reproduce (3W/76T/1L,
> p=0.63 on a re-run of the same rig, same tasks, same pool), and the shipped
> path measured a *loss* at every small width. All-small pools now serve their
> best member instead of convening a committee. The rest of this section is kept
> for the record; do not cite the 6× 8B number.

Findings (as originally written; the first is withdrawn):

- ~~**Width is the small-model lever.** A committee of six diverse 8B models
  beats the single best member (12W/2L, p=0.013); four is only marginal
  (p=0.06); two/three are null.~~ Withdrawn: rested on ~14 decided trials out of
  80 and did not replicate. The tier-aware cap (6 all-small / 4 with a verified
  big) is retained, but an all-small pool no longer forms a committee at all.
- **The refine round never earns its serial cost.** `refine vs single-agg` is
  null in every cell (2/77/1 at N6). Hermes' single-aggregation cadence (one
  synth, no refine pass) is >= Together's layered shape here, at half the
  serial latency. The `RefinementPolicy` default should reconsider the extra
  round for small pools.
- **At 8B, diversity matters** (6 diverse 12W/2L vs 6 same 4W/1L, p=0.38),
  unlike mid-scale where Self ≈ Mixed. Weaker models decorrelate their errors
  better when they are genuinely different families; repeated draws of one 8B
  do not supply enough independent signal.

Open: 4B pools (is the floor lower or does it need even more width?), and
whether dropping the refine round for small pools recovers latency without
losing the width win.

## Withdrawal: the 6x8B small-pool win did not replicate (2026-08-06)

The width-sprint headline ("6x diverse 8B beats its best member, 12W/65T/2L,
p=0.013") is **withdrawn**. Re-running the same rig on the same 40 tasks, same
pool, same judge gives **3W/76T/1L (p=0.63)**, with 11 of 40 tasks flipping
verdict. The original rested on ~14 decided trials out of 80 (the rest ties) and
was a single unreplicated run selected from a width sweep; correcting for the
four pool widths compared puts it at roughly p=0.05 even on its own terms.

It was real arithmetic on real data, but never a robust effect, and it should
not have been published as a headline from one run.

### What the shipped path actually does at small scale

Through `moa::handle_turn` at production defaults, 8B-class peers with an 8B
reducer, vs the pool's best member alone (40 prompts x 2 draws, out-of-family
position-swapped judge, length logged):

| pool | result | decided | p |
|---|---|---|---|
| 2x 8B | 0W/43T/37L | 37 | <0.0001 |
| 4x 8B | 5W/63T/12L | 17 | 0.14 |
| 6x 8B | 5W/52T/23L | 28 | 0.0009 |

MoA answers are consistently shorter (3236-3372 chars vs ~4070 solo). The
committee never won at any width. Note the capable pool wins 28/29 of decided
trials where the MoA answer is *shorter*, which rules out "this judge simply
prefers longer answers" as the explanation.

**Action:** all-small pools no longer convene a committee; they collapse to the
best member and the caller degrades to serving it directly.

**Scope of the claim:** this measures a weak reducer synthesizing weak drafts.
It is not evidence that small-model MoA cannot work. The untested cell is small
peers with a *strong* reducer (the capable pool confounds peer strength with
reducer strength). If a mesh gains a big-tier model the pool is no longer
all-small and MoA engages again.

### Known methodology limitations

- `judge_pair` collapses genuine ties, position disagreements, and API/parse
  failures into the same `0` result, so "tie" is a garbage bucket and tie counts
  cannot be interpreted as agreement.
- Significance is computed per draw (80) rather than per prompt cluster (40),
  which overstates confidence; the capable-pool result survives either way, the
  small-pool ones are marginal.
- `pool[0]` is assumed to be the strongest member rather than verified.
- Single judge model.

## Grounded rubric metric + big-tier pool (2026-08-27, $2.30)

The judge was measuring length (r=+0.39 on *both* judges), so the metric was
replaced rather than the pool swept further. `rubric.py` scores answers against
per-task checkable items + wrong-claim decoys, judge-free, and **calibrates
itself**: of 135 hand-written items, 45 fired on 100% of answers and 9 on <10%
(broken regex, not hard items) — only the 81 discriminating items are scored.

Re-scoring the 397 saved 9B trials for free:

| arm | rubric W/T/L | p | length-matched (<10%) |
|---|---|---|---|
| refine | 54/55/36 | 0.073 | 15W/11L p=0.56 |
| single-round | 54/60/31 | 0.017 | 17W/14L p=0.72 |

Coverage *density* (items per 1k chars) is a dead heat both arms. **The 9B lift
does not survive length control on a grounded metric either.** Rubric agrees with
the judge on only 39-45% of rows, so it is measuring something else and still says
flat.

Only surviving directional signal, and it is not a length artifact — failure
rescue past a coverage floor of 0.34: single-round 19 rescued / 9 regressed
(p=0.087); below-floor fraction 28.3% solo -> 21.4% MoA.

**Big diverse tier — the expectation does not hold.** Pool `qwen3.8-27b,
gemma-3-27b-it, mistral-small-3.2-24b, gpt-oss-20b`, solo baseline the strongest
member, 120 trials through shipped `handle_turn`:

| metric | result |
|---|---|
| judge 1 | 28W/59T/33L p=0.61 |
| judge 2 | 39W/37T/44L p=0.66 |
| rubric (n=87) | 28W/38T/21L p=0.39 |
| rescue/regress | 8/9 |
| r(len, score) | +0.097 |

Both judges numerically negative; length confound nearly absent here, so there is
no bias to explain a win away — there is no win. **Mixing at the large tier is not
supported by this data.** Do not lift the all-small collapse guard either.

Detail: `RESEARCH/MOA_GROUNDED_METRIC_AND_BIG_TIER_2026_08_27.md` in the Buzz
workspace; artifacts in `RESEARCH/artifacts/moa-bigtier-2026-08-27/`.
