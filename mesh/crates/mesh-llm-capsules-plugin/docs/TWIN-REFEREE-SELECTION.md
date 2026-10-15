<!-- SPDX-License-Identifier: Apache-2.0 -->
# Choosing a referee: who is eligible

Sometimes the host sends one request to two machines (twins). When their
answers differ, this node can ask a third machine, a referee, for an
independent check. The referee re-answers the same request and signs a
verdict: which twin its answer agrees with, or that it cannot tell.

This page says when a referee is asked, who can be asked, and which one is.

## 1. When a referee is asked

Only when all of these hold:

- a client marked the two exchanges as one pair (the `x-mesh-twin-bracket` header, which the host passes on as a twin bracket id);
- both twins answered the same request, at temperature 0;
- both served the same model and the same weights, as their signed records say;
- their answers differ. Any difference counts: there is no similarity
  threshold.

At most once per pair. A pair is the twin bracket id, the request and the two
machines, so re-sealed records of the same exchange are the same pair. A call
that was made and not answered has used the pair's one call; it is not
retried, and another referee is not tried.

It is off by default; the operator turns it on (`adjudicate_differing_twins: on`).

## 2. Who is eligible

A node is eligible only if all of these hold:

- it serves the twins' model hash and weights digest;
- it is neither twin;
- it announced a key, so it can sign a verdict;
- this node has not stopped routing to it;
- it has no referee-signed contradiction for that model, in this node's own
  counts, inside the bar window.

**The bar window.** After a contradiction for a model, a node is not asked to
referee that model for D days: D is `referee_bar_days`, 30 unless the operator
sets it. The window is `[t, t + D)`, where `t` is when THIS node recorded the
verdict, never a time the referee wrote, so a referee cannot backdate a bar
away. A later corroboration does not end it early. It is per model: a bar for
one model leaves the node eligible for another. The contradiction stays in the
counts and on the page.

## 3. Which eligible node is asked

- **Tier 1:** eligible nodes with a corroborated verdict for that model on
  this node's own chain.
- **Tier 2:** eligible nodes with none yet.

The pick is random within tier 1. Tier 2 is used only when tier 1 is empty.
The tier is sealed in the verdict: the referee signs the tier it was asked
from, a verdict with an altered tier fails verification, and a verdict that
seals another tier than this node asked at is refused.

A node whose bar has lapsed is in tier 2 until it has a corroboration for that
model dated after the lapse.

## 4. When nobody is eligible, or nobody answers

The pair reads "Not adjudicated" with the reason, and nothing counts against
either twin:

| Reason | The line |
| --- | --- |
| the check is off | Not adjudicated: the independent check is turned off on this node. |
| no twin bracket id | Not adjudicated: no client marked these as a pair. |
| the answers agree | Not adjudicated: the two answers agree. |
| sampled, another or unnamed model, other or unknown weights | Not adjudicated: not comparable, with why. |
| nobody eligible | Not adjudicated: no eligible referee. |
| the referee cannot sign | Not adjudicated: the referee asked cannot sign a verdict. |
| the referee did not answer | Not adjudicated: the referee asked did not answer. |

A pair that found no eligible referee made no call, so its one call is not
used, but it is not retried on its own. The operator may ask again
(`referee_ask_again`): that looks for a referee again, and a call it makes is
the pair's one call.

Without a twin bracket id from the host nothing is ever called: pairs are never
guessed from timing.

The re-answer needs the pair's request, and the referee both answers. A host
passes no text to plugins, so the client that marked the pair supplies them
(`referee_supply_twin_texts`). They are kept only when each is the body this
node's own records sealed (the request's digest is both records'
`request_digest`, each answer's is one record's `response_digest`) and the
answers differ; then the pair is considered as above.

A pair is decided once at a time, and its one call is recorded before it is
made: if this node stops mid-call, the call counts as made and not answered,
and no further call follows on its own.

## Known limits

- This node records a verdict's time to the minute, so a bar can end up to
  59 seconds before exactly `t + D`.
- A contradiction that names no model hash bars nothing: a bar is per model,
  and such a verdict names none. Every verdict this plugin's referee signs
  seals the model hash.
- The referee call (the re-answer, then the adjudicate request) runs in the
  handler of the exchange that completed the pair, so that handler can wait up
  to the re-answer and stream timeouts.

## 5. What this is not

These are yes/no rules applied to this node's own verified records. Nothing is
ordered by merit, weighted or combined into a number, and choosing sends
nothing to other nodes. Asking the chosen referee does: it receives the twins'
request, including the prompt, and both twins' answers (off by default:
`adjudicate_differing_twins`). A verdict is a ruling on one pair of answers; it proves nothing
about a node's future answers.

## 6. Settings

| Setting | Environment variable | Default |
| --- | --- | --- |
| `adjudicate_differing_twins` | `CAPSULES_ADJUDICATE_DIFFERING_TWINS` | off (`on`, `1`, `true` or `yes` turns it on) |
| `referee_bar_days` | `CAPSULES_REFEREE_BAR_DAYS` | 30 |
| `stop_routing_after_contradictions` | `CAPSULES_STOP_ROUTING_AFTER_CONTRADICTIONS` | off |
| `stop_routing_window_days` | `CAPSULES_STOP_ROUTING_WINDOW_DAYS` | 30 |

The last two are the opt-in stop-routing rule: with N set, this node stops
routing to a peer after N referee-signed contradictions within D days, at most
N - 1 of them from any one referee, through the host's own stop-routing path.
Undo it the same way as a manual block; the verdicts that met the rule never
count toward it again.

The referee's re-answer goes through the host's OpenAI-compatible API
(`CAPSULES_OPENAI_API_URL`, default `http://127.0.0.1:9337`), routed to
the chosen node.

## Where this is checked

The rules above are held by the referee parity corpus
(`vectors/parity/referee/`, run by the plugin's tests) and by unit tests of
their own names; the three rules for the bar's edge, a lapsed bar and a pair
with nobody eligible are also each pinned by a named constant
(`referee::bar::BAR_WINDOW_INCLUDES_ITS_END`,
`referee::bar::LAPSED_BAR_STARTS_WITHOUT_HISTORY`,
`referee::request::NO_ELIGIBLE_IS_RETRIED_AUTOMATICALLY`).
