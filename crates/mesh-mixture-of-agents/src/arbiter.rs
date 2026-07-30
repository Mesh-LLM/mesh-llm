//! Deterministic arbitration of worker outputs.
//!
//! The arbiter uses code, not models, to decide the outcome.
//! Models are only called (via the reducer) when there's genuine ambiguity.
//!
//! Decision priority:
//! 1. Unanimous tool proposal → emit tool call
//! 2. High-confidence tool proposal with no dissent → emit tool call
//! 3. Unanimous answers → pick highest confidence
//! 4. Conflicting outputs → escalate to reducer
//! 5. All uncertainty → escalate to reducer

use crate::normalize::{OutputKind, WorkerOutput};
use crate::worker::WorkerRole;
use serde_json::Value;

/// Pick the best tool proposal: prefer proposals that have arguments,
/// then by confidence.  A proposal without arguments (e.g. from a fast
/// worker that only got tool names in the system prompt) should lose to
/// one that has actual arguments.
/// Pick the tool proposal that the most workers actually back.
///
/// Agreeing on a tool *name* is not the same as agreeing on its
/// *arguments*. Recorded traces from 9 open-weight models (see
/// `evals/moa-openrouter/agentic.jsonl`) show all 9 workers choosing the
/// same tool while emitting up to 6 distinct argument sets — including one
/// worker hallucinating a `rust_project/` path prefix no other worker
/// used. Every OpenAI-shape tool call is normalized to a fixed 0.9
/// confidence by `extract_text_from_response`, so the confidence-only
/// tiebreak in [`best_tool_proposal`] was free to return that outlier.
///
/// Group proposals by canonically-compared arguments and let the largest
/// cluster win. `serde_json::Value` object equality is key-order
/// independent, so `{"path":".","pattern":"x"}` and
/// `{"pattern":"x","path":"."}` cluster together.
///
/// A no-arguments cluster never outvotes a real-arguments cluster — a
/// majority emitting bare calls shouldn't suppress the workers that
/// actually filled the schema in. Within the winning cluster, and for
/// ties, we defer to [`best_tool_proposal`] so behavior is unchanged
/// when every worker disagrees.
fn best_tool_proposal_by_consensus<'a>(proposals: &[&'a WorkerOutput]) -> &'a WorkerOutput {
    match decisive_argument_cluster(proposals) {
        Some(group) => best_tool_proposal(&group),
        // No decisive cluster (all argument-free, or an exact tie): fall back
        // to the old confidence/has-args ordering.
        None => best_tool_proposal(proposals),
    }
}

/// Group proposals by canonical arguments and return the strictly-largest
/// group, or `None` when there is no clear winner.
///
/// `None` means "these proposals do not agree on arguments": either every
/// proposal was argument-free, or the top two clusters are the same size. A
/// tie is the case that matters — with two workers proposing two different
/// argument sets, any pick is arbitrary, so callers should prefer waiting for
/// more workers over guessing.
fn decisive_argument_cluster<'a>(proposals: &[&'a WorkerOutput]) -> Option<Vec<&'a WorkerOutput>> {
    let mut clusters: std::collections::BTreeMap<String, Vec<&'a WorkerOutput>> =
        std::collections::BTreeMap::new();
    for p in proposals {
        // Canonical key: serialized Value. serde_json's default Map is a
        // BTreeMap, so serialization is key-sorted and deterministic, which
        // makes `{"a":1,"b":2}` and `{"b":2,"a":1}` cluster together.
        let key = match p.tool_arguments.as_ref() {
            Some(args) if args != &Value::Object(Default::default()) => {
                serde_json::to_string(args).unwrap_or_default()
            }
            _ => String::new(), // no/empty arguments
        };
        clusters.entry(key).or_default().push(p);
    }

    // Only clusters with real arguments can win: a majority emitting bare
    // calls must not suppress the workers that actually filled the schema in.
    let mut sized: Vec<(usize, Vec<&'a WorkerOutput>)> = clusters
        .into_iter()
        .filter(|(key, _)| !key.is_empty())
        .map(|(_, group)| (group.len(), group))
        .collect();
    sized.sort_by_key(|(n, _)| std::cmp::Reverse(*n));

    match sized.as_slice() {
        [] => None,
        [(_, only)] => Some(only.clone()),
        [(first_n, first), (second_n, _), ..] => (first_n > second_n).then(|| first.clone()),
    }
}

fn best_tool_proposal<'a>(proposals: &[&'a WorkerOutput]) -> &'a WorkerOutput {
    proposals
        .iter()
        .copied()
        .max_by(|a, b| {
            let a_has_args = a.tool_arguments.is_some()
                && a.tool_arguments.as_ref() != Some(&Value::Object(Default::default()));
            let b_has_args = b.tool_arguments.is_some()
                && b.tool_arguments.as_ref() != Some(&Value::Object(Default::default()));
            a_has_args
                .cmp(&b_has_args)
                .then(a.confidence.total_cmp(&b.confidence))
        })
        .unwrap()
}

/// What the arbiter decided.
#[derive(Debug)]
pub enum Decision {
    /// Emit a text answer.
    Answer(String),
    /// Emit a tool call.
    ToolCall { name: String, arguments: Value },
    /// Ambiguous — needs the reducer model.
    NeedsReducer { reason: String },
}

/// Arbitrate worker outputs into a single decision.
pub fn arbitrate(outputs: &[WorkerOutput], has_tools: bool) -> Decision {
    if outputs.is_empty() {
        return Decision::NeedsReducer {
            reason: "no worker outputs".into(),
        };
    }

    if outputs.len() == 1 {
        return single_output_decision(&outputs[0], has_tools);
    }

    let tool_proposals: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::ToolProposal)
        .collect();
    let answers: Vec<&WorkerOutput> = outputs.iter().filter(|o| is_usable_answer(o)).collect();
    let critiques: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::Critique)
        .collect();
    let uncertainties: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::Uncertainty)
        .collect();

    // If everyone is uncertain, reducer
    if uncertainties.len() == outputs.len() {
        return Decision::NeedsReducer {
            reason: "all workers uncertain".into(),
        };
    }

    // ── Tool call arbitration ────────────────────────────────────

    if has_tools && !tool_proposals.is_empty() {
        // Check if any critique opposes the tool call
        let has_tool_dissent = critiques.iter().any(|c| {
            c.payload.to_lowercase().contains("don't")
                || c.payload.to_lowercase().contains("should not")
                || c.payload.to_lowercase().contains("no tool")
        });

        if has_tool_dissent {
            return Decision::NeedsReducer {
                reason: "tool proposal with dissenting critique".into(),
            };
        }

        // All tool proposals agree on the same tool?
        let tool_names: Vec<&str> = tool_proposals
            .iter()
            .filter_map(|o| o.tool_name.as_deref())
            .collect();

        if !tool_names.is_empty() {
            // If some workers propose tools and others answer directly, conflict
            if !answers.is_empty() {
                return Decision::NeedsReducer {
                    reason: "some workers propose tools, others answer directly".into(),
                };
            }

            let first = tool_names[0];
            let unanimous = tool_names.iter().all(|n| *n == first);

            if unanimous {
                let best = best_tool_proposal_by_consensus(&tool_proposals);
                return Decision::ToolCall {
                    name: first.to_string(),
                    arguments: best
                        .tool_arguments
                        .clone()
                        .unwrap_or(Value::Object(Default::default())),
                };
            }

            // Different tools proposed — check if one is clearly dominant
            let max_conf = best_tool_proposal(&tool_proposals);
            let others_low = tool_proposals
                .iter()
                .filter(|o| o.tool_name != max_conf.tool_name)
                .all(|o| o.confidence < 0.5);

            if max_conf.confidence > 0.7 && others_low {
                return Decision::ToolCall {
                    name: max_conf.tool_name.clone().unwrap_or_default(),
                    arguments: max_conf
                        .tool_arguments
                        .clone()
                        .unwrap_or(Value::Object(Default::default())),
                };
            }

            return Decision::NeedsReducer {
                reason: format!("conflicting tool proposals: {}", tool_names.join(" vs ")),
            };
        }

        // Tool proposals without extractable names — single high-confidence?
        if tool_proposals.len() == 1 && tool_proposals[0].confidence > 0.6 {
            return Decision::NeedsReducer {
                reason: "tool proposal without parseable tool name".into(),
            };
        }
    }

    // ── Answer arbitration ───────────────────────────────────────

    if !answers.is_empty() {
        // Pick the highest-confidence answer
        let best = answers
            .iter()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            .unwrap();

        // If confidence is low and there's critique, reducer
        if best.confidence < 0.5 && !critiques.is_empty() {
            return Decision::NeedsReducer {
                reason: "low confidence answer with critique".into(),
            };
        }

        // Diverging answers are synthesized, not picked.
        //
        // Returning `best.payload` verbatim was close to arbitrary: models
        // rarely emit our `kind:/confidence:` envelope, so `normalize`
        // defaults them all to `confidence: 0.5` and `max_by` just returns
        // the first maximum — i.e. whichever worker happened to finish
        // first. That discarded every other worker's contribution while
        // still paying to run them.
        //
        // This is the pattern Together's MoA is built on and benchmarks
        // well with: fan out, then have an aggregator read all responses
        // and write one. We only take that path when the answers actually
        // disagree — when they agree, the existing consensus/early-exit
        // paths are cheaper and already correct.
        if answers.len() >= 2 && largest_agreeing_cluster(&answers).is_none() {
            return Decision::NeedsReducer {
                reason: format!("{} workers answered with no agreement", answers.len()),
            };
        }

        return Decision::Answer(best.payload.clone());
    }

    // Only critiques and/or uncertainty — reducer
    Decision::NeedsReducer {
        reason: "no clear answer or tool proposal".into(),
    }
}

/// Tier-gap gating for early decisions.
///
/// When the worker pool mixes a big-tier Strong worker with small-tier
/// workers (the "MiniMax + small Qwens" shape), small-tier-only
/// consensus must not finalize *against* the strong worker while it is
/// still running — two fast small models agreeing on a wrong answer
/// would otherwise outvote the model most likely to be right. The
/// fan-out loop bounds how long the gate can hold via `strong_patience`,
/// so this never becomes an unbounded wait.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrongGate {
    /// No quality gap, patience disabled, or patience expired: decide
    /// on count and confidence alone (previous behavior).
    Off,
    /// Quality gap active. `strong_pending` is true while the Strong
    /// worker has not yet finished (succeeded or failed).
    Active { strong_pending: bool },
}

/// Try to decide early with a partial set of worker outputs.
///
/// Returns `Some(decision)` if we can confidently resolve without waiting
/// for more workers.  Returns `None` if we need to keep waiting.
///
/// `total_workers` is how many were dispatched.
/// `total_finished` includes both successful outputs AND failed workers,
/// so we know when there's no point waiting for more.
pub fn try_early_decision(
    outputs: &[WorkerOutput],
    total_workers: usize,
    total_finished: usize,
    has_tools: bool,
    strong_gate: StrongGate,
) -> Option<Decision> {
    if outputs.is_empty() {
        // All workers finished but none succeeded
        if total_finished >= total_workers {
            return None; // Let the caller handle the empty-outputs case
        }
        return None;
    }

    let remaining = total_workers.saturating_sub(total_finished);
    let strong_pending = matches!(
        strong_gate,
        StrongGate::Active {
            strong_pending: true
        }
    );

    // ── Only one worker will ever respond ───────────────────────────
    // If we have 1 successful output and no more workers are coming,
    // return it immediately — no point waiting.
    if outputs.len() == 1 && remaining == 0 {
        return Some(single_output_decision(&outputs[0], has_tools));
    }

    // ── Single output with others still pending ─────────────────────
    // Wait for at least one more so we have a chance to detect disagreement —
    // UNLESS most other workers have already failed, in which case return
    // the sole survivor immediately rather than waiting for stragglers.
    if outputs.len() < 2 && remaining > 0 {
        let failed_count = total_finished - outputs.len();
        let majority_failed = failed_count > 0 && failed_count >= total_workers / 2;
        if !majority_failed {
            return None;
        }
        // Tier gate: a small-tier sole-survivor *answer* must not
        // finalize while the strong worker is still running. The
        // fan-out loop bounds this wait via `strong_patience`. Tool
        // proposals are exempt — they are schema-verified and holding
        // them would slow agent loops.
        if strong_pending
            && outputs[0].role != WorkerRole::Strong
            && outputs[0].kind == OutputKind::Answer
        {
            return None;
        }
        // Majority failed — return the sole survivor
        tracing::info!(
            "moa: early exit — sole survivor, {failed_count}/{total_workers} workers failed",
        );
        return Some(single_output_decision(&outputs[0], has_tools));
    }

    // ── 2+ outputs: check for consensus ─────────────────────────────

    let answers: Vec<&WorkerOutput> = outputs.iter().filter(|o| is_usable_answer(o)).collect();
    let tool_proposals: Vec<&WorkerOutput> = outputs
        .iter()
        .filter(|o| o.kind == OutputKind::ToolProposal)
        .collect();

    // Workers agree on an answer — but agreement means the *content*
    // overlaps, not just that they all produced an Answer-kind output.
    // Two workers saying "Paris" and "Berlin" must not be treated as
    // consensus. Find the largest cluster of content-similar answers
    // and only early-exit if it's ≥2 workers AND a majority of answers.
    let agreeing_cluster = if answers.len() >= 2 && tool_proposals.is_empty() {
        largest_agreeing_cluster(&answers)
    } else {
        None
    };
    if let Some((cluster_size, best)) = agreeing_cluster {
        let majority = cluster_size * 2 >= answers.len();
        let qualifies = majority && best.confidence >= 0.5;
        if let Some(decision) = qualifies
            .then(|| {
                tier_aware_consensus_decision(&answers, cluster_size, best, strong_gate, remaining)
            })
            .flatten()
        {
            return Some(decision);
        }
    }

    // All agree on the same tool call
    if has_tools && tool_proposals.len() >= 2 && answers.is_empty() {
        let tool_names: Vec<&str> = tool_proposals
            .iter()
            .filter_map(|o| o.tool_name.as_deref())
            .collect();
        if !tool_names.is_empty() {
            let first = tool_names[0];
            let unanimous = tool_names.iter().all(|n| *n == first);
            if unanimous {
                // Agreeing on the tool *name* is not agreeing on the call.
                //
                // Early exit runs on whichever workers have arrived so far, so
                // the first two responders can be two fast small models that
                // picked the same tool with different arguments. Committing
                // then means shipping one of two arbitrary argument sets and
                // aborting the workers that would have broken the tie —
                // recorded traces show 9 workers agreeing on `search` while
                // emitting 6 distinct patterns.
                //
                // With no decisive argument cluster, keep waiting: either a
                // later worker tips the balance, or `strong_patience` /
                // `first_answer_grace` lapses and full arbitration decides
                // with everything in hand.
                let Some(cluster) = decisive_argument_cluster(&tool_proposals) else {
                    tracing::info!(
                        "moa: {} workers agree on tool '{}' but arguments are tied — \
                         waiting for {} more rather than guessing",
                        tool_proposals.len(),
                        first,
                        remaining,
                    );
                    return None;
                };
                let best = best_tool_proposal(&cluster);
                tracing::info!(
                    "moa: early exit — {} workers agree on tool '{}' ({} share arguments), \
                     {} still pending",
                    tool_proposals.len(),
                    first,
                    cluster.len(),
                    remaining,
                );
                return Some(Decision::ToolCall {
                    name: first.to_string(),
                    arguments: best
                        .tool_arguments
                        .clone()
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                });
            }
        }
    }

    // Conflict detected early — some say tool, some say answer.
    // Escalate to reducer now, don't wait for more conflicting opinions.
    if !tool_proposals.is_empty() && !answers.is_empty() {
        tracing::info!(
            "moa: early escalation — {} tool proposals vs {} answers, {} still pending",
            tool_proposals.len(),
            answers.len(),
            remaining,
        );
        return Some(Decision::NeedsReducer {
            reason: "some workers propose tools, others answer directly".into(),
        });
    }

    // Not enough signal yet — keep waiting
    None
}

/// Resolve a majority answer cluster against the strong-worker tier gate.
///
/// Consensus that includes the strong worker's answer passes through —
/// that is agreement *with* the strong model, not against it. Small-tier-
/// only consensus is held back (returns `None`) while the strong worker
/// is still pending so the model most likely to be right gets to weigh
/// in. The fan-out loop bounds how long the hold can last.
fn tier_aware_consensus_decision(
    answers: &[&WorkerOutput],
    cluster_size: usize,
    best: &WorkerOutput,
    strong_gate: StrongGate,
    remaining: usize,
) -> Option<Decision> {
    let strong_pending = matches!(
        strong_gate,
        StrongGate::Active {
            strong_pending: true
        }
    );
    let strong_answer = answers
        .iter()
        .find(|a| a.role == WorkerRole::Strong && is_usable_answer(a));
    if strong_pending && strong_answer.is_none() {
        tracing::info!(
            "moa: consensus held — {}/{} small-tier workers agree but strong worker \
             still pending (patience window active)",
            cluster_size,
            answers.len(),
        );
        return None;
    }

    // Strong landed but does NOT share the small-tier cluster: prefer the
    // strong worker's answer over small-model consensus. Holding for the
    // strong worker only buys it a seat; this is what makes its answer
    // actually win on disagreement. When the strong worker IS in the
    // cluster (its content agrees with the representative under the same
    // bidirectional-subset rule the clusterer uses), the cluster
    // representative already reflects its content, so ship that.
    if let Some(strong) = strong_answer {
        let strong_tokens = content_tokens(&strong.payload);
        let best_tokens = content_tokens(&best.payload);
        let in_cluster = !strong_tokens.is_empty()
            && !best_tokens.is_empty()
            && !has_negation_mismatch(&strong_tokens, &best_tokens)
            && (strong_tokens.is_subset(&best_tokens) || best_tokens.is_subset(&strong_tokens));
        if !in_cluster {
            tracing::info!(
                "moa: strong worker dissents from {}/{} small-tier consensus — preferring \
                 strong answer (conf={:.2})",
                cluster_size,
                answers.len(),
                strong.confidence,
            );
            return Some(Decision::Answer(strong.payload.clone()));
        }
    }

    tracing::info!(
        "moa: early exit — {}/{} workers agree on answer (conf={:.2}), {} still pending",
        cluster_size,
        answers.len(),
        best.confidence,
        remaining,
    );
    Some(Decision::Answer(best.payload.clone()))
}

/// Tokens that flip meaning. Always preserved as content (even if they'd
/// fail the length / stopword filters), AND a leftover negation in either
/// side of a comparison blocks clustering regardless of subset relation.
const NEGATION_TOKENS: &[&str] = &[
    // Standalone negations
    "no", "not", "never", "none", "nor", "without", "neither", "cannot",
    // Joined contractions (apostrophe stripped)
    "dont", "doesnt", "didnt", "wont", "wouldnt", "cant", "shouldnt", "isnt", "arent", "wasnt",
    "werent",
    // Split contractions (apostrophe → space → two tokens; the prefix
    // is what survives because the `t` half is <3 chars and gets filtered).
    "don", "doesn", "didn", "won", "wouldn", "shouldn", "isn", "aren", "wasn", "weren", "hadn",
    "haven", "hasn",
];

/// Content-bearing tokens extracted from an answer payload.
///
/// Pipeline:
/// 1. Lowercase the text and replace non-alphanumeric chars with spaces.
/// 2. Split on whitespace.
/// 3. Drop stopwords and tokens shorter than 3 chars — UNLESS they're
///    in `NEGATION_TOKENS` (always kept, because they flip meaning) or
///    pure digits (kept so "42" survives).
///
/// The output is a deduplicated set: token *presence* matters, not
/// frequency. That matches how the subset-containment rule below
/// reasons about agreement.
fn content_tokens(text: &str) -> std::collections::HashSet<String> {
    // Common English stopwords. Intentionally small — scaffolding words
    // we don't want to dominate the comparison.
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "in", "into", "is",
        "it", "its", "of", "on", "or", "out", "so", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "those", "to", "was", "were", "will", "with",
    ];

    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|tok| {
            if NEGATION_TOKENS.contains(tok) {
                return true;
            }
            if STOPWORDS.contains(tok) {
                return false;
            }
            // Keep digit-only tokens regardless of length (so "42" survives).
            if tok.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
            tok.len() >= 3
        })
        .map(|t| t.to_string())
        .collect()
}

/// Does the symmetric difference between `a` and `b` contain any
/// negation token? If yes, one side has a negation the other lacks,
/// and the two answers must not cluster even when subset would hold.
fn has_negation_mismatch(
    a: &std::collections::HashSet<String>,
    b: &std::collections::HashSet<String>,
) -> bool {
    a.symmetric_difference(b)
        .any(|tok| NEGATION_TOKENS.contains(&tok.as_str()))
}

/// Find the largest cluster of answers that agree on content.
///
/// Two answers agree iff:
/// - the smaller content-token set is a subset of the larger, AND
/// - their symmetric difference contains no negation tokens
///
/// Rationale: a terse answer ("Paris") and a verbose one ("Paris is
/// the capital of France") share all of the short answer's content
/// tokens, so the terse one is a subset of the verbose one. Two
/// genuinely conflicting answers ("…is Paris" vs "…is Berlin") each
/// have a leftover content token the other lacks. And an answer that
/// adds a negation ("use grep" vs "do not use grep") is rejected by
/// the negation-mismatch guard even when subset would otherwise hold.
///
/// Returns the cluster size and the highest-confidence member, or `None`
/// if no cluster has ≥ 2 members. The chosen representative is the
/// member with the *largest* content-token set (preferring the more
/// complete answer), tie-broken by confidence.
///
/// Greedy single-pass clustering — fine for the N ≤ ~8 worker case
/// MoA actually sees.
fn largest_agreeing_cluster<'a>(answers: &[&'a WorkerOutput]) -> Option<(usize, &'a WorkerOutput)> {
    let token_sets: Vec<_> = answers.iter().map(|o| content_tokens(&o.payload)).collect();
    let mut best: Option<(usize, &WorkerOutput)> = None;

    for (i, anchor_tokens) in token_sets.iter().enumerate() {
        // Build cluster anchored at i: include j if i's tokens ⊆ j's
        // tokens OR j's tokens ⊆ i's tokens.
        let mut members: Vec<&WorkerOutput> = vec![answers[i]];
        let mut member_token_counts: Vec<usize> = vec![anchor_tokens.len()];
        for (j, other_tokens) in token_sets.iter().enumerate() {
            if i == j {
                continue;
            }
            // Negation mismatch in either direction blocks clustering
            // even when subset would otherwise hold: "use grep" is a
            // subset of "do not use grep" by tokens, but they obviously
            // disagree.
            if has_negation_mismatch(anchor_tokens, other_tokens) {
                continue;
            }
            // An empty token set means the answer was all stopwords /
            // sub-3-char filler. That has no content signal — don't let
            // it cluster with anything, even though `{} ⊆ X` is
            // technically true for every X.
            if anchor_tokens.is_empty() || other_tokens.is_empty() {
                continue;
            }
            if other_tokens.is_subset(anchor_tokens) || anchor_tokens.is_subset(other_tokens) {
                members.push(answers[j]);
                member_token_counts.push(other_tokens.len());
            }
        }
        if members.len() < 2 {
            continue;
        }
        // Representative = member with the most content tokens (most
        // complete answer); tie-break on confidence.
        let representative = members
            .iter()
            .zip(member_token_counts.iter())
            .max_by(|(a, a_n), (b, b_n)| {
                a_n.cmp(b_n)
                    .then_with(|| a.confidence.total_cmp(&b.confidence))
            })
            .map(|(o, _)| *o)
            .unwrap();
        match best {
            Some((size, _)) if size >= members.len() => {}
            _ => best = Some((members.len(), representative)),
        }
    }
    best
}

fn single_output_decision(output: &WorkerOutput, has_tools: bool) -> Decision {
    if output.kind == OutputKind::Answer && !is_usable_answer(output) {
        // Two distinct unusable shapes reach here, and the reason string is
        // surfaced to the reducer as context, so keep them apart.
        return Decision::NeedsReducer {
            reason: if output.truncated {
                "single worker answer truncated at token limit".into()
            } else {
                "single worker returned silent reply sentinel".to_string()
            },
        };
    }

    match output.kind {
        OutputKind::ToolProposal if has_tools => {
            if let Some(ref name) = output.tool_name {
                Decision::ToolCall {
                    name: name.clone(),
                    arguments: output
                        .tool_arguments
                        .clone()
                        .unwrap_or(Value::Object(Default::default())),
                }
            } else {
                Decision::Answer(output.payload.clone())
            }
        }
        OutputKind::Uncertainty => Decision::NeedsReducer {
            reason: "single worker uncertain".into(),
        },
        _ => Decision::Answer(output.payload.clone()),
    }
}

/// Is this output an answer we can return to the caller verbatim?
///
/// Excludes truncated answers. A response the backend cut off at the token
/// limit is a half-finished sentence: it must not win the confidence pick,
/// must not anchor consensus, and must not be shipped as-is.
///
/// Truncated answers are *not* dropped from the turn, though — they stay in
/// `outputs` and so are still packed into the reducer's context by
/// `pack_for_reducer_selected`, which labels them as incomplete. Partial text
/// is usable material for synthesis; it just can't be the final answer.
fn is_usable_answer(output: &WorkerOutput) -> bool {
    output.kind == OutputKind::Answer
        && !output.truncated
        && !crate::normalize::is_silent_reply_sentinel(&output.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_output(kind: OutputKind, confidence: f32, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind,
            confidence,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: "test".to_string(),
            role: WorkerRole::Generalist,
            elapsed_ms: 0,
            truncated: false,
        }
    }

    fn make_tool_output(confidence: f32, tool: &str, args: Value) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence,
            tool_name: Some(tool.to_string()),
            tool_arguments: Some(args),
            payload: "propose tool".to_string(),
            model: "test".to_string(),
            role: WorkerRole::Generalist,
            elapsed_ms: 0,
            truncated: false,
        }
    }

    #[test]
    fn unanimous_answer_picks_highest_confidence() {
        let outputs = vec![
            make_output(OutputKind::Answer, 0.7, "Paris"),
            make_output(OutputKind::Answer, 0.9, "Paris is the capital"),
        ];
        match arbitrate(&outputs, false) {
            Decision::Answer(text) => assert!(text.contains("Paris")),
            other => panic!("expected Answer, got {other:?}"),
        }
    }

    #[test]
    fn unanimous_tool_proposal() {
        let outputs = vec![
            make_tool_output(0.8, "read_file", serde_json::json!({"path": "a.rs"})),
            make_tool_output(0.7, "read_file", serde_json::json!({"path": "a.rs"})),
        ];
        match arbitrate(&outputs, true) {
            Decision::ToolCall { name, .. } => assert_eq!(name, "read_file"),
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn conflicting_tools_needs_reducer() {
        let outputs = vec![
            make_tool_output(0.6, "read_file", serde_json::json!({})),
            make_tool_output(0.6, "web_search", serde_json::json!({})),
        ];
        match arbitrate(&outputs, true) {
            Decision::NeedsReducer { reason } => assert!(reason.contains("conflicting")),
            other => panic!("expected NeedsReducer, got {other:?}"),
        }
    }

    #[test]
    fn tool_vs_answer_needs_reducer() {
        let outputs = vec![
            make_tool_output(0.7, "read_file", serde_json::json!({})),
            make_output(OutputKind::Answer, 0.8, "I can answer that directly"),
        ];
        match arbitrate(&outputs, true) {
            Decision::NeedsReducer { reason } => assert!(reason.contains("some workers")),
            other => panic!("expected NeedsReducer, got {other:?}"),
        }
    }

    // ── Early decision tests ────────────────────────────────────

    #[test]
    fn early_decision_none_with_one_of_three() {
        let outputs = vec![make_output(OutputKind::Answer, 0.9, "Paris")];
        // 1 of 3 — too early to decide
        assert!(try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off).is_none());
    }

    #[test]
    fn early_decision_consensus_two_of_three() {
        let outputs = vec![
            make_output(OutputKind::Answer, 0.8, "Paris"),
            make_output(OutputKind::Answer, 0.9, "Paris is the capital"),
        ];
        // 2 of 3 agree — early exit
        match try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off) {
            Some(Decision::Answer(text)) => assert!(text.contains("Paris")),
            other => panic!("expected early Answer, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_tool_consensus() {
        let outputs = vec![
            make_tool_output(0.8, "read_file", serde_json::json!({"path": "a.rs"})),
            make_tool_output(0.7, "read_file", serde_json::json!({"path": "a.rs"})),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), true, StrongGate::Off) {
            Some(Decision::ToolCall { name, .. }) => assert_eq!(name, "read_file"),
            other => panic!("expected early ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_conflict_escalates() {
        let outputs = vec![
            make_tool_output(0.7, "read_file", serde_json::json!({})),
            make_output(OutputKind::Answer, 0.8, "I know the answer"),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), true, StrongGate::Off) {
            Some(Decision::NeedsReducer { .. }) => {}
            other => panic!("expected early NeedsReducer, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_requires_content_agreement() {
        // Two high-confidence answers that disagree on the answer noun.
        // The old code wrongly early-exited on the highest-confidence
        // one. The new subset rule sees `{paris}` and `{berlin}` as
        // distinct leftovers, so neither side is a subset of the other.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "The capital of France is Paris"),
            make_output(OutputKind::Answer, 0.8, "The capital of France is Berlin"),
        ];
        let res = try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off);
        assert!(
            res.is_none(),
            "disagreeing answers should not trigger early-exit, got {res:?}"
        );
    }

    #[test]
    fn early_decision_agrees_on_terse_vs_verbose() {
        // The most common real-world agreement pattern: one worker is
        // terse, another is verbose, both correct. Terse content tokens
        // ⊆ verbose content tokens → cluster.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.7, "Paris"),
            make_output(OutputKind::Answer, 0.9, "Paris is the capital of France"),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off) {
            // Representative picks the most complete (most tokens) member.
            Some(Decision::Answer(text)) => {
                let lower = text.to_lowercase();
                assert!(lower.contains("paris"), "expected Paris, got {text:?}");
                assert!(
                    lower.contains("capital") || lower == "paris",
                    "expected verbose representative or single 'paris', got {text:?}"
                );
            }
            other => panic!("expected agreement on terse-vs-verbose, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_majority_cluster_wins() {
        // Two agreeing answers + one outlier. Cluster of 2 is a majority
        // of 3 finished answers → early-exit fires.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "Paris"),
            make_output(OutputKind::Answer, 0.8, "Paris is the capital"),
            make_output(OutputKind::Answer, 0.6, "I think it's Lyon"),
        ];
        match try_early_decision(&outputs, 4, outputs.len(), false, StrongGate::Off) {
            Some(Decision::Answer(text)) => assert!(
                text.to_lowercase().contains("paris"),
                "should pick from the agreeing cluster, got {text:?}"
            ),
            other => panic!("expected early Answer on majority cluster, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_disagreement_with_shared_scaffolding_still_blocks() {
        // High token overlap from shared scaffolding ("the capital of
        // France is X") used to false-positive a similarity check.
        // With subset containment, each answer has a distinct leftover
        // (paris, berlin, madrid) so no cluster forms.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "The capital of France is Paris"),
            make_output(OutputKind::Answer, 0.9, "The capital of France is Berlin"),
            make_output(OutputKind::Answer, 0.5, "The capital of France is Madrid"),
        ];
        let res = try_early_decision(&outputs, 4, outputs.len(), false, StrongGate::Off);
        assert!(
            res.is_none(),
            "three disagreeing answers should not early-exit, got {res:?}"
        );
    }

    #[test]
    fn early_decision_negation_blocks_agreement() {
        // The negation guard keeps "not" / "dont" etc. as content tokens,
        // so an answer with negation is not a subset of the affirmative
        // version even when all other tokens match.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "You should use grep"),
            make_output(OutputKind::Answer, 0.8, "You should not use grep"),
        ];
        let res = try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off);
        assert!(
            res.is_none(),
            "affirmative vs negated answer should not cluster, got {res:?}"
        );
    }

    #[test]
    fn early_decision_dont_blocks_agreement() {
        // Same idea with a contraction. After punctuation stripping
        // "don't" → "dont", which is in NEGATION_TOKENS.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "Do that"),
            make_output(OutputKind::Answer, 0.8, "Don't do that"),
        ];
        let res = try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off);
        assert!(
            res.is_none(),
            "affirmative vs negated should not cluster, got {res:?}"
        );
    }

    #[test]
    fn early_decision_numeric_answers_cluster() {
        // Numeric answers survive the length filter and cluster cleanly.
        let outputs = vec![
            make_output(OutputKind::Answer, 0.9, "42"),
            make_output(OutputKind::Answer, 0.8, "The answer is 42"),
        ];
        match try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off) {
            Some(Decision::Answer(text)) => assert!(text.contains("42")),
            other => panic!("expected numeric agreement, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_single_survivor() {
        // 1 success out of 3, other 2 failed — should return the single answer
        let outputs = vec![make_output(OutputKind::Answer, 0.8, "Paris")];
        // total_workers=3, total_finished=3 (1 success + 2 failures), remaining=0
        match try_early_decision(&outputs, 3, 3, false, StrongGate::Off) {
            Some(Decision::Answer(text)) => assert!(text.contains("Paris")),
            other => panic!("expected early Answer for sole survivor, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_low_confidence_waits() {
        let outputs = vec![
            make_output(OutputKind::Answer, 0.3, "maybe Paris"),
            make_output(OutputKind::Answer, 0.4, "could be Paris"),
        ];
        // Both answers but low confidence — should wait for more
        assert!(try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off).is_none());
    }

    #[test]
    fn no_reply_sentinel_does_not_win_answer_arbitration() {
        let outputs = vec![
            make_output(OutputKind::Answer, 0.99, "NO_REPLY"),
            make_output(OutputKind::Answer, 0.6, "I can help with that."),
        ];
        match arbitrate(&outputs, false) {
            Decision::Answer(text) => assert_eq!(text, "I can help with that."),
            other => panic!("expected usable answer, got {other:?}"),
        }
    }

    #[test]
    fn no_reply_sentinel_for_single_output_needs_reducer() {
        let outputs = vec![make_output(OutputKind::Answer, 0.99, "NO_REPLY")];
        match arbitrate(&outputs, false) {
            Decision::NeedsReducer { reason } => {
                assert!(reason.contains("silent reply sentinel"));
            }
            other => panic!("expected reducer escalation, got {other:?}"),
        }
    }

    #[test]
    fn all_uncertain_needs_reducer() {
        let outputs = vec![
            make_output(OutputKind::Uncertainty, 0.2, "not sure"),
            make_output(OutputKind::Uncertainty, 0.3, "hard to say"),
        ];
        match arbitrate(&outputs, false) {
            Decision::NeedsReducer { reason } => assert!(reason.contains("uncertain")),
            other => panic!("expected NeedsReducer, got {other:?}"),
        }
    }

    #[test]
    fn early_decision_sole_survivor_majority_failed() {
        // 1 success, 3 failures, 1 still pending — majority failed, return sole survivor
        let outputs = vec![make_output(OutputKind::Answer, 0.8, "Paris")];
        // total_workers=5, total_finished=4 (1 success + 3 failures), remaining=1
        match try_early_decision(&outputs, 5, 4, false, StrongGate::Off) {
            Some(Decision::Answer(text)) => assert!(text.contains("Paris")),
            other => {
                panic!("expected early Answer for sole survivor (majority failed), got {other:?}")
            }
        }
    }

    #[test]
    fn early_decision_sole_survivor_minority_failed_waits() {
        // 1 success, 1 failure, 3 still pending — minority failed, wait for more
        let outputs = vec![make_output(OutputKind::Answer, 0.8, "Paris")];
        // total_workers=5, total_finished=2 (1 success + 1 failure), remaining=3
        assert!(try_early_decision(&outputs, 5, 2, false, StrongGate::Off).is_none());
    }

    #[test]
    fn best_tool_proposal_prefers_arguments() {
        let without_args = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.9,
            tool_name: Some("read_file".into()),
            tool_arguments: None,
            payload: "calling read_file".into(),
            model: "fast-model".into(),
            role: crate::worker::WorkerRole::Fast,
            elapsed_ms: 100,
            truncated: false,
        };
        let with_args = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.6,
            tool_name: Some("read_file".into()),
            tool_arguments: Some(serde_json::json!({"path": "/tmp/test.txt"})),
            payload: "calling read_file".into(),
            model: "strong-model".into(),
            role: crate::worker::WorkerRole::Strong,
            elapsed_ms: 3000,
            truncated: false,
        };
        let proposals = vec![&without_args, &with_args];
        let best = best_tool_proposal(&proposals);
        assert_eq!(best.model, "strong-model");
        assert!(best.tool_arguments.is_some());
    }

    // ── Argument consensus ───────────────────────────────────────────
    //
    // Recorded traces (`evals/moa-openrouter/agentic.jsonl`): 9 open-weight
    // models unanimously call the same tool while emitting up to 6 distinct
    // argument sets. Every OpenAI-shape tool call is normalized to a fixed
    // 0.9 confidence, so a confidence-only tiebreak cannot tell a majority
    // argument set from a lone outlier.

    fn tool_from(model: &str, tool: &str, args: Value) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::ToolProposal,
            // The fixed confidence every native tool call gets.
            confidence: 0.9,
            tool_name: Some(tool.to_string()),
            tool_arguments: Some(args),
            payload: format!("calling {tool}"),
            model: model.to_string(),
            role: WorkerRole::Generalist,
            elapsed_ms: 0,
            truncated: false,
        }
    }

    #[test]
    fn majority_arguments_beat_a_lone_outlier() {
        // The real `explore_error_handling` step-0 shape: 8 workers agree on
        // `{"path": "src"}`, mistral-small hallucinates a `rust_project/`
        // prefix. Equal confidence, so only cluster size can separate them.
        let a = tool_from("qwen3-8b", "list_dir", json!({"path": "src"}));
        let b = tool_from("ministral-8b", "list_dir", json!({"path": "src"}));
        let outlier = tool_from(
            "mistral-small",
            "list_dir",
            json!({"path": "rust_project/src"}),
        );

        // Outlier first, so a "first maximum wins" tiebreak would pick it.
        let proposals = vec![&outlier, &a, &b];
        let best = best_tool_proposal_by_consensus(&proposals);
        assert_eq!(
            best.tool_arguments,
            Some(json!({"path": "src"})),
            "the argument set two workers proposed must win over the singleton"
        );
    }

    #[test]
    fn argument_clustering_ignores_key_order() {
        // Recorded workers emit the same call with keys in different orders
        // (`{"pattern":..,"path":..}` vs `{"path":..,"pattern":..}`). Those
        // must cluster together, otherwise a real majority looks like a tie.
        let a = tool_from(
            "qwen3-8b",
            "search",
            json!({"pattern": "MeshError::Timeout", "path": "."}),
        );
        let b = tool_from(
            "minimax",
            "search",
            json!({"path": ".", "pattern": "MeshError::Timeout"}),
        );
        let other = tool_from(
            "qwen3-14b",
            "search",
            json!({"path": ".", "pattern": "\\bMeshError::Timeout\\b"}),
        );

        let proposals = vec![&other, &a, &b];
        let cluster = decisive_argument_cluster(&proposals)
            .expect("two key-order variants of the same call are a decisive majority");
        assert_eq!(cluster.len(), 2);
    }

    #[test]
    fn tied_argument_clusters_are_not_decisive() {
        // Two workers, two different argument sets. Any pick is arbitrary, so
        // `try_early_decision` must keep waiting rather than commit.
        let a = tool_from("qwen3-8b", "search", json!({"pattern": "a"}));
        let b = tool_from("qwen3-14b", "search", json!({"pattern": "b"}));
        let proposals = vec![&a, &b];
        assert!(
            decisive_argument_cluster(&proposals).is_none(),
            "a 1-1 split on arguments is not a decision"
        );
    }

    #[test]
    fn argument_free_proposals_never_outvote_real_arguments() {
        // Two bare calls and one with real arguments: the filled-in call wins
        // despite being outnumbered, because an empty object carries no
        // information the caller can act on.
        let bare_a = tool_from("a", "read_file", json!({}));
        let bare_b = tool_from("b", "read_file", json!({}));
        let real = tool_from("c", "read_file", json!({"path": "src/error.rs"}));
        let proposals = vec![&bare_a, &bare_b, &real];
        let best = best_tool_proposal_by_consensus(&proposals);
        assert_eq!(best.tool_arguments, Some(json!({"path": "src/error.rs"})));
    }

    #[test]
    fn early_exit_waits_when_tool_arguments_are_tied() {
        // Two fast workers agree on the tool name but not the arguments, with
        // more workers still pending. Committing here would ship a coin-flip
        // and abort the workers that would have broken the tie.
        let a = tool_from("fast-a", "search", json!({"pattern": "a"}));
        let b = tool_from("fast-b", "search", json!({"pattern": "b"}));
        let decision = try_early_decision(&[a, b], 9, 2, true, StrongGate::Off);
        assert!(
            decision.is_none(),
            "tied arguments must not produce an early exit; got {decision:?}"
        );
    }

    #[test]
    fn early_exit_fires_when_arguments_agree() {
        // Same shape, but the two arrivals agree on arguments — that is a real
        // consensus and should still short-circuit.
        let a = tool_from("fast-a", "search", json!({"pattern": "x"}));
        let b = tool_from("fast-b", "search", json!({"pattern": "x"}));
        match try_early_decision(&[a, b], 9, 2, true, StrongGate::Off) {
            Some(Decision::ToolCall { name, arguments }) => {
                assert_eq!(name, "search");
                assert_eq!(arguments, json!({"pattern": "x"}));
            }
            other => panic!("expected an early ToolCall decision, got {other:?}"),
        }
    }

    // ── Truncation ───────────────────────────────────────────────────
    //
    // 39/140 recorded responses came back `finish_reason == "length"`, and 24
    // of those carry partial text. Such an answer parses as normal prose at
    // the default 0.5 confidence, so before truncation was tracked it could
    // win the pick and be returned verbatim as a half-finished sentence.

    #[test]
    fn truncated_answer_is_not_returned_verbatim() {
        let mut cut_off = make_output(
            OutputKind::Answer,
            0.9, // higher than the complete answer, so confidence alone would pick it
            "**Island Magic: My Unforgettable Journey Through the Heart of",
        );
        cut_off.truncated = true;
        let complete = make_output(OutputKind::Answer, 0.5, "Hawaii is worth visiting.");

        match arbitrate(&[cut_off, complete], false) {
            Decision::Answer(text) => assert_eq!(
                text, "Hawaii is worth visiting.",
                "a truncated answer must not win even with higher confidence"
            ),
            other => panic!("expected the complete answer, got {other:?}"),
        }
    }

    #[test]
    fn sole_truncated_answer_goes_to_synthesis() {
        // Nothing complete to fall back on: escalate rather than ship a
        // dangling sentence.
        let mut cut_off = make_output(OutputKind::Answer, 0.9, "The first step is to");
        cut_off.truncated = true;

        match arbitrate(&[cut_off], false) {
            Decision::NeedsReducer { reason } => assert!(
                reason.contains("truncated"),
                "reason should name truncation so the reducer gets useful context; got {reason:?}"
            ),
            other => panic!("expected NeedsReducer, got {other:?}"),
        }
    }

    #[test]
    fn truncated_answers_do_not_anchor_consensus() {
        // Two truncated workers agreeing is not consensus — they agree on a
        // prefix, and neither finished its thought.
        let mut a = make_output(OutputKind::Answer, 0.9, "The capital of Japan is");
        a.truncated = true;
        let mut b = make_output(OutputKind::Answer, 0.9, "The capital of Japan is");
        b.truncated = true;

        assert!(
            matches!(
                try_early_decision(&[a, b], 4, 2, false, StrongGate::Off),
                None | Some(Decision::NeedsReducer { .. })
            ),
            "truncated agreement must not short-circuit the turn"
        );
    }

    // ── Synthesis on disagreement ────────────────────────────────────

    #[test]
    fn diverging_answers_go_to_synthesis_rather_than_an_arbitrary_pick() {
        // Models rarely emit our confidence envelope, so `normalize` defaults
        // everything to 0.5 and `max_by` returns whichever worker happened to
        // land first. Picking there discards every other worker's output while
        // still having paid to run them.
        let a = make_output(OutputKind::Answer, 0.5, "Use ripgrep for this search task");
        let b = make_output(OutputKind::Answer, 0.5, "Postgres indexes are B-trees");
        let c = make_output(
            OutputKind::Answer,
            0.5,
            "Kubernetes schedules pods on nodes",
        );

        match arbitrate(&[a, b, c], false) {
            Decision::NeedsReducer { reason } => assert!(
                reason.contains("no agreement"),
                "reason should explain the escalation; got {reason:?}"
            ),
            other => panic!("three unrelated answers should be synthesized, got {other:?}"),
        }
    }

    #[test]
    fn agreeing_answers_still_short_circuit_without_the_reducer() {
        // Synthesis is for disagreement. When workers agree, the cheap path
        // must stay cheap — otherwise every turn pays for an extra inference.
        let a = make_output(OutputKind::Answer, 0.5, "The capital of Japan is Tokyo");
        let b = make_output(OutputKind::Answer, 0.5, "The capital of Japan is Tokyo");

        assert!(
            matches!(arbitrate(&[a, b], false), Decision::Answer(_)),
            "agreeing answers must not be sent to the reducer"
        );
    }

    #[test]
    fn best_tool_proposal_falls_back_to_confidence() {
        let a = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.6,
            tool_name: Some("read_file".into()),
            tool_arguments: Some(serde_json::json!({"path": "/a.txt"})),
            payload: "calling read_file".into(),
            model: "model-a".into(),
            role: crate::worker::WorkerRole::Specialist,
            elapsed_ms: 2000,
            truncated: false,
        };
        let b = WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.9,
            tool_name: Some("read_file".into()),
            tool_arguments: Some(serde_json::json!({"path": "/b.txt"})),
            payload: "calling read_file".into(),
            model: "model-b".into(),
            role: crate::worker::WorkerRole::Strong,
            elapsed_ms: 3000,
            truncated: false,
        };
        let proposals = vec![&a, &b];
        let best = best_tool_proposal(&proposals);
        // Both have args, so confidence wins
        assert_eq!(best.model, "model-b");
    }

    // ── Tier gate (StrongGate) ───────────────────────────────────────

    fn make_role_output(
        kind: OutputKind,
        confidence: f32,
        payload: &str,
        role: WorkerRole,
    ) -> WorkerOutput {
        WorkerOutput {
            role,
            ..make_output(kind, confidence, payload)
        }
    }

    const GATE_PENDING: StrongGate = StrongGate::Active {
        strong_pending: true,
    };

    #[test]
    fn gate_holds_small_tier_consensus_while_strong_pending() {
        // Two small-tier workers agree — without the gate this would
        // early-exit. With the strong worker still running, hold.
        let outputs = vec![
            make_role_output(OutputKind::Answer, 0.8, "Paris", WorkerRole::Fast),
            make_role_output(
                OutputKind::Answer,
                0.9,
                "Paris is the capital",
                WorkerRole::Specialist,
            ),
        ];
        assert!(
            try_early_decision(&outputs, 3, outputs.len(), false, GATE_PENDING).is_none(),
            "small-tier consensus must be held while the strong worker is pending"
        );
        // Same outputs, gate off → previous behavior (early exit).
        assert!(
            try_early_decision(&outputs, 3, outputs.len(), false, StrongGate::Off).is_some(),
            "gate off must preserve pre-gate early-exit behavior"
        );
    }

    #[test]
    fn gate_passes_consensus_that_includes_strong_worker() {
        // Strong worker has answered and agrees — that's agreement WITH
        // the strong model. Ship it even though another worker is pending.
        let outputs = vec![
            make_role_output(OutputKind::Answer, 0.8, "Paris", WorkerRole::Fast),
            make_role_output(
                OutputKind::Answer,
                0.9,
                "Paris is the capital",
                WorkerRole::Strong,
            ),
        ];
        let gate = StrongGate::Active {
            strong_pending: false,
        };
        match try_early_decision(&outputs, 3, outputs.len(), false, gate) {
            Some(Decision::Answer(text)) => assert!(text.contains("Paris")),
            other => panic!("expected early Answer with strong agreement, got {other:?}"),
        }
    }

    #[test]
    fn gate_holds_small_tier_sole_survivor_answer() {
        // Majority failed, sole survivor is a small-tier Answer, strong
        // still pending → hold (the fan-out patience timer bounds this).
        let outputs = vec![make_role_output(
            OutputKind::Answer,
            0.9,
            "Tokyo",
            WorkerRole::Fast,
        )];
        // 3 dispatched, 2 finished (1 ok + 1 failed) → majority_failed
        assert!(
            try_early_decision(&outputs, 3, 2, false, GATE_PENDING).is_none(),
            "small-tier sole-survivor answer must be held while strong is pending"
        );
        // Gate off → pre-gate behavior: sole survivor ships.
        assert!(try_early_decision(&outputs, 3, 2, false, StrongGate::Off).is_some());
    }

    #[test]
    fn gate_does_not_hold_tool_proposals() {
        // Tool proposals are schema-verified and exempt from the gate —
        // agent loops must stay snappy.
        let outputs = vec![WorkerOutput {
            role: WorkerRole::Fast,
            ..make_tool_output(0.9, "read_file", serde_json::json!({"path": "x"}))
        }];
        // Majority failed → sole-survivor path, but it's a ToolProposal.
        match try_early_decision(&outputs, 3, 2, true, GATE_PENDING) {
            Some(Decision::ToolCall { name, .. }) => assert_eq!(name, "read_file"),
            other => panic!("tool proposals must not be gated, got {other:?}"),
        }
    }

    #[test]
    fn strong_dissent_wins_over_small_consensus() {
        // Two small workers agree on "Sydney"; the strong worker landed
        // with a different answer ("Canberra"). Gate no longer pending.
        // The strong worker's answer must win, not the small consensus.
        let outputs = vec![
            make_role_output(OutputKind::Answer, 0.9, "Sydney", WorkerRole::Fast),
            make_role_output(OutputKind::Answer, 0.9, "Sydney", WorkerRole::Specialist),
            make_role_output(OutputKind::Answer, 0.7, "Canberra", WorkerRole::Strong),
        ];
        let gate = StrongGate::Active {
            strong_pending: false,
        };
        match try_early_decision(&outputs, 3, outputs.len(), false, gate) {
            Some(Decision::Answer(text)) => assert!(
                text.contains("Canberra"),
                "strong dissent must win over small consensus, got {text:?}"
            ),
            other => panic!("expected strong's Answer, got {other:?}"),
        }
    }

    #[test]
    fn gate_releases_when_strong_finished() {
        // Strong worker finished (failed or succeeded without usable
        // answer) — gate no longer pending, consensus ships.
        let outputs = vec![
            make_role_output(OutputKind::Answer, 0.8, "Paris", WorkerRole::Fast),
            make_role_output(
                OutputKind::Answer,
                0.9,
                "Paris is the capital",
                WorkerRole::Specialist,
            ),
        ];
        let gate = StrongGate::Active {
            strong_pending: false,
        };
        assert!(
            try_early_decision(&outputs, 4, outputs.len(), false, gate).is_some(),
            "consensus must ship once the strong worker has finished"
        );
    }
}
