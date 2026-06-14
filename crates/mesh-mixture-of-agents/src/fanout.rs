//! Worker fan-out and incremental gathering.
//!
//! Workers run in parallel via [`tokio::task::JoinSet`]. As each one
//! completes, the result is normalized, allowed-tool filtered, and fed
//! to the arbiter's early-exit check. If the arbiter can already decide
//! from the responses collected so far (consensus, sole survivor, etc.)
//! the remaining workers are cancelled and the decision is returned
//! immediately.

use std::time::{Duration, Instant};

use crate::enforce_tool_call_contract;
use crate::worker::WorkerRole;
use crate::{WorkerSummary, arbiter, normalize};
use normalize::WorkerOutput;
use serde_json::Value;

/// Min confidence for the time-based grace path; matches the consensus rule.
const GRACE_MIN_CONFIDENCE: f32 = 0.5;
const TOOL_GRACE_MIN_CONFIDENCE: f32 = 0.6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraceMode {
    Disabled,
    Answer,
    Tool,
}

/// Time-based decision policy for a single fan-out: how long to wait
/// before shipping a partial result, and how long to hold small-tier
/// decisions for a pending big-tier strong worker.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GatherPolicy {
    /// See [`crate::GatewayConfig::first_answer_grace`].
    pub first_answer_grace: Duration,
    pub grace_mode: GraceMode,
    /// See [`crate::GatewayConfig::strong_patience`].
    pub strong_patience: Duration,
}

/// Identifier for a worker we dispatched. Used to reconcile the
/// per-worker accounting at the end of fan-out so the — possibly
/// aborted or panicked — task's existence still shows up in
/// `worker_summaries`.
pub(crate) struct DispatchedWorker {
    pub model: String,
    pub role: WorkerRole,
}

pub(crate) async fn gather_workers_incremental(
    join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
    dispatched: &[DispatchedWorker],
    has_tools: bool,
    allowed_tools: &[String],
    tools: Option<&Value>,
    policy: GatherPolicy,
) -> (
    Vec<WorkerOutput>,
    Vec<WorkerSummary>,
    Option<arbiter::Decision>,
) {
    let GatherPolicy {
        first_answer_grace,
        grace_mode,
        strong_patience,
    } = policy;
    let total_workers = dispatched.len();
    let mut outputs = Vec::new();
    let mut summaries = Vec::new();
    let mut total_finished: usize = 0;
    let dispatched_at = Instant::now();
    let grace_enabled = grace_mode != GraceMode::Disabled && !first_answer_grace.is_zero();

    // Tier gate: when the pool mixes a big-tier Strong worker with
    // small-tier workers, small-tier-only consensus is held until the
    // strong worker finishes OR the patience window expires. The window
    // is a hard bound — when it lapses the gate switches off entirely
    // and every decision rule reverts to pre-gate behavior, so a stuck
    // strong worker can never hold the turn hostage (the failure mode
    // that sank PR #820).
    let mut strong_finished = false;
    let gate_enabled = !strong_patience.is_zero()
        && crate::worker::has_quality_gap(dispatched.iter().map(|d| (d.model.as_str(), d.role)));
    let strong_gate = |strong_finished: bool, elapsed: Duration| -> arbiter::StrongGate {
        if !gate_enabled || elapsed >= strong_patience {
            return arbiter::StrongGate::Off;
        }
        arbiter::StrongGate::Active {
            strong_pending: !strong_finished,
        }
    };

    // Grace eligibility: once the grace window has elapsed and a qualifying
    // partial decision exists, ship it instead of waiting for the slow tail.
    // Answer grace handles ordinary chat, including tool-enabled clients that
    // attach schemas to every request. Tool grace handles obvious tool-intent
    // prompts, but only when a worker has actually proposed a valid tool.
    let grace_eligible = |outs: &[WorkerOutput]| -> bool {
        if !grace_enabled {
            return false;
        }
        match grace_mode {
            GraceMode::Disabled => false,
            GraceMode::Answer => outs.iter().any(|o| {
                o.kind == normalize::OutputKind::Answer && o.confidence >= GRACE_MIN_CONFIDENCE
            }),
            GraceMode::Tool => outs.iter().any(|o| {
                o.kind == normalize::OutputKind::ToolProposal
                    && o.tool_name.is_some()
                    && o.confidence >= TOOL_GRACE_MIN_CONFIDENCE
            }),
        }
    };

    loop {
        let grace_remaining = if grace_eligible(&outputs) {
            first_answer_grace.saturating_sub(dispatched_at.elapsed())
        } else {
            Duration::from_secs(60 * 60)
        };
        // While the tier gate is actively holding (strong worker pending,
        // patience not yet expired), arm a wake-up at patience expiry so a
        // held consensus decision is re-evaluated even if no further worker
        // event arrives. Without this, a stuck strong worker would mean the
        // next re-check only happens at worker_timeout.
        let gate_holding = gate_enabled
            && !strong_finished
            && dispatched_at.elapsed() < strong_patience
            && !outputs.is_empty();
        let patience_remaining = strong_patience.saturating_sub(dispatched_at.elapsed());

        // The answer grace is also tier-gated: a single small-tier answer
        // must not ship at grace expiry while the strong worker is still
        // inside its patience window — that was the dominant path where a
        // small model's answer pre-empted the strong one. Tool grace is
        // exempt (schema-verified proposals; agent loops stay snappy).
        let armed = grace_eligible(&outputs) && !(gate_holding && grace_mode == GraceMode::Answer);

        let join_result = tokio::select! {
            biased;
            join = join_set.join_next() => join,
            _ = tokio::time::sleep(grace_remaining), if armed => {
                tracing::info!(
                    "moa: grace early-exit after {}ms (grace={}ms), {} pending",
                    dispatched_at.elapsed().as_millis(),
                    first_answer_grace.as_millis(),
                    total_workers.saturating_sub(total_finished),
                );
                let decision = match grace_mode {
                    GraceMode::Answer => grace_answer_decision(&outputs),
                    GraceMode::Tool => grace_tool_decision(&outputs),
                    GraceMode::Disabled => unreachable!("disabled grace cannot be armed"),
                };
                drain_after_early_exit(join_set, &mut summaries).await;
                reconcile_dispatched(dispatched, &mut summaries);
                return (outputs, summaries, Some(decision));
            }
            _ = tokio::time::sleep(patience_remaining), if gate_holding => {
                tracing::info!(
                    "moa: strong patience expired after {}ms — re-evaluating held outputs",
                    dispatched_at.elapsed().as_millis(),
                );
                // Gate is now Off (elapsed >= strong_patience); re-run the
                // early-decision check over what we already have.
                if let Some(decision) = arbiter::try_early_decision(
                    &outputs,
                    total_workers,
                    total_finished,
                    has_tools,
                    arbiter::StrongGate::Off,
                ) {
                    drain_after_early_exit(join_set, &mut summaries).await;
                    reconcile_dispatched(dispatched, &mut summaries);
                    return (outputs, summaries, Some(decision));
                }
                continue;
            }
        };

        let Some(join_result) = join_result else {
            break;
        };

        match join_result {
            Ok((model, role, Ok(text), elapsed)) => {
                total_finished += 1;
                if role == WorkerRole::Strong {
                    strong_finished = true;
                }
                let mut normalized =
                    normalize::normalize_worker_output(&text, &model, role, elapsed);
                enforce_tool_call_contract(&mut normalized, allowed_tools, tools, &model);
                tracing::info!(
                    "moa: worker {} ({}) → {:?} conf={:.2} ({}ms, {} chars)",
                    model,
                    role.label(),
                    normalized.kind,
                    normalized.confidence,
                    elapsed,
                    text.len(),
                );
                summaries.push(WorkerSummary {
                    model: model.clone(),
                    role,
                    succeeded: true,
                    elapsed_ms: elapsed,
                    output_kind: Some(normalized.kind),
                    confidence: Some(normalized.confidence),
                });
                outputs.push(normalized);

                if let Some(decision) = arbiter::try_early_decision(
                    &outputs,
                    total_workers,
                    total_finished,
                    has_tools,
                    strong_gate(strong_finished, dispatched_at.elapsed()),
                ) {
                    drain_after_early_exit(join_set, &mut summaries).await;
                    reconcile_dispatched(dispatched, &mut summaries);
                    return (outputs, summaries, Some(decision));
                }
            }
            Ok((model, role, Err(e), elapsed)) => {
                total_finished += 1;
                if role == WorkerRole::Strong {
                    strong_finished = true;
                }
                tracing::warn!(
                    "moa: worker {} ({}) failed after {}ms: {}",
                    model,
                    role.label(),
                    elapsed,
                    e,
                );
                summaries.push(WorkerSummary {
                    model,
                    role,
                    succeeded: false,
                    elapsed_ms: elapsed,
                    output_kind: None,
                    confidence: None,
                });

                if let Some(decision) = arbiter::try_early_decision(
                    &outputs,
                    total_workers,
                    total_finished,
                    has_tools,
                    strong_gate(strong_finished, dispatched_at.elapsed()),
                ) {
                    drain_after_early_exit(join_set, &mut summaries).await;
                    reconcile_dispatched(dispatched, &mut summaries);
                    return (outputs, summaries, Some(decision));
                }
            }
            Err(e) => {
                total_finished += 1;
                tracing::warn!("moa: worker task panicked or was cancelled: {e}");
                // No (model, role) payload available from a JoinError, so
                // we cannot attribute this slot here — including whether it
                // was the Strong worker. If a panicking Strong leaves
                // `strong_finished` false, the tier gate simply holds until
                // `strong_patience` expires (its bounded fallback) rather
                // than releasing immediately. Panicking workers are rare and
                // the worst case is one extra patience window of latency, so
                // we don't add fragile JoinError↔slot correlation to shave
                // it. `reconcile_dispatched` still attributes the slot by
                // name at the end for accounting.
            }
        }
    }

    reconcile_dispatched(dispatched, &mut summaries);
    (outputs, summaries, None)
}

fn grace_answer_decision(outputs: &[WorkerOutput]) -> arbiter::Decision {
    // Prefer the Strong worker's qualifying answer when it has landed:
    // if the biggest model already answered, shipping a smaller model's
    // marginally-higher-confidence answer instead would defeat the point
    // of waiting for it. Confidence is self-reported and not comparable
    // across models, so role is the better tie-breaker here.
    let strong = outputs.iter().find(|o| {
        o.kind == normalize::OutputKind::Answer
            && o.role == WorkerRole::Strong
            && o.confidence >= GRACE_MIN_CONFIDENCE
    });
    let answer = strong.unwrap_or_else(|| {
        outputs
            .iter()
            .filter(|o| o.kind == normalize::OutputKind::Answer)
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("answer grace requires at least one Answer")
    });
    let answer_count = outputs
        .iter()
        .filter(|o| o.kind == normalize::OutputKind::Answer)
        .count();
    tracing::info!(
        "moa: answer grace picked {} answer(s), conf={:.2}",
        answer_count,
        answer.confidence,
    );
    arbiter::Decision::Answer(answer.payload.clone())
}

fn grace_tool_decision(outputs: &[WorkerOutput]) -> arbiter::Decision {
    let proposal = outputs
        .iter()
        .filter(|o| {
            o.kind == normalize::OutputKind::ToolProposal
                && o.tool_name.is_some()
                && o.confidence >= TOOL_GRACE_MIN_CONFIDENCE
        })
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("tool grace requires at least one valid ToolProposal");
    let name = proposal
        .tool_name
        .clone()
        .expect("tool grace filters proposals without names");
    tracing::info!(
        "moa: tool grace picked {} conf={:.2}",
        name,
        proposal.confidence,
    );
    arbiter::Decision::ToolCall {
        name,
        arguments: proposal
            .tool_arguments
            .clone()
            .unwrap_or(serde_json::Value::Object(Default::default())),
    }
}

/// After `abort_all`, drain any tasks that did finish before the abort
/// reached them, recording each as a summary. Aborted tasks produce a
/// `JoinError::cancelled` which carries no `(model, role)` payload —
/// those are reconciled by [`reconcile_dispatched`] using the dispatch
/// list.
async fn drain_after_early_exit(
    join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
    summaries: &mut Vec<WorkerSummary>,
) {
    join_set.abort_all();
    while let Some(leftover) = join_set.join_next().await {
        if let Ok((m, r, result, el)) = leftover {
            summaries.push(WorkerSummary {
                model: m,
                role: r,
                succeeded: result.is_ok(),
                elapsed_ms: el,
                output_kind: None,
                confidence: None,
            });
        }
    }
}

/// Ensure every dispatched worker appears in `summaries`. Anything we
/// dispatched that didn't produce a summary by name (aborted by
/// early-exit, panicked, or otherwise lost) gets a synthesized
/// `succeeded: false` entry so the `x-moa-workers` header faithfully
/// reflects the dispatched count.
fn reconcile_dispatched(dispatched: &[DispatchedWorker], summaries: &mut Vec<WorkerSummary>) {
    for w in dispatched {
        if summaries.iter().any(|s| s.model == w.model) {
            continue;
        }
        summaries.push(WorkerSummary {
            model: w.model.clone(),
            role: w.role,
            succeeded: false,
            elapsed_ms: 0,
            output_kind: None,
            confidence: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRole;

    /// Build a MoA normalization envelope (the `{"kind":"answer",...}`
    /// shape that `normalize_worker_output` parses via strategy #1)
    /// for use as a stubbed worker payload. Not a chat-completion JSON.
    ///
    /// Uses `serde_json::json!` so payloads containing quotes,
    /// backslashes, or newlines round-trip correctly.
    fn answer_text(payload: &str, confidence: f32) -> String {
        serde_json::json!({
            "kind": "answer",
            "confidence": confidence,
            "payload": payload,
        })
        .to_string()
    }

    fn tool_text(name: &str, confidence: f32) -> String {
        serde_json::json!({
            "kind": "tool_proposal",
            "tool": name,
            "arguments": {"path": "/tmp/openclaw-tool-baseline.txt"},
            "confidence": confidence,
            "payload": "Use the requested tool.",
        })
        .to_string()
    }

    fn spawn_worker(
        join_set: &mut tokio::task::JoinSet<(String, WorkerRole, Result<String, String>, u64)>,
        model: &str,
        role: WorkerRole,
        delay_ms: u64,
        result: Result<String, String>,
    ) -> DispatchedWorker {
        let model_owned = model.to_string();
        let result_clone = result.clone();
        join_set.spawn(async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            (model_owned, role, result_clone, delay_ms)
        });
        DispatchedWorker {
            model: model.to_string(),
            role,
        }
    }

    #[tokio::test]
    async fn grace_fires_when_lone_answer_qualifies_and_grace_elapsed() {
        // One fast worker answers quickly. Two more are pending and won't
        // return until well after the grace window. With grace=50ms,
        // gather should bail with the sole answer instead of waiting.
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "fast",
                WorkerRole::Fast,
                10,
                Ok(answer_text("hi", 0.7)),
            ),
            spawn_worker(
                &mut js,
                "slow1",
                WorkerRole::Specialist,
                5_000,
                Ok(answer_text("agreed", 0.6)),
            ),
            spawn_worker(
                &mut js,
                "slow2",
                WorkerRole::Strong,
                5_000,
                Ok(answer_text("agreed", 0.6)),
            ),
        ];

        let started = std::time::Instant::now();
        let (outputs, summaries, decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            false, // has_tools
            &[],
            None,
            GatherPolicy {
                first_answer_grace: Duration::from_millis(50),
                grace_mode: GraceMode::Answer,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "grace should bail well under 1s; got {elapsed:?}"
        );
        let decision = decision.expect("grace must yield a Decision");
        assert!(matches!(decision, arbiter::Decision::Answer(_)));
        assert_eq!(outputs.len(), 1, "only the fast worker landed");
        assert_eq!(summaries.iter().filter(|s| s.succeeded).count(), 1);
    }

    #[tokio::test]
    async fn answer_grace_can_fire_when_tools_are_present() {
        // OpenClaw-style clients attach tool schemas to ordinary chat turns.
        // When the caller has classified the prompt as non-tool intent, answer
        // grace should still avoid waiting for the slow tail.
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "fast",
                WorkerRole::Fast,
                10,
                Ok(answer_text("hi", 0.7)),
            ),
            spawn_worker(
                &mut js,
                "slow1",
                WorkerRole::Specialist,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
            spawn_worker(
                &mut js,
                "slow2",
                WorkerRole::Strong,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
        ];

        let started = std::time::Instant::now();
        let (outputs, _summaries, decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            true, // has_tools
            &[],
            None,
            GatherPolicy {
                first_answer_grace: Duration::from_millis(50),
                grace_mode: GraceMode::Answer,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "answer grace should still fire with schemas attached; got {elapsed:?}"
        );
        assert_eq!(outputs.len(), 1, "grace should leave the slow tail pending");
        assert!(matches!(decision, Some(arbiter::Decision::Answer(_))));
    }

    #[tokio::test]
    async fn disabled_grace_waits_even_when_answer_qualifies() {
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "fast",
                WorkerRole::Fast,
                10,
                Ok(answer_text("hi", 0.7)),
            ),
            spawn_worker(
                &mut js,
                "slow1",
                WorkerRole::Specialist,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
            spawn_worker(
                &mut js,
                "slow2",
                WorkerRole::Strong,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
        ];

        let started = std::time::Instant::now();
        let (outputs, _summaries, _decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            true,
            &[],
            None,
            GatherPolicy {
                first_answer_grace: Duration::from_millis(50),
                grace_mode: GraceMode::Disabled,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed >= Duration::from_millis(150),
            "disabled grace must not short-circuit; got {elapsed:?}"
        );
        assert!(outputs.len() >= 2);
    }

    #[tokio::test]
    async fn tool_grace_fires_on_high_confidence_tool_proposal() {
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "tool_worker",
                WorkerRole::Specialist,
                10,
                Ok(tool_text("read", 0.85)),
            ),
            spawn_worker(
                &mut js,
                "slow1",
                WorkerRole::Strong,
                5_000,
                Ok(tool_text("read", 0.9)),
            ),
        ];

        let started = std::time::Instant::now();
        let (_outputs, _summaries, decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            true,
            &["read".to_string()],
            None,
            GatherPolicy {
                first_answer_grace: Duration::from_millis(50),
                grace_mode: GraceMode::Tool,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "tool grace should not wait for the slow tail; got {elapsed:?}"
        );
        match decision.expect("tool grace should decide") {
            arbiter::Decision::ToolCall { name, arguments } => {
                assert_eq!(name, "read");
                assert_eq!(arguments["path"], "/tmp/openclaw-tool-baseline.txt");
            }
            other => panic!("expected tool call, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn grace_zero_disables_the_check() {
        // grace=0 means the timer arm never arms — gather behaves
        // exactly like pre-grace event-driven shape.
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "fast",
                WorkerRole::Fast,
                10,
                Ok(answer_text("hi", 0.7)),
            ),
            spawn_worker(
                &mut js,
                "slow1",
                WorkerRole::Specialist,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
            spawn_worker(
                &mut js,
                "slow2",
                WorkerRole::Strong,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
        ];

        let started = std::time::Instant::now();
        let (outputs, _summaries, _decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            false, // has_tools
            &[],
            None,
            GatherPolicy {
                first_answer_grace: Duration::ZERO,
                grace_mode: GraceMode::Answer,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed >= Duration::from_millis(150),
            "grace=0 must not short-circuit; got {elapsed:?}"
        );
        assert!(outputs.len() >= 2);
    }

    #[tokio::test]
    async fn grace_does_not_fire_below_confidence_threshold() {
        // A lone answer with confidence < 0.5 must NOT trigger grace.
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "fast",
                WorkerRole::Fast,
                10,
                Ok(answer_text("hi", 0.3)),
            ),
            spawn_worker(
                &mut js,
                "slow1",
                WorkerRole::Specialist,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
            spawn_worker(
                &mut js,
                "slow2",
                WorkerRole::Strong,
                200,
                Ok(answer_text("agreed", 0.6)),
            ),
        ];

        let started = std::time::Instant::now();
        let (outputs, _summaries, _decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            false,
            &[],
            None,
            GatherPolicy {
                first_answer_grace: Duration::from_millis(50),
                grace_mode: GraceMode::Answer,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed >= Duration::from_millis(150),
            "low-confidence sole answer must not grace-exit; got {elapsed:?}"
        );
        assert!(outputs.len() >= 2);
    }

    #[tokio::test]
    async fn grace_fires_with_multiple_diverse_answers() {
        // Real lab scenario: many fast workers all return short answers
        // in <1s but textually DON'T agree (Hello / Yes / Ready / Okay).
        // Arbiter consensus rule requires textual agreement so it
        // doesn't fire. Without diverse-answer grace, MoA waits for the
        // slow tail worker. With it, grace catches this case too —
        // grace window elapses, we pick the highest-confidence answer.
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "fast1",
                WorkerRole::Fast,
                10,
                Ok(answer_text("Hello", 0.6)),
            ),
            spawn_worker(
                &mut js,
                "fast2",
                WorkerRole::Specialist,
                20,
                Ok(answer_text("Yes", 0.7)),
            ),
            spawn_worker(
                &mut js,
                "slow_strong",
                WorkerRole::Strong,
                5_000,
                Ok(answer_text("Ready", 0.5)),
            ),
        ];

        let started = std::time::Instant::now();
        let (outputs, _summaries, decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            false,
            &[],
            None,
            GatherPolicy {
                first_answer_grace: Duration::from_millis(50),
                grace_mode: GraceMode::Answer,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "grace should fire well under 1s once multiple fast answers are in; got {elapsed:?}"
        );
        let decision = decision.expect("grace must yield a Decision");
        match decision {
            arbiter::Decision::Answer(text) => {
                // Should pick the highest-confidence one (fast2, 0.7).
                assert!(
                    text.contains("Yes"),
                    "expected highest-confidence answer 'Yes' (conf=0.7); got {text:?}"
                );
            }
            other => panic!("expected Decision::Answer, got {other:?}"),
        }
        // We had at least 2 answers when grace fired.
        assert!(outputs.len() >= 2);
    }

    #[tokio::test]
    async fn grace_picks_highest_confidence_when_multiple_qualify() {
        // Three workers, different confidences. Two return fast, the
        // third stays slow so grace gets a chance to fire on the two
        // diverse fast answers. Grace must pick the most confident.
        let mut js = tokio::task::JoinSet::new();
        let dispatched = vec![
            spawn_worker(
                &mut js,
                "w_low",
                WorkerRole::Fast,
                10,
                Ok(answer_text("low", 0.5)),
            ),
            spawn_worker(
                &mut js,
                "w_high",
                WorkerRole::Specialist,
                20,
                Ok(answer_text("best", 0.9)),
            ),
            spawn_worker(
                &mut js,
                "w_slow",
                WorkerRole::Strong,
                5_000,
                Ok(answer_text("slow_low", 0.4)),
            ),
        ];

        let (_outputs, _summaries, decision) = gather_workers_incremental(
            &mut js,
            &dispatched,
            false,
            &[],
            None,
            GatherPolicy {
                first_answer_grace: Duration::from_millis(100),
                grace_mode: GraceMode::Answer,
                strong_patience: Duration::ZERO,
            },
        )
        .await;
        let decision = decision.expect("grace must yield a Decision");
        match decision {
            arbiter::Decision::Answer(text) => {
                assert!(
                    text.contains("best"),
                    "expected highest-confidence answer 'best' (conf=0.9); got {text:?}"
                );
            }
            other => panic!("expected Decision::Answer, got {other:?}"),
        }
    }
}
