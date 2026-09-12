//! Control-loop half of supervised (mesh-demand) model loading.
//!
//! Automatic serving may download many gigabytes before a model can run.
//! Doing that inline on the runtime control loop blocks Shutdown, Join,
//! Unload, and the shutdown-signal branch for the whole download, so the slow
//! phases run in owned tasks and report back as runtime events.
//!
//! The control loop keeps everything it exclusively owns: the capacity
//! ledger, the published serving assignment, and registration. Each transfer
//! back to the loop re-checks the launch boundary, because selection-time
//! facts (manual intent, admission, peer coverage) can go stale while a
//! download runs.

use super::{
    IntentSource, RunAutoRuntimeLoopContext, RuntimeEvent, SupervisedLaunchInputs,
    SupervisedLaunchOutcome, SupervisedLoadRequest, SupervisedLoadResolution,
    SupervisedResolveInputs, add_serving_assignment, current_time_secs, next_runtime_instance_id,
    remove_serving_assignment, reserve_runtime_capacity_for_model,
    supervised_discard_runtime_model, supervised_launch_runtime_model,
    supervised_register_runtime_model, supervised_resolve_runtime_model,
};
use mesh_llm_events::{OutputEvent, emit_event};
use std::time::Instant;

/// Join every in-flight native-start task, then drain the outcomes they
/// already queued.
///
/// Two separate hazards are covered here. Joining (rather than aborting) means
/// a `spawn_blocking` native load is allowed to finish instead of being
/// orphaned. Draining afterwards means an outcome that was delivered to the
/// event channel *before* the loop exited still has its handle stopped,
/// instead of being dropped with the receiver while the model stays up.
pub(super) async fn drain_supervised_launch_tasks(ctx: &mut RunAutoRuntimeLoopContext<'_>) {
    join_supervised_launch_tasks(&mut ctx.supervised_launch_tasks).await;
}

/// The join primitive itself, separated from the runtime context so tests
/// exercise the same code the shutdown path runs.
///
/// This must JOIN rather than abort: a native start reaches `spawn_blocking`
/// work that cannot be cancelled once running, so aborting the wrapper would
/// orphan a live load instead of draining it.
pub(super) async fn join_supervised_launch_tasks(tasks: &mut tokio::task::JoinSet<()>) {
    while tasks.join_next().await.is_some() {}
}

/// Withdraw everything a supervised load published *before* its native
/// runtime started: the gossiped serving assignment and the strict-local
/// source policy.
///
/// Both are claimed on the control loop ahead of the launch so peers see this
/// node committed, which means every abandonment path — shutdown, failed
/// launch, or a closed launch boundary — has to give them back. Split out
/// from handle teardown so the withdrawal is testable without a live native
/// runtime.
pub(super) async fn withdraw_supervised_load_footprint(
    node: &crate::mesh::Node,
    runtime_model_name: &str,
    profile: &str,
) {
    remove_serving_assignment(node, runtime_model_name).await;
    crate::inference::skippy::unregister_local_source_policy(runtime_model_name, profile);
}

/// Abandon one supervised launch outcome, whether it succeeded or failed.
///
/// Both cases have already published a serving assignment and a source
/// policy on the control loop before the native start, so both must withdraw
/// them. Only the success case additionally owns a live handle to stop. The
/// failure case carries no outcome, so the canonical identity is taken from
/// the request instead.
pub(super) async fn abandon_supervised_launch(
    node: &crate::mesh::Node,
    request: &SupervisedLoadRequest,
    result: std::result::Result<SupervisedLaunchOutcome, String>,
) {
    let (runtime_model_name, profile) = match result {
        Ok(outcome) => {
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Stopping automatic model '{}' completed during shutdown",
                    outcome.loaded_name
                ),
                context: None,
            });
            outcome.handle.shutdown().await;
            (
                outcome.resolution.runtime_model_name,
                outcome.resolution.profile,
            )
        }
        // A failed launch published the same footprint before it failed.
        Err(_) => {
            let Some(runtime_model_name) = request.runtime_model_name.clone() else {
                return;
            };
            (runtime_model_name, request.profile.clone())
        }
    };
    withdraw_supervised_load_footprint(node, &runtime_model_name, &profile).await;
}

/// Stop any supervised launch outcome still sitting in the event queue at
/// shutdown. Called after the loop has exited, so registration is no longer
/// possible and the only correct action is to abandon the work.
pub(super) async fn drain_pending_supervised_launch_events(
    node: &crate::mesh::Node,
    runtime_event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
) {
    while let Ok(event) = runtime_event_rx.try_recv() {
        if let RuntimeEvent::SupervisedLoadLaunched { request, result } = event {
            abandon_supervised_launch(node, &request, *result).await;
        }
    }
}

/// Reasons a supervised automatic load must stop at a launch boundary.
///
/// Selection-time checks can go stale while a multi-gigabyte download runs, so
/// every boundary is re-evaluated on the control loop immediately before the
/// next irreversible step.
/// Pure launch-boundary decision, independent of runtime wiring.
///
/// Factored out so every boundary rule is directly testable; the callers
/// below only gather the current facts and pass them in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SupervisedLaunchFacts {
    pub(super) shutdown_requested: bool,
    pub(super) manual_intent_present: bool,
    pub(super) intent_still_effective: bool,
    pub(super) explicit_join: bool,
    pub(super) admitted_peers: usize,
    pub(super) peer_http_coverage: bool,
}

pub(super) fn supervised_block_reason(facts: SupervisedLaunchFacts) -> Option<&'static str> {
    if facts.shutdown_requested {
        return Some("runtime shutdown was requested");
    }
    // A human-driven intent outranks mesh demand. An effective-intent check
    // alone compares only the same model/profile, so it would miss a manual
    // load of a *different* model. Automatic contribution yields to any
    // manual intent, matching the producer's own guard.
    if facts.manual_intent_present {
        return Some("a manual model intent owns this session");
    }
    if !facts.intent_still_effective {
        return Some("a higher-priority desired state superseded automatic serving");
    }
    // An explicit private join must never serve before admission succeeds.
    if facts.explicit_join && facts.admitted_peers == 0 {
        return Some("the requested mesh has not admitted this node");
    }
    // Only an actually HTTP-routable peer counts as coverage; a peer that has
    // merely been assigned a model cannot answer a request yet.
    if facts.peer_http_coverage {
        return Some("another mesh node already serves a usable model");
    }
    None
}

/// Gather the current launch-boundary facts and apply the decision above.
///
/// Peer state is async, so this is only consulted at boundaries where
/// awaiting is acceptable: after resolution and again after native start.
async fn supervised_mesh_block_reason(
    ctx: &RunAutoRuntimeLoopContext<'_>,
    request: &SupervisedLoadRequest,
) -> Option<String> {
    let peers = ctx.node.peers().await;
    let manual_intent_present = ctx
        .node
        .runtime_intents
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .iter()
        .any(|intent| intent.source != IntentSource::MeshDemand);
    supervised_block_reason(SupervisedLaunchFacts {
        shutdown_requested: ctx.startup_ready_reporter.is_shutdown_requested(),
        manual_intent_present,
        intent_still_effective: ctx.model_target_reconciliation_state.is_effective_intent(
            &request.intent_id,
            &request.spec,
            &request.profile,
        ),
        explicit_join: !ctx.options.join.is_empty(),
        admitted_peers: peers.len(),
        // A paused peer still advertises routable models, but cannot serve
        // a request. Counting it as coverage would let a single paused host
        // suppress automatic contribution indefinitely.
        peer_http_coverage: peers.iter().any(|peer| {
            !matches!(
                peer.inference_admission_state,
                Some(
                    crate::proto::node::InferenceAdmissionState::RemotePaused
                        | crate::proto::node::InferenceAdmissionState::AllPaused
                )
            ) && !peer.http_routable_models().is_empty()
        }),
    })
    .map(str::to_string)
}

/// Start phase 1 (resolve/plan) of a supervised load in an owned task.
pub(super) fn spawn_supervised_load_resolve(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    mut request: SupervisedLoadRequest,
) {
    ctx.model_target_reconciliation_state
        .mark_load_started(&request.spec, &request.profile);
    // One instance identity and one start time span both phases, so the audit
    // trail matches an inline load rather than splitting into two instances.
    request.instance_id = next_runtime_instance_id(ctx.next_runtime_instance_sequence);
    request.load_started = Instant::now();
    let inputs = SupervisedResolveInputs {
        config: ctx.config.clone(),
        spec: request.spec.clone(),
        config_model_id: request.config_model_id.clone(),
        profile: request.profile.clone(),
        instance_id: request.instance_id.clone(),
        load_started: request.load_started,
    };
    let event_tx = ctx.runtime_event_tx.clone();
    ctx.supervised_resolve_tasks.spawn(async move {
        let result = supervised_resolve_runtime_model(inputs)
            .await
            .map_err(|error| format!("{error:#}"));
        // Nothing has been launched or reserved yet, so a closed event
        // channel needs no teardown here.
        let _ = event_tx.send(RuntimeEvent::SupervisedLoadResolved {
            request: Box::new(request),
            result: Box::new(result),
        });
    });
}

/// Control-loop half between phase 1 and phase 2: re-check the launch
/// boundary, then reserve capacity and publish the serving assignment before
/// the native runtime starts.
pub(super) async fn run_auto_handle_supervised_load_resolved(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    mut request: SupervisedLoadRequest,
    result: std::result::Result<SupervisedLoadResolution, String>,
) {
    let resolution = match result {
        Ok(resolution) => resolution,
        Err(error) => {
            finish_supervised_load_failure(ctx, &request, error);
            return;
        }
    };
    // Record the canonical identity now, so every later teardown path
    // withdraws the assignment actually published below.
    request.runtime_model_name = Some(resolution.runtime_model_name.clone());
    if let Some(reason) = supervised_mesh_block_reason(ctx, &request).await {
        finish_supervised_load_abandoned(ctx, &request, &reason);
        return;
    }

    let instance_id = request.instance_id.clone();
    let capacity_reservation = match reserve_runtime_capacity_for_model(
        ctx.runtime_capacity_ledger,
        &instance_id,
        &resolution.runtime_model_name,
        None,
        ctx.node.local_runtime_capacity_bytes(),
        resolution.model_bytes,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            crate::runtime::model_lifecycle::unregister_local_source_policy_if_unused(
                ctx,
                &resolution.runtime_model_name,
                &resolution.profile,
            );
            finish_supervised_load_failure(ctx, &request, format!("{error:#}"));
            return;
        }
    };
    // Register the source policy here rather than inside the cancellable
    // resolve task, keeping the inline loader's guarantee that the policy is
    // in place before the assignment is gossiped and before any stage-control
    // request can arrive.
    crate::inference::skippy::register_local_source_policy(
        &resolution.runtime_model_name,
        &resolution.profile,
        resolution.local_source_required,
    );
    // Publish the assignment before the native start, so a concurrently
    // joining node sees this node committed rather than racing it.
    add_serving_assignment(
        ctx.node,
        ctx.primary_model_name,
        &resolution.runtime_model_name,
    )
    .await;

    let inputs = SupervisedLaunchInputs {
        node: ctx.node.clone(),
        config: ctx.config.clone(),
        options: ctx.options.clone(),
        resolution,
        spec: request.spec.clone(),
        config_model_id: request.config_model_id.clone(),
        instance_id,
        capacity_reservation,
        openai_guardrail_policy: ctx.openai_guardrail_policy.clone(),
        survey_telemetry: ctx.survey_telemetry.clone(),
        load_started: request.load_started,
    };
    let event_tx = ctx.runtime_event_tx.clone();
    let node_for_cleanup = ctx.node.clone();
    ctx.supervised_launch_tasks.spawn(async move {
        let result = match supervised_launch_runtime_model(inputs).await {
            Ok(outcome) => Ok(outcome),
            // Dropping the returned reservation here releases the capacity.
            Err((error, reservation)) => {
                drop(reservation);
                Err(error)
            }
        };
        deliver_supervised_launch_outcome(&node_for_cleanup, &event_tx, request, result).await;
    });
}

/// Hand a completed launch back to the control loop, abandoning it locally if
/// the loop is already gone.
///
/// Extracted so the undeliverable path can be exercised against a genuinely
/// closed channel rather than by calling the abandonment helper directly.
async fn deliver_supervised_launch_outcome(
    node: &crate::mesh::Node,
    event_tx: &tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    request: SupervisedLoadRequest,
    result: std::result::Result<SupervisedLaunchOutcome, String>,
) {
    let Err(undelivered) = event_tx.send(RuntimeEvent::SupervisedLoadLaunched {
        request: Box::new(request),
        result: Box::new(result),
    }) else {
        return;
    };
    // The control loop is gone, so nothing will register or drain this work.
    // Abandon it here rather than leaking a served model or a stranded
    // assignment past runtime shutdown. A failed launch leaks the same
    // footprint, so it is handled too.
    if let RuntimeEvent::SupervisedLoadLaunched { request, result } = undelivered.0 {
        abandon_supervised_launch(node, &request, *result).await;
    }
}

/// Control-loop half after phase 2: register the live handle, or drain it if
/// the launch boundary closed while the native runtime was starting.
pub(super) async fn run_auto_handle_supervised_load_launched(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    request: SupervisedLoadRequest,
    result: std::result::Result<SupervisedLaunchOutcome, String>,
) {
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => {
            // Withdraw the canonical identity that was published before the
            // launch, not the requested spec: an HF reference or absolute
            // path would not match the advertised assignment.
            if let Some(runtime_model_name) = request.runtime_model_name.as_deref() {
                withdraw_supervised_load_footprint(ctx.node, runtime_model_name, &request.profile)
                    .await;
            }
            finish_supervised_load_failure(ctx, &request, error);
            return;
        }
    };
    if let Some(reason) = supervised_mesh_block_reason(ctx, &request).await {
        supervised_discard_runtime_model(ctx, outcome, &reason).await;
        finish_supervised_load_abandoned(ctx, &request, &reason);
        return;
    }
    let response = supervised_register_runtime_model(
        ctx,
        outcome,
        request.spec.clone(),
        request.config_model_id.clone(),
    )
    .await;
    ctx.model_target_reconciliation_state
        .record_load_success(&request.spec, &request.profile);
    ctx.model_target_reconciliation_state.notify_load_success(
        &request.spec,
        &request.profile,
        response,
    );
    ctx.model_target_reconciliation_state
        .retire_one_shot_present(&request.intent_id);
}

fn finish_supervised_load_failure(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    request: &SupervisedLoadRequest,
    error: String,
) {
    let anyhow_error = anyhow::Error::msg(error.clone());
    ctx.model_target_reconciliation_state.record_load_failure(
        &request.spec,
        &request.profile,
        current_time_secs(),
        &ctx.model_target_reconciliation_policy,
    );
    ctx.model_target_reconciliation_state
        .set_intent_error(&request.intent_id, error);
    ctx.model_target_reconciliation_state
        .retire_one_shot_present(&request.intent_id);
    ctx.model_target_reconciliation_state.notify_load_failure(
        &request.spec,
        &request.profile,
        &anyhow_error,
    );
}

fn finish_supervised_load_abandoned(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    request: &SupervisedLoadRequest,
    reason: &str,
) {
    let _ = emit_event(OutputEvent::Info {
        message: "Automatic serving stopped before loading a model".to_string(),
        context: Some(reason.to_string()),
    });
    ctx.model_target_reconciliation_state
        .record_load_success(&request.spec, &request.profile);
    ctx.model_target_reconciliation_state
        .retire_one_shot_present(&request.intent_id);
    ctx.model_target_reconciliation_state.notify_load_failure(
        &request.spec,
        &request.profile,
        &anyhow::Error::msg(reason.to_string()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::inference::skippy::{local_source_required_for_model, register_local_source_policy};
    use crate::mesh::Node;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Publishing footprint claimed on the control loop before a native start.
    async fn claim_footprint(node: &Node, model: &str, profile: &str) {
        register_local_source_policy(model, profile, true);
        add_serving_assignment(node, "primary", model).await;
    }

    #[tokio::test]
    async fn withdrawing_a_supervised_load_releases_assignment_and_source_policy() {
        let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model = "org/withdraw-me.gguf";
        let profile = "strict-supervised";
        claim_footprint(&node, model, profile).await;
        assert!(
            node.serving_models().await.iter().any(|m| m == model),
            "the assignment must be published before the native start"
        );
        assert!(local_source_required_for_model(model, Some(profile)));

        withdraw_supervised_load_footprint(&node, model, profile).await;

        assert!(
            !node.serving_models().await.iter().any(|m| m == model),
            "an abandoned supervised load must not leave this node advertising a model it does not serve"
        );
        assert!(
            !local_source_required_for_model(model, Some(profile)),
            "an abandoned supervised load must not strand a global source-policy entry"
        );
    }

    #[tokio::test]
    async fn withdrawal_uses_canonical_identity_not_the_requested_spec() {
        // The spec may be an HF reference or an absolute path; only the
        // canonical runtime name matches the published assignment.
        let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let canonical = "org/model.gguf";
        let spec = "/abs/cache/snapshots/deadbeef/model.gguf";
        claim_footprint(&node, canonical, "p").await;

        withdraw_supervised_load_footprint(&node, spec, "p").await;
        assert!(
            node.serving_models().await.iter().any(|m| m == canonical),
            "withdrawing by the requested spec must not match the canonical assignment"
        );

        withdraw_supervised_load_footprint(&node, canonical, "p").await;
        assert!(!node.serving_models().await.iter().any(|m| m == canonical));
    }

    #[tokio::test]
    async fn shutdown_joins_slow_native_tasks_instead_of_orphaning_them() {
        // A native start reaches `spawn_blocking` work that cannot be
        // cancelled once running, so shutdown must let it finish. This calls
        // the production join primitive, so changing it to abort fails here.
        let finished = Arc::new(AtomicUsize::new(0));
        let mut launch_tasks = tokio::task::JoinSet::new();
        for _ in 0..3 {
            let finished = Arc::clone(&finished);
            launch_tasks.spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                finished.fetch_add(1, Ordering::SeqCst);
            });
        }

        join_supervised_launch_tasks(&mut launch_tasks).await;

        assert_eq!(
            finished.load(Ordering::SeqCst),
            3,
            "every native-start task must run to completion at shutdown"
        );
    }

    #[tokio::test]
    async fn aborting_resolve_tasks_does_not_wait_for_them() {
        // Resolve/plan tasks hold no reservation, assignment, or policy, so
        // shutdown may abort them rather than waiting out a long download.
        let finished = Arc::new(AtomicUsize::new(0));
        let mut resolve_tasks = tokio::task::JoinSet::new();
        let counter = Arc::clone(&finished);
        resolve_tasks.spawn(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            counter.fetch_add(1, Ordering::SeqCst);
        });

        tokio::time::timeout(Duration::from_secs(5), resolve_tasks.shutdown())
            .await
            .expect("aborting resolve tasks must not block on a long download");

        assert_eq!(
            finished.load(Ordering::SeqCst),
            0,
            "an aborted resolve task must not complete"
        );
    }

    #[tokio::test]
    async fn closed_receiver_stops_queued_supervised_events_from_being_missed() {
        // Shutdown closes the sender side, then drains. A queued event must
        // still be observable by the drain rather than silently dropped.
        let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
        tx.send(RuntimeEvent::SupervisedLoadResolved {
            request: Box::new(test_request()),
            result: Box::new(Err("resolve failed".to_string())),
        })
        .unwrap();
        rx.close();

        // Closing must not discard already-queued events; the drain relies
        // on still being able to see them.
        drain_pending_supervised_launch_events(&node, &mut rx).await;
        assert!(
            rx.try_recv().is_err(),
            "the drain must consume every queued event before the receiver is dropped"
        );
    }

    /// A request whose canonical identity is already known, i.e. one that has
    /// published its footprint and can still fail during native start.
    fn launched_request(model: &str, profile: &str) -> SupervisedLoadRequest {
        SupervisedLoadRequest {
            runtime_model_name: Some(model.to_string()),
            profile: profile.to_string(),
            ..test_request()
        }
    }

    #[tokio::test]
    async fn queued_failed_launch_is_cleaned_up_by_the_shutdown_drain() {
        // A failed launch has already published its assignment and policy
        // before failing. Dropping it at shutdown without withdrawing them
        // leaves this node advertising a model it never served.
        let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model = "org/failed-launch.gguf";
        let profile = "strict-failed";
        claim_footprint(&node, model, profile).await;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
        tx.send(RuntimeEvent::SupervisedLoadLaunched {
            request: Box::new(launched_request(model, profile)),
            result: Box::new(Err("native start failed".to_string())),
        })
        .unwrap();
        rx.close();

        drain_pending_supervised_launch_events(&node, &mut rx).await;

        assert!(
            !node.serving_models().await.iter().any(|m| m == model),
            "a queued failed launch must not leave its assignment published"
        );
        assert!(
            !local_source_required_for_model(model, Some(profile)),
            "a queued failed launch must not strand its source policy"
        );
    }

    #[tokio::test]
    async fn undeliverable_failed_launch_is_cleaned_up_by_the_producer() {
        // Same leak via the other abandonment path. This forces a genuinely
        // closed channel — the receiver is dropped before the send — rather
        // than calling the abandonment helper directly, so it covers the
        // real delivery failure the producer must handle.
        let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model = "org/undelivered-failure.gguf";
        let profile = "strict-undelivered";
        claim_footprint(&node, model, profile).await;

        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
        drop(event_rx);

        deliver_supervised_launch_outcome(
            &node,
            &event_tx,
            launched_request(model, profile),
            Err("native start failed".to_string()),
        )
        .await;

        assert!(
            !node.serving_models().await.iter().any(|m| m == model),
            "an undeliverable failed launch must withdraw its assignment"
        );
        assert!(
            !local_source_required_for_model(model, Some(profile)),
            "an undeliverable failed launch must withdraw its source policy"
        );
    }

    #[tokio::test]
    async fn delivered_launch_is_left_for_the_control_loop() {
        // The complement: when delivery succeeds, the producer must NOT
        // withdraw anything, or it would tear down a load the control loop
        // is about to register.
        let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model = "org/delivered.gguf";
        let profile = "strict-delivered";
        claim_footprint(&node, model, profile).await;

        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
        deliver_supervised_launch_outcome(
            &node,
            &event_tx,
            launched_request(model, profile),
            Err("native start failed".to_string()),
        )
        .await;

        assert!(
            node.serving_models().await.iter().any(|m| m == model),
            "a delivered outcome must leave cleanup to the control loop"
        );
        assert!(local_source_required_for_model(model, Some(profile)));

        // Finish the lifecycle through the normal consumer instead of
        // abandoning a registered footprint: the source-policy registry is
        // process-global, so a test that claims one and leaves it behind
        // contaminates every later test in this binary.
        event_rx.close();
        drain_pending_supervised_launch_events(&node, &mut event_rx).await;
        assert!(
            !node.serving_models().await.iter().any(|m| m == model),
            "the delivered outcome must be cleaned up once the consumer runs"
        );
        assert!(
            !local_source_required_for_model(model, Some(profile)),
            "the delivered outcome must not strand a global source policy"
        );
    }

    fn test_request() -> SupervisedLoadRequest {
        SupervisedLoadRequest {
            intent_id: "intent".to_string(),
            spec: "org/model".to_string(),
            config_model_id: None,
            profile: String::new(),
            source: IntentSource::MeshDemand,
            instance_id: "runtime-control-1".to_string(),
            load_started: Instant::now(),
            runtime_model_name: None,
        }
    }

    /// Facts describing a healthy solo node that should proceed to load.
    fn clear() -> SupervisedLaunchFacts {
        SupervisedLaunchFacts {
            shutdown_requested: false,
            manual_intent_present: false,
            intent_still_effective: true,
            explicit_join: false,
            admitted_peers: 0,
            peer_http_coverage: false,
        }
    }

    #[test]
    fn clear_boundary_allows_automatic_load() {
        assert_eq!(supervised_block_reason(clear()), None);
    }

    #[test]
    fn shutdown_stops_a_resolved_automatic_load() {
        let facts = SupervisedLaunchFacts {
            shutdown_requested: true,
            ..clear()
        };
        assert_eq!(
            supervised_block_reason(facts),
            Some("runtime shutdown was requested")
        );
    }

    #[test]
    fn manual_intent_for_a_different_model_suppresses_automatic_serving() {
        // The effective-intent check only compares the same model/profile, so
        // this case is covered solely by the global manual-intent guard.
        let facts = SupervisedLaunchFacts {
            manual_intent_present: true,
            intent_still_effective: true,
            ..clear()
        };
        assert_eq!(
            supervised_block_reason(facts),
            Some("a manual model intent owns this session")
        );
    }

    #[test]
    fn superseded_intent_stops_automatic_load() {
        let facts = SupervisedLaunchFacts {
            intent_still_effective: false,
            ..clear()
        };
        assert_eq!(
            supervised_block_reason(facts),
            Some("a higher-priority desired state superseded automatic serving")
        );
    }

    #[test]
    fn explicit_join_waits_for_admission_before_serving() {
        let facts = SupervisedLaunchFacts {
            explicit_join: true,
            admitted_peers: 0,
            ..clear()
        };
        assert_eq!(
            supervised_block_reason(facts),
            Some("the requested mesh has not admitted this node")
        );
        // Once admitted with no usable coverage, this node should contribute.
        assert_eq!(
            supervised_block_reason(SupervisedLaunchFacts {
                admitted_peers: 1,
                ..facts
            }),
            None
        );
    }

    #[test]
    fn usable_peer_coverage_prevents_a_duplicate_load() {
        let facts = SupervisedLaunchFacts {
            admitted_peers: 1,
            peer_http_coverage: true,
            ..clear()
        };
        assert_eq!(
            supervised_block_reason(facts),
            Some("another mesh node already serves a usable model")
        );
    }

    #[test]
    fn assigned_but_unusable_peers_do_not_count_as_coverage() {
        // A peer that has been assigned a model but is not yet HTTP-routable
        // (or is paused) must not keep this node idle indefinitely.
        let facts = SupervisedLaunchFacts {
            admitted_peers: 2,
            peer_http_coverage: false,
            ..clear()
        };
        assert_eq!(supervised_block_reason(facts), None);
    }

    #[test]
    fn shutdown_outranks_every_other_boundary_reason() {
        let facts = SupervisedLaunchFacts {
            shutdown_requested: true,
            manual_intent_present: true,
            intent_still_effective: false,
            explicit_join: true,
            admitted_peers: 0,
            peer_http_coverage: true,
        };
        assert_eq!(
            supervised_block_reason(facts),
            Some("runtime shutdown was requested")
        );
    }
}
