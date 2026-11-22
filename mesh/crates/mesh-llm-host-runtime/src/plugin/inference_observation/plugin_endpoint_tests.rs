use super::*;
use proto::inference_observation_event::Kind;

fn model_set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

fn summary_with(
    kind: &str,
    available: bool,
    protocol: Option<&str>,
    models: &[&str],
) -> PluginEndpointSummary {
    PluginEndpointSummary {
        plugin_name: "p".into(),
        plugin_status: "running".into(),
        endpoint_id: "e".into(),
        state: if available { "healthy" } else { "unhealthy" }.into(),
        available,
        kind: kind.into(),
        transport_kind: "http".into(),
        protocol: protocol.map(str::to_string),
        address: Some("http://127.0.0.1:1".into()),
        args: Vec::new(),
        namespace: None,
        supports_streaming: true,
        managed_by_plugin: true,
        detail: None,
        models: models.iter().map(|m| (*m).to_string()).collect(),
    }
}

#[test]
fn ready_models_for_endpoints_filters_by_availability_and_protocol() {
    let models = ready_models_for_endpoints(&[
        summary_with(
            "inference",
            true,
            Some("openai_compatible"),
            &["chat-a", ""],
        ),
        summary_with("inference", true, Some("transcription_v1"), &["whisper"]),
        summary_with("inference", true, Some("document_ocr_v1"), &["ocr"]),
        summary_with("inference", false, Some("openai_compatible"), &["down"]),
        summary_with("mcp", true, None, &["mcp-not-model"]),
    ]);
    assert_eq!(models, model_set(&["chat-a", "whisper"]));
}

#[test]
fn readiness_tracker_reconciles_transitions_blocks_after_close_and_cleans_up_on_drop() {
    let hub = Arc::new(ObservationHub::new());
    let (sub, _, _) = hub.subscribe();

    {
        let tracker = PluginModelReadinessTracker::new(hub.clone());
        tracker.sync_plugin_models("p1", model_set(&["a", "b"]));
        // Steady refresh with the same models emits nothing.
        tracker.sync_plugin_models("p1", model_set(&["a", "b"]));
        // A second plugin serving "b" shares the refcount: unloading p1's "b" keeps it ready.
        tracker.sync_plugin_models("p2", model_set(&["b"]));
        tracker.sync_plugin_models("p1", model_set(&["a", "c"]));

        let drained = hub.drain(sub, usize::MAX).unwrap();
        let events: Vec<(&str, String)> = drained
            .events
            .into_iter()
            .map(|e| match e.kind.unwrap() {
                Kind::ModelReady(m) => ("ready", m),
                Kind::ModelUnloaded(m) => ("unloaded", m),
                other => panic!("unexpected event: {other:?}"),
            })
            .collect();
        assert_eq!(
            events,
            vec![
                ("ready", "a".into()),
                ("ready", "b".into()),
                ("ready", "c".into()),
            ]
        );

        // Once closed (shutdown), late health probe completions cannot resurrect models.
        tracker.close();
        tracker.sync_plugin_models("p1", model_set(&["zombified"]));
    }

    let drained = hub.drain(sub, usize::MAX).unwrap();
    let mut unloaded: Vec<String> = drained
        .events
        .into_iter()
        .map(|e| match e.kind.unwrap() {
            Kind::ModelUnloaded(m) => m,
            other => panic!("unexpected event after close/drop: {other:?}"),
        })
        .collect();
    unloaded.sort();
    assert_eq!(unloaded, vec!["a", "b", "c"]);
    let (snap_sub, _, snapshot) = hub.subscribe();
    hub.unsubscribe(snap_sub);
    assert!(snapshot.ready_models.is_empty());
}

#[test]
fn executed_operation_maps_inference_paths_and_skips_non_executions() {
    use proto::InferenceOperation as Op;
    assert_eq!(
        executed_operation_for_path("/v1/chat/completions", false),
        Some((Op::ChatCompletion, false))
    );
    assert_eq!(
        executed_operation_for_path("/v1/chat/completions?foo=1", true),
        Some((Op::ChatCompletion, true))
    );
    assert_eq!(
        executed_operation_for_path("/v1/completions", true),
        Some((Op::Completion, true))
    );
    assert_eq!(
        executed_operation_for_path("/v1/responses", true),
        Some((Op::Responses, true))
    );
    assert_eq!(
        executed_operation_for_path("/v1/messages", false),
        Some((Op::Messages, false))
    );
    assert_eq!(
        executed_operation_for_path("/v1/embeddings", true),
        Some((Op::Embeddings, false))
    );
    assert_eq!(
        executed_operation_for_path("/v1/rerank", false),
        Some((Op::Rerank, false))
    );
    assert_eq!(
        executed_operation_for_path("/v1/audio/transcriptions", false),
        Some((Op::AudioTranscription, false))
    );
    assert_eq!(
        executed_operation_for_path("/v1/systemone", false),
        Some((Op::SystemOne, false))
    );
    for skipped in [
        "/v1/models",
        "/models",
        "/v1/messages/count_tokens",
        "/v1/tokenize",
        "/mesh/v1/documents/ocr",
    ] {
        assert_eq!(
            executed_operation_for_path(skipped, false),
            None,
            "expected no execution for {skipped}"
        );
    }
}

#[test]
fn concurrent_observations_stay_isolated_and_report_all_outcomes() {
    let hub = Arc::new(ObservationHub::new());
    let (sub, _, _) = hub.subscribe();

    // Two concurrent executions on the same model must get distinct execution IDs
    // and ending one must never affect the other.
    let first = PluginExecutionObservation::start(
        hub.clone(),
        Some("dev/echo"),
        "/v1/chat/completions",
        true,
    )
    .expect("first observation");
    let second = PluginExecutionObservation::start(
        hub.clone(),
        Some("dev/echo"),
        "/v1/chat/completions",
        true,
    )
    .expect("second observation");
    second.dropped();
    first.first_output();
    let usage = TokenUsage::from_counts(Some(20), Some(5), Some(25))
        .map(|u| u.with_cached_prompt_tokens(Some(8)));
    first.complete(usage.as_ref());

    // Upstream failure maps to Failed.
    let failed_obs = PluginExecutionObservation::start(
        hub.clone(),
        Some("dev/echo"),
        "/v1/chat/completions",
        false,
    )
    .unwrap();
    failed_obs.fail();

    // Dropping without finishing maps to Cancelled.
    let cancelled_obs =
        PluginExecutionObservation::start(hub.clone(), Some("dev/echo"), "/v1/responses", false)
            .unwrap();
    drop(cancelled_obs);

    let drained = hub.drain(sub, usize::MAX).unwrap();
    let summary: Vec<(u64, &str)> = drained
        .events
        .iter()
        .map(|e| {
            let label = match e.kind.as_ref().unwrap() {
                Kind::Started(_) => "started",
                Kind::FirstOutput(_) => "first_output",
                Kind::Usage(u) => {
                    assert_eq!(u.prompt_tokens, 20);
                    assert_eq!(u.cached_prompt_tokens, Some(8));
                    assert_eq!(u.completion_tokens, 5);
                    "usage"
                }
                Kind::ComputeFinished(_) => "compute_finished",
                Kind::Ended(end) => match end.outcome() {
                    proto::InferenceExecutionOutcome::Completed => "ended:completed",
                    proto::InferenceExecutionOutcome::Dropped => "ended:dropped",
                    proto::InferenceExecutionOutcome::Failed => "ended:failed",
                    proto::InferenceExecutionOutcome::Cancelled => "ended:cancelled",
                    other => panic!("unexpected outcome: {other:?}"),
                },
                other => panic!("unexpected event: {other:?}"),
            };
            (e.execution_id, label)
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            (1, "started"),
            (2, "started"),
            (2, "compute_finished"),
            (2, "ended:dropped"),
            (1, "first_output"),
            (1, "usage"),
            (1, "compute_finished"),
            (1, "ended:completed"),
            (3, "started"),
            (3, "compute_finished"),
            (3, "ended:failed"),
            (4, "started"),
            (4, "compute_finished"),
            (4, "ended:cancelled"),
        ]
    );
}
