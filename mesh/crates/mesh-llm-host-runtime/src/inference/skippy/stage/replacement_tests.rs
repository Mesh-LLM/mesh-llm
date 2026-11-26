//! Replacing a running stage: who may replace it, and that a replacement
//! refused before launch leaves it serving.

use super::tests::{coordinator_id, load_request};
use super::*;

/// Answers nothing; it only gives a placeholder running stage a live server
/// handle without loading a model.
struct PlaceholderStageBackend;

#[async_trait::async_trait]
impl skippy_inference_api::InferenceBackend for PlaceholderStageBackend {
    async fn models(
        &self,
    ) -> skippy_inference_api::InferenceResult<Vec<skippy_inference_api::ModelObject>> {
        Ok(Vec::new())
    }

    async fn chat_completion(
        &self,
        _request: skippy_inference_api::generation::GenerationRequest,
    ) -> skippy_inference_api::InferenceResult<skippy_inference_api::generation::GenerationResponse>
    {
        Err(skippy_inference_api::InferenceError::unsupported(
            "placeholder stage",
        ))
    }

    async fn chat_completion_stream(
        &self,
        _request: skippy_inference_api::generation::GenerationRequest,
        _context: skippy_inference_api::InferenceRequestContext,
    ) -> skippy_inference_api::InferenceResult<
        skippy_inference_api::generation::GenerationEventStream,
    > {
        Err(skippy_inference_api::InferenceError::unsupported(
            "placeholder stage",
        ))
    }
}

/// Register `load` as a running stage served by a placeholder server.
fn insert_placeholder_running_stage(state: &mut StageControlState, load: StageLoadRequest) {
    let server = skippy_serving::start_openai_backend(
        "127.0.0.1:0".parse().unwrap(),
        Arc::new(PlaceholderStageBackend),
    );
    state.stages.insert(
        stage_key(&load.topology_id, &load.run_id, &load.stage_id),
        RunningStage {
            load,
            server,
            compute_meter: Arc::default(),
            package: None,
        },
    );
}

fn other_coordinator_id() -> iroh::EndpointId {
    iroh::EndpointId::from(iroh::SecretKey::from_bytes(&[0x6b; 32]).public())
}

#[test]
fn coordinator_bound_stage_is_replaceable_only_by_its_coordinator() {
    let mut owned = load_request();
    owned.coordinator_id = Some(coordinator_id());
    let mut same = load_request();
    same.coordinator_id = Some(coordinator_id());
    let mut other = load_request();
    other.coordinator_id = Some(other_coordinator_id());
    let unbound = load_request();

    assert_eq!(stage_replacement_rejection(&owned, &same), None);
    assert_eq!(
        stage_replacement_rejection(&owned, &unbound),
        Some("an unbound stage load cannot replace a coordinator-bound stage")
    );
    assert_eq!(stage_replacement_rejection(&unbound, &unbound), None);
    assert_eq!(stage_replacement_rejection(&unbound, &other), None);
}

#[tokio::test]
async fn load_from_another_coordinator_leaves_the_bound_stage_running() {
    let mut state = StageControlState::default();
    let mut owned = load_request();
    owned.coordinator_id = Some(coordinator_id());
    owned.coordinator_term = 3;
    insert_placeholder_running_stage(&mut state, owned);

    let mut takeover = load_request();
    takeover.coordinator_id = Some(other_coordinator_id());
    takeover.coordinator_term = 9;
    let response = state
        .load(takeover, other_coordinator_id(), StageSourceFetch::Allowed)
        .await
        .expect("a rejected takeover must answer, not fail");

    assert!(!response.accepted);
    assert!(!response.removed_running_stage);
    assert_eq!(
        response.error.as_deref(),
        Some("a stage load cannot replace a stage bound to a different coordinator")
    );
    let running = state
        .stages
        .values()
        .next()
        .expect("the bound stage must keep running");
    assert_eq!(running.load.coordinator_id, Some(coordinator_id()));
    state.shutdown().await.unwrap();
}

#[tokio::test]
#[serial_test::serial(runtime_event_engine_state)]
async fn invalid_replacement_load_leaves_the_running_stage_serving() {
    // Calls the launch step directly so the replacement ordering is tested
    // without a native runtime, which `load` requires before it gets here.
    let error = launch_rejected_replacement(|replacement| {
        replacement.bind_addr = "0.0.0.0:0".to_string();
    })
    .await;
    assert!(error.contains("not a loopback address"), "{error}");
}

#[tokio::test]
#[serial_test::serial(runtime_event_engine_state)]
async fn replacement_with_unsupported_runtime_options_leaves_the_stage_serving() {
    // A runtime slice of a GGUF materializes nothing, so preparing it leaves
    // the process environment alone.
    let error = launch_rejected_replacement(|replacement| {
        replacement.cache_type_k = "bogus".to_string();
    })
    .await;
    assert!(error.contains("unsupported KV cache type"), "{error}");
}

/// Launch a replacement for a running placeholder stage, edited by `edit` so
/// launching it fails, and return the failure after checking that the
/// running stage was not shut down. Its stage-load operation reserves a slot
/// in the process-wide runtime event engine, so callers serialize on it.
async fn launch_rejected_replacement(edit: impl FnOnce(&mut StageLoadRequest)) -> String {
    let mut state = StageControlState::default();
    let mut running = load_request();
    running.shutdown_generation = 1;
    let key = stage_key(&running.topology_id, &running.run_id, &running.stage_id);
    insert_placeholder_running_stage(&mut state, running);

    let mut replacement = load_request();
    replacement.shutdown_generation = 2;
    edit(&mut replacement);
    let stage_op = runtime_events::StageLoadOperation::begin(
        &replacement.topology_id,
        &replacement.stage_id,
        replacement.layer_start,
        replacement.layer_end,
    );
    let error = state
        .launch_stage(key, replacement, StageSourceFetch::LocalOnly, stage_op)
        .await
        .expect_err("the invalid replacement must be rejected");
    let stage = state
        .stages
        .values()
        .next()
        .expect("the stage must keep serving when its replacement is invalid");
    assert_eq!(stage.load.shutdown_generation, 1);
    assert!(
        matches!(
            stage.server.status().state,
            skippy_serving::EmbeddedState::Starting | skippy_serving::EmbeddedState::Ready
        ),
        "the replaced stage must not have been shut down"
    );
    state.shutdown().await.unwrap();
    format!("{error:#}")
}

/// The stage control loop with a placeholder stage running `load`, for tests
/// that drive a node's stage-control handler against a running stage.
pub(crate) fn spawn_stage_control_loop_with_placeholder_stage(
    load: StageLoadRequest,
) -> StageControlHandle {
    let mut state = StageControlState::default();
    insert_placeholder_running_stage(&mut state, load);
    spawn_stage_control_loop_with_state(state)
}
