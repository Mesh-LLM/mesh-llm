//! Encoder-decoder lane admission for the embedded OpenAI frontend (#1954).
//!
//! Native serializes encoder-decoder work onto a single lane, so the loaded
//! [`RuntimeState`] reports one lane even when `StageConfig::lane_count` is
//! higher. OpenAI generation admission must be measured against that effective
//! bound; measured against the configured bound it lets startup pass with N
//! concurrent generations the runtime can never serve.

use super::*;

use crate::frontend::EmbeddedOpenAiRequestDefaults;
use crate::frontend::generation::ensure_generation_concurrency_fits_lanes;
use crate::frontend::speculative::SpeculativeDecodeConfig;
use crate::frontend::{EmbeddedOpenAiArgs, embedded_openai_backend};
use crate::runtime_state::RuntimeState;
use crate::telemetry::TelemetryLevel;

/// Lanes the stage config asks for.
const CONFIGURED_LANES: u32 = 4;
/// Lanes the runtime serves for an encoder-decoder model loaded from that
/// config: `lane_count_for_workload(4, ModelWorkload::EncoderDecoder) == 1`,
/// pinned by
/// `runtime_state::tests::encoder_decoder_lane_admission_clamps_to_the_native_single_lane`.
const ENCODER_DECODER_RUNTIME_LANES: u32 = 1;

/// Launch arguments for an encoder-decoder stage whose config asks for four
/// lanes while the loaded runtime serves one.
fn encoder_decoder_openai_args() -> EmbeddedOpenAiArgs {
    let config = StageConfig {
        stage_id: "stage-0".to_string(),
        lane_count: CONFIGURED_LANES,
        ..StageConfig::default()
    };
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(
        ENCODER_DECODER_RUNTIME_LANES,
    )));
    let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
    EmbeddedOpenAiArgs {
        bind_addr: "127.0.0.1:0".parse::<SocketAddr>().expect("bind addr"),
        config,
        runtime,
        model_id: Some("encoder-decoder-fixture".to_string()),
        default_max_tokens: 16,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        live_defaults_scope: 0,
        // The OpenAI default is `config.lane_count`, so an unset
        // `--generation-concurrency` on this stage arrives here as four.
        generation_concurrency: CONFIGURED_LANES as usize,
        continuous_batching: false,
        pipeline_decode_groups: None,
        adaptive_generation_min_concurrency: None,
        generation_queue_capacity: 8,
        generation_admission_timeout_secs: 1,
        prefill_chunk_size: 512,
        prefill_chunk_policy: "fixed".to_string(),
        prefill_chunk_schedule: None,
        prefill_adaptive_start: 0,
        prefill_adaptive_step: 0,
        prefill_adaptive_max: 0,
        prefill_adaptive_target_ms: 1.0,
        draft_model_path: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        draft_n_gpu_layers: None,
        speculative: SpeculativeDecodeConfig::default(),
        native_mtp_enabled: false,
        native_mtp_draft_model_path: None,
        native_mtp_max_tokens: 0,
        native_mtp_min_tokens: 0,
        activation_width: 0,
        reply_credit_limit: None,
        downstream_connect_timeout_secs: 1,
        downstream_wire_condition: WireCondition::new(0.0, None).expect("wire condition"),
        prediction_returns: None,
        telemetry,
        hook_policy: None,
        generation_receipt: None,
        generation_lifecycle: None,
        linear_proposal_ingress: None,
        openai_guardrails: None,
        kv_lifecycle_observer: None,
        l3_manager: None,
    }
}

/// The lane bound the guard refuses `CONFIGURED_LANES` generations against.
fn refusal_lane_bound(lane_count: u32) -> Option<String> {
    ensure_generation_concurrency_fits_lanes(
        CONFIGURED_LANES as usize,
        lane_count,
        "--generation-concurrency",
    )
    .err()
    .map(|error| error.to_string())
}

#[test]
/// The guard itself already refuses above the bound it is given.
fn guard_refuses_concurrency_above_the_lane_bound_it_is_given() {
    let refusal = refusal_lane_bound(ENCODER_DECODER_RUNTIME_LANES)
        .expect("four generations must be refused on one lane");
    assert!(
        refusal.contains("lane_count"),
        "refusal must name the lane bound, got: {refusal}"
    );
    assert!(
        refusal.contains("1"),
        "refusal must report the one served lane, got: {refusal}"
    );
    // A causal model keeps its configured lanes.
    assert!(
        refusal_lane_bound(CONFIGURED_LANES).is_none(),
        "a causal runtime with the configured lanes must stay admissible"
    );
}

#[test]
/// The frontend must give the guard the runtime's bound, not the config's.
fn encoder_decoder_admission_measures_concurrency_against_the_served_lane_count() {
    let args = encoder_decoder_openai_args();
    let served_lanes = {
        let runtime = args
            .runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        runtime.lane_count()
    };
    let configured_lanes = args.config.lane_count;
    let requested = args.generation_concurrency;
    assert_eq!(
        (configured_lanes, served_lanes),
        (CONFIGURED_LANES, ENCODER_DECODER_RUNTIME_LANES),
        "fixture must be an encoder-decoder stage configured for four lanes on a one-lane runtime"
    );

    let outcome = match embedded_openai_backend(encoder_decoder_openai_args()) {
        Ok(_) => "admitted".to_string(),
        Err(error) => {
            let mut chain = error.to_string();
            let mut source = std::error::Error::source(&*error);
            while let Some(cause) = source {
                chain.push_str(": ");
                chain.push_str(&cause.to_string());
                source = cause.source();
            }
            chain
        }
    };
    let expected = refusal_lane_bound(served_lanes)
        .expect("the guard must refuse the requested concurrency on one lane");
    assert!(
        outcome.contains(&expected),
        "lane-count mismatch: requested {requested} generations, config lane_count \
         {configured_lanes}, runtime serves {served_lanes}. The frontend admitted past \
         admission and failed later instead of refusing; expected the guard refusal \
         \"{expected}\", got: {outcome}"
    );
}
