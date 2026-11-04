//! Local serving of speech-recognition (Whisper) models.
//!
//! A Whisper model opens through its own native entry point, not the llama
//! loader, and has no stages, sessions, or prompt cache. Each concurrent
//! transcription holds its own decoder state, so the capacity ledger charges
//! the weights plus one decoder state per lane.

use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};

use super::capacity::runtime_model_required_bytes;
use super::context_planning::{RuntimeResourcePlanBreakdown, RuntimeResourcePlanSource};
use super::local::{
    LocalOpenAiModelStartSpec, LocalRuntimeBackendHandle, LocalRuntimeModelHandle, ModelLoadSource,
    report_model_loaded_analytics,
};
use super::local_memory_plan::{MemoryPlanStartPath, emit_memory_plan_resolved};
use super::split_planning::format_gb;
use crate::inference::skippy;
use crate::mesh;
use crate::models;

/// Concurrent transcriptions per model. Each holds its own decoder state, so
/// lanes trade memory for throughput.
const SPEECH_LANES_ENV: &str = "MESH_LLM_SPEECH_LANES";
const DEFAULT_SPEECH_LANES: usize = 2;
/// Native device name for "the first GPU, CPU when there is none".
const FIRST_ACCELERATOR: &str = "auto";

/// Where a speech model runs: the configured device (`--device`, then a pinned
/// GPU), otherwise the first accelerator. Whisper's encoder is a dense
/// transformer over 30 s windows and is many times faster there.
fn speech_device(device_override: Option<&str>, pinned_device: Option<&str>) -> String {
    device_override
        .or(pinned_device)
        .unwrap_or(FIRST_ACCELERATOR)
        .to_string()
}

fn speech_lanes(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|lanes| *lanes > 0)
        .unwrap_or(DEFAULT_SPEECH_LANES)
}

/// Bytes a speech model occupies once serving: its weights plus one decoder
/// state per lane.
fn speech_resident_bytes(
    weight_bytes: u64,
    header: &skippy_runtime::SpeechModelHeader,
    lanes: usize,
) -> u64 {
    weight_bytes.saturating_add(header.decoder_state_bytes().saturating_mul(lanes as u64))
}

/// Loads a speech-recognition model. It answers `/v1/audio/transcriptions`
/// and `/v1/audio/translations` only and advertises the speech-recognition
/// workload class with verified audio input.
pub(super) async fn start_local_speech_model(
    spec: LocalOpenAiModelStartSpec<'_>,
    model_name: String,
    weight_bytes: u64,
    header: skippy_runtime::SpeechModelHeader,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let lanes = speech_lanes(std::env::var(SPEECH_LANES_ENV).ok().as_deref());
    let state_bytes = header.decoder_state_bytes();
    let resident = speech_resident_bytes(weight_bytes, &header, lanes);
    let required = runtime_model_required_bytes(resident);
    anyhow::ensure!(
        spec.capacity_budget_bytes >= required,
        "speech model requires {} (weights {} plus {} lanes of {} decoder state), local capacity is {}",
        format_gb(required),
        format_gb(weight_bytes),
        lanes,
        format_gb(state_bytes),
        format_gb(spec.capacity_budget_bytes)
    );
    emit_memory_plan_resolved(
        &model_name,
        Some(&RuntimeResourcePlanBreakdown {
            vram_bytes: spec.capacity_budget_bytes,
            model_bytes: weight_bytes,
            projector_bytes: 0,
            speculative_bytes: 0,
            kv_budget_bytes: 0,
            planned_kv_bytes: 0,
            kv_bytes_per_token: 0,
            compute_charge_bytes: resident.saturating_sub(weight_bytes),
            planning_source: RuntimeResourcePlanSource::StaticEstimate,
            measured_fit: None,
            slots: lanes,
            context_length: header.text_ctx,
            slots_auto: false,
            context_auto: false,
        }),
        // A direct file load; the decoder states are the plan's compute
        // charge, and the load line below names the device.
        MemoryPlanStartPath::Direct,
    );

    let device = speech_device(
        spec.device_override.as_deref(),
        spec.pinned_gpu.map(|gpu| gpu.backend_device.as_str()),
    );
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_name.clone(),
        source: None,
    });
    let model_path = spec.model_path.to_path_buf();
    let load_device = device.clone();
    let model = tokio::task::spawn_blocking(move || {
        skippy::load_speech_recognition_model(&model_path, Some(&load_device))
    })
    .await
    .context("join load speech model task")??;
    let info = *model.info();
    tracing::info!(
        model = model_name,
        speech.device = device,
        speech.on_accelerator = info.on_accelerator,
        speech.multilingual = info.multilingual,
        speech.lanes = lanes,
        speech.decoder_state_bytes = state_bytes,
        "speech-recognition model loaded"
    );
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_name.clone(),
        bytes: None,
    });
    report_model_loaded_analytics(&model_name, ModelLoadSource::WhisperFile, None);
    let capabilities = models::ModelCapabilities {
        audio: models::CapabilityLevel::Supported,
        ..Default::default()
    }
    .normalize();
    let http = skippy::start_speech_recognition_http_on(
        &model_name,
        model.clone(),
        lanes,
        spec.http_bind_addr,
    );
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();
    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length: header.text_ctx,
            slots: lanes,
            capabilities,
            workload_class: mesh::ModelWorkloadClass::SpeechRecognition,
            thinking: None,
            inner: LocalRuntimeBackendHandle::Dedicated {
                model: super::local::DedicatedModel::Speech(model),
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> skippy_runtime::SpeechModelHeader {
        skippy_runtime::SpeechModelHeader {
            vocab_size: 51_866,
            audio_ctx: 1500,
            audio_state: 1280,
            audio_layer_count: 32,
            text_ctx: 448,
            text_state: 1280,
            text_layer_count: 4,
            mel_count: 128,
        }
    }

    #[test]
    fn residency_charges_one_decoder_state_per_lane() {
        let header = header();
        let state = header.decoder_state_bytes();
        assert_eq!(speech_resident_bytes(1_000, &header, 1), 1_000 + state);
        assert_eq!(speech_resident_bytes(1_000, &header, 3), 1_000 + 3 * state);
    }

    #[test]
    fn lanes_default_to_two_and_ignore_nonsense() {
        assert_eq!(speech_lanes(None), DEFAULT_SPEECH_LANES);
        assert_eq!(speech_lanes(Some("4")), 4);
        assert_eq!(speech_lanes(Some("0")), DEFAULT_SPEECH_LANES);
        assert_eq!(speech_lanes(Some("many")), DEFAULT_SPEECH_LANES);
    }

    #[test]
    fn the_configured_device_wins_and_the_first_accelerator_is_the_default() {
        assert_eq!(speech_device(Some("CUDA1"), Some("CUDA0")), "CUDA1");
        assert_eq!(speech_device(None, Some("CUDA0")), "CUDA0");
        assert_eq!(speech_device(None, None), "auto");
    }
}
