//! Deterministic test-only TTS output for comparison with pinned llama-tts.
//!
//! The public speech endpoint deliberately uses a random seed. An independent
//! waveform oracle therefore has to drive the same local model execution with
//! fixed sampling parameters rather than compare two random HTTP responses.

use super::*;
use skippy_runtime::{SpeechOutputFormat, SpeechSynthesisConfig};

const OUTPUT_ENV: &str = "SKIPPY_TTS_ORACLE_CANDIDATE_WAV";

fn fixture_path(name: &str) -> Result<PathBuf> {
    let path = PathBuf::from(env::var_os(name).context(format!("{name} is required"))?);
    if !path.is_file() {
        bail!("{name} does not point at a file: {}", path.display());
    }
    Ok(path)
}

fn fixture_number<T>(name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    env::var(name)
        .with_context(|| format!("{name} is required"))?
        .parse::<T>()
        .map_err(|error| anyhow!("parse {name}: {error}"))
}

fn local_tts_config(
    model_id: &str,
    model_path: &Path,
    projector_path: &Path,
    layer_end: u32,
) -> StageConfig {
    StageConfig {
        run_id: "tts-monolithic-oracle".to_string(),
        topology_id: "tts-monolithic-oracle-local".to_string(),
        model_id: model_id.to_string(),
        model_path: Some(model_path.to_string_lossy().to_string()),
        projector_path: Some(projector_path.to_string_lossy().to_string()),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end,
        ctx_size: 2048,
        lane_count: 1,
        n_batch: Some(2048),
        n_ubatch: Some(2048),
        n_gpu_layers: 0,
        kv_offload: Some(false),
        op_offload: Some(false),
        selected_device: Some(StageDevice {
            backend_device: "CPU".to_string(),
            stable_id: None,
            index: None,
            vram_bytes: None,
        }),
        filter_tensors_on_load: false,
        native_mtp_enabled: false,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        ..StageConfig::default()
    }
}

#[test]
fn deterministic_tts_candidate_when_fixture_is_set() -> Result<()> {
    let Some(output_path) = env::var_os(OUTPUT_ENV) else {
        return Ok(());
    };
    let output_path = PathBuf::from(output_path);
    let model_path = fixture_path("SKIPPY_WORKLOAD_MODEL")?;
    let projector_path = fixture_path("SKIPPY_WORKLOAD_PROJECTOR")?;
    let model_id =
        env::var("SKIPPY_WORKLOAD_MODEL_ID").context("SKIPPY_WORKLOAD_MODEL_ID is required")?;
    let layer_end: u32 = fixture_number("SKIPPY_WORKLOAD_LAYER_END")?;
    if layer_end == 0 {
        bail!("SKIPPY_WORKLOAD_LAYER_END must be positive");
    }
    let prompt =
        env::var("SKIPPY_TTS_ORACLE_PROMPT").context("SKIPPY_TTS_ORACLE_PROMPT is required")?;
    let seed: u32 = fixture_number("SKIPPY_TTS_ORACLE_SEED")?;
    let top_k: i32 = fixture_number("SKIPPY_TTS_ORACLE_TOP_K")?;
    let top_p: f32 = fixture_number("SKIPPY_TTS_ORACLE_TOP_P")?;
    let max_frames: usize = fixture_number("SKIPPY_TTS_ORACLE_MAX_FRAMES")?;
    if prompt.is_empty() || top_k < 1 || !(0.0..=1.0).contains(&top_p) || max_frames == 0 {
        bail!("invalid deterministic TTS oracle parameters");
    }

    let config = local_tts_config(&model_id, &model_path, &projector_path, layer_end);
    let runtime = load_runtime(&config)?.context("load full-model TTS oracle candidate")?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("TTS oracle candidate runtime lock poisoned"))?;
    let audio = runtime.synthesize_speech(
        "tts-oracle",
        &SpeechSynthesisConfig {
            prompt,
            language: None,
            top_k,
            top_p,
            seed,
            output_format: SpeechOutputFormat::Wav,
            max_frames,
        },
        || false,
    )?;
    if audio.bytes.len() <= 44 || audio.sample_rate == 0 || audio.sample_count == 0 {
        bail!("deterministic TTS candidate produced no WAV samples");
    }
    fs::write(&output_path, &audio.bytes)
        .with_context(|| format!("write candidate WAV to {}", output_path.display()))?;
    Ok(())
}
