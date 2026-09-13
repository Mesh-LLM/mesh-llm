//! Real-projector lifecycle regressions, opt-in through the workload fixture.

use super::*;
use crate::{ModelInfo, RuntimeConfig, TensorRole};

fn speech_fixture() -> Result<Option<StageModel>> {
    if std::env::var("SKIPPY_WORKLOAD_CLASS").as_deref() != Ok("speech_synthesis") {
        return Ok(None);
    }
    let path = std::env::var("SKIPPY_WORKLOAD_MODEL").context("speech fixture model")?;
    let projector =
        std::env::var("SKIPPY_WORKLOAD_PROJECTOR").context("speech fixture projector")?;
    let layer_end = ModelInfo::open(&path)?
        .tensors()?
        .into_iter()
        .filter(|tensor| tensor.role == TensorRole::Layer)
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .context("speech fixture layers")?
        + 1;
    let config = RuntimeConfig {
        layer_end,
        ctx_size: 2048,
        n_batch: Some(2048),
        n_ubatch: Some(2048),
        projector_path: Some(projector),
        projector_use_gpu: Some(false),
        kv_offload: Some(false),
        op_offload: Some(false),
        ..RuntimeConfig::default()
    };
    StageModel::open(path, &config).map(Some)
}

fn assert_generation_reusable(model: &StageModel, session: &mut StageSession) -> Result<()> {
    session.reset()?;
    let tokens = model.tokenize("The mesh is ready.", true)?;
    let (last, prefix) = tokens.split_last().context("generation fixture tokens")?;
    session.prefill_chunked(prefix)?;
    // Plain prefill deliberately produces no logits. Decode the final token
    // with an output row so this checks real generation, not a sampler fallback.
    let token = session.decode_step(*last)?;
    assert!(token >= 0, "normal generation must produce a valid token");
    assert_eq!(session.sample_current(None)?, token);
    Ok(())
}

#[test]
fn speech_success_cancellation_and_native_failure_leave_session_reusable() -> Result<()> {
    let Some(model) = speech_fixture()? else {
        return Ok(());
    };
    let mut session = model.create_session()?;
    let mut config = SpeechSynthesisConfig {
        prompt: "Hello.".into(),
        language: None,
        top_k: 1,
        top_p: 1.0,
        seed: 42,
        output_format: SpeechOutputFormat::Wav,
        max_frames: 2,
    };
    let cancelled = model
        .synthesize_speech(&mut session, &config, || true)
        .unwrap_err();
    assert!(cancelled.to_string().contains("cancelled"), "{cancelled:#}");
    assert_generation_reusable(&model, &mut session)?;
    config.language = Some("not-a-supported-language".into());
    let rejected = model
        .synthesize_speech(&mut session, &config, || false)
        .unwrap_err();
    assert!(
        rejected.to_string().contains("rejected the input"),
        "{rejected:#}"
    );
    assert_generation_reusable(&model, &mut session)?;
    config.language = None;
    let capped = model
        .synthesize_speech(&mut session, &config, || false)
        .unwrap_err();
    assert!(capped.to_string().contains("frame limit"), "{capped:#}");
    assert_generation_reusable(&model, &mut session)?;
    config.max_frames = 512;
    let audio = model.synthesize_speech(&mut session, &config, || false)?;
    assert!(audio.sample_count > 0 && audio.bytes.len() > 44);
    assert_generation_reusable(&model, &mut session)?;
    Ok(())
}
