//! Native output-buffer contracts for sampled prefill with no output head.

use crate::{ModelInfo, RuntimeConfig, StageModel, TensorRole};
use anyhow::{Context as _, Result};

/// The raw wrapper starts its token buffer at zero; native prefill must replace
/// it with the no-token sentinel on every successful output-disabled call.
#[test]
fn output_disabled_prefill_clears_sampled_token_when_fixture_is_set() -> Result<()> {
    let Some(path) = std::env::var_os("SKIPPY_CORRECTNESS_MODEL") else {
        return Ok(());
    };
    let _native_log_guard = crate::logging::native_log_test_guard();
    let info = ModelInfo::open(&path)?;
    let layer_end = info
        .tensors()?
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Layer)
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .context("correctness fixture has no transformer layers")?
        + 1;
    let model = StageModel::open(
        &path,
        &RuntimeConfig {
            layer_end,
            ctx_size: 256,
            n_gpu_layers: 0,
            include_output: false,
            filter_tensors_on_load: false,
            ..RuntimeConfig::default()
        },
    )?;
    let tokens = model.tokenize("A short native prefill regression.", true)?;
    let mut session = model.create_session()?;
    for positions in [Vec::new(), (0..i32::try_from(tokens.len())?).collect()] {
        session.reset()?;
        let (token, _, _) =
            session.prefill_chunk_frame_sampled_raw(&tokens, &positions, None, None, 0)?;
        assert_eq!(
            token, -1,
            "an output-disabled prefill must never leak a token"
        );
    }
    Ok(())
}
