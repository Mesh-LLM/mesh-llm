//! Compatibility adapter from the existing llama.cpp `RuntimeState` to the
//! engine-neutral dense stage contract.
//!
//! The mature binary server continues to call `RuntimeState` directly while
//! its broader cache/MTP/multimodal surface is migrated. This adapter keeps the
//! new contract honest: it is implementable by the existing runtime without
//! changing the native Skippy ABI or teaching MLX about llama types.

use std::sync::Mutex;

use anyhow::{Result, bail, ensure};
use half::{bf16, f16};
use skippy_engine::{
    StageActivation, StageEngine, StageEngineInfo, StageExecutionKind, StageExecutionOutput,
    StageExecutionRequest,
};
use skippy_runtime::{
    ActivationDesc, ActivationFrame, LogitBias, MAX_LOGIT_BIAS, RuntimeActivationDType,
    RuntimeActivationLayout, SamplingConfig,
};

use crate::runtime_state::RuntimeState;

pub struct LlamaStageEngine {
    info: StageEngineInfo,
    runtime: Mutex<RuntimeState>,
}

impl LlamaStageEngine {
    pub fn new(info: StageEngineInfo, runtime: RuntimeState) -> Result<Self> {
        info.validate()?;
        Ok(Self {
            info,
            runtime: Mutex::new(runtime),
        })
    }

    pub fn into_runtime(self) -> Result<RuntimeState> {
        self.runtime
            .into_inner()
            .map_err(|_| anyhow::anyhow!("llama stage runtime lock poisoned"))
    }

    fn runtime(&self) -> Result<std::sync::MutexGuard<'_, RuntimeState>> {
        self.runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("llama stage runtime lock poisoned"))
    }

    fn activation_frame(&self, input: Option<StageActivation>) -> Result<Option<ActivationFrame>> {
        let Some(input) = input else {
            return Ok(None);
        };
        ensure!(
            input.width == self.info.activation_width as usize,
            "input activation width mismatch"
        );
        let producer_stage_index = self.info.stage_index.saturating_sub(1);
        Ok(Some(ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: i32::try_from(producer_stage_index)?,
                layer_start: 0,
                layer_end: i32::try_from(self.info.layer_start)?,
                token_count: u32::try_from(input.token_count)?,
                sequence_count: u32::from(input.token_count > 0),
                payload_bytes: u64::try_from(input.f32_le_bytes.len())?,
                flags: 0,
            },
            payload: input.f32_le_bytes,
        }))
    }

    fn output(
        &self,
        frame: ActivationFrame,
        predicted_tokens: Vec<i32>,
    ) -> Result<StageExecutionOutput> {
        let activation = if self.info.is_final() {
            None
        } else {
            Some(runtime_activation(
                frame,
                self.info.activation_width as usize,
            )?)
        };
        Ok(StageExecutionOutput {
            activation,
            predicted_tokens: if self.info.is_final() {
                predicted_tokens
            } else {
                Vec::new()
            },
        })
    }
}

impl StageEngine for LlamaStageEngine {
    fn info(&self) -> &StageEngineInfo {
        &self.info
    }

    fn execute(&self, request: StageExecutionRequest) -> Result<StageExecutionOutput> {
        let input = self.activation_frame(request.input)?;
        let sampling = runtime_sampling_config(request.sampling.as_ref());
        let session_id = request.session_id.to_string();
        let mut runtime = self.runtime()?;
        match request.kind {
            StageExecutionKind::Prefill => {
                let output = runtime.prefill_frame_with_positions(
                    &session_id,
                    &request.token_ids,
                    &request.positions,
                    input.as_ref(),
                )?;
                self.output(output, Vec::new())
            }
            StageExecutionKind::PrefillFinal if self.info.is_final() => {
                let (predicted, output) = runtime.prefill_final_frame_sampled(
                    &session_id,
                    &request.token_ids,
                    &request.positions,
                    sampling.as_ref(),
                    input.as_ref(),
                )?;
                self.output(output, vec![predicted])
            }
            StageExecutionKind::PrefillFinal => {
                let output = runtime.prefill_frame_with_positions(
                    &session_id,
                    &request.token_ids,
                    &request.positions,
                    input.as_ref(),
                )?;
                self.output(output, Vec::new())
            }
            StageExecutionKind::Decode => {
                let token_id = one_token(&request.token_ids, "decode")?;
                let (predicted, output) = runtime.decode_frame_sampled(
                    &session_id,
                    token_id,
                    sampling.as_ref(),
                    input.as_ref(),
                    0,
                )?;
                self.output(output, vec![predicted])
            }
            StageExecutionKind::Verify => {
                let (predicted, output) = runtime.verify_frame_sampled(
                    &session_id,
                    &request.token_ids,
                    sampling.as_ref(),
                    input.as_ref(),
                    0,
                )?;
                self.output(output, predicted)
            }
        }
    }

    fn reset_session(&self, session_id: u64) -> Result<()> {
        self.runtime()?
            .drop_session_timed(&session_id.to_string())?;
        Ok(())
    }

    fn checkpoint_session(&self, session_id: u64) -> Result<()> {
        self.runtime()?.checkpoint_session(&session_id.to_string())
    }

    fn restore_session(&self, session_id: u64) -> Result<()> {
        self.runtime()?.restore_session(&session_id.to_string())
    }

    fn trim_session(&self, session_id: u64, token_count: u64) -> Result<()> {
        self.runtime()?
            .trim_session(&session_id.to_string(), token_count)
    }
}

fn one_token(tokens: &[i32], operation: &str) -> Result<i32> {
    match tokens {
        [token] => Ok(*token),
        _ => bail!("{operation} requires exactly one token"),
    }
}

fn runtime_sampling_config(
    sampling: Option<&skippy_protocol::binary::StageSamplingConfig>,
) -> Option<SamplingConfig> {
    let sampling = sampling?;
    let mut config = SamplingConfig {
        enabled: true,
        seed: sampling.seed,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        presence_penalty: sampling.presence_penalty,
        frequency_penalty: sampling.frequency_penalty,
        repeat_penalty: sampling.repeat_penalty,
        penalty_last_n: sampling.penalty_last_n,
        ..SamplingConfig::default()
    };
    config.logit_bias = sampling
        .logit_bias
        .iter()
        .take(MAX_LOGIT_BIAS)
        .map(|source| LogitBias {
            token_id: source.token_id,
            bias: source.bias,
        })
        .collect();
    sampling.enabled().then_some(config)
}

fn runtime_activation(frame: ActivationFrame, width: usize) -> Result<StageActivation> {
    ensure!(
        frame.desc.layout == RuntimeActivationLayout::TokenMajor,
        "llama stage output activation is not token-major"
    );
    ensure!(
        frame.payload.len() as u64 == frame.desc.payload_bytes,
        "llama stage output payload length mismatch"
    );
    let token_count = usize::try_from(frame.desc.token_count)?;
    let f32_le_bytes = match frame.desc.dtype {
        RuntimeActivationDType::F32 => frame.payload,
        RuntimeActivationDType::F16 => {
            decode_16_bit_activation(&frame.payload, |bytes| f16::from_le_bytes(bytes).to_f32())?
        }
        RuntimeActivationDType::Bf16 => {
            decode_16_bit_activation(&frame.payload, |bytes| bf16::from_le_bytes(bytes).to_f32())?
        }
        RuntimeActivationDType::Unknown => bail!("llama stage output activation dtype is unknown"),
    };
    StageActivation::new(token_count, width, f32_le_bytes)
}

fn decode_16_bit_activation(payload: &[u8], decode: impl Fn([u8; 2]) -> f32) -> Result<Vec<u8>> {
    ensure!(
        payload.len().is_multiple_of(2),
        "16-bit activation has odd byte length"
    );
    Ok(payload
        .chunks_exact(2)
        .flat_map(|chunk| decode([chunk[0], chunk[1]]).to_le_bytes())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_activation_converts_to_f32_contract_bytes() {
        let values = [f16::from_f32(1.5), f16::from_f32(-2.0)];
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let frame = ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F16,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 1,
                token_count: 1,
                sequence_count: 1,
                payload_bytes: payload.len() as u64,
                flags: 0,
            },
            payload,
        };
        let activation = runtime_activation(frame, 2).unwrap();
        assert_eq!(activation.values(), vec![1.5, -2.0]);
    }
}
