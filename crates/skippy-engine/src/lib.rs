//! Engine-neutral staged execution contract.
//!
//! This crate deliberately owns no model runtime. It is the narrow boundary
//! between Skippy's stage transport and concrete engines such as llama.cpp or
//! MLX: token IDs and F32 residual bytes enter, residual bytes and optional
//! predicted token IDs leave. Native arrays and cache handles never cross it.

use anyhow::{Result, bail, ensure};
use skippy_protocol::binary::StageSamplingConfig;

/// Static facts needed by the stage transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageEngineInfo {
    pub engine: String,
    pub model_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub activation_width: u32,
}

impl StageEngineInfo {
    pub fn is_first(&self) -> bool {
        self.layer_start == 0
    }

    pub fn is_final(&self) -> bool {
        self.layer_end == self.total_layers
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.layer_start < self.layer_end,
            "stage layer range must be non-empty"
        );
        ensure!(
            self.layer_end <= self.total_layers,
            "stage layer range exceeds model layer count"
        );
        ensure!(
            self.activation_width > 0,
            "activation width must be positive"
        );
        Ok(())
    }
}

/// One decoded residual activation tensor in token-major F32 form.
#[derive(Clone, Debug, PartialEq)]
pub struct StageActivation {
    pub token_count: usize,
    pub width: usize,
    pub f32_le_bytes: Vec<u8>,
}

impl StageActivation {
    pub fn new(token_count: usize, width: usize, f32_le_bytes: Vec<u8>) -> Result<Self> {
        let expected = token_count
            .checked_mul(width)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("stage activation size overflow"))?;
        ensure!(
            f32_le_bytes.len() == expected,
            "stage activation has {} bytes; expected {expected}",
            f32_le_bytes.len()
        );
        Ok(Self {
            token_count,
            width,
            f32_le_bytes,
        })
    }

    pub fn values(&self) -> Vec<f32> {
        self.f32_le_bytes
            .chunks_exact(size_of::<f32>())
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    pub fn from_values(token_count: usize, width: usize, values: &[f32]) -> Result<Self> {
        let expected = token_count
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("stage activation element count overflow"))?;
        ensure!(
            values.len() == expected,
            "stage activation has {} values; expected {expected}",
            values.len()
        );
        Self::new(
            token_count,
            width,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageExecutionKind {
    Prefill,
    PrefillFinal,
    Decode,
    Verify,
}

/// A single stage operation after wire decoding.
#[derive(Clone, Debug, PartialEq)]
pub struct StageExecutionRequest {
    pub session_id: u64,
    pub kind: StageExecutionKind,
    pub token_ids: Vec<i32>,
    pub positions: Vec<i32>,
    pub input: Option<StageActivation>,
    pub sampling: Option<StageSamplingConfig>,
}

/// Result of one stage operation before wire encoding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageExecutionOutput {
    pub activation: Option<StageActivation>,
    pub predicted_tokens: Vec<i32>,
}

impl StageExecutionOutput {
    pub fn predicted(&self) -> Option<i32> {
        self.predicted_tokens.first().copied()
    }
}

/// Concrete staged model execution behind a transport-neutral interface.
pub trait StageEngine: Send + Sync + 'static {
    fn info(&self) -> &StageEngineInfo;

    fn execute(&self, request: StageExecutionRequest) -> Result<StageExecutionOutput>;

    fn reset_session(&self, session_id: u64) -> Result<()>;

    fn checkpoint_session(&self, _session_id: u64) -> Result<()> {
        bail!(
            "{} stage engine does not support checkpoints",
            self.info().engine
        )
    }

    fn restore_session(&self, _session_id: u64) -> Result<()> {
        bail!(
            "{} stage engine does not support checkpoint restore",
            self.info().engine
        )
    }

    fn trim_session(&self, _session_id: u64, _token_count: u64) -> Result<()> {
        bail!(
            "{} stage engine does not support cache trim",
            self.info().engine
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_round_trips_values() {
        let activation = StageActivation::from_values(2, 2, &[1.0, -2.0, 3.5, 4.0]).unwrap();
        assert_eq!(activation.values(), vec![1.0, -2.0, 3.5, 4.0]);
    }

    #[test]
    fn activation_rejects_wrong_size() {
        assert!(StageActivation::new(2, 2, vec![0; 4]).is_err());
    }
}
