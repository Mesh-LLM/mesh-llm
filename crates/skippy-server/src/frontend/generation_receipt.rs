use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use openai_frontend::{OpenAiError, OpenAiResult};

use crate::frontend::{StageOpenAiBackend, openai_backend_error};

const TOKEN_ID_DIGEST_DOMAIN: &[u8] = b"skippy-generation-token-ids-v1\0";

/// Why a successful local generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationTermination {
    /// The token callback requested a stop, including for an end-of-generation token.
    CallbackStop,
    /// The request consumed its resolved completion-token budget.
    MaxTokens,
    /// The local generation loop observed request cancellation.
    Cancelled,
}

/// Optional digest of the target runtime's full exported state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationStateDigest {
    /// Number of bytes in the exported runtime state.
    pub byte_length: u64,
    /// BLAKE3 digest of the exported runtime-state bytes.
    pub blake3_digest: [u8; 32],
}

/// Target-authoritative result captured immediately before local session teardown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationReceipt {
    /// OpenAI request identity.
    pub request_id: u64,
    /// OpenAI session identity.
    pub session_id: u64,
    /// Number of prompt token IDs supplied to local generation.
    pub prompt_token_count: usize,
    /// Stable digest of the prompt token IDs.
    pub prompt_token_digest: [u8; 32],
    /// Target-authoritative generated token IDs in callback order.
    pub generated_token_ids: Box<[i32]>,
    /// Canonical runtime position captured before session teardown.
    pub final_session_position: u64,
    /// Why generation stopped successfully.
    pub termination: GenerationTermination,
    /// Time spent in model generation, excluding receipt delivery.
    pub model_generation_elapsed_us: u64,
    /// Optional digest of the target runtime's full exported state.
    pub full_state: Option<GenerationStateDigest>,
}

/// Receives one successful local-generation result before its runtime session is dropped.
pub trait GenerationReceiptSink: Send + Sync {
    /// Records one successful local-generation receipt.
    fn record(&self, receipt: &GenerationReceipt) -> Result<()>;
}

/// Optional local-generation observation.
///
/// The default configuration records exact token IDs and positions without exporting
/// full model state. Full-state export is intended for exactness checks only: it is
/// deliberately opt-in and must remain disabled for timed measurements.
#[derive(Clone)]
pub struct GenerationReceiptConfig {
    sink: Arc<dyn GenerationReceiptSink>,
    export_full_state: bool,
}

impl GenerationReceiptConfig {
    /// Creates receipt delivery with full-state export disabled.
    pub fn new(sink: Arc<dyn GenerationReceiptSink>) -> Self {
        Self {
            sink,
            export_full_state: false,
        }
    }

    #[must_use]
    /// Enables or disables the optional full-state digest.
    pub fn with_full_state_digest(mut self, enabled: bool) -> Self {
        self.export_full_state = enabled;
        self
    }

    /// Reports whether receipt delivery exports and hashes full runtime state.
    pub fn exports_full_state(&self) -> bool {
        self.export_full_state
    }

    pub(crate) fn sink(&self) -> &dyn GenerationReceiptSink {
        self.sink.as_ref()
    }
}

/// Stable, platform-independent digest of signed token IDs.
///
/// The encoding is a domain tag, a little-endian `u64` token count, and each
/// signed token ID in little-endian `i32` form.
pub fn generation_token_id_digest(token_ids: &[i32]) -> [u8; 32] {
    let token_count =
        u64::try_from(token_ids.len()).expect("supported targets have at most u64::MAX tokens");
    let mut hasher = blake3::Hasher::new();
    hasher.update(TOKEN_ID_DIGEST_DOMAIN);
    hasher.update(&token_count.to_le_bytes());
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

pub(crate) struct GenerationReceiptObservation {
    generated_token_ids: Vec<i32>,
    max_tokens: usize,
    termination: Option<GenerationTermination>,
    model_generation_elapsed: Option<Duration>,
}

impl GenerationReceiptObservation {
    pub(crate) fn new(max_tokens: usize) -> Self {
        Self {
            generated_token_ids: Vec::with_capacity(max_tokens.min(4_096)),
            max_tokens,
            termination: None,
            model_generation_elapsed: None,
        }
    }

    pub(crate) fn record_token(&mut self, token_id: i32) -> OpenAiResult<()> {
        if self.generated_token_ids.len() >= self.max_tokens {
            return Err(OpenAiError::backend(
                "generation receipt observed more tokens than the request budget",
            ));
        }
        self.generated_token_ids.push(token_id);
        Ok(())
    }

    pub(crate) fn mark_callback_stop(&mut self) {
        self.termination = Some(GenerationTermination::CallbackStop);
    }

    pub(crate) fn mark_cancelled(&mut self) {
        if self.termination.is_none() {
            self.termination = Some(GenerationTermination::Cancelled);
        }
    }

    pub(crate) fn set_model_generation_elapsed(&mut self, elapsed: Duration) {
        self.model_generation_elapsed = Some(elapsed);
    }

    fn finish(self) -> OpenAiResult<FinishedGenerationObservation> {
        let model_generation_elapsed = self.model_generation_elapsed.ok_or_else(|| {
            OpenAiError::backend("generation receipt is missing model-generation timing")
        })?;
        Ok(FinishedGenerationObservation {
            generated_token_ids: self.generated_token_ids.into_boxed_slice(),
            termination: self.termination.unwrap_or(GenerationTermination::MaxTokens),
            model_generation_elapsed_us: duration_us(model_generation_elapsed),
        })
    }
}

struct FinishedGenerationObservation {
    generated_token_ids: Box<[i32]>,
    termination: GenerationTermination,
    model_generation_elapsed_us: u64,
}

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deliver_local_generation_receipt(
        &self,
        config: &GenerationReceiptConfig,
        session_label: &str,
        request_id: u64,
        session_id: u64,
        prompt_token_ids: &[i32],
        observation: GenerationReceiptObservation,
    ) -> OpenAiResult<()> {
        let observation = observation.finish()?;
        let (final_session_position, full_state) = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let final_session_position = runtime
                .canonical_session_position(session_label)
                .map_err(openai_backend_error)?;
            let full_state = if config.exports_full_state() {
                let bytes = runtime
                    .export_full_state(session_label)
                    .map_err(openai_backend_error)?;
                Some(state_digest(&bytes)?)
            } else {
                None
            };
            (final_session_position, full_state)
        };
        let receipt = GenerationReceipt {
            request_id,
            session_id,
            prompt_token_count: prompt_token_ids.len(),
            prompt_token_digest: generation_token_id_digest(prompt_token_ids),
            generated_token_ids: observation.generated_token_ids,
            final_session_position,
            termination: observation.termination,
            model_generation_elapsed_us: observation.model_generation_elapsed_us,
            full_state,
        };
        config.sink().record(&receipt).map_err(openai_backend_error)
    }
}

fn state_digest(bytes: &[u8]) -> OpenAiResult<GenerationStateDigest> {
    let byte_length = u64::try_from(bytes.len())
        .map_err(|_| OpenAiError::backend("full-state byte length exceeds u64"))?;
    Ok(GenerationStateDigest {
        byte_length,
        blake3_digest: *blake3::hash(bytes).as_bytes(),
    })
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_digest_is_stable_and_order_sensitive() {
        let digest = generation_token_id_digest(&[-1, 0, 1, i32::MAX]);
        assert_eq!(
            digest,
            [
                0x1a, 0xe4, 0xc4, 0x37, 0x7c, 0xce, 0x52, 0xaa, 0x76, 0x66, 0x8c, 0x07, 0xd0, 0x16,
                0xaa, 0x7b, 0x19, 0xfe, 0xd5, 0x8c, 0xbd, 0x35, 0x89, 0x06, 0xe6, 0x10, 0x8f, 0x03,
                0xf7, 0xbf, 0x33, 0x3a,
            ]
        );
        assert_ne!(digest, generation_token_id_digest(&[0, -1, 1, i32::MAX]));
    }

    #[test]
    fn observation_keeps_the_callback_stopping_token() {
        let mut observation = GenerationReceiptObservation::new(3);
        observation.record_token(7).unwrap();
        observation.record_token(8).unwrap();
        observation.mark_callback_stop();
        observation.set_model_generation_elapsed(Duration::from_micros(42));
        let finished = observation.finish().unwrap();
        assert_eq!(&*finished.generated_token_ids, &[7, 8]);
        assert_eq!(finished.termination, GenerationTermination::CallbackStop);
        assert_eq!(finished.model_generation_elapsed_us, 42);
    }

    #[test]
    fn observation_is_bounded_by_the_resolved_token_budget() {
        let mut observation = GenerationReceiptObservation::new(1);
        observation.record_token(7).unwrap();
        assert!(
            observation
                .record_token(8)
                .unwrap_err()
                .to_string()
                .contains("more tokens than the request budget")
        );
    }

    #[test]
    fn cancellation_precedes_default_max_token_termination() {
        let mut observation = GenerationReceiptObservation::new(1);
        observation.mark_cancelled();
        observation.set_model_generation_elapsed(Duration::ZERO);
        assert_eq!(
            observation.finish().unwrap().termination,
            GenerationTermination::Cancelled
        );

        let mut max_tokens = GenerationReceiptObservation::new(0);
        max_tokens.set_model_generation_elapsed(Duration::ZERO);
        assert_eq!(
            max_tokens.finish().unwrap().termination,
            GenerationTermination::MaxTokens
        );
    }

    #[test]
    fn state_digest_binds_length_and_bytes() {
        let digest = state_digest(b"state").unwrap();
        assert_eq!(digest.byte_length, 5);
        assert_eq!(digest.blake3_digest, *blake3::hash(b"state").as_bytes());
        assert_ne!(
            digest.blake3_digest,
            state_digest(b"state!").unwrap().blake3_digest
        );
    }
}
