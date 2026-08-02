use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use openai_frontend::{OpenAiError, OpenAiResult};

use crate::frontend::{StageOpenAiBackend, openai_backend_error};
use crate::runtime_state::RuntimeState;

const TOKEN_ID_DIGEST_DOMAIN: &[u8] = b"skippy-generation-token-ids-v1\0";

/// Why a successful local generation stopped.
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationStateDigest {
    /// Number of bytes in the exported runtime state.
    pub byte_length: u64,
    /// BLAKE3 digest of the exported runtime-state bytes.
    pub blake3_digest: [u8; 32],
}

/// Target-authoritative result captured immediately before local session teardown.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationReceipt {
    /// OpenAI request identity.
    pub request_id: u64,
    /// OpenAI session identity.
    pub session_id: u64,
    /// Number of target-tokenized prompt-text IDs supplied to local generation.
    ///
    /// For multimodal requests, media embeddings have no token IDs and are not included.
    pub prompt_token_count: usize,
    /// Stable digest of the target-tokenized prompt-text IDs.
    pub prompt_token_digest: [u8; 32],
    /// Exact target-tokenized prompt-text IDs supplied to local generation.
    pub prompt_token_ids: Box<[i32]>,
    /// Target-authoritative generated token IDs in callback order.
    pub generated_token_ids: Box<[i32]>,
    /// Canonical runtime position captured before session teardown.
    pub final_session_position: u64,
    /// Why generation stopped successfully.
    pub termination: GenerationTermination,
    /// Time spent in model generation, excluding receipt delivery.
    pub model_generation_elapsed_us: u64,
    /// Backend request-start to first generated-token availability.
    pub request_to_first_token_us: Option<u64>,
    /// Backend request-start to each generated-token availability, in token order.
    pub request_to_token_emission_us: Box<[u64]>,
    /// Optional digest of the target runtime's full exported state.
    pub full_state: Option<GenerationStateDigest>,
}

/// Target-authoritative beginning of one local generation lifecycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationStart {
    pub request_id: u64,
    pub session_id: u64,
    pub prompt_token_ids: Box<[i32]>,
}

/// Target-authoritative termination of a generation that produced no final
/// receipt. The proposal/session adapter uses this boundary to close durable
/// request state instead of leaving later requests blocked behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationAbort {
    pub request_id: u64,
    pub session_id: u64,
}

/// A canonical target-token delta committed during an active generation.
///
/// This is a model-neutral lifecycle event. It deliberately carries no
/// consumer-specific state or evidence concept: consumers receive only the
/// target-owned request identity, canonical position, and token delta.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationCommit {
    pub request_id: u64,
    pub session_id: u64,
    /// Total generated canonical tokens after applying `token_ids`.
    pub generated_token_count: usize,
    pub token_ids: Box<[i32]>,
}

/// Receives the complete local-generation lifecycle before runtime teardown.
///
/// A successful `begin` is followed exactly once by either `record` or
/// `abort`. Implementations can therefore advance durable request state
/// without relying on an agent-specific hook.
pub trait GenerationReceiptSink: Send + Sync {
    fn begin(&self, start: &GenerationStart) -> Result<()>;

    /// Delivers canonical target tokens before the next proposal lookup.
    fn committed(&self, commit: &GenerationCommit) -> Result<()>;

    fn abort(&self, abort: &GenerationAbort) -> Result<()>;

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
    token_emission_elapsed: Vec<Duration>,
    max_tokens: usize,
    termination: Option<GenerationTermination>,
    model_generation_elapsed: Option<Duration>,
}

pub(crate) struct LocalGenerationReceiptDelivery<'a> {
    pub(crate) config: &'a GenerationReceiptConfig,
    pub(crate) session_label: &'a str,
    pub(crate) request_id: u64,
    pub(crate) session_id: u64,
    pub(crate) prompt_token_ids: &'a [i32],
    pub(crate) observation: GenerationReceiptObservation,
}

trait GenerationReceiptRuntime {
    fn canonical_session_position(&self, session_label: &str) -> Result<u64>;
    fn export_full_state(&mut self, session_label: &str) -> Result<Vec<u8>>;
}

impl GenerationReceiptRuntime for RuntimeState {
    fn canonical_session_position(&self, session_label: &str) -> Result<u64> {
        self.canonical_session_position(session_label)
    }

    fn export_full_state(&mut self, session_label: &str) -> Result<Vec<u8>> {
        self.export_full_state(session_label)
    }
}

impl GenerationReceiptObservation {
    pub(crate) fn new(max_tokens: usize) -> Self {
        Self {
            generated_token_ids: Vec::with_capacity(max_tokens.min(4_096)),
            token_emission_elapsed: Vec::with_capacity(max_tokens.min(4_096)),
            max_tokens,
            termination: None,
            model_generation_elapsed: None,
        }
    }

    pub(crate) fn record_token(
        &mut self,
        token_id: i32,
        request_elapsed: Duration,
    ) -> OpenAiResult<()> {
        if self.generated_token_ids.len() >= self.max_tokens {
            return Err(OpenAiError::backend(
                "generation receipt observed more tokens than the request budget",
            ));
        }
        if self
            .token_emission_elapsed
            .last()
            .is_some_and(|prior| request_elapsed < *prior)
        {
            return Err(OpenAiError::backend(
                "generation receipt observed non-monotonic token timing",
            ));
        }
        self.generated_token_ids.push(token_id);
        self.token_emission_elapsed.push(request_elapsed);
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
        let request_to_token_emission_us = self
            .token_emission_elapsed
            .into_iter()
            .map(duration_us)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(FinishedGenerationObservation {
            generated_token_ids: self.generated_token_ids.into_boxed_slice(),
            termination: self.termination.unwrap_or(GenerationTermination::MaxTokens),
            model_generation_elapsed_us: duration_us(model_generation_elapsed),
            request_to_first_token_us: request_to_token_emission_us.first().copied(),
            request_to_token_emission_us,
        })
    }
}

struct FinishedGenerationObservation {
    generated_token_ids: Box<[i32]>,
    termination: GenerationTermination,
    model_generation_elapsed_us: u64,
    request_to_first_token_us: Option<u64>,
    request_to_token_emission_us: Box<[u64]>,
}

impl StageOpenAiBackend {
    pub(crate) fn deliver_local_generation_receipt(
        &self,
        delivery: LocalGenerationReceiptDelivery<'_>,
    ) -> OpenAiResult<()> {
        let config = delivery.config;
        let receipt = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            build_generation_receipt(&mut *runtime, delivery)?
        };
        record_generation_receipt(config, &receipt)
    }
}

fn build_generation_receipt(
    runtime: &mut dyn GenerationReceiptRuntime,
    delivery: LocalGenerationReceiptDelivery<'_>,
) -> OpenAiResult<GenerationReceipt> {
    let observation = delivery.observation.finish()?;
    let final_session_position = runtime
        .canonical_session_position(delivery.session_label)
        .map_err(openai_backend_error)?;
    let full_state = if delivery.config.exports_full_state() {
        let bytes = runtime
            .export_full_state(delivery.session_label)
            .map_err(openai_backend_error)?;
        Some(state_digest(&bytes)?)
    } else {
        None
    };
    Ok(GenerationReceipt {
        request_id: delivery.request_id,
        session_id: delivery.session_id,
        prompt_token_count: delivery.prompt_token_ids.len(),
        prompt_token_digest: generation_token_id_digest(delivery.prompt_token_ids),
        prompt_token_ids: delivery.prompt_token_ids.to_vec().into_boxed_slice(),
        generated_token_ids: observation.generated_token_ids,
        final_session_position,
        termination: observation.termination,
        model_generation_elapsed_us: observation.model_generation_elapsed_us,
        request_to_first_token_us: observation.request_to_first_token_us,
        request_to_token_emission_us: observation.request_to_token_emission_us,
        full_state,
    })
}

fn record_generation_receipt(
    config: &GenerationReceiptConfig,
    receipt: &GenerationReceipt,
) -> OpenAiResult<()> {
    config.sink().record(receipt).map_err(openai_backend_error)
}

pub(crate) fn complete_generation_before_cleanup<T>(
    generation_result: OpenAiResult<T>,
    deliver_receipt: impl FnOnce() -> OpenAiResult<()>,
    cleanup: impl FnOnce(),
) -> OpenAiResult<T> {
    let receipt_result = deliver_receipt();
    cleanup();
    match generation_result {
        Ok(output) => {
            receipt_result?;
            Ok(output)
        }
        Err(primary) => {
            if receipt_result.is_err() {
                eprintln!(
                    "generation lifecycle abort failed; preserving the primary generation error"
                );
            }
            Err(primary)
        }
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
    use std::sync::Mutex;

    use super::*;

    struct FakeRuntime {
        position: Result<u64, &'static str>,
        full_state: Result<Vec<u8>, &'static str>,
    }

    impl GenerationReceiptRuntime for FakeRuntime {
        fn canonical_session_position(&self, _session_label: &str) -> Result<u64> {
            self.position.map_err(anyhow::Error::msg)
        }

        fn export_full_state(&mut self, _session_label: &str) -> Result<Vec<u8>> {
            self.full_state.clone().map_err(anyhow::Error::msg)
        }
    }

    #[derive(Default)]
    struct RecordingSink {
        receipts: Mutex<Vec<GenerationReceipt>>,
        error: Option<&'static str>,
    }

    impl GenerationReceiptSink for RecordingSink {
        fn begin(&self, _start: &GenerationStart) -> Result<()> {
            Ok(())
        }

        fn committed(&self, _commit: &GenerationCommit) -> Result<()> {
            Ok(())
        }

        fn abort(&self, _abort: &GenerationAbort) -> Result<()> {
            Ok(())
        }

        fn record(&self, receipt: &GenerationReceipt) -> Result<()> {
            self.receipts.lock().unwrap().push(receipt.clone());
            self.error
                .map_or(Ok(()), |error| Err(anyhow::anyhow!(error)))
        }
    }

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
    fn receipt_prompt_evidence_preserves_exact_signed_token_ids() {
        let prompt = [-1, 0, 7, i32::MAX];
        let receipt = GenerationReceipt {
            request_id: 1,
            session_id: 2,
            prompt_token_count: prompt.len(),
            prompt_token_digest: generation_token_id_digest(&prompt),
            prompt_token_ids: prompt.to_vec().into_boxed_slice(),
            generated_token_ids: vec![9].into_boxed_slice(),
            final_session_position: 4,
            termination: GenerationTermination::MaxTokens,
            model_generation_elapsed_us: 3,
            request_to_first_token_us: Some(1),
            request_to_token_emission_us: vec![1].into_boxed_slice(),
            full_state: None,
        };
        assert_eq!(receipt.prompt_token_ids.as_ref(), prompt);
        assert_eq!(receipt.prompt_token_count, receipt.prompt_token_ids.len());
        assert_eq!(
            receipt.prompt_token_digest,
            generation_token_id_digest(&receipt.prompt_token_ids)
        );
        assert_eq!(
            receipt.generated_token_ids.len(),
            receipt.request_to_token_emission_us.len()
        );
        assert_eq!(
            receipt.request_to_first_token_us,
            receipt.request_to_token_emission_us.first().copied()
        );
    }

    #[test]
    fn observation_keeps_the_callback_stopping_token() {
        let mut observation = GenerationReceiptObservation::new(3);
        observation
            .record_token(7, Duration::from_micros(11))
            .unwrap();
        observation
            .record_token(8, Duration::from_micros(17))
            .unwrap();
        observation.mark_callback_stop();
        observation.set_model_generation_elapsed(Duration::from_micros(42));
        let finished = observation.finish().unwrap();
        assert_eq!(&*finished.generated_token_ids, &[7, 8]);
        assert_eq!(finished.termination, GenerationTermination::CallbackStop);
        assert_eq!(finished.model_generation_elapsed_us, 42);
        assert_eq!(finished.request_to_first_token_us, Some(11));
        assert_eq!(&*finished.request_to_token_emission_us, &[11, 17]);
    }

    #[test]
    fn observation_is_bounded_by_the_resolved_token_budget() {
        let mut observation = GenerationReceiptObservation::new(1);
        observation.record_token(7, Duration::ZERO).unwrap();
        assert!(
            observation
                .record_token(8, Duration::from_micros(1))
                .unwrap_err()
                .to_string()
                .contains("more tokens than the request budget")
        );
    }

    #[test]
    fn observation_rejects_non_monotonic_token_timing() {
        let mut observation = GenerationReceiptObservation::new(2);
        observation
            .record_token(7, Duration::from_micros(2))
            .unwrap();
        assert!(
            observation
                .record_token(8, Duration::from_micros(1))
                .unwrap_err()
                .to_string()
                .contains("non-monotonic token timing")
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
        let finished = max_tokens.finish().unwrap();
        assert_eq!(finished.termination, GenerationTermination::MaxTokens);
        assert_eq!(finished.request_to_first_token_us, None);
        assert!(finished.request_to_token_emission_us.is_empty());
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

    #[test]
    fn model_free_delivery_validates_position_exports_state_and_propagates_sink_errors() {
        let sink = Arc::new(RecordingSink::default());
        let config = GenerationReceiptConfig::new(sink.clone()).with_full_state_digest(true);
        let mut observation = GenerationReceiptObservation::new(1);
        observation
            .record_token(9, Duration::from_micros(5))
            .unwrap();
        observation.set_model_generation_elapsed(Duration::from_micros(17));
        let mut runtime = FakeRuntime {
            position: Ok(4),
            full_state: Ok(b"state".to_vec()),
        };
        let receipt = build_generation_receipt(
            &mut runtime,
            LocalGenerationReceiptDelivery {
                config: &config,
                session_label: "session",
                request_id: 2,
                session_id: 3,
                prompt_token_ids: &[4, 5, 6],
                observation,
            },
        )
        .unwrap();
        record_generation_receipt(&config, &receipt).unwrap();
        let receipts = sink.receipts.lock().unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].final_session_position, 4);
        assert_eq!(receipts[0].generated_token_ids.as_ref(), &[9]);
        assert_eq!(
            receipts[0].full_state.as_ref().unwrap().blake3_digest,
            *blake3::hash(b"state").as_bytes()
        );
        drop(receipts);

        let failing_position = build_generation_receipt(
            &mut FakeRuntime {
                position: Err("position mismatch"),
                full_state: Ok(Vec::new()),
            },
            LocalGenerationReceiptDelivery {
                config: &config,
                session_label: "session",
                request_id: 2,
                session_id: 3,
                prompt_token_ids: &[],
                observation: {
                    let mut observation = GenerationReceiptObservation::new(0);
                    observation.set_model_generation_elapsed(Duration::ZERO);
                    observation
                },
            },
        )
        .unwrap_err();
        assert!(failing_position.to_string().contains("position mismatch"));

        let failing_sink = Arc::new(RecordingSink {
            receipts: Mutex::new(Vec::new()),
            error: Some("sink failed"),
        });
        let failing_config = GenerationReceiptConfig::new(failing_sink);
        let mut observation = GenerationReceiptObservation::new(0);
        observation.set_model_generation_elapsed(Duration::ZERO);
        let receipt = build_generation_receipt(
            &mut runtime,
            LocalGenerationReceiptDelivery {
                config: &failing_config,
                session_label: "session",
                request_id: 2,
                session_id: 3,
                prompt_token_ids: &[],
                observation,
            },
        )
        .unwrap();
        let sink_error = record_generation_receipt(&failing_config, &receipt).unwrap_err();
        assert!(sink_error.to_string().contains("sink failed"));
    }

    #[test]
    fn receipt_delivery_precedes_cleanup_and_cleanup_survives_sink_failure() {
        let events = Mutex::new(Vec::new());
        let error = complete_generation_before_cleanup(
            Ok(()),
            || {
                events.lock().unwrap().push("receipt");
                Err(OpenAiError::backend("sink failed"))
            },
            || events.lock().unwrap().push("cleanup"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("sink failed"));
        assert_eq!(*events.lock().unwrap(), ["receipt", "cleanup"]);

        let events = Mutex::new(Vec::new());
        let generation_error = complete_generation_before_cleanup::<()>(
            Err(OpenAiError::backend("generation failed")),
            || {
                events.lock().unwrap().push("abort");
                Ok(())
            },
            || events.lock().unwrap().push("cleanup"),
        )
        .unwrap_err();
        assert!(generation_error.to_string().contains("generation failed"));
        assert_eq!(*events.lock().unwrap(), ["abort", "cleanup"]);
    }

    #[test]
    fn receipt_lifecycle_begins_before_generation_and_closes_before_cleanup() {
        let source = include_str!("local_generation.rs");
        let begin = source
            .find(".begin(&GenerationStart")
            .expect("receipt lifecycle must begin before model execution");
        let generation = source
            .find("let result = (||")
            .expect("local generation body should remain explicit");
        let record = source
            .find("observation.record_token(token_id, request.ids.request_started_at.elapsed())?")
            .expect("receipt should record each generated token");
        let callback = source
            .find("let control = on_token(token_id)?")
            .expect("generation callback should still control stopping");
        let finalization = source
            .find("self.finalize_generation_receipt(")
            .expect("receipt lifecycle should close on every generation outcome");
        let cleanup = source
            .find("self.cleanup_local_generation_session")
            .expect("session should still be dropped");
        assert!(begin < generation);
        assert!(record < callback);
        assert!(callback < finalization);
        assert!(finalization < cleanup);
    }
}
