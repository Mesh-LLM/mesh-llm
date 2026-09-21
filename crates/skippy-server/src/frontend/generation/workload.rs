use super::StageOpenAiBackend;
use crate::frontend::util::openai_backend_error;
use openai_frontend::{OpenAiError, OpenAiResult};
use skippy_runtime::ModelWorkload;
use std::sync::OnceLock;

/// A backend owns one immutable loaded model; clones share its probe result.
/// Preserve probe errors rather than assuming an unsupported model is causal.
#[derive(Default)]
pub(in crate::frontend) struct CachedModelWorkload(OnceLock<OpenAiResult<ModelWorkload>>);

impl CachedModelWorkload {
    /// Serialize the first probe and reuse its success or failure for this loaded model.
    fn get_or_probe(
        &self,
        probe: impl FnOnce() -> OpenAiResult<ModelWorkload>,
    ) -> OpenAiResult<ModelWorkload> {
        self.0.get_or_init(probe).clone()
    }
}

impl StageOpenAiBackend {
    /// Classify text generation without locking the runtime after its first request.
    pub(in crate::frontend) fn model_workload(&self) -> OpenAiResult<ModelWorkload> {
        self.workload.get_or_probe(|| {
            self.runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?
                .workload_info()
                .map(|info| info.kind)
                .map_err(openai_backend_error)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    /// Cloned backends share one native probe even when requests arrive concurrently.
    fn concurrent_requests_share_one_loaded_model_probe() {
        let cache = Arc::new(CachedModelWorkload::default());
        let probes = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cache = cache.clone();
                let probes = &probes;
                scope.spawn(move || {
                    let workload = cache.get_or_probe(|| {
                        probes.fetch_add(1, Ordering::SeqCst);
                        Ok(ModelWorkload::EncoderDecoder)
                    });
                    assert_eq!(workload.unwrap(), ModelWorkload::EncoderDecoder);
                });
            }
        });
        assert_eq!(probes.load(Ordering::SeqCst), 1);
        assert_eq!(
            cache
                .get_or_probe(|| panic!("cached requests must not lock or probe the runtime"))
                .unwrap(),
            ModelWorkload::EncoderDecoder,
        );
    }

    #[test]
    /// A failed native descriptor must never become an implicit causal-generation default.
    fn failed_probe_remains_fail_closed_and_preserves_the_error() {
        let cache = CachedModelWorkload::default();
        let first = cache
            .get_or_probe(|| Err(OpenAiError::backend("native workload probe failed")))
            .unwrap_err();
        let cached = cache
            .get_or_probe(|| panic!("failed immutable-model probes must not be retried"))
            .unwrap_err();
        assert_eq!(first.status(), cached.status());
        assert_eq!(first.body().error.message, cached.body().error.message);
    }
}
