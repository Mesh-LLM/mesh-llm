//! Nonblocking context execution activity for completion-aware serving.
use super::StageModel;

impl StageModel {
    /// Nonblocking native activity marker. Bit zero means device work may be pending.
    /// Read only while holding the same lock used for session/context mutations.
    pub fn execution_epoch(&self) -> Option<u64> {
        self.has_native_model().then(|| {
            // SAFETY: this model owns the native handle; the caller serializes context access.
            unsafe { skippy_ffi::skippy_model_execution_epoch(self.inner.raw) }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_model_has_no_native_completion_marker() {
        assert_eq!(StageModel::new_dummy().execution_epoch(), None);
    }
}
