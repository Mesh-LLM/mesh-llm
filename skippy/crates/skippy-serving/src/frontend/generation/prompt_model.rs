use super::StageOpenAiBackend;
use crate::runtime_state::RuntimeState;
use crate::runtime_state::panic_recovery::lock_runtime;
use skippy_runtime::StageModelReader;
use std::sync::{Mutex, OnceLock};

struct PromptModel {
    reader: StageModelReader,
    media_marker: String,
}

/// Immutable vocabulary and prompt metadata belong to the loaded model, not
/// to its mutable inference context. Backend clones share this snapshot.
#[derive(Default)]
pub(in crate::frontend) struct CachedPromptModel(OnceLock<PromptModel>);

impl CachedPromptModel {
    fn get(&self, runtime: &Mutex<RuntimeState>) -> &PromptModel {
        self.0.get_or_init(|| {
            let runtime = lock_runtime(runtime);
            PromptModel {
                reader: runtime.model.reader(),
                media_marker: runtime.media_marker(),
            }
        })
    }

    pub(super) fn initialize(&self, runtime: &Mutex<RuntimeState>) {
        self.get(runtime);
    }
}

impl StageOpenAiBackend {
    pub(in crate::frontend) fn prompt_model_reader(&self) -> StageModelReader {
        self.prompt_model.get(&self.runtime).reader.clone()
    }

    pub(in crate::frontend) fn prompt_media_marker(&self) -> &str {
        &self.prompt_model.get(&self.runtime).media_marker
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    #[test]
    fn initialized_prompt_access_does_not_wait_for_inference_lock() {
        skippy_runtime::test_support::require_native_runtime();
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(2)));
        let access = Arc::new(CachedPromptModel::default());
        access.initialize(&runtime);
        let expected_marker = access.get(&runtime).media_marker.clone();
        let guard = runtime.lock().unwrap();
        let (sent, received) = mpsc::channel();
        let worker = {
            let runtime = runtime.clone();
            let access = access.clone();
            std::thread::spawn(move || {
                let model = access.get(&runtime);
                let _reader = model.reader.clone();
                sent.send(model.media_marker.clone()).unwrap();
            })
        };
        let observed = received.recv_timeout(Duration::from_secs(2));
        drop(guard);
        worker.join().unwrap();
        assert_eq!(observed.unwrap(), expected_marker);
    }
}
