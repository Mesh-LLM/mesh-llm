use crate::errors::FfiError;
use crate::handles::ConsoleHandle;
use crate::runtime_blocking::block_on;

#[uniffi::export]
impl ConsoleHandle {
    pub fn url(&self) -> String {
        self.url.clone()
    }

    pub fn stop(&self) -> Result<(), FfiError> {
        let handle = self
            .inner
            .lock()
            .map_err(|error| FfiError::ConsoleFailed(error.to_string()))?
            .take();
        if let Some(handle) = handle {
            block_on(handle.stop());
        }
        Ok(())
    }
}
