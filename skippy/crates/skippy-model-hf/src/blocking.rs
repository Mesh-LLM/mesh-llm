//! Isolation for the blocking HF client when called from async runtimes.
use anyhow::Result;

pub fn run_hf_sync<T, F>(operation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        std::thread::spawn(operation).join().map_err(|panic| {
            if let Some(message) = panic.downcast_ref::<&str>() {
                anyhow::anyhow!("Hugging Face sync task panicked: {message}")
            } else if let Some(message) = panic.downcast_ref::<String>() {
                anyhow::anyhow!("Hugging Face sync task panicked: {message}")
            } else {
                anyhow::anyhow!("Hugging Face sync task panicked")
            }
        })?
    } else {
        operation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn sync_operation_leaves_runtime_context_and_preserves_panic_detail() {
        run_hf_sync(|| {
            assert!(tokio::runtime::Handle::try_current().is_err());
            Ok(())
        })
        .unwrap();
        let error = run_hf_sync::<(), _>(|| panic!("transfer panic")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Hugging Face sync task panicked: transfer panic"
        );
    }
}
