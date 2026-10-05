//! A detached terminal owner survives cancellation of the ingress waiter.
pub(super) async fn complete(work: impl std::future::Future<Output = ()> + Send + 'static) {
    // Dropping this JoinHandle detaches rather than cancels its task.
    let _ = tokio::spawn(work).await;
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn cancelling_waiter_preserves_one_terminal_attempt() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel();
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_attempts = attempts.clone();
        let waiter = tokio::spawn(super::complete(async move {
            started_tx.send(()).unwrap();
            release_rx.await.unwrap();
            worker_attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            terminal_tx.send(()).unwrap();
        }));
        started_rx.await.unwrap();
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        release_tx.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), terminal_rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
