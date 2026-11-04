//! Scoped identity and lifetime of blocking inference work.

use super::{InferenceCacheHints, InferenceGenerationIds};
use skippy_inference_api::InferenceRequestContext;
use tokio::task;

/// Combine frontend request identity, trusted session metadata and execution scope.
pub(in crate::frontend) fn generation_ids(
    cache: InferenceCacheHints,
    agent_session_id: Option<&str>,
    context: &InferenceRequestContext,
) -> InferenceGenerationIds {
    let frontend_request_id = context
        .request_id()
        .map(|request_id| request_id.as_uuid().into_bytes());
    InferenceGenerationIds::new_with_trust(
        cache,
        agent_session_id,
        context.has_trusted_agent_session(),
        frontend_request_id,
    )
    .with_execution_scope(context.execution_scope())
}

/// Run blocking inference while owning its permit and full request context.
/// Abandoning the awaiting future does not release capacity while native work continues.
pub(in crate::frontend) async fn run_blocking_generation_worker<T, F, P>(
    permit: P,
    context: InferenceRequestContext,
    work: F,
) -> Result<T, task::JoinError>
where
    T: Send + 'static,
    F: FnOnce(skippy_inference_api::CancellationToken) -> T + Send + 'static,
    P: Send + 'static,
{
    task::spawn_blocking(move || {
        let _permit = permit;
        let result = work(context.cancellation_token());
        // The outer future may have been dropped while native execution was
        // unwinding. Shared-owner capacity remains held until this point.
        drop(context);
        result
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_inference_api::{InferenceExecutionLease, InferenceExecutionScope};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::Duration;

    #[derive(Debug)]
    struct ObservedLease(Arc<AtomicBool>);
    impl InferenceExecutionLease for ObservedLease {}
    impl Drop for ObservedLease {
        /// Record the instant the final test execution lease is released.
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    /// Aborting the async waiter must retain the lease until its blocking worker has exited.
    async fn abandoned_future_retains_execution_lease_until_blocking_work_exits() {
        let released = Arc::new(AtomicBool::new(false));
        let context = InferenceRequestContext::new().with_execution_scope(
            InferenceExecutionScope::new(),
            Arc::new(ObservedLease(released.clone())),
        );
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel();
        let waiting = tokio::spawn(run_blocking_generation_worker((), context, move |_| {
            started_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }));
        tokio::time::timeout(Duration::from_secs(5), started_rx)
            .await
            .unwrap()
            .unwrap();
        waiting.abort();
        assert!(waiting.await.unwrap_err().is_cancelled());
        assert!(!released.load(Ordering::Acquire));
        finish_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !released.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
