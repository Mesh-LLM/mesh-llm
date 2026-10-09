use super::*;
use async_trait::async_trait;
use skippy_inference_api::InferenceHookPolicy;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct PreparedPolicy {
    seen: Mutex<Vec<String>>,
    terminals: Mutex<Vec<String>>,
}

#[async_trait]
impl InferenceHookPolicy for PreparedPolicy {
    fn observes_dispatched_request(&self) -> bool {
        true
    }
    fn requires_exchange_lifecycle(&self) -> bool {
        true
    }
    async fn admit_effective_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) -> InferenceResult<()> {
        self.seen.lock().unwrap().push(route.exchange_id.clone());
        if request.model == "prepared-denied" {
            return Err(InferenceError::invalid_request("prepared request denied"));
        }
        Ok(())
    }
    async fn on_chat_completion_terminal(
        &self,
        request: &ChatCompletionRequest,
        _exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        assert!(matches!(outcome, ChatCompletionOutcome::Denied { .. }));
        self.terminals.lock().unwrap().push(request.model.clone());
    }
}

#[tokio::test]
async fn prepared_admission_denies_after_preparation_before_generation() {
    let policy = Arc::new(PreparedPolicy::default());
    let backend = super::tests::hooks_test_backend(Some(policy.clone()));
    let generated = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let marker = generated.clone();
    let result = backend
        .chat_completion_with_prepared_hooks_for_exchange(
            ChatCompletionRequest::default(),
            Some("exchange".into()),
            move |mut request, admission| async move {
                // This models defaults/template preparation inside the dispatch closure.
                request.model = "prepared-denied".into();
                admission.admit(&request).await?;
                marker.store(true, std::sync::atomic::Ordering::SeqCst);
                unreachable!("denied effective request must never reach generation")
            },
        )
        .await;
    assert!(result.is_err());
    assert!(!generated.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(*policy.seen.lock().unwrap(), ["exchange"]);
    assert_eq!(*policy.terminals.lock().unwrap(), ["prepared-denied"]);
}
