//! A task-owned demand hint, separate from configured/human requests.
use super::Node;
use std::sync::{Arc, Mutex};

/// Dropping an automatic producer withdraws only its own hint, synchronously.
/// The normal heartbeat advertises withdrawal even when the task was aborted.
pub(crate) struct AutomaticModelRequest {
    slot: Arc<Mutex<Option<Arc<String>>>>,
    value: Arc<String>,
}

impl Drop for AutomaticModelRequest {
    fn drop(&mut self) {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        if slot
            .as_ref()
            .is_some_and(|value| Arc::ptr_eq(value, &self.value))
        {
            *slot = None;
        }
    }
}

impl Node {
    pub(crate) fn request_automatic_model(&self, canonical_ref: String) -> AutomaticModelRequest {
        let value = Arc::new(canonical_ref);
        *self
            .automatic_model_request
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(value.clone());
        AutomaticModelRequest {
            slot: self.automatic_model_request.clone(),
            value,
        }
    }

    pub(super) fn append_automatic_model_request(&self, models: &mut Vec<String>) {
        let slot = self
            .automatic_model_request
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(value) = slot.as_ref()
            && !models.contains(value.as_ref())
        {
            models.push(value.as_ref().clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn aborting_producer_withdraws_only_its_hint() {
        let node = Node::new_for_tests(super::super::NodeRole::Worker)
            .await
            .unwrap();
        let cloned = node.clone();
        let (ready, rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let _request = cloned.request_automatic_model("automatic".into());
            ready.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        rx.await.unwrap();
        node.set_requested_models(vec!["human".into()]).await;
        let human = node.requested_models.lock().await.clone();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(node.requested_models().await, human);
        let announcement =
            node.build_local_announcement(node.snapshot_local_announcement_data().await);
        assert_eq!(announcement.requested_models, human);
    }

    #[tokio::test]
    async fn withdrawal_preserves_later_human_request_and_newer_automatic_owner() {
        let node = Node::new_for_tests(super::super::NodeRole::Worker)
            .await
            .unwrap();
        let old = node.request_automatic_model("model".into());
        assert_eq!(node.requested_models().await, ["model"]);
        let new = node.request_automatic_model("new".into());
        drop(old);
        assert_eq!(node.requested_models().await, ["new"]);
        node.set_requested_models(vec!["new".into()]).await;
        let human = node.requested_models.lock().await.clone();
        drop(new);
        assert_eq!(node.requested_models().await, human);
    }
}
