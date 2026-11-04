//! Native model discovery and provider-specific inference.

use super::EmbeddedServeHandle;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub use skippy_inference_api::common::ReasoningEffort;
pub use skippy_inference_api::{ChatCompletionRequest, ChatMessage, MessageContent};

/// One provider's exact serving ID, returned by native model discovery.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelTarget {
    pub provider_node_id: String,
    pub serving_id: String,
}

/// An advertised model, without merging capabilities across providers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub target: ModelTarget,
    pub descriptor: crate::mesh::ServedModelDescriptor,
    /// Model-specific render-probe observations in their reported order.
    /// `None` means unknown; an empty list means no effort was observed.
    pub thinking_efforts: Option<Vec<String>>,
}

/// An inference request bound to the provider whose metadata was selected.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeChatRequest {
    pub target: ModelTarget,
    pub request: ChatCompletionRequest,
}

/// Native chat deltas. Dropping the stream cancels its producer.
pub struct ChatCompletionStream {
    receiver: tokio::sync::mpsc::Receiver<Result<serde_json::Value>>,
    task: tokio::task::JoinHandle<()>,
    shutdown_task: Option<tokio::task::JoinHandle<()>>,
    shutdown: Option<tokio::sync::watch::Receiver<bool>>,
    finished: bool,
}

impl ChatCompletionStream {
    pub(crate) fn new(
        receiver: tokio::sync::mpsc::Receiver<Result<serde_json::Value>>,
        task: tokio::task::JoinHandle<()>,
    ) -> Self {
        Self {
            receiver,
            task,
            shutdown_task: None,
            shutdown: None,
            finished: false,
        }
    }

    /// Read the next OpenAI-shaped chat delta or terminal error.
    pub async fn next_chunk(&mut self) -> Option<Result<serde_json::Value>> {
        if self.finished {
            return None;
        }
        let (chunk, stopped) = if let Some(shutdown) = self.shutdown.as_mut() {
            let stopping = *shutdown.borrow_and_update();
            if stopping {
                (None, true)
            } else {
                tokio::select! {
                    biased;
                    _ = shutdown.changed() => (None, true),
                    chunk = self.receiver.recv() => (chunk, false),
                }
            }
        } else {
            (self.receiver.recv().await, false)
        };
        if (stopped || chunk.is_none())
            && let Some(watcher) = self.shutdown_task.take()
        {
            watcher.abort();
        }
        if stopped {
            self.finished = true;
            self.task.abort();
            self.receiver.close();
            return Some(Err(anyhow::anyhow!("embedded runtime is shutting down")));
        }
        self.finished = chunk.is_none();
        chunk
    }

    fn cancel_on_shutdown(mut self, mut shutdown: tokio::sync::watch::Receiver<bool>) -> Self {
        let producer = self.task.abort_handle();
        self.shutdown = Some(shutdown.clone());
        self.shutdown_task = Some(tokio::spawn(async move {
            if !*shutdown.borrow_and_update() {
                let _ = shutdown.changed().await;
            }
            producer.abort();
        }));
        self
    }
}

impl Drop for ChatCompletionStream {
    fn drop(&mut self) {
        self.task.abort();
        if let Some(watcher) = self.shutdown_task.take() {
            watcher.abort();
        }
    }
}

impl EmbeddedServeHandle {
    fn native_node(&self) -> Result<&crate::mesh::Node> {
        anyhow::ensure!(
            !*self.shutdown.borrow(),
            "embedded runtime is shutting down"
        );
        anyhow::ensure!(
            !self
                .task
                .as_ref()
                .is_some_and(std::thread::JoinHandle::is_finished),
            "embedded mesh runtime has stopped"
        );
        self.native_node
            .as_ref()
            .context("embedded mesh runtime is unavailable")
    }

    /// Read the local and discovered peer model catalog in process.
    ///
    /// Discovery is asynchronous; call again after peer/model updates. Entries
    /// retain their provider identity so capabilities are never combined across
    /// different providers of the same serving ID.
    pub async fn models(&self) -> Result<Vec<Model>> {
        self.native_node()?.native_models().await
    }

    /// Execute against the exact discovered provider without loopback HTTP.
    ///
    /// A local target executes in process. A remote target uses native mesh
    /// transport. There is no fallback to a provider with different controls.
    pub async fn chat_completion(
        &self,
        target: ModelTarget,
        request: ChatCompletionRequest,
    ) -> Result<serde_json::Value> {
        let node = self.native_node()?;
        let mut shutdown = self.shutdown.subscribe();
        anyhow::ensure!(
            !*shutdown.borrow_and_update(),
            "embedded runtime is shutting down"
        );
        // Keep the transport's async state machine behind the host boundary;
        // embedding crates should not inherit its deep future layout.
        let inference: std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send + '_>,
        > = Box::pin(node.native_chat_completion(NativeChatRequest { target, request }));
        tokio::select! {
            biased;
            _ = shutdown.changed() => anyhow::bail!("embedded runtime is shutting down"),
            result = inference => result,
        }
    }

    /// Stream native chat deltas from the exact discovered provider.
    /// Dropping the stream or shutting down this handle cancels execution.
    pub async fn chat_completion_stream(
        &self,
        target: ModelTarget,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream> {
        let stream = self
            .native_node()?
            .native_chat_completion_stream(NativeChatRequest { target, request })
            .await?;
        Ok(stream.cancel_on_shutdown(self.shutdown.subscribe()))
    }
}

pub(super) async fn wait_for_embedded_node(
    handle: &EmbeddedServeHandle,
    timeout: Duration,
) -> Result<crate::mesh::Node> {
    let ready = handle.control(|resp| crate::api::RuntimeControlRequest::NativeNode { resp });
    tokio::pin!(ready);
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    let mut runtime_watch = tokio::time::interval(Duration::from_millis(25));
    loop {
        tokio::select! {
            biased;
            result = &mut ready => return result,
            _ = &mut deadline => anyhow::bail!("timed out waiting for embedded mesh runtime readiness"),
            _ = runtime_watch.tick() => {
                if handle.task.as_ref().is_none_or(std::thread::JoinHandle::is_finished) {
                    anyhow::bail!("embedded mesh runtime exited before becoming ready");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::RuntimeControlRequest;
    use tokio::sync::{mpsc, watch};

    fn handle(control_tx: mpsc::UnboundedSender<RuntimeControlRequest>) -> EmbeddedServeHandle {
        EmbeddedServeHandle {
            api_base_url: "http://invalid.invalid/v1".into(),
            console_url: "http://invalid.invalid".into(),
            invite_token: None,
            control_tx: Some(control_tx),
            task: None,
            shutdown: watch::channel(false).0,
            native_node: None,
        }
    }

    fn hold_runtime(handle: &mut EmbeddedServeHandle) -> std::sync::mpsc::Sender<()> {
        let (stop, stopped) = std::sync::mpsc::channel();
        handle.task = Some(std::thread::spawn(move || {
            let _ = stopped.recv();
            Ok(())
        }));
        stop
    }

    #[tokio::test]
    async fn readiness_uses_the_native_control_channel() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut handle = handle(tx);
        let stop = hold_runtime(&mut handle);
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
            .await
            .unwrap();
        let expected = node.id();
        let respond = async {
            let Some(RuntimeControlRequest::NativeNode { resp }) = rx.recv().await else {
                panic!("expected native readiness request");
            };
            assert!(resp.send(node).is_ok());
        };
        let (result, ()) = tokio::join!(
            wait_for_embedded_node(&handle, Duration::from_secs(1)),
            respond
        );
        stop.send(()).unwrap();
        handle.task.take().unwrap().join().unwrap().unwrap();
        assert_eq!(result.unwrap().id(), expected);
    }

    #[tokio::test]
    async fn readiness_reports_a_closed_runtime_channel() {
        let (tx, rx) = mpsc::unbounded_channel();
        let handle = handle(tx);
        drop(rx);
        let error = wait_for_embedded_node(&handle, Duration::from_secs(1))
            .await
            .err()
            .expect("closed control channel must fail");
        assert!(error.to_string().contains("closed"));
    }

    #[tokio::test]
    async fn readiness_has_a_startup_deadline() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut handle = handle(tx);
        let stop = hold_runtime(&mut handle);
        let result = wait_for_embedded_node(&handle, Duration::from_millis(10)).await;
        stop.send(()).unwrap();
        handle.task.take().unwrap().join().unwrap().unwrap();
        let error = result.err().expect("stalled runtime must time out");
        assert!(error.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn native_queries_reject_shutdown() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let handle = handle(tx);
        handle.shutdown.send_replace(true);
        assert!(
            handle
                .models()
                .await
                .unwrap_err()
                .to_string()
                .contains("shutting down")
        );
    }

    struct ProducerStopped(Option<tokio::sync::oneshot::Sender<()>>);

    impl Drop for ProducerStopped {
        fn drop(&mut self) {
            if let Some(stopped) = self.0.take() {
                let _ = stopped.send(());
            }
        }
    }

    fn stalled_stream() -> (ChatCompletionStream, tokio::sync::oneshot::Receiver<()>) {
        let (sender, receiver) = mpsc::channel(1);
        let (stopped, observed) = tokio::sync::oneshot::channel();
        let signal = ProducerStopped(Some(stopped));
        let task = tokio::spawn(async move {
            let _signal = signal;
            let _sender = sender;
            std::future::pending::<()>().await;
        });
        (ChatCompletionStream::new(receiver, task), observed)
    }

    #[tokio::test]
    async fn dropping_a_native_stream_cancels_its_producer() {
        let (stream, stopped) = stalled_stream();
        drop(stream);
        tokio::time::timeout(Duration::from_secs(1), stopped)
            .await
            .expect("producer must be cancelled")
            .unwrap();
    }

    #[tokio::test]
    async fn shutdown_cancels_an_unpolled_native_stream() {
        let (stream, stopped) = stalled_stream();
        let (shutdown, receiver) = watch::channel(false);
        let mut stream = stream.cancel_on_shutdown(receiver);
        shutdown.send_replace(true);
        tokio::time::timeout(Duration::from_secs(1), stopped)
            .await
            .expect("shutdown must cancel without reading another chunk")
            .unwrap();
        let error = stream.next_chunk().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("shutting down"));
        assert!(stream.next_chunk().await.is_none());
    }

    #[test]
    fn native_request_preserves_provider_and_effort() {
        let request = NativeChatRequest {
            target: ModelTarget {
                provider_node_id: "provider-a".into(),
                serving_id: "example".into(),
            },
            request: ChatCompletionRequest {
                reasoning_effort: Some(ReasoningEffort::High),
                ..Default::default()
            },
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["target"]["provider_node_id"], "provider-a");
        assert_eq!(value["request"]["reasoning_effort"], "high");
        let restored: NativeChatRequest = serde_json::from_value(value).unwrap();
        assert_eq!(restored.target, request.target);
        assert_eq!(
            restored.request.reasoning_effort,
            Some(ReasoningEffort::High)
        );
    }
}
