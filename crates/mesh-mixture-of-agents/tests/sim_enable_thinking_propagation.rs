//! Simulated-mesh integration test for `enable_thinking` propagation.
//!
//! When the MoA gateway is called with `reasoning_effort: "none"` (or any
//! other recognized "don't think" knob), every worker AND the reducer
//! must receive `chat_template_kwargs.enable_thinking: false` in their
//! outbound request body. That's how the reasoning-template flag reaches
//! llama.cpp on the worker side and gets the model to skip its `<think>`
//! phase entirely.
//!
//! Background — see #617 / `~/Desktop/think.md`. For `model: "mesh"`, the
//! OpenAI-style reasoning knobs were being silently dropped, so the MoA
//! fast worker (256-token budget) burned its entire budget inside an
//! unclosed `<think>` block and never reached the answer. The fix
//! propagates the caller's preference through `GatewayConfig::enable_thinking`
//! down to `SamplingParams` and the backends.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

/// Backend that records every request body it receives and returns a
/// canned successful chat-completion response. Used to assert what each
/// worker (and the reducer) actually sees on the wire.
struct RecordingBackend {
    received: Arc<Mutex<Vec<Value>>>,
    response_text: String,
}

impl RecordingBackend {
    fn new(response_text: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            received: Arc::new(Mutex::new(Vec::new())),
            response_text: response_text.into(),
        })
    }
}

#[async_trait]
impl moa::ModelBackend for RecordingBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        _timeout: Duration,
        sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        // Reconstruct the request body the same way the backends do,
        // so the assertion is on the actual wire shape.
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
        });
        if let Some(t) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), t.clone());
        }
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);

        self.received.lock().await.push(body);

        Ok(json!({
            "choices": [{
                "message": { "role": "assistant", "content": self.response_text }
            }]
        }))
    }
}

fn build_config(
    backend: Arc<RecordingBackend>,
    enable_thinking: Option<bool>,
) -> moa::GatewayConfig {
    moa::GatewayConfig {
        backends: vec![backend.clone() as Arc<dyn moa::ModelBackend>],
        models: vec![
            moa::ModelEntry {
                name: "test-fast".into(),
                backend_index: 0,
            },
            moa::ModelEntry {
                name: "test-specialist".into(),
                backend_index: 0,
            },
            moa::ModelEntry {
                name: "test-strong".into(),
                backend_index: 0,
            },
        ],
        worker_timeout: Duration::from_secs(5),
        reducer_timeout: Duration::from_secs(5),
        hedge_delay: Duration::from_secs(1),
        first_answer_grace: Duration::ZERO,
        enable_thinking,
    }
}

#[tokio::test]
async fn enable_thinking_false_reaches_every_worker() {
    let backend = RecordingBackend::new("Hi.");
    let config = build_config(backend.clone(), Some(false));

    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "say hi"}],
    });

    let _ = moa::handle_turn(&config, &body).await;

    let received = backend.received.lock().await;
    assert!(
        !received.is_empty(),
        "no worker calls recorded; MoA fanout didn't fire"
    );
    for (i, body) in received.iter().enumerate() {
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            json!(false),
            "worker call #{i} missing chat_template_kwargs.enable_thinking=false; body: {body}"
        );
        assert_eq!(
            body["reasoning_effort"],
            json!("none"),
            "worker call #{i} missing reasoning_effort='none'; body: {body}"
        );
    }
}

#[tokio::test]
async fn enable_thinking_true_reaches_every_worker_without_clobbering_effort() {
    let backend = RecordingBackend::new("Hi.");
    let config = build_config(backend.clone(), Some(true));

    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "say hi"}],
    });

    let _ = moa::handle_turn(&config, &body).await;

    let received = backend.received.lock().await;
    assert!(!received.is_empty());
    for (i, body) in received.iter().enumerate() {
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            json!(true),
            "worker call #{i} missing chat_template_kwargs.enable_thinking=true"
        );
        // Explicit "on" must not impose a specific reasoning_effort on
        // the caller's behalf.
        assert!(
            body.get("reasoning_effort").is_none(),
            "worker call #{i} unexpectedly set reasoning_effort: {body}"
        );
    }
}

#[tokio::test]
async fn no_thinking_override_leaves_body_clean() {
    let backend = RecordingBackend::new("Hi.");
    let config = build_config(backend.clone(), None);

    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "say hi"}],
    });

    let _ = moa::handle_turn(&config, &body).await;

    let received = backend.received.lock().await;
    assert!(!received.is_empty());
    for (i, body) in received.iter().enumerate() {
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "worker call #{i} got spurious chat_template_kwargs: {body}"
        );
        assert!(
            body.get("reasoning_effort").is_none(),
            "worker call #{i} got spurious reasoning_effort: {body}"
        );
    }
}
