//! Buzz terminal-send rescue remains active when Mesh has a real committee.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

struct TerminalBackend;

#[async_trait]
impl moa::ModelBackend for TerminalBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        Ok(json!({
            "choices": [{"message": {"role": "assistant", "content": "Committee result."}}]
        }))
    }
}

#[tokio::test]
async fn committee_terminal_prose_is_wrapped_as_buzz_send() {
    let backends: Vec<Arc<dyn moa::ModelBackend>> =
        vec![Arc::new(TerminalBackend), Arc::new(TerminalBackend)];
    let config = moa::GatewayConfig {
        backends,
        models: vec![
            moa::ModelEntry::new("Qwen3-32B", 0),
            moa::ModelEntry::new("Llama-70B", 1),
        ],
        worker_timeout: Duration::from_secs(1),
        hedge_delay: Duration::from_millis(10),
        reducer_timeout: Duration::from_secs(1),
        first_answer_grace: Duration::ZERO,
        strong_patience: Duration::ZERO,
        enable_thinking: Some(false),
        actor_candidates: vec![0, 1],
        reference_policy: moa::ReferencePolicy::Never,
        refinement_policy: moa::RefinementPolicy::Never,
    };
    let body = json!({
        "model": "mesh",
        "messages": [{
            "role": "user",
            "content": "[Context]\nChannel: demo (#11111111-1111-1111-1111-111111111111)\nIMPORTANT: For ordinary replies use `--reply-to aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` on `buzz messages send`.\n\n[Buzz event]\nReport the result."
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "buzz-dev-mcp__shell",
                "description": "Run a shell command",
                "parameters": {"type": "object"}
            }
        }]
    });

    let result = moa::handle_turn(&config, &body).await;
    let call = &result.response_body["choices"][0]["message"]["tool_calls"][0];

    assert_eq!(
        result.response_body["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert_eq!(call["function"]["name"], "buzz-dev-mcp__shell");
    let arguments: Value = serde_json::from_str(
        call["function"]["arguments"]
            .as_str()
            .expect("tool arguments"),
    )
    .expect("valid arguments JSON");
    let command = arguments["command"].as_str().expect("shell command");
    assert!(command.contains("buzz messages send"));
    assert!(command.contains("11111111-1111-1111-1111-111111111111"));
    assert!(command.contains("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    assert!(command.contains("Committee result."));
}
