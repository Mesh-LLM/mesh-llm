//! Diagnostic: dump the exact prompts `handle_turn` sends on a text turn.
//!
//! Two prompt-level hypotheses for the harness-vs-shipped gap were tested live
//! and rejected. Rather than guess a third, this captures what production
//! actually sends — every role/content pair, in order — so it can be compared
//! byte-for-byte against the harness's `plain_answer` / `refine` /
//! `synth_messages`.
//!
//! Deterministic and offline: canned backends, no network.
//!
//! Run with:
//!   cargo test -p mesh-mixture-of-agents --test diag_prompt_diff -- --nocapture

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Records every request it is asked to serve, then returns canned prose.
struct Recorder {
    label: String,
    reply: String,
    log: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl moa::ModelBackend for Recorder {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        _tools: Option<&Value>,
        max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        let rendered = messages
            .iter()
            .map(|m| {
                let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
                let content = m
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("<non-string>");
                format!("  [{role}]\n{content}")
            })
            .collect::<Vec<_>>()
            .join("\n");
        self.log.lock().unwrap().push((
            format!("{} -> {model} (max_tokens={max_tokens})", self.label),
            rendered,
        ));
        Ok(json!({"choices":[{"message":{"content": self.reply},"finish_reason":"stop"}]}))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn dump_text_turn_prompts() {
    let log = Arc::new(Mutex::new(Vec::new()));
    // Distinct replies so round-1 answers disagree and the turn must synthesize.
    let replies = [
        "Backpressure means the consumer signals the producer to slow down.",
        "It is a flow-control mechanism preventing unbounded queue growth.",
        "Without it, queues grow until memory is exhausted.",
        "It propagates load information upstream through the pipeline.",
    ];
    let names = ["Qwen3-8B", "Llama-3.1-8B", "Granite-4.1-8B", "Ministral-8B"];

    let mut backends: Vec<Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for (i, name) in names.iter().enumerate() {
        models.push(moa::ModelEntry {
            name: (*name).to_string(),
            backend_index: i,
        });
        backends.push(Arc::new(Recorder {
            label: (*name).to_string(),
            reply: replies[i].to_string(),
            log: log.clone(),
        }));
    }

    let cfg = moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(5),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(5),
        // Production defaults for a text turn.
        first_answer_grace: Duration::from_secs(3),
        strong_patience: Duration::from_secs(20),
        enable_thinking: Some(false),
        actor_candidates: Vec::new(),
        reference_policy: Default::default(),
        refinement_policy: Default::default(),
    };

    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "Explain backpressure."}],
    });
    let result = moa::handle_turn(&cfg, &body).await;

    println!("\n================ PRODUCTION handle_turn: every call ================");
    for (i, (who, prompt)) in log.lock().unwrap().iter().enumerate() {
        println!("\n--- call {} : {} ---\n{}", i + 1, who, prompt);
    }
    println!(
        "\n================ turn_kind={:?} reducer_used={} ================",
        result.turn_kind, result.reducer_used
    );

    println!("\n================ HARNESS equivalents (for comparison) ================");
    println!("\n--- harness plain_answer (round 1 draft) ---");
    println!("  [user]\nExplain backpressure.");
    println!("  (no system message; max_tokens=1024)");
    println!("\n--- harness refine / synthesize ---");
    let harness_synth = "You have been given a user request and several candidate responses \
         from other models. Synthesize them into one high-quality response. Critically \
         evaluate them — some may be biased or incorrect, and agreement is not proof of \
         correctness. Do not merely copy the longest or most confident; produce the most \
         accurate, well-structured reply. Be direct.";
    println!("  [system]\n{harness_synth}\n\nCandidate responses:\n[Response 1]:\n<draft>\n...");
    println!("  [user]\nExplain backpressure.");
    println!("  (max_tokens=1024)");
}
