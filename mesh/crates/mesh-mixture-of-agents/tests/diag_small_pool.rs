//! Diagnostic: why does an all-small pool win in the eval rig but not through
//! `handle_turn`? Dumps every request the shipped path sends (worker prompts +
//! reducer prompt), plus how many drafts actually reached synthesis, so the
//! difference can be *seen* rather than hypothesised.
//!
//! Not an assertion test — run with --nocapture and read the output.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};

/// Records every request body it is handed, then returns a canned answer.
struct RecordingBackend {
    name: String,
    reply: String,
    delay: Duration,
    log: Arc<Mutex<Vec<(String, Value)>>>,
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
        params: moa::SamplingParams,
    ) -> Result<Value, String> {
        self.log.lock().unwrap().push((
            self.name.clone(),
            json!({
                "model": model,
                "max_tokens": max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "enable_thinking": params.enable_thinking,
                "has_tools": tools.is_some(),
                "messages": messages,
            }),
        ));
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        Ok(json!({
            "choices": [{"message": {"content": self.reply}, "finish_reason": "stop"}]
        }))
    }
}

fn user_turn(content: &str) -> Value {
    json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": content}],
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn dump_all_small_pool_requests() {
    let log: Arc<Mutex<Vec<(String, Value)>>> = Arc::new(Mutex::new(Vec::new()));

    // Six same-tier 8B-class peers, distinct answers so synthesis has real
    // material and nothing can be mistaken for consensus.
    let names = [
        "qwen3-8b",
        "llama-3.1-8b",
        "granite-4.1-8b",
        "ministral-8b",
        "qwen2.5-7b",
        "qwen3.5-9b",
    ];
    let mut backends: Vec<Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for (i, n) in names.iter().enumerate() {
        models.push(moa::ModelEntry::new((*n).to_string(), i));
        backends.push(Arc::new(RecordingBackend {
            name: (*n).to_string(),
            reply: format!("DRAFT-FROM-{n}: backpressure means slowing the producer."),
            delay: Duration::ZERO,
            log: log.clone(),
        }));
    }

    let config = moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(90),
        hedge_delay: Duration::from_secs(5),
        reducer_timeout: Duration::from_secs(60),
        first_answer_grace: Duration::from_secs(10),
        strong_patience: Duration::from_secs(20),
        enable_thinking: None,
        actor_candidates: Vec::new(),
        reference_policy: Default::default(),
        refinement_policy: Default::default(),
    };

    let prompt = "Explain backpressure in distributed systems.";
    let result = moa::handle_turn(&config, &user_turn(prompt)).await;

    let entries = log.lock().unwrap().clone();
    println!("\n================ SHIPPED handle_turn ================");
    println!(
        "turn_kind={:?}  reducer_used={}  reducer_attempts={}  workers_dispatched={}",
        result.turn_kind,
        result.reducer_used,
        result.reducer_attempts,
        result.worker_summaries.len()
    );
    println!("total backend calls: {}", entries.len());

    for (i, (who, body)) in entries.iter().enumerate() {
        let msgs = body["messages"].as_array().cloned().unwrap_or_default();
        let is_reducer = msgs.iter().any(|m| {
            m["content"]
                .as_str()
                .map(|s| s.contains("Worker outputs") || s.contains("Synthesize"))
                .unwrap_or(false)
        });
        println!(
            "\n--- call {i}: {who} {} | max_tokens={} temp={} thinking={:?} ---",
            if is_reducer { "[REDUCER]" } else { "[worker]" },
            body["max_tokens"],
            body["temperature"],
            body["enable_thinking"],
        );
        for m in &msgs {
            let role = m["role"].as_str().unwrap_or("?");
            let content = m["content"].as_str().unwrap_or("");
            println!("  [{role}] {content}");
        }
    }

    // How many drafts actually reached synthesis?
    let reducer_call = entries.iter().find(|(_, b)| {
        b["messages"]
            .as_array()
            .map(|ms| {
                ms.iter().any(|m| {
                    m["content"]
                        .as_str()
                        .map(|s| s.contains("Worker outputs"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    });
    match reducer_call {
        Some((_, b)) => {
            let sys = b["messages"][0]["content"].as_str().unwrap_or("");
            let n = sys.matches("[Response").count();
            println!(
                "\n>>> DRAFTS THAT REACHED SYNTHESIS: {n} of {}",
                names.len()
            );
        }
        None => println!("\n>>> NO REDUCER CALL — synthesis never ran"),
    }

    println!("\n================ RIG (what won 12W/2L) ================");
    println!("peer draft call  : messages=[user: <prompt>]  max_tokens=1024 temp=0.8");
    println!(
        "synthesis call   : messages=[system: COMMITTEE_SYNTH_PROMPT + \"Candidate responses:\" \
         + [Response N] x6, user: <prompt>]  max_tokens=1024 temp=0.3"
    );
    println!("(no MoA preamble on peers, no agent system prompt, no truncation)\n");
}
