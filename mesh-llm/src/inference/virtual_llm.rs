//! Virtual LLM — the decision engine behind mesh hook callbacks.
//!
//! llama-server calls POST /mesh/hook on the management API (port 3131)
//! at key points during inference. Each hook blocks the slot until we
//! respond. We can consult other models in the mesh over QUIC and tell
//! the C++ side to inject context or replace the response.
//!
//! # Hooks
//!
//! | Hook             | When                          | What we can do              |
//! |------------------|-------------------------------|-----------------------------|
//! | pre_inference    | Before tokenization           | Inject (e.g. image caption) |
//! | post_prefill     | After prefill, before gen     | Inject second opinion       |
//! | mid_generation   | During gen, entropy spike     | Inject mid-course correction|
//! | pre_response     | After gen, before sending     | Replace or pass through     |
//!
//! # Consultation
//!
//! All peer consultation goes through `consult.rs` which opens a QUIC
//! stream directly to the peer — same path as normal mesh inference.
//! Fan-out: race 2 peers, take the first response, abort the loser.
//! 10s timeout on all consultations.

use crate::inference::consult;
use crate::mesh;
use serde_json::{json, Value};

// ===========================================================================
// Hook 1: pre_inference — before tokenization
// ===========================================================================

/// Fires on media triggers: images on a text-only model, audio on a
/// non-audio model. We find a capable peer, get a caption/transcript,
/// and inject it as text so the model can "see" the media.
pub async fn handle_pre_inference(node: &mesh::Node, payload: &Value) -> Value {
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let model = payload["model"].as_str().unwrap_or("");

    tracing::info!("virtual: pre_inference trigger={trigger} model={model}");

    match trigger {
        "images_no_multimodal" => caption_image(node, payload, model).await,
        "audio_no_support" => {
            tracing::info!("virtual: audio — not yet implemented");
            json!({ "action": "none" })
        }
        _ => json!({ "action": "none" }),
    }
}

async fn caption_image(node: &mesh::Node, payload: &Value, current_model: &str) -> Value {
    let vision_peer = consult::find_vision_peer(node, current_model).await;
    let peer_id = match vision_peer {
        Some(id) => id,
        None => {
            tracing::info!("virtual: no vision peer available");
            return json!({ "action": "none" });
        }
    };

    let peers = node.peers().await;
    let vision_model = peers
        .iter()
        .find(|p| p.id == peer_id)
        .and_then(|p| {
            p.served_model_descriptors
                .iter()
                .find(|d| d.capabilities.supports_vision_runtime())
                .map(|d| d.identity.model_name.clone())
        })
        .unwrap_or_default();

    let (image_url, user_text) = extract_image_from_payload(payload);
    if image_url.is_empty() {
        tracing::warn!("virtual: images trigger but no image in payload");
        return json!({ "action": "none" });
    }

    tracing::info!(
        "virtual: captioning via {} model={vision_model}",
        peer_id.fmt_short()
    );

    match consult::caption_image(node, peer_id, &vision_model, &image_url, &user_text).await {
        Ok(caption) => {
            tracing::info!("virtual: caption ({} chars)", caption.len());
            json!({
                "action": "inject",
                "text": format!("[Image description: {caption}]\n\n"),
            })
        }
        Err(e) => {
            tracing::warn!("virtual: caption failed: {e}");
            json!({ "action": "none" })
        }
    }
}

// ===========================================================================
// Hook 2: post_prefill — model processed prompt, first token is uncertain
// ===========================================================================

/// The model doesn't know how to start its answer (high entropy on first
/// token). Ask a different model the same question, inject its answer as
/// context. The value is diversity — a different architecture's perspective.
///
/// Returns `{"action": "inject", "text": "..."}` or `{"action": "none"}`.
/// C++ tokenizes the inject text and decodes it into KV cache — the model
/// "reads" it as part of the prompt, then generates its own answer.
pub async fn handle_post_prefill(node: &mesh::Node, payload: &Value) -> Value {
    let entropy = payload["signals"]["first_token_entropy"]
        .as_f64()
        .unwrap_or(0.0);
    let margin = payload["signals"]["first_token_margin"]
        .as_f64()
        .unwrap_or(1.0);
    let model = payload["model"].as_str().unwrap_or("");

    tracing::info!("virtual: post_prefill entropy={entropy:.2} margin={margin:.3} model={model}");

    let messages = match extract_messages(payload) {
        Some(m) => m,
        None => return json!({ "action": "none" }),
    };

    race_second_opinion(node, model, &messages).await
}

// ===========================================================================
// Hook 2b: mid_generation — sustained entropy spike during generation
// ===========================================================================

/// The model is generating but has been uncertain for many tokens in a row
/// (sustained entropy spike). It may be going off the rails. Same approach
/// as post_prefill: ask a peer the original question, inject the answer.
///
/// The payload includes `generated_text` (partial output so far) but we
/// don't send that to the peer — we just re-ask the original question.
/// The injection goes into KV cache at the current position, so the model
/// reads it and course-corrects from there.
pub async fn handle_mid_generation(node: &mesh::Node, payload: &Value) -> Value {
    let n_decoded = payload["n_decoded"].as_i64().unwrap_or(0);
    let model = payload["model"].as_str().unwrap_or("");

    tracing::info!("virtual: mid_generation n_decoded={n_decoded} model={model}");

    let messages = match extract_messages(payload) {
        Some(m) => m,
        None => return json!({ "action": "none" }),
    };

    race_second_opinion(node, model, &messages).await
}

// ===========================================================================
// Hook 3: pre_response — generation complete, about to send to client
// ===========================================================================

/// The model finished generating but signals suggest the output is bad:
/// tail entropy spike, high overall uncertainty, suspiciously short, or
/// Hook 1 explicitly requested verification.
///
/// We send the last user message + tail of the generated text to a peer
/// and ask "is this coherent?" If the peer says no, we replace the
/// response entirely with the peer's corrected version.
///
/// Returns `{"action": "replace", "text": "..."}` or `{"action": "none"}`.
pub async fn handle_pre_response(node: &mesh::Node, payload: &Value) -> Value {
    let trigger = payload["trigger"].as_str().unwrap_or("unknown");
    let generated_text = payload["generated_text"].as_str().unwrap_or("");
    let n_decoded = payload["n_decoded"].as_i64().unwrap_or(0);
    let mean_entropy = payload["signals"]["mean_entropy"].as_f64().unwrap_or(0.0);
    let model = payload["model"].as_str().unwrap_or("");

    tracing::info!(
        "virtual: pre_response trigger={trigger} n_decoded={n_decoded} \
         mean_entropy={mean_entropy:.2}"
    );

    match trigger {
        "very_short" => {
            tracing::info!("virtual: short response ({n_decoded} tokens), passing through");
            json!({ "action": "none" })
        }
        "tail_entropy_spike" | "high_uncertainty" | "verify" => {
            verify_and_maybe_replace(node, payload, model, trigger, generated_text).await
        }
        _ => json!({ "action": "none" }),
    }
}

async fn verify_and_maybe_replace(
    node: &mesh::Node,
    payload: &Value,
    current_model: &str,
    trigger: &str,
    generated_text: &str,
) -> Value {
    let peer = consult::find_different_model_peer(node, current_model).await;
    let (peer_id, peer_model) = match peer {
        Some(p) => p,
        None => {
            tracing::info!("virtual: no peer available for verification");
            return json!({ "action": "none" });
        }
    };

    let messages = match payload["messages"].as_array() {
        Some(m) => m.as_slice(),
        None => return json!({ "action": "none" }),
    };

    tracing::info!(
        "virtual: verifying ({trigger}) via {} model={peer_model}",
        peer_id.fmt_short()
    );

    match consult::verify_response(node, peer_id, &peer_model, messages, generated_text).await {
        Ok(verdict) => {
            if verdict.contains("LOOKS_GOOD") {
                tracing::info!("virtual: verification passed");
                json!({ "action": "none" })
            } else {
                let trimmed = if verdict.len() > 2048 {
                    format!("{}...", &verdict[..2048])
                } else {
                    verdict
                };
                tracing::info!("virtual: replacing response ({} chars)", trimmed.len());
                json!({
                    "action": "replace",
                    "text": trimmed,
                })
            }
        }
        Err(e) => {
            tracing::warn!("virtual: verification failed: {e}");
            json!({ "action": "none" })
        }
    }
}

// ===========================================================================
// Shared: race 2 peers for a second opinion, return inject or none
// ===========================================================================

async fn race_second_opinion(node: &mesh::Node, current_model: &str, messages: &[Value]) -> Value {
    let peers = consult::find_different_model_peers(node, current_model, 2).await;
    if peers.is_empty() {
        tracing::info!("virtual: no different model available");
        return json!({ "action": "none" });
    }

    let peer_names: Vec<_> = peers
        .iter()
        .map(|(id, m)| format!("{}={m}", id.fmt_short()))
        .collect();
    tracing::info!(
        "virtual: racing {} peers: [{}]",
        peers.len(),
        peer_names.join(", ")
    );

    match consult::race_second_opinion(node, &peers, messages).await {
        Some((opinion, winner_id, winner_model)) => {
            let trimmed = if opinion.len() > 512 {
                format!("{}...", &opinion[..512])
            } else {
                opinion
            };
            tracing::info!(
                "virtual: injecting from {} ({}) ({} chars)",
                winner_id.fmt_short(),
                winner_model,
                trimmed.len()
            );
            json!({
                "action": "inject",
                "text": format!("\n[Context: {trimmed}]\n\n"),
            })
        }
        None => {
            tracing::warn!("virtual: all peers failed");
            json!({ "action": "none" })
        }
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn extract_messages(payload: &Value) -> Option<Vec<Value>> {
    match payload["messages"].as_array() {
        Some(m) if !m.is_empty() => Some(m.clone()),
        _ => {
            tracing::debug!("virtual: no messages in payload");
            None
        }
    }
}

fn extract_image_from_payload(payload: &Value) -> (String, String) {
    let messages = match payload["messages"].as_array() {
        Some(m) => m,
        None => return (String::new(), String::new()),
    };

    for msg in messages.iter().rev() {
        if msg["role"].as_str() != Some("user") {
            continue;
        }
        if let Some(parts) = msg["content"].as_array() {
            let mut image_url = String::new();
            let mut text = String::new();
            for part in parts {
                match part["type"].as_str() {
                    Some("image_url") => {
                        if image_url.is_empty() {
                            image_url = part["image_url"]["url"].as_str().unwrap_or("").to_string();
                        }
                    }
                    Some("text") => {
                        text = part["text"].as_str().unwrap_or("").to_string();
                    }
                    _ => {}
                }
            }
            if !image_url.is_empty() {
                return (image_url, text);
            }
        }
    }

    (String::new(), String::new())
}
