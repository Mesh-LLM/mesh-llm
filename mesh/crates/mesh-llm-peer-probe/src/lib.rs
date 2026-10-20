//! Experimental artifact-only peer probe. The controller is real WASM, but the
//! broker is controlled by B; no hardware attestation or locality claim exists.
pub mod appraisal;
pub mod broker;
pub mod controller;
pub mod model;
pub mod protocol;

use crate::{
    broker::Broker,
    protocol::{ProbeInput, ProbeResponse, ProbeStatus},
};
use anyhow::Result;
use std::time::Instant;

/// A non-agentic comparator that uses the same broker and challenge, without WASM.
pub fn fixed_probe(input: &ProbeInput, mut broker: Box<dyn Broker>) -> Result<ProbeResponse> {
    let start = Instant::now();
    let mut trace = Vec::new();
    for tool in &input.challenge.intents {
        trace.push(broker.observe(&input.challenge, *tool)?);
    }
    Ok(ProbeResponse {
        challenge_id: input.challenge.id.clone(),
        peer: input.challenge.peer.clone(),
        nonce: input.challenge.nonce.clone(),
        answer: "Fixed probe: no agent explanation".into(),
        status: ProbeStatus::Completed,
        tool_calls: trace.len() as u32,
        trace,
        model_actions: vec![],
        error: None,
        wasm_executed: false,
        model_calls: 0,
        elapsed_ms: start.elapsed().as_millis(),
    })
}
