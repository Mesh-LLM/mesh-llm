//! Audits of the turns this node's peers serve with receipts, when
//! `[receipts] audit` is on or the request's trust policy requires receipts
//! (`mesh/docs/design/RECEIPTS.md`, `network/request_trust.rs`).
//!
//! A chat completion a local client sends through this node to a peer names
//! a fresh receipts VRF key (`x-receipt-client-key`), so a peer that issues
//! receipts binds the turn to it and returns the signed receipt with the
//! response. The VRF of the turn root picks about one turn in ten
//! ([`AUDIT_RATE`]) to audit, so the peer cannot tell which. The audit runs
//! off the request path on one worker thread: it challenges the peer within
//! the challenge window, reads the bytes the checks need from the claimed
//! Hugging Face GGUF by range, and checks the opening on the CPU.
//!
//! Every watched turn ends in one of four verdicts (`verdict`), and each is
//! logged:
//!
//! - **passed:** nothing happens.
//! - **failed:** the peer signed a defect. The evidence is kept
//!   (`evidence_store`) and the peer is blocked locally for seven days.
//! - **strike:** the peer sent no usable receipt though it advertises
//!   them, named another model than the request asked for, gave no usable
//!   signed answer to the challenge, or failed a check only this node saw.
//!   Three strikes within a day block the peer for seven days (`strikes`).
//! - **no audit:** the VRF did not draw the turn, or this node could not
//!   audit it: it cannot tie the model's name to a Hugging Face file
//!   (`requested_model`), Hugging Face is out of reach, the runtime or the
//!   backend is one it cannot check, the worker was busy, the challenge
//!   window passed before the challenge was ready. It says nothing about
//!   the peer.
//!
//! What the audit holds a peer to is what this node knows without it: the
//! request it sent (`chat_request`), the peer's identity key, and the
//! Hugging Face file the requested model's name stands for. Everything a
//! peer sends is bounded before it is kept (`collector`, `answer_limits`)
//! and escaped before it is logged (`peer_text`).

use std::sync::{Arc, Mutex};

use iroh::EndpointId;
use skippy_receipts::sampler_request::RequestedSampling;
use skippy_receipts::vrf;

use crate::mesh;
use chat_request::ChatRequest;

mod answer_limits;
mod chat_request;
mod claimed_models;
mod collector;
mod drawn_turn;
mod enforcement;
mod evidence_store;
mod http_answer;
mod intake;
mod opener;
mod peer_text;
#[cfg(test)]
mod policy_tests;
mod requested_model;
mod strikes;
#[cfg(test)]
mod test_support;
mod verdict;
mod worker;

/// The share of receipted turns this node audits.
pub(crate) const AUDIT_RATE: f64 = 0.1;

/// What this node asked a peer for, and what it knew of the peer when it
/// asked. The audit holds the peer to this, and to nothing the peer says
/// of itself afterwards.
struct Asked {
    /// The peer the request was sent to, whose identity key signs its
    /// receipts.
    host: EndpointId,
    /// The receipts VRF key the request named.
    key: vrf::SecretKey,
    /// The model the request named.
    model: String,
    /// The id the request asked under.
    id: Option<String>,
    /// The last user message's text, when the receipt's tokens must hold it.
    prompt: Option<String>,
    sampling: RequestedSampling,
    /// Why the response's text need not be the sampled tokens' text.
    text_unavailable: Option<&'static str>,
    /// The peer advertised, when the request was sent, that it serves the
    /// model with receipts.
    advertised: bool,
}

/// A turn this node watches for a receipt to audit.
pub(crate) struct WatchedTurn {
    node: mesh::Node,
    asked: Asked,
}

/// What forwarding a watched request needs: the request naming this node's
/// key, and the observer that collects the response for the audit.
pub(crate) struct Watch {
    pub(crate) request: Vec<u8>,
    pub(crate) observer: Arc<dyn skippy_inference_api::wire_bytes::WireBytesObserver>,
}

/// When this node audits the turn `request` asks `host` for: the request
/// to forward instead, and the observer to attach to the client stream.
/// `None` for a request this node leaves alone: it audits no turn of the
/// request (`request_trust::audits_turn`: auditing is off and no policy
/// requires receipts, or the node is offline), it is not a chat completion,
/// or its client named a receipt key of its own.
pub(crate) async fn watch(node: &mesh::Node, host: EndpointId, request: &[u8]) -> Option<Watch> {
    if !crate::network::request_trust::audits_turn(node, request).await {
        return None;
    }
    let parsed = ChatRequest::parse(request)?;
    let key = vrf::SecretKey::generate(&mut rand::rng());
    let fresh_seed = rand::RngExt::random_range(&mut rand::rng(), 1..=u32::MAX);
    let (headers, sampling) = chat_request::receipt_headers(
        &hex::encode(key.public()),
        parsed.sampling,
        parsed.seed_header,
        fresh_seed,
    );
    let request = chat_request::with_headers(request, parsed.header_end, &headers);
    let advertised =
        crate::network::request_trust::peer_advertises_receipts(node, &host, &parsed.model).await;
    let asked = Asked {
        host,
        key,
        model: parsed.model,
        id: parsed.asked_id,
        prompt: parsed.prompt,
        sampling,
        text_unavailable: parsed.text_unavailable,
        advertised,
    };
    let turn = WatchedTurn {
        node: node.clone(),
        asked,
    };
    Some(Watch {
        request,
        observer: Arc::new(collector::Collector::new(turn)),
    })
}

/// This node's audits of its peers: the worker that runs them and the
/// strikes they count. A node and its mesh memberships share one, so a
/// peer's strikes count wherever it was asked.
#[derive(Default)]
pub(crate) struct Audits {
    strikes: Mutex<strikes::Strikes>,
    worker: worker::Slot<drawn_turn::Job>,
}

impl Audits {
    /// Ends the audits with the node: turns still waiting are dropped, the
    /// worker ends after the step it is in and its claimed models are
    /// released. Later turns are no audit.
    pub(crate) async fn shut_down(&self) {
        self.worker.shut_down().await;
    }
}
