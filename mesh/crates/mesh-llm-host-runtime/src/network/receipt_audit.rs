//! Audits of the turns this node's peers serve with receipts, when
//! `[receipts] audit` is on or the request's trust policy requires receipts
//! (`mesh/docs/design/RECEIPTS.md`, `network/request_trust.rs`).
//!
//! A chat completion a local client sends through this node to a peer names
//! a fresh receipts VRF key (`x-receipt-client-key`), so a peer that issues
//! receipts binds the turn to it and returns the signed receipt with the
//! response. The VRF of the turn root picks one turn in ten, or one in
//! `[receipts] audit_one_in`, to audit, so the peer cannot tell which. The audit runs
//! off the request path on one worker thread: it challenges the peer within
//! the challenge window, reads the bytes the checks need from the claimed
//! Hugging Face GGUF by range, and checks the opening on the CPU.
//!
//! Every watched turn ends in one of these verdicts (`verdict`), and each is
//! logged:
//!
//! - **passed:** nothing happens.
//! - **failed:** the peer signed a defect. The evidence is kept
//!   (`evidence_store`) and the peer is blocked locally for seven days.
//! - **strike:** the peer sent no usable receipt though it advertises
//!   them, named another model than a request that pinned its repository
//!   and commit asked for, gave no usable signed answer to the challenge,
//!   answered in a shape the request did not allow, or failed a check only
//!   this node saw.
//!   Three strikes within a day block the peer for seven days (`strikes`).
//! - **no audit:** the VRF did not draw the turn, or this node could not
//!   audit it: it cannot tie the model's name to a Hugging Face file
//!   (`requested_model`), Hugging Face is out of reach, the worker was busy,
//!   the challenge window passed before the challenge was ready. It says nothing about
//!   the peer.
//! - **unauditable:** the receipt states something no audit covers. The
//!   peer loses the `receipts` tier on this node for a day, and is struck
//!   when this node audited it for the model within that day
//!   (`auditability`).
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
mod auditability;
mod chat_request;
mod claimed_models;
mod collector;
mod commit_budget;
#[cfg(test)]
pub(crate) use collector::probe::CollectorProbe;
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
#[cfg(test)]
pub(crate) mod two_node_seam;
mod verdict;
mod worker;

/// Whether this node can tie the model name `name` to a Hugging Face file
/// by its own word (`requested_model`), which an audit of a turn asked
/// under that name needs. Under any other name every audit ends as no
/// audit, so such a request cannot satisfy the `receipts` tier
/// (`network/request_trust.rs`).
pub(crate) fn ties_model_name(name: &str) -> bool {
    requested_model::RequestedModel::of(name, &requested_model::node_catalog).is_ok()
}

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
    /// The receipt id this node chose for the attempt
    /// (`x-receipt-id`), which a peer that knows the header issues its
    /// receipt under.
    id: String,
    /// The request's own id (`x-request-id`), which a peer that does not
    /// know `x-receipt-id` issues its receipt under; `None` when the
    /// request names no one id, and the peer then names one of its own.
    request_id: Option<String>,
    /// The last user message's text, when the receipt's tokens must hold it.
    prompt: Option<String>,
    sampling: RequestedSampling,
    /// Why the response's text need not be the sampled tokens' text.
    text_unavailable: Option<&'static str>,
    /// Why the answer may call tools or hold several choices, when the
    /// request allows it.
    shaped_answer: Option<&'static str>,
    /// The peer advertised, when the request was sent, that it serves the
    /// model with receipts.
    advertised: bool,
    /// The request's trust policy requires receipts.
    tier: bool,
    /// The share of turns the VRF draws for an audit (`[receipts]
    /// audit_one_in`).
    rate: f64,
}

/// A turn this node watches for a receipt to audit.
pub(crate) struct WatchedTurn {
    node: mesh::Node,
    asked: Asked,
}

/// What forwarding a watched request needs: the request naming this node's
/// key, the receipt id the peer is asked to issue its receipt under, and
/// the observer that collects the response for the audit.
pub(crate) struct Watch {
    pub(crate) request: Vec<u8>,
    pub(crate) receipt_id: String,
    pub(crate) observer: Arc<dyn skippy_inference_api::wire_bytes::WireBytesObserver>,
}

/// When this node audits the turn `request` asks `host` for: the request
/// to forward instead, and the observer to attach to the client stream.
/// `None` for a request this node leaves alone: it audits no turn of the
/// request (`request_trust::audits_turn`: auditing is off and no policy
/// requires receipts, or the node is offline), it is not a chat completion,
/// or its client named a receipt key of its own. An error for a chat
/// completion this node cannot read when its policy requires receipts
/// (`tier`): such a request is never served unwatched.
pub(crate) async fn watch(
    node: &mesh::Node,
    host: EndpointId,
    request: &[u8],
    tier: bool,
) -> Result<Option<Watch>, String> {
    if !crate::network::request_trust::audits_turn(node, request).await {
        return Ok(None);
    }
    let parsed = match ChatRequest::read(request) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => return Ok(None),
        Err(why) if tier => return Err(why.to_owned()),
        Err(why) => {
            tracing::info!(
                "receipt audit: a turn on {} is not watched: {why}",
                host.fmt_short()
            );
            return Ok(None);
        }
    };
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
    // each attempt its own id, so no other turn, of this client or
    // another, holds it at the peer
    let receipt_id = uuid::Uuid::new_v4().hyphenated().to_string();
    let asked = Asked {
        host,
        key,
        model: parsed.model,
        id: receipt_id.clone(),
        request_id: parsed.asked_id,
        prompt: parsed.prompt,
        sampling,
        text_unavailable: parsed.text_unavailable,
        shaped_answer: parsed.shaped_answer,
        advertised,
        tier,
        rate: node
            .config_state
            .lock()
            .await
            .config()
            .receipts
            .audit_rate(),
    };
    let turn = WatchedTurn {
        node: node.clone(),
        asked,
    };
    Ok(Some(Watch {
        request,
        receipt_id,
        observer: Arc::new(collector::Collector::new(turn)),
    }))
}

/// This node's audits of its peers: the worker that runs them and the
/// strikes they count. A node and its mesh memberships share one, so a
/// peer's strikes count wherever it was asked.
#[derive(Default)]
pub(crate) struct Audits {
    strikes: Mutex<strikes::Strikes>,
    auditability: Mutex<auditability::Auditability>,
    commits: Mutex<commit_budget::CommitBudget>,
    worker: worker::Slot<drawn_turn::Job>,
    /// Drawn turns not audited because the worker was busy.
    busy: std::sync::atomic::AtomicU64,
    /// What the worker opens a claimed model with, in a test that audits
    /// the toy model (`two_node_seam`); Hugging Face otherwise.
    #[cfg(test)]
    claimed_opener: std::sync::OnceLock<claimed_models::Opener>,
    /// A test's stand-in for a node with no native runtime loaded.
    #[cfg(test)]
    auditor_unavailable: std::sync::atomic::AtomicBool,
}

impl Audits {
    /// Ends the audits with the node: turns still waiting are dropped, the
    /// worker ends after the step it is in and its claimed models are
    /// released. Later turns are no audit.
    pub(crate) async fn shut_down(&self) {
        self.worker.shut_down().await;
    }

    /// Whether this node can audit a peer's turn: an audit replays it on the
    /// native runtime, which a `client` loads only with `[receipts] audit`
    /// or `require` set. Tests have no runtime loaded and stand in for one.
    pub(crate) fn auditor_ready(&self) -> bool {
        #[cfg(test)]
        return !self
            .auditor_unavailable
            .load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        skippy_runtime::native_runtime_loaded()
    }

    /// Makes [`Self::auditor_ready`] false, as on a node without a runtime.
    #[cfg(test)]
    pub(crate) fn set_auditor_unavailable(&self) {
        self.auditor_unavailable
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether `peer` stopped satisfying the `receipts` tier on this node:
    /// a receipt of it this node could not audit, for a reason in the
    /// receipt, came within the last day (`auditability`).
    pub(crate) fn demoted(&self, peer: &EndpointId, now_ms: u64) -> bool {
        self.auditability
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .demoted(peer, now_ms)
    }

    /// What the audit worker opens a claimed model with.
    fn claimed_opener(&self) -> claimed_models::Opener {
        #[cfg(test)]
        if let Some(open) = self.claimed_opener.get() {
            return *open;
        }
        claimed_models::open_from_hub
    }
}
