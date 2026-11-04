//! What a test of the audits across two nodes joined over real QUIC
//! (`network/openai/transport_tests/receipt_audits.rs`) needs from this
//! module: a host backend that serves the toy turn and opens it as the toy
//! peer does, the toy model in place of the claimed model's Hugging Face
//! file, and what the audits came to.
//!
//! The seam replaces two things, which the test therefore does not cover:
//! the peer's runtime (skippy issues and opens real receipts) and the
//! claimed model this node reads from Hugging Face. Everything between them
//! is the node's own: routing, the tunnel, the watch and its collector, the
//! intake, the worker, the challenge over the tunnel, and enforcement.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};
use iroh::EndpointId;
use skippy_receipt_audit::claimed_file::HfFile;
use skippy_receipt_audit::claimed_model::ClaimedModel;
use skippy_receipt_audit::turn_audit::ClaimedSetup;
use skippy_receipts::toy_turn::Turn;
use skippy_receipts::turn;

use super::opener::Transport;
use super::test_support::{Answers, ToyClaimed, ToyPeer, toy_setup};
use crate::mesh;

/// The model the toy turn names, which the test's request asks for.
pub(crate) const TOY_MODEL: &str = super::test_support::TOY_MODEL;
/// The toy turn's prompt.
pub(crate) const PROMPT: &str = super::test_support::PROMPT;

/// What the toy peer does with the turns it serves.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ToyServing {
    /// Serves the toy turn with its receipt, and opens it as committed.
    Honest,
    /// Serves it the same, and opens something else than it committed.
    Cheating,
    /// Serves the answer without a receipt, though it advertises them.
    WithoutReceipt,
}

/// The receipt key and id of the turn the toy peer served last, so it can
/// open that turn.
type Served = Option<([u8; 32], String)>;

/// A host backend that serves chat completions and opens their receipts
/// as the toy peer does.
pub(crate) struct ToyBackend {
    serving: ToyServing,
    served: Mutex<Served>,
}

impl ToyBackend {
    pub(crate) fn new(serving: ToyServing) -> Self {
        Self {
            serving,
            served: Mutex::new(None),
        }
    }

    /// The HTTP answer to `request`, a whole request with its body.
    pub(crate) fn answer(&self, request: &[u8]) -> Vec<u8> {
        let (head, body) = split_request(request);
        let result = if head.starts_with("POST /v1/receipts/open ") {
            self.open(body)
        } else if head.starts_with("POST /v1/chat/completions ") {
            self.serve(&head)
        } else {
            Ok(http(404, &[], b"{}"))
        };
        result.unwrap_or_else(|error| http(400, &[], format!("{error:#}").as_bytes()))
    }

    fn serve(&self, head: &str) -> Result<Vec<u8>> {
        let key: [u8; 32] = hex::decode(header(head, "x-receipt-client-key").context("no key")?)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("a receipt key is 32 bytes"))?;
        // an older auditor names no attempt id; the request id stands in
        let id = header(head, "x-receipt-id")
            .or_else(|| header(head, "x-request-id"))
            .context("no receipt id")?;
        let turn = toy_turn(key, &id);
        let content =
            ToyClaimed { unloadable: false }.detokenize(&turn.receipt.sampled_tokens())?;
        let mut body = serde_json::json!({
            "object": "chat.completion",
            "model": TOY_MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        });
        if !matches!(self.serving, ToyServing::WithoutReceipt) {
            body["receipt"] = serde_json::to_value(&turn.receipt)?;
        }
        *self.served.lock().unwrap_or_else(|e| e.into_inner()) = Some((key, id));
        Ok(http(
            200,
            &[("Content-Type", "application/json")],
            &serde_json::to_vec(&body)?,
        ))
    }

    fn open(&self, challenge: &[u8]) -> Result<Vec<u8>> {
        let (key, id) = self
            .served
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .context("no turn was served")?;
        let answers = match self.serving {
            ToyServing::Cheating => Answers::Changed(Box::new(|opening| {
                opening.stages[0].layers[0].leaves[0].matrices.clear();
            })),
            ToyServing::Honest | ToyServing::WithoutReceipt => Answers::Honest,
        };
        let peer = ToyPeer::new(toy_turn(key, &id), answers);
        let answer = peer.post("open", challenge, Duration::from_secs(60), usize::MAX)?;
        let signature = answer.signature.unwrap_or_default();
        Ok(http(
            answer.status,
            &[
                ("Content-Type", "application/json"),
                (turn::OPENING_SIGNATURE_HEADER, &signature),
            ],
            &answer.body,
        ))
    }
}

/// The toy turn the toy peer serves for the receipt key `key`, under `id`.
fn toy_turn(key: [u8; 32], id: &str) -> Turn {
    let mut turn = Turn::new();
    turn.receipt.model = TOY_MODEL.to_owned();
    turn.receipt.receipt_id = id.to_owned();
    turn.bound(key, turn::GREEDY)
}

fn split_request(request: &[u8]) -> (String, &[u8]) {
    let end = request
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map_or(request.len(), |at| at + 4);
    (
        String::from_utf8_lossy(&request[..end]).into_owned(),
        &request[end..],
    )
}

fn header(head: &str, name: &str) -> Option<String> {
    head.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_owned())
    })
}

fn http(status: u16, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
    let mut head = format!(
        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    let mut answer = head.into_bytes();
    answer.extend_from_slice(body);
    answer
}

/// Where the toy model keeps its head key, for every test of the process.
fn toy_state() -> &'static Path {
    static STATE: OnceLock<PathBuf> = OnceLock::new();
    STATE.get_or_init(|| {
        let directory = tempfile::tempdir().expect("toy model state");
        directory.keep()
    })
}

fn open_toy(_: &HfFile) -> Result<ClaimedSetup> {
    Ok(toy_setup(toy_state()))
}

/// Makes `node` audit with the toy model in place of the claimed model's
/// Hugging Face file.
pub(crate) fn audit_with_the_toy_model(node: &mesh::Node) {
    let _ = node.receipt_audits.claimed_opener.set(open_toy);
}

/// The strikes `node` counts against `peer` now.
pub(crate) fn strikes(node: &mesh::Node, peer: &EndpointId) -> usize {
    node.receipt_audits
        .strikes
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .counting(peer, crate::network::peer_blocks::now_ms())
}

/// Whether `node` audited a receipt `peer` issued for the toy model.
pub(crate) fn audited(node: &mesh::Node, peer: EndpointId) -> bool {
    node.receipt_audits
        .auditability
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .was_audited(peer, TOY_MODEL, crate::network::peer_blocks::now_ms())
}

/// The evidence files `node` kept under `state_dir`.
pub(crate) fn evidence_files(state_dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(state_dir.join("receipt-evidence"))
        .map(|entries| entries.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default()
}
