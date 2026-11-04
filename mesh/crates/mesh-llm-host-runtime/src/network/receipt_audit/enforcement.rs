//! What this node does about a verdict (`verdict`).
//!
//! - **passed:** an info line.
//! - **no audit:** an info line with the reason. Never silent: an operator
//!   who turned audits on can read why a turn was not audited.
//! - **strike:** a warning with the reason, and one strike against the
//!   peer. The strike that reaches the limit blocks the peer for seven
//!   days, and the block's reason lists the strikes.
//! - **failed:** a warning, the evidence kept, and the peer blocked for
//!   seven days, with the failures and the evidence file as the reason.
//!
//! Every line and reason is built of text made safe to print
//! (`peer_text::Shown`), so a peer puts no line of its own into the log.

use std::sync::Mutex;

use iroh::EndpointId;

use super::evidence_store::EvidenceStore;
use super::peer_text::Shown;
use super::strikes::{Count, STRIKE_LIMIT, STRIKE_WINDOW, Strike, Strikes};
use super::verdict::{Proof, Verdict};
use crate::mesh;
use crate::network::peer_blocks::{self, BlockLength, PeerBlocks, Requester, RoutingChoice};

/// The peer and the turn a verdict is about.
pub(super) struct Subject {
    host: EndpointId,
    receipt_id: Shown,
    /// The turn's model: the one its receipt names once that is the one
    /// the request asked for, else the one the request named.
    model: Shown,
}

impl Subject {
    /// The subject of a verdict on `host`'s turn. The id, and the model
    /// when it is out of the receipt, are the peer's text.
    pub(super) fn of(host: EndpointId, receipt_id: &str, model: &str) -> Self {
        Self {
            host,
            receipt_id: Shown::of(receipt_id),
            model: Shown::of(model),
        }
    }

    fn turn(&self) -> String {
        format!("receipt {} from {}", self.receipt_id, self.host.fmt_short())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Level {
    Info,
    Warn,
}

/// What a verdict came to: the lines to log, and the block to publish when
/// the peer was blocked.
#[derive(Default)]
pub(super) struct Concluded {
    pub(super) lines: Vec<(Level, String)>,
    pub(super) blocked: Option<RoutingChoice>,
}

impl Concluded {
    fn say(&mut self, level: Level, line: String) {
        self.lines.push((level, line));
    }

    fn log(&self) {
        for (level, line) in &self.lines {
            match level {
                Level::Info => tracing::info!("{line}"),
                Level::Warn => tracing::warn!("{line}"),
            }
        }
    }
}

/// The state a verdict is acted on with.
pub(super) struct Enforcement<'a> {
    pub(super) strikes: &'a Mutex<Strikes>,
    pub(super) blocks: &'a PeerBlocks,
    /// Where evidence is kept, when the node has a state directory.
    pub(super) evidence: Option<EvidenceStore>,
}

impl Enforcement<'_> {
    /// Acts on `verdict` about `subject` at `now_ms`.
    pub(super) fn conclude(&self, subject: &Subject, verdict: Verdict, now_ms: u64) -> Concluded {
        let mut out = Concluded::default();
        let turn = subject.turn();
        match verdict {
            Verdict::Passed => out.say(Level::Info, format!("{turn}: audit passed")),
            Verdict::NoAudit(reason) => {
                out.say(Level::Info, format!("{turn}: no audit: {reason}"));
            }
            Verdict::Strike(reason) => self.strike(subject, reason, now_ms, &mut out),
            Verdict::Failed { failures, proof } => {
                self.failed(subject, &failures, &proof, now_ms, &mut out);
            }
        }
        out
    }

    fn strike(&self, subject: &Subject, reason: Shown, now_ms: u64, out: &mut Concluded) {
        let strike = Strike {
            at_ms: now_ms,
            receipt_id: subject.receipt_id.as_str().to_owned(),
            reason: reason.as_str().to_owned(),
        };
        let counted = self
            .strikes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record(subject.host, strike);
        let (count, blocking) = match counted {
            Count::Counted(count) => (count, None),
            Count::Limit(strikes) => (strikes.len(), Some(strikes)),
        };
        let hours = STRIKE_WINDOW.as_secs() / 3600;
        out.say(
            Level::Warn,
            format!(
                "{}: strike {count} of {STRIKE_LIMIT} within {hours} h: {reason}",
                subject.turn()
            ),
        );
        let Some(strikes) = blocking else {
            return;
        };
        let listed: Vec<serde_json::Value> = strikes
            .iter()
            .map(|strike| {
                serde_json::json!({
                    "at_ms": strike.at_ms,
                    "receipt_id": strike.receipt_id,
                    "reason": strike.reason,
                })
            })
            .collect();
        let reason = serde_json::json!({
            "receipt_audit": {
                "strikes": listed,
                "strike_limit": STRIKE_LIMIT,
                "strike_window_hours": hours,
            }
        });
        let why = format!("{STRIKE_LIMIT} strikes within {hours} h");
        self.block(subject, reason, &why, now_ms, out);
    }

    fn failed(
        &self,
        subject: &Subject,
        failures: &[Shown],
        proof: &Proof,
        now_ms: u64,
        out: &mut Concluded,
    ) {
        let listed: Vec<&str> = failures.iter().map(Shown::as_str).collect();
        out.say(
            Level::Warn,
            format!("{}: audit failed: {}", subject.turn(), listed.join("; ")),
        );
        let evidence = self.keep(subject, proof, now_ms, out);
        let reason = serde_json::json!({
            "receipt_audit": {
                "receipt_id": subject.receipt_id.as_str(),
                "model": subject.model.as_str(),
                "failures": listed,
                "evidence": evidence,
            }
        });
        self.block(subject, reason, "a receipt audit failed", now_ms, out);
    }

    /// Keeps the proof of a failure, and returns its file. A proof that is
    /// not kept does not stop the block.
    fn keep(
        &self,
        subject: &Subject,
        proof: &Proof,
        now_ms: u64,
        out: &mut Concluded,
    ) -> Option<String> {
        let Some(store) = &self.evidence else {
            out.say(
                Level::Warn,
                format!(
                    "{}: evidence not kept: this node has no state directory of its own",
                    subject.turn()
                ),
            );
            return None;
        };
        let (host, id) = (&subject.host, subject.receipt_id.as_str());
        let kept = match proof {
            Proof::Evidence(evidence) => store.keep(host, id, evidence, now_ms),
            Proof::RefusedReceipt(receipt) => {
                let refused = serde_json::json!({ "refused_receipt": receipt });
                store.keep(host, id, &refused, now_ms)
            }
        };
        match kept {
            Ok(path) => Some(path.display().to_string()),
            Err(error) => {
                // the error names this node's own paths, and no peer text
                out.say(
                    Level::Warn,
                    format!("{}: evidence not kept: {error:#}", subject.turn()),
                );
                None
            }
        }
    }

    fn block(
        &self,
        subject: &Subject,
        reason: serde_json::Value,
        why: &str,
        now_ms: u64,
        out: &mut Concluded,
    ) {
        let host = subject.host.fmt_short();
        let blocked = self.blocks.block(
            &subject.host,
            BlockLength::SevenDays,
            Requester::Receipts,
            Some(reason),
            now_ms,
        );
        match blocked {
            Ok(choice) => {
                out.say(
                    Level::Warn,
                    format!("blocked peer {host} for seven days: {why}"),
                );
                out.blocked = Some(choice);
            }
            Err(error) => out.say(Level::Warn, format!("peer {host} not blocked: {error}")),
        }
    }
}

/// Where `node` keeps evidence: under its identity state directory.
fn evidence_store(node: &mesh::Node) -> Option<EvidenceStore> {
    let state_dir = node
        .identity_state_dir
        .clone()
        .or_else(|| mesh::identity_state_dir().ok());
    EvidenceStore::under(state_dir.as_deref())
}

/// Acts on `verdict` about a peer of `node`, logs it, and tells plugins of
/// a new block. A strike or a failure may block the peer, which writes the
/// block store's file and waits on `runtime`: such a verdict is for a
/// thread that may wait, the audit worker or a blocking task.
pub(super) fn conclude_for(
    node: &mesh::Node,
    subject: &Subject,
    verdict: Verdict,
    runtime: Option<&tokio::runtime::Handle>,
) {
    let enforcement = Enforcement {
        strikes: &node.receipt_audits.strikes,
        blocks: &node.peer_blocks,
        // only a failure keeps evidence
        evidence: match &verdict {
            Verdict::Failed { .. } => evidence_store(node),
            _ => None,
        },
    };
    let concluded = enforcement.conclude(subject, verdict, peer_blocks::now_ms());
    concluded.log();
    if let (Some(choice), Some(runtime)) = (&concluded.blocked, runtime) {
        runtime.block_on(peer_blocks::publish(node, choice));
    }
}
