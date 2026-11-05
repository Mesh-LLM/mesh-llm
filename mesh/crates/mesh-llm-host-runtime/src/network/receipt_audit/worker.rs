//! The audit worker: one thread with a short queue, so audits never take
//! more than one core from serving, and a turn that would wait past the
//! challenge window is dropped as no audit instead.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use skippy_commands::receipts::claimed_file::{self, HfFile};
use skippy_commands::receipts::turn_audit::{
    self, AuditOptions, ClaimedSetup, ReceivedTurn, TurnAudit,
};
use skippy_commands::receipts::{AuditOutput, ChatReply, load_claimed_model};
use skippy_receipts::evidence::Evidence;
use skippy_receipts::wire::Receipt;

use super::collector::Reply;
use super::opener::TunnelOpener;
use super::{AUDIT_RATE, WatchedTurn};
use crate::network::peer_blocks::{self, BlockLength, Requester};

/// Turns waiting for the worker; a full queue drops the newest.
const QUEUE: usize = 2;
/// A challenge sent later than this after the receipt may find the
/// node's replay state gone.
const CHALLENGE_WINDOW: Duration = Duration::from_secs(10);
/// Claimed models kept open between audits.
const KEPT_MODELS: usize = 2;
/// The most failures a block's reason lists, and the most of each.
const REASON_FAILURES: usize = 8;
const REASON_CHARS: usize = 240;

struct Job {
    turn: WatchedTurn,
    reply: Reply,
    receipt: Receipt,
    received: Instant,
    runtime: tokio::runtime::Handle,
}

/// Queues the turn for an audit when the VRF of its root draws it.
pub(super) fn submit(turn: WatchedTurn, mut reply: Reply) {
    let Some(receipt) = reply
        .receipt
        .take()
        .and_then(|r| serde_json::from_value::<Receipt>(r).ok())
    else {
        return;
    };
    if !turn_audit::audit_drawn(&receipt, &turn.key, AUDIT_RATE) {
        return;
    }
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let job = Job {
        turn,
        reply,
        receipt,
        received: Instant::now(),
        runtime,
    };
    let Some(queue) = queue() else {
        return;
    };
    if let Err(TrySendError::Full(job)) = queue.try_send(job) {
        tracing::info!(
            "receipt {}: no audit, the audit worker is busy",
            job.receipt.receipt_id
        );
    }
}

fn queue() -> Option<&'static SyncSender<Job>> {
    static QUEUE_TX: OnceLock<Option<SyncSender<Job>>> = OnceLock::new();
    QUEUE_TX
        .get_or_init(|| {
            let (tx, rx) = sync_channel(QUEUE);
            std::thread::Builder::new()
                .name("receipt-audit".into())
                .spawn(move || run(&rx))
                .map_err(|error| tracing::warn!("receipt audits off: no worker thread: {error}"))
                .ok()
                .map(|_| tx)
        })
        .as_ref()
}

fn run(jobs: &Receiver<Job>) {
    let mut models: Vec<(String, ClaimedSetup)> = Vec::new();
    for job in jobs {
        // a panic in one audit must not end the audits after it
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| audit(&job, &mut models)))
                .unwrap_or_else(|_| Outcome::NoAudit("the audit panicked".into()));
        report(&job, outcome);
    }
}

/// What an audit says about the peer.
enum Outcome {
    /// The audit did not run, or a check could not: nothing about the peer.
    NoAudit(String),
    Passed,
    Failed {
        failures: Vec<String>,
        evidence: Option<Box<Evidence>>,
    },
}

fn audit(job: &Job, models: &mut Vec<(String, ClaimedSetup)>) -> Outcome {
    if job.received.elapsed() > CHALLENGE_WINDOW {
        return Outcome::NoAudit("it waited past the challenge window".into());
    }
    let setup = match claimed_model(&job.receipt.model, models) {
        Ok(setup) => setup,
        Err(error) => return Outcome::NoAudit(format!("{error:#}")),
    };
    let opener = TunnelOpener {
        node: job.turn.node.clone(),
        host: job.turn.host,
        model: job.turn.model.clone(),
        node_key: job.receipt.binding.as_ref().map(|b| b.node_key.clone()),
        runtime: job.runtime.clone(),
    };
    let reply = ChatReply {
        model: job.turn.model.clone(),
        content: job.reply.content.clone(),
        reasoning: job.reply.reasoning.clone(),
    };
    let prompt = job.turn.prompt.clone().unwrap_or_default();
    // text a stop string, a tool call or a format shaped is not the
    // sampled tokens' text
    let plain = !job.turn.not_plain_text && !job.reply.not_plain_text;
    let received = ReceivedTurn {
        receipt: job.receipt.clone(),
        key: &job.turn.key,
        response: plain.then_some((prompt.as_str(), &reply)),
        asked_greedy: false,
    };
    match turn_audit::audit_turn(setup, received, &opener, &AuditOptions::default()) {
        Err(error) => Outcome::NoAudit(format!("{error:#}")),
        Ok(TurnAudit::NoAudit(output)) => Outcome::NoAudit(output.reason),
        Ok(TurnAudit::Audited { output, evidence }) => judge(&output, evidence),
    }
}

/// A check that ran and failed fails the audit; one that could not run
/// here makes it no audit.
fn judge(output: &AuditOutput, evidence: Option<Box<Evidence>>) -> Outcome {
    let failures = output.failures();
    if !failures.is_empty() {
        return Outcome::Failed { failures, evidence };
    }
    let unchecked = output.unchecked();
    match unchecked.is_empty() {
        true => Outcome::Passed,
        false => Outcome::NoAudit(format!("checks could not run: {}", unchecked.join("; "))),
    }
}

fn report(job: &Job, outcome: Outcome) {
    let turn = format!(
        "receipt {} from {}",
        job.receipt.receipt_id,
        job.turn.host.fmt_short()
    );
    let message = match &outcome {
        Outcome::NoAudit(reason) => format!("no audit: {reason}"),
        Outcome::Passed => "audit passed".to_owned(),
        Outcome::Failed { failures, .. } => format!("audit failed: {}", failures.join("; ")),
    };
    let Outcome::Failed { failures, evidence } = outcome else {
        return tracing::info!("{turn}: {message}");
    };
    tracing::warn!("{turn}: {message}");
    block_peer(job, &failures, evidence.as_deref());
}

fn claimed_model<'a>(
    name: &str,
    models: &'a mut Vec<(String, ClaimedSetup)>,
) -> Result<&'a ClaimedSetup> {
    if let Some(at) = models.iter().position(|(n, _)| n == name) {
        return Ok(&models[at].1);
    }
    let file = HfFile::parse(name)
        .with_context(|| format!("the receipt names {name}, not one Hugging Face file"))?;
    let claimed = claimed_file::open_remote(&file, &claimed_file::default_cache_root())?;
    let setup = ClaimedSetup::from_file(claimed, &load_claimed_model)?;
    // a node that cannot load the model here cannot check it
    setup
        .model
        .detokenize(&[])
        .context("load the claimed model here")?;
    if models.len() >= KEPT_MODELS {
        models.remove(0);
    }
    models.push((name.to_owned(), setup));
    Ok(&models.last().expect("just pushed").1)
}

/// Blocks the peer for seven days, with the failures and the kept
/// evidence as the reason.
fn block_peer(job: &Job, failures: &[String], evidence: Option<&Evidence>) {
    let id = &job.receipt.receipt_id;
    let evidence = evidence.and_then(|evidence| {
        save_evidence(id, evidence)
            .map_err(|error| tracing::warn!("receipt {id}: evidence not kept: {error:#}"))
            .ok()
    });
    let failures: Vec<String> = failures
        .iter()
        .take(REASON_FAILURES)
        .map(|f| f.chars().take(REASON_CHARS).collect())
        .collect();
    let reason = serde_json::json!({
        "receipt_audit": {
            "receipt_id": id,
            "model": job.receipt.model,
            "failures": failures,
            "evidence": evidence,
        }
    });
    let node = &job.turn.node;
    let blocked = node.peer_blocks.block(
        &job.turn.host,
        BlockLength::SevenDays,
        Requester::Receipts,
        Some(reason),
        peer_blocks::now_ms(),
    );
    let host = job.turn.host.fmt_short().to_string();
    match blocked {
        Ok(choice) => {
            tracing::warn!("blocked peer {host} for seven days: a receipt audit failed");
            job.runtime.block_on(peer_blocks::publish(node, &choice));
        }
        Err(error) => tracing::warn!("peer {host} not blocked: {error}"),
    }
}

/// Keeps the evidence of a failed audit in the node's state directory, for
/// `skippy recheck-receipt --evidence`.
fn save_evidence(receipt_id: &str, evidence: &Evidence) -> Result<PathBuf> {
    let directory = crate::mesh::identity_state_dir()?.join("receipt-evidence");
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create {}", directory.display()))?;
    let name: String = receipt_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    let path = directory.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(evidence)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}
