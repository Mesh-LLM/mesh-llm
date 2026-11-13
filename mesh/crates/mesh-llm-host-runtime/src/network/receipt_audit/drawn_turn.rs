//! The audit of one turn the VRF drew: open the claimed model, challenge
//! the peer within the window, check its opening, and say what that came
//! to ([`Verdict`]).
//!
//! By the time a turn is here, its receipt is the peer's own, bound to
//! this request's key and id, and names the file the request asked for
//! (`intake`). What can still go wrong is told apart by whose doing it is:
//! this node's (the claimed model does not open, the window passed while it
//! opened) is no audit; the peer's is a strike or a failure.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use skippy_receipt_audit::chat_reply::ChatReply;
use skippy_receipt_audit::claimed_file::{HfFile, NotOnBranch};
use skippy_receipt_audit::claimed_model::ClaimedModel;
use skippy_receipt_audit::turn_audit::{self, AuditOptions, ClaimedSetup, ReceivedTurn};
use skippy_receipts::coverage;
use skippy_receipts::wire::Receipt;

use super::answer_limits::AnswerLimits;
use super::claimed_models::{self, ClaimedModels, MODEL_IDLE};
use super::collector::Reply;
use super::enforcement::Subject;
use super::opener::{
    CHALLENGE_WINDOW, Late, OPEN_WAIT, ROWS_WAIT, Transport, TunnelTransport, WindowedOpener,
};
use super::peer_text::shown_error;
use super::verdict::Verdict;
use super::worker::Runner;
use super::{Asked, WatchedTurn};
use crate::network::peer_blocks::now_ms;

/// The longest the peer can make one audit hold the worker: it is busy
/// until the end of the window, answers the challenge sent then at the
/// end of its wait, and the rows at the end of theirs. Time past it is
/// this node's own: opening the claimed model, which reads it from
/// Hugging Face the first time, and checking the turn. Neither has a
/// bound here, so an audit that takes longer says so ([`held_too_long`]).
const PEER_HOLD: Duration = CHALLENGE_WINDOW
    .saturating_add(OPEN_WAIT)
    .saturating_add(ROWS_WAIT);

/// What the log says of an audit that held the worker for `held`, when
/// that is past what the peer can make it take ([`PEER_HOLD`]); turns
/// drawn meanwhile were refused as busy.
fn held_too_long(held: Duration) -> Option<String> {
    (held > PEER_HOLD).then(|| {
        format!(
            "receipt audit: one audit held the audit worker for {} s, past the {} s a peer can make it take; turns drawn meanwhile were not audited (opening a claimed model the first time reads it from Hugging Face)",
            held.as_secs(),
            PEER_HOLD.as_secs()
        )
    })
}

/// A drawn turn, as the audit needs it.
pub(super) struct DrawnTurn<'a> {
    pub(super) asked: &'a Asked,
    pub(super) reply: &'a Reply,
    pub(super) receipt: &'a Receipt,
    /// The claimed file: the one the request's model name stands for, at
    /// the commit the receipt names.
    pub(super) file: &'a HfFile,
    /// When the request named no commit, the branch the commit must be on.
    pub(super) branch: Option<&'a str>,
    /// When the response, and the receipt with it, arrived.
    pub(super) received: Instant,
}

/// The text of the tokens the receipt says were sampled, without the
/// tokens that end generation, as the text check reads it.
fn sampled_text(receipt: &Receipt, model: &dyn ClaimedModel) -> Result<String> {
    let mut sampled = receipt.sampled_tokens();
    while let Some(&last) = sampled.last() {
        if !model.is_end_of_generation(last)? {
            break;
        }
        sampled.pop();
    }
    model.detokenize(&sampled)
}

/// Why the answered text need not be the sampled tokens' text although the
/// node is honest, when it need not: the request or the answer says so, or
/// the sampled text holds markers a node's output parser removes that the
/// text check does not know. It knows `<think>` and `</think>`. The text
/// check then does not run.
fn text_unavailable(turn: &DrawnTurn<'_>, setup: &ClaimedSetup) -> Option<&'static str> {
    // an answer of another shape reaches an audit only when its request
    // allowed it (`intake::contradiction`)
    let known = turn
        .asked
        .text_unavailable
        .or(turn.reply.text_unavailable)
        .or(turn.reply.shape.and(turn.asked.shaped_answer));
    if known.is_some() {
        return known;
    }
    let Ok(sampled) = sampled_text(turn.receipt, setup.model.as_ref()) else {
        return Some("the sampled tokens' text could not be read here");
    };
    let thinks = sampled.contains("<think>") || sampled.contains("</think>");
    if sampled.contains("<|") {
        // a channel or turn marker, which a parser takes out of the answer
        Some("the sampled tokens hold control markers")
    } else if !turn.reply.reasoning.is_empty() && !thinks {
        // the reasoning was cut out of the sampled text by other markers
        Some("the model reasons in a format the text check does not know")
    } else {
        None
    }
}

/// Audits `turn` against `setup`, the claimed model, already open.
fn audit_against(
    turn: &DrawnTurn<'_>,
    setup: &ClaimedSetup,
    transport: &dyn Transport,
    clock: &dyn Fn() -> Instant,
) -> Verdict {
    // what the receipt says this node cannot audit, before anything is
    // asked: told apart from a plain no audit (`auditability`)
    let runtime = setup.model.runtime();
    if let Some(why) = coverage::unauditable(turn.receipt, &setup.weights, &runtime) {
        return Verdict::unauditable(why, &turn.asked.model);
    }
    let limits = match AnswerLimits::of(&setup.weights, turn.receipt) {
        Ok(limits) => limits,
        Err(error) => return Verdict::no_audit(shown_error(&error)),
    };
    let asked = turn.asked;
    let opener = WindowedOpener::new(transport, &asked.model, &limits, turn.received, clock);
    let reply = ChatReply {
        model: asked.model.clone(),
        content: turn.reply.content.clone(),
        reasoning: turn.reply.reasoning.clone(),
        cut_short: turn.reply.cut_short,
    };
    let unavailable = text_unavailable(turn, setup);
    if let Some(why) = unavailable {
        tracing::debug!("receipt audit: the text is not checked: {why}");
    }
    // a prompt this node does not check is found in any text
    let prompt = asked.prompt.as_deref().unwrap_or_default();
    let received = ReceivedTurn {
        receipt: turn.receipt.clone(),
        key: &asked.key,
        response: unavailable.is_none().then_some((prompt, &reply)),
        sampling: asked.sampling.clone(),
        // the intake held the receipt to an id this node asked under
        asked_id: Some(turn.receipt.receipt_id.as_str()),
    };
    let audit = turn_audit::audit_turn(setup, received, &opener, &AuditOptions::default());
    Verdict::of(audit, opener.late())
}

/// Audits `turn`: opens its claimed model with `models`, then challenges
/// the peer over `transport`.
pub(super) fn audit<O: Fn(&HfFile) -> Result<ClaimedSetup>>(
    turn: &DrawnTurn<'_>,
    models: &mut ClaimedModels<O>,
    transport: &dyn Transport,
    clock: &dyn Fn() -> Instant,
) -> Verdict {
    // a turn that waited in the queue past the window is not worth opening
    // a model for
    let waited = clock().saturating_duration_since(turn.received);
    if waited > CHALLENGE_WINDOW {
        return Verdict::no_audit(Late::Challenge(waited).to_string());
    }
    // opening the model may take long the first time; the window is
    // checked again when the challenge is sent
    let setup = match models.get(turn.file, turn.branch, clock()) {
        Ok(setup) => setup,
        // the peer served a commit that is not of the model asked for
        Err(error) if NotOnBranch::caused(&error) => return Verdict::strike(shown_error(&error)),
        Err(error) => {
            return Verdict::no_audit(format!(
                "this node could not open the claimed model: {}",
                shown_error(&error)
            ));
        }
    };
    audit_against(turn, setup, transport, clock)
}

/// A drawn turn waiting for the audit worker.
pub(super) struct Job {
    pub(super) turn: WatchedTurn,
    pub(super) reply: Reply,
    pub(super) receipt: Receipt,
    pub(super) file: HfFile,
    pub(super) branch: Option<String>,
    pub(super) received: Instant,
    pub(super) runtime: tokio::runtime::Handle,
}

impl Job {
    pub(super) fn subject(&self) -> Subject {
        Subject::of(
            self.turn.asked.host,
            &self.receipt.receipt_id,
            &self.receipt.model,
        )
    }
}

/// Runs the audits of a node on its worker thread, against the claimed
/// models it keeps open there.
pub(super) struct AuditRunner {
    models: ClaimedModels<claimed_models::Opener>,
    /// The node is shutting down: turns still queued are not audited.
    stopping: Arc<AtomicBool>,
}

impl AuditRunner {
    /// A runner that opens claimed models with `open`.
    pub(super) fn new(stopping: Arc<AtomicBool>, open: claimed_models::Opener) -> Self {
        Self {
            models: ClaimedModels::opened_by(open),
            stopping,
        }
    }

    /// Audits `turn`; a panic in the audit is this node's fault, so its
    /// turn is no audit, and the models it may have left half-made are
    /// released.
    fn audit_contained(&mut self, turn: &DrawnTurn<'_>, transport: &dyn Transport) -> Verdict {
        let audited = catch_unwind(AssertUnwindSafe(|| {
            audit(turn, &mut self.models, transport, &Instant::now)
        }));
        audited.unwrap_or_else(|_| {
            self.models.release_all();
            if !skippy_runtime::native_runtime_loaded() {
                // a `client` started without `[receipts] audit` or `require`
                tracing::error!(
                    "receipt audit: the turn is not audited: this node has no native runtime loaded, which an audit runs on; set [receipts] audit = true so it loads at startup"
                );
                return Verdict::no_audit("this node has no native runtime loaded");
            }
            tracing::error!("receipt audit: the audit of a turn panicked; the turn is not audited");
            Verdict::no_audit("the audit panicked on this node")
        })
    }
}

impl Runner for AuditRunner {
    type Job = Job;

    fn run(&mut self, job: Job) {
        let subject = job.subject();
        let verdict = if self.stopping.load(Ordering::Acquire) {
            Verdict::no_audit("this node is shutting down")
        } else {
            let transport = TunnelTransport {
                node: job.turn.node.clone(),
                host: job.turn.asked.host,
                runtime: job.runtime.clone(),
            };
            let turn = DrawnTurn {
                asked: &job.turn.asked,
                reply: &job.reply,
                receipt: &job.receipt,
                file: &job.file,
                branch: job.branch.as_deref(),
                received: job.received,
            };
            let started = Instant::now();
            let verdict = self.audit_contained(&turn, &transport);
            if let Some(warning) = held_too_long(started.elapsed()) {
                tracing::warn!("{warning}");
            }
            verdict
        };
        if matches!(
            verdict,
            Verdict::Passed | Verdict::Failed { .. } | Verdict::Strike(_)
        ) {
            // the audit ran past what the receipt says: the peer is auditable
            job.turn
                .node
                .receipt_audits
                .auditability
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .audited(job.turn.asked.host, &job.turn.asked.model, now_ms());
        }
        super::enforcement::conclude_for(&job.turn.node, &subject, verdict, Some(&job.runtime));
    }

    fn idle(&mut self) {
        self.models.release_idle(Instant::now(), MODEL_IDLE);
    }

    fn recover(&mut self) {
        // a model a panic interrupted may be half-loaded
        self.models.release_all();
    }
}

#[cfg(test)]
mod tests {
    use super::{Duration, PEER_HOLD, held_too_long};

    #[test]
    fn an_audit_that_holds_the_worker_past_what_a_peer_can_make_it_take_says_so() {
        assert_eq!(PEER_HOLD, Duration::from_secs(35));
        assert_eq!(held_too_long(PEER_HOLD), None);
        let warning = held_too_long(Duration::from_secs(90)).expect("said");
        assert!(warning.contains("90 s, past the 35 s"), "{warning}");
    }
}
