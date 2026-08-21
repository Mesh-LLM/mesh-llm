//! Bounded dispatch for one native serving plugin's callback threads.
//!
//! Lifecycle and proposal callbacks share one ordered queue so every proposal
//! observes the committed state that preceded it. Passive terminal callbacks
//! (`report` / `discard`) run on their own worker so they can never consume a
//! proposal's deadline. Proposal dispatch fences that passive queue first, so
//! commits -> report -> next proposal remains causal.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_server::frontend::{
    GenerationAbort, GenerationCommit, GenerationReceipt, GenerationStart, LinearProposal,
    LinearProposalDiscardReason, LinearProposalQuery, LinearProposalReceipt,
    LinearProposalSourceOutcome, LinearProposalSourceTelemetry,
};

use crate::ActivePlugin;

/// Commands a caller may enqueue for either worker.
const PLUGIN_COMMAND_CAPACITY: usize = 1_024;
/// Headroom above [`PLUGIN_COMMAND_CAPACITY`] kept exclusively for terminal
/// dispositions, so a deadline discard is never lost to a full queue.
const PLUGIN_TERMINAL_RESERVE: usize = 64;
/// Bound on how long a drop waits for a worker to observe its closed queue.
const CLEAN_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) enum PluginCommand {
    Begin(GenerationStart),
    Committed(GenerationCommit),
    Abort(GenerationAbort),
    Finish(GenerationReceipt),
    Proposal(LinearProposalQuery, SyncSender<ProposalResponse>),
    /// Primary-queue handoff for a report. The primary worker enqueues the
    /// actual passive report only after all earlier lifecycle commits, then
    /// waits for the passive callback before it accepts later primary work.
    ReportHandoff(LinearProposalReceipt),
    Report(LinearProposalReceipt, SyncSender<Result<()>>),
    Discard(Vec<u8>, LinearProposalDiscardReason),
    /// A passive-queue barrier. The primary worker waits for this barrier
    /// before invoking a proposal or finalizing a generation, so all earlier
    /// reports/discards have completed before the plugin sees the next
    /// callback.
    Fence(SyncSender<()>),
}

pub(crate) struct ProposalResponse {
    pub(crate) proposal: std::result::Result<Option<LinearProposal>, String>,
    pub(crate) telemetry: LinearProposalSourceTelemetry,
}

struct QueuedPluginCommand {
    enqueued_at: Instant,
    command: PluginCommand,
}

struct QueueState {
    commands: VecDeque<QueuedPluginCommand>,
    closed: bool,
}

/// Bounded, closable command queue owned by exactly one worker thread.
pub(crate) struct PluginCommandQueue {
    state: Mutex<QueueState>,
    available: Condvar,
}

#[derive(Debug)]
pub(crate) enum PluginCommandQueueError {
    Full,
    Stopped,
    Poisoned,
}

impl PluginCommandQueue {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                commands: VecDeque::with_capacity(PLUGIN_COMMAND_CAPACITY),
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    fn enqueue(&self, command: PluginCommand) -> Result<()> {
        self.try_enqueue(command).map_err(|error| match error {
            PluginCommandQueueError::Full => anyhow!("native serving plugin command queue is full"),
            PluginCommandQueueError::Stopped => anyhow!("native serving plugin worker stopped"),
            PluginCommandQueueError::Poisoned => {
                anyhow!("native serving plugin command queue lock poisoned")
            }
        })
    }

    pub(crate) fn try_enqueue(
        &self,
        command: PluginCommand,
    ) -> std::result::Result<(), PluginCommandQueueError> {
        self.enqueue_within(command, PLUGIN_COMMAND_CAPACITY)
    }

    /// Enqueues a terminal disposition using the reserved headroom.
    ///
    /// A late candidate is withheld from decode, so its `discard` is the only
    /// remaining way the plugin can resolve that decision ID. It must not be
    /// dropped just because ordinary traffic filled the queue.
    fn try_enqueue_terminal(
        &self,
        command: PluginCommand,
    ) -> std::result::Result<(), PluginCommandQueueError> {
        let capacity = if matches!(command, PluginCommand::Fence(_)) {
            PLUGIN_COMMAND_CAPACITY + PLUGIN_TERMINAL_RESERVE
        } else {
            // Keep one slot available for a fence after ordinary terminal
            // callbacks fill the reserved headroom.
            PLUGIN_COMMAND_CAPACITY + PLUGIN_TERMINAL_RESERVE - 1
        };
        self.enqueue_within(command, capacity)
    }

    fn enqueue_within(
        &self,
        command: PluginCommand,
        capacity: usize,
    ) -> std::result::Result<(), PluginCommandQueueError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PluginCommandQueueError::Poisoned)?;
        if state.closed {
            return Err(PluginCommandQueueError::Stopped);
        }
        if state.commands.len() >= capacity {
            return Err(PluginCommandQueueError::Full);
        }
        state.commands.push_back(QueuedPluginCommand {
            enqueued_at: Instant::now(),
            command,
        });
        self.available.notify_one();
        Ok(())
    }

    /// Refuses new work and wakes the worker so it can drain and exit.
    ///
    /// Closing does not require queue capacity, so a full queue can still be
    /// shut down cleanly.
    pub(crate) fn close(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.closed = true;
        }
        self.available.notify_all();
    }

    /// Returns the next command, or `None` once the queue is closed and drained.
    fn next(&self) -> Option<QueuedPluginCommand> {
        let mut state = self
            .state
            .lock()
            .expect("native serving plugin command queue lock must not be poisoned");
        loop {
            // Lifecycle and proposal callbacks share this FIFO so every
            // proposal observes its earlier committed state. Passive callbacks
            // use their own worker and cannot delay either class.
            if let Some(command) = state.commands.pop_front() {
                return Some(command);
            }
            if state.closed {
                return None;
            }
            state = self
                .available
                .wait(state)
                .expect("native serving plugin command queue lock must not be poisoned");
        }
    }
}

/// One-shot notification that a worker thread has left its loop.
struct WorkerExit {
    exited: Mutex<bool>,
    signal: Condvar,
}

impl WorkerExit {
    fn new() -> Self {
        Self {
            exited: Mutex::new(false),
            signal: Condvar::new(),
        }
    }

    fn mark_exited(&self) {
        if let Ok(mut exited) = self.exited.lock() {
            *exited = true;
        }
        self.signal.notify_all();
    }

    fn wait_for_exit(&self, timeout: Duration) -> bool {
        let Ok(exited) = self.exited.lock() else {
            return false;
        };
        self.signal
            .wait_timeout_while(exited, timeout, |exited| !*exited)
            .is_ok_and(|(exited, _)| *exited)
    }
}

/// Closes the queue and signals exit even when a callback unwinds.
struct WorkerStopGuard {
    queue: Arc<PluginCommandQueue>,
    exit: Arc<WorkerExit>,
}

impl Drop for WorkerStopGuard {
    fn drop(&mut self) {
        self.queue.close();
        self.exit.mark_exited();
    }
}

pub(crate) struct PluginDriver {
    pub(crate) queue: Arc<PluginCommandQueue>,
    passive_queue: Arc<PluginCommandQueue>,
    active: Arc<ActivePlugin>,
    pub(crate) fatal_error: Arc<Mutex<Option<String>>>,
    lifecycle_delivery_failures: Arc<AtomicU64>,
    report_delivery_failures: Arc<AtomicU64>,
    worker: WorkerHandle,
    passive_worker: WorkerHandle,
}

struct WorkerHandle {
    exit: Arc<WorkerExit>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl PluginDriver {
    pub(crate) fn spawn(active: ActivePlugin) -> Result<Self> {
        let queue = Arc::new(PluginCommandQueue::new());
        let passive_queue = Arc::new(PluginCommandQueue::new());
        let active = Arc::new(active);
        let fatal_error = Arc::new(Mutex::new(None));
        let lifecycle_delivery_failures = Arc::new(AtomicU64::new(0));
        let report_delivery_failures = Arc::new(AtomicU64::new(0));
        let exit = Arc::new(WorkerExit::new());
        let passive_exit = Arc::new(WorkerExit::new());

        let worker_queue = Arc::clone(&queue);
        let worker_passive_queue = Arc::clone(&passive_queue);
        let worker_active = Arc::clone(&active);
        let worker_failures = Arc::clone(&lifecycle_delivery_failures);
        let worker_report_failures = Arc::clone(&report_delivery_failures);
        let worker_exit = Arc::clone(&exit);
        let handle = thread::Builder::new()
            .name("mesh-native-serving-plugin".to_string())
            .spawn(move || {
                plugin_worker(
                    worker_active,
                    worker_queue,
                    worker_passive_queue,
                    worker_failures,
                    worker_report_failures,
                    worker_exit,
                );
            })
            .context("spawn native serving plugin worker")?;

        let passive_worker_queue = Arc::clone(&passive_queue);
        let passive_worker_active = Arc::clone(&active);
        let passive_worker_exit = Arc::clone(&passive_exit);
        let passive_handle = thread::Builder::new()
            .name("mesh-native-serving-plugin-passive".to_string())
            .spawn(move || {
                plugin_passive_worker(
                    passive_worker_active,
                    passive_worker_queue,
                    passive_worker_exit,
                );
            })
            .context("spawn native serving plugin passive worker")?;

        Ok(Self {
            queue,
            passive_queue,
            active,
            fatal_error,
            lifecycle_delivery_failures,
            report_delivery_failures,
            worker: WorkerHandle {
                exit,
                handle: Mutex::new(Some(handle)),
            },
            passive_worker: WorkerHandle {
                exit: passive_exit,
                handle: Mutex::new(Some(passive_handle)),
            },
        })
    }

    pub(crate) fn ensure_healthy(&self) -> Result<()> {
        let error = self
            .fatal_error
            .lock()
            .map_err(|_| anyhow!("native serving plugin health lock poisoned"))?;
        if let Some(error) = error.as_deref() {
            bail!("native serving plugin worker failed: {error}");
        }
        Ok(())
    }

    pub(crate) fn enqueue(&self, command: PluginCommand) -> Result<()> {
        self.ensure_healthy()?;
        self.enqueue_recovery(command)
    }

    /// Deliver lifecycle cleanup even after an earlier callback failed.
    ///
    /// This bypasses only the health gate. The command still uses its bounded
    /// callback queue and fails if that worker has stopped or is full.
    pub(crate) fn enqueue_recovery(&self, command: PluginCommand) -> Result<()> {
        self.queue_for(&command).enqueue(command)
    }

    /// Enqueues a terminal proposal disposition using reserved capacity.
    pub(crate) fn enqueue_terminal(&self, command: PluginCommand) -> Result<()> {
        self.ensure_healthy()?;
        self.queue_for(&command)
            .try_enqueue_terminal(command)
            .map_err(|error| match error {
                PluginCommandQueueError::Full => {
                    anyhow!("native serving plugin terminal queue is full")
                }
                PluginCommandQueueError::Stopped => {
                    anyhow!("native serving plugin passive worker stopped")
                }
                PluginCommandQueueError::Poisoned => {
                    anyhow!("native serving plugin terminal queue lock poisoned")
                }
            })
    }

    fn queue_for(&self, command: &PluginCommand) -> &Arc<PluginCommandQueue> {
        if matches!(
            command,
            PluginCommand::Report(_, _) | PluginCommand::Discard(_, _) | PluginCommand::Fence(_)
        ) {
            &self.passive_queue
        } else {
            &self.queue
        }
    }

    pub(crate) fn lifecycle_delivery_failures(&self) -> u64 {
        self.lifecycle_delivery_failures.load(Ordering::Relaxed)
    }

    pub(crate) fn report_delivery_failures(&self) -> u64 {
        self.report_delivery_failures.load(Ordering::Relaxed)
    }

    pub(crate) fn propose(&self, query: LinearProposalQuery) -> Result<ProposalResponse> {
        self.ensure_healthy()?;
        let deadline = query.deadline;
        let submitted_at = Instant::now();
        if submitted_at >= deadline {
            return Ok(abstention(
                0,
                LinearProposalSourceOutcome::HostDeadlineExceeded,
            ));
        }
        // A proposal is accepted only when the caller receives it. A buffered
        // reply could be written after recv_timeout returned, then orphaned
        // when the receiver was dropped without giving the worker a chance to
        // discard the candidate.
        let (reply, response) = sync_channel(0);
        match self
            .queue
            .try_enqueue(PluginCommand::Proposal(query, reply))
        {
            Ok(()) => {}
            Err(PluginCommandQueueError::Full) => {
                return Ok(abstention(0, LinearProposalSourceOutcome::QueueFull));
            }
            Err(PluginCommandQueueError::Stopped) => {
                bail!("native serving plugin worker stopped before accepting proposal")
            }
            Err(PluginCommandQueueError::Poisoned) => {
                bail!("native serving plugin command queue lock poisoned")
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(abstention(
                elapsed_us(submitted_at),
                LinearProposalSourceOutcome::HostDeadlineExceeded,
            ));
        }
        match response.recv_timeout(remaining) {
            Ok(result) => Ok(result),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(abstention(
                elapsed_us(submitted_at),
                LinearProposalSourceOutcome::HostDeadlineExceeded,
            )),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                bail!("native serving plugin worker stopped before replying")
            }
        }
    }
}

fn abstention(queue_wait_us: u64, outcome: LinearProposalSourceOutcome) -> ProposalResponse {
    ProposalResponse {
        proposal: Ok(None),
        telemetry: LinearProposalSourceTelemetry {
            queue_wait_us,
            callback_elapsed_us: 0,
            outcome,
        },
    }
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl Drop for PluginDriver {
    fn drop(&mut self) {
        stop_worker(&self.queue, &self.worker, "callback");
        stop_worker(&self.passive_queue, &self.passive_worker, "passive");
        if let Some(active) = Arc::get_mut(&mut self.active)
            && let Err(error) = active.shutdown()
        {
            eprintln!("native serving plugin shutdown failed: {error:#}");
        }
    }
}

/// Closes a worker's queue and joins it once it has actually left its loop.
///
/// Closing never needs queue capacity, and the exit signal is set by the
/// worker's stop guard, so neither a full queue nor the gap between the last
/// callback and thread teardown can skip the join.
fn stop_worker(queue: &Arc<PluginCommandQueue>, worker: &WorkerHandle, label: &str) {
    queue.close();
    if !worker.exit.wait_for_exit(CLEAN_SHUTDOWN_TIMEOUT) {
        eprintln!(
            "native serving plugin {label} worker did not stop within {CLEAN_SHUTDOWN_TIMEOUT:?}; \
             deferring plugin shutdown to that thread"
        );
        return;
    }
    if let Ok(mut handle) = worker.handle.lock()
        && let Some(handle) = handle.take()
    {
        let _ = handle.join();
    }
}

fn plugin_worker(
    active: Arc<ActivePlugin>,
    queue: Arc<PluginCommandQueue>,
    passive_queue: Arc<PluginCommandQueue>,
    lifecycle_delivery_failures: Arc<AtomicU64>,
    report_delivery_failures: Arc<AtomicU64>,
    exit: Arc<WorkerExit>,
) {
    let _stop_guard = WorkerStopGuard {
        queue: Arc::clone(&queue),
        exit,
    };
    while let Some(QueuedPluginCommand {
        enqueued_at,
        command,
    }) = queue.next()
    {
        let (result, lifecycle) = match command {
            PluginCommand::Begin(event) => (active.begin(&event), true),
            PluginCommand::Committed(event) => (active.committed(&event), true),
            PluginCommand::Abort(event) => (active.abort(&event), true),
            PluginCommand::Finish(event) => (
                finish_after_passive_fence(&passive_queue, || active.finish(&event)),
                true,
            ),
            PluginCommand::ReportHandoff(event) => {
                let result = report_after_passive_handoff(&passive_queue, event);
                (result, false)
            }
            PluginCommand::Proposal(query, reply) => {
                run_proposal(&active, &passive_queue, enqueued_at, query, &reply);
                continue;
            }
            PluginCommand::Report(_, _)
            | PluginCommand::Discard(_, _)
            | PluginCommand::Fence(_) => {
                unreachable!("passive plugin callbacks must use the passive worker queue")
            }
        };
        if let Err(error) = result {
            if lifecycle {
                lifecycle_delivery_failures.fetch_add(1, Ordering::Relaxed);
            } else {
                report_delivery_failures.fetch_add(1, Ordering::Relaxed);
                eprintln!("native serving plugin report handoff failed: {error:#}");
            }
        }
    }
}

fn run_proposal(
    active: &ActivePlugin,
    passive_queue: &PluginCommandQueue,
    enqueued_at: Instant,
    query: LinearProposalQuery,
    reply: &SyncSender<ProposalResponse>,
) {
    let deadline = query.deadline;
    let queue_wait_us = elapsed_us(enqueued_at);
    if Instant::now() >= deadline {
        send_proposal_response(
            passive_queue,
            reply,
            abstention(
                queue_wait_us,
                LinearProposalSourceOutcome::DeadlineExceededBeforeDispatch,
            ),
        );
        return;
    }

    // Reports and discards run on the passive worker so they cannot consume
    // proposal callback time. This fence serializes the passive terminal
    // callbacks with the primary proposal queue. The wait is not abandoned;
    // after it completes, the original absolute deadline is checked again.
    let remaining = deadline.saturating_duration_since(Instant::now());
    let fence_timeout = remaining.min(CLEAN_SHUTDOWN_TIMEOUT);
    let fence_consumes_query_deadline = remaining <= CLEAN_SHUTDOWN_TIMEOUT;
    if let Err(error) = fence_passive(passive_queue, fence_timeout) {
        if fence_consumes_query_deadline && matches!(&error, PassiveFenceError::Timeout(_)) {
            send_proposal_response(
                passive_queue,
                reply,
                abstention(
                    queue_wait_us,
                    LinearProposalSourceOutcome::HostDeadlineExceeded,
                ),
            );
            return;
        }
        eprintln!("native serving plugin proposal fence failed: {error}");
        send_proposal_response(
            passive_queue,
            reply,
            ProposalResponse {
                proposal: Err(error.to_string()),
                telemetry: LinearProposalSourceTelemetry {
                    queue_wait_us,
                    callback_elapsed_us: 0,
                    outcome: LinearProposalSourceOutcome::SourceError,
                },
            },
        );
        return;
    }
    if Instant::now() >= deadline {
        send_proposal_response(
            passive_queue,
            reply,
            abstention(
                queue_wait_us,
                LinearProposalSourceOutcome::DeadlineExceededBeforeDispatch,
            ),
        );
        return;
    }

    let callback_started = Instant::now();
    let result = active.propose(query);
    // One timestamp classifies both the forwarding decision and the telemetry,
    // so a candidate can never be forwarded while telemetry calls it late.
    let callback_finished = Instant::now();
    let callback_elapsed_us = elapsed_us(callback_started);
    let deadline_missed = callback_finished >= deadline;

    let (proposal, outcome) = match result {
        Ok(Some(proposal)) if deadline_missed => {
            discard_late_candidate(passive_queue, &proposal);
            (
                Ok(None),
                LinearProposalSourceOutcome::CandidateReturnedTooLate,
            )
        }
        Ok(Some(proposal)) => (Ok(Some(proposal)), LinearProposalSourceOutcome::Ready),
        Ok(None) if deadline_missed => (
            Ok(None),
            LinearProposalSourceOutcome::DeadlineExceededInPlugin,
        ),
        Ok(None) => (Ok(None), LinearProposalSourceOutcome::Abstained),
        Err(error) => {
            // Fail open, but surface the plugin's message instead of
            // silently degrading a failure into an abstention.
            eprintln!("native serving plugin proposal failed: {error:#}");
            (
                Err(format!("{error:#}")),
                LinearProposalSourceOutcome::SourceError,
            )
        }
    };

    send_proposal_response(
        passive_queue,
        reply,
        ProposalResponse {
            proposal,
            telemetry: LinearProposalSourceTelemetry {
                queue_wait_us,
                callback_elapsed_us,
                outcome,
            },
        },
    );
}

/// Delivers a proposal response, resolving an on-time candidate if the
/// bounded caller reply has already been dropped or timed out.
fn send_proposal_response(
    passive_queue: &PluginCommandQueue,
    reply: &SyncSender<ProposalResponse>,
    response: ProposalResponse,
) {
    let candidate_decision_id = response
        .proposal
        .as_ref()
        .ok()
        .and_then(|proposal| proposal.as_ref())
        .map(|proposal| proposal.decision_id.as_bytes().to_vec());
    if reply.send(response).is_ok() {
        return;
    }

    if let Some(decision_id) = candidate_decision_id
        && let Err(error) = passive_queue.try_enqueue_terminal(PluginCommand::Discard(
            decision_id,
            LinearProposalDiscardReason::DeadlineExceeded,
        ))
    {
        eprintln!(
            "native serving plugin could not deliver the terminal discard for a detached proposal reply: {error:?}"
        );
    }
}

/// Resolves a decision the host refuses to forward after its deadline.
fn discard_late_candidate(passive_queue: &PluginCommandQueue, proposal: &LinearProposal) {
    if let Err(error) = passive_queue.try_enqueue_terminal(PluginCommand::Discard(
        proposal.decision_id.as_bytes().to_vec(),
        LinearProposalDiscardReason::DeadlineExceeded,
    )) {
        eprintln!(
            "native serving plugin could not deliver the terminal discard for a late proposal: \
             {error:?}"
        );
    }
}

fn report_after_passive_handoff(
    passive_queue: &PluginCommandQueue,
    receipt: LinearProposalReceipt,
) -> Result<()> {
    let (ack, result) = sync_channel(0);
    passive_queue
        .try_enqueue_terminal(PluginCommand::Report(receipt, ack))
        .map_err(|error| match error {
            PluginCommandQueueError::Full => {
                anyhow!("native serving plugin terminal queue is full")
            }
            PluginCommandQueueError::Stopped => {
                anyhow!("native serving plugin passive worker stopped")
            }
            PluginCommandQueueError::Poisoned => {
                anyhow!("native serving plugin terminal queue lock poisoned")
            }
        })?;
    match result.recv_timeout(CLEAN_SHUTDOWN_TIMEOUT) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => bail!(
            "native serving plugin report callback did not complete within {CLEAN_SHUTDOWN_TIMEOUT:?}"
        ),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            bail!("native serving plugin passive worker stopped before report ack")
        }
    }
}

/// Waits until every passive terminal callback queued before this point has
/// completed. This preserves report/discard causality without putting slow
/// callbacks on the proposal-deadline queue.
#[derive(Debug)]
enum PassiveFenceError {
    Timeout(Duration),
    Failure(anyhow::Error),
}

impl std::fmt::Display for PassiveFenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout(timeout) => write!(
                formatter,
                "native serving plugin passive callbacks did not complete within {timeout:?}"
            ),
            Self::Failure(error) => write!(formatter, "{error:#}"),
        }
    }
}

fn fence_passive(
    passive_queue: &PluginCommandQueue,
    timeout: Duration,
) -> std::result::Result<(), PassiveFenceError> {
    let (reply, response) = sync_channel(1);
    passive_queue
        .try_enqueue_terminal(PluginCommand::Fence(reply))
        .map_err(|error| {
            PassiveFenceError::Failure(match error {
                PluginCommandQueueError::Full => {
                    anyhow!("native serving plugin terminal queue is full")
                }
                PluginCommandQueueError::Stopped => {
                    anyhow!("native serving plugin passive worker stopped")
                }
                PluginCommandQueueError::Poisoned => {
                    anyhow!("native serving plugin terminal queue lock poisoned")
                }
            })
        })?;
    match response.recv_timeout(timeout) {
        Ok(()) => Ok(()),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(PassiveFenceError::Timeout(timeout)),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(PassiveFenceError::Failure(
            anyhow!("native serving plugin passive worker stopped before fence"),
        )),
    }
}

fn finish_after_passive_fence<T>(
    passive_queue: &PluginCommandQueue,
    finish: impl FnOnce() -> Result<T>,
) -> Result<T> {
    fence_passive(passive_queue, CLEAN_SHUTDOWN_TIMEOUT).map_err(|error| anyhow!("{error}"))?;
    finish()
}

fn plugin_passive_worker(
    active: Arc<ActivePlugin>,
    queue: Arc<PluginCommandQueue>,
    exit: Arc<WorkerExit>,
) {
    let _stop_guard = WorkerStopGuard {
        queue: Arc::clone(&queue),
        exit,
    };
    while let Some(queued) = queue.next() {
        match queued.command {
            PluginCommand::Report(event, ack) => {
                let result = active.report(&event);
                if ack.send(result).is_err() {
                    eprintln!(
                        "native serving plugin report callback acknowledgement receiver dropped"
                    );
                }
            }
            PluginCommand::Discard(decision_id, reason) => {
                let _ = active.discard(&decision_id, reason);
            }
            PluginCommand::Fence(reply) => {
                let _ = reply.send(());
            }
            PluginCommand::Begin(_)
            | PluginCommand::Committed(_)
            | PluginCommand::Abort(_)
            | PluginCommand::Finish(_)
            | PluginCommand::ReportHandoff(_)
            | PluginCommand::Proposal(_, _) => {
                unreachable!("lifecycle and proposal callbacks must use the primary worker queue")
            }
        }
    }
}

#[cfg(test)]
mod tests;
