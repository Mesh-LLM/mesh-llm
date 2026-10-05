//! Test-only deterministic branch/gate counters for the exact-state capture
//! path. The bounded telemetry sink drops events, so absent telemetry does
//! not prove a branch never ran; these counters do. Process-global: tests
//! must snapshot before their request and report deltas.
use std::sync::atomic::{AtomicI64, AtomicUsize};

pub(crate) static POST_DECODE_ENTERED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_GATES_PASSED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_SCHEDULER_REJECTED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_TASK_EXECUTED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_SKIPPED_POSITION_ERROR: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_SKIPPED_POSITION_MISMATCH: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_RECORDED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_RECORD_NONE: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_RECORD_ERROR: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_NONE_SHOULD_RECORD: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_NONE_MIN_TOKENS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_NONE_BEGIN_RECORD: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_NONE_RADIX_BUSY: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_NONE_ALREADY_RECORDED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_NONE_CAPACITY: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_NONE_PAYLOAD_DISABLED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static POST_DECODE_ADMISSION_DECLINED: AtomicUsize = AtomicUsize::new(0);
/// Admission declines for EVERY capture class, so prefill-ladder
/// (BestEffort) declines are visible next to the post-decode ones.
pub(crate) static ANY_ADMISSION_DECLINED: AtomicUsize = AtomicUsize::new(0);
pub(crate) static LAST_RUNTIME_POSITION: AtomicI64 = AtomicI64::new(-1);
pub(crate) static LAST_CHECKPOINT_COUNT: AtomicI64 = AtomicI64::new(-1);

/// One observed identity on the record or lookup side. Stored entries are
/// logged by the recorder at storage completion and later annotated with
/// `retained` (a read-only radix peek after the drain); lookup candidates
/// carry whether `acquire_recurrent` matched and how many tokens the
/// radix entry actually holds. Lets tests compare RETAINED radix keys
/// against second-request lookup candidates and name the first divergence
/// before assigning any root cause.
#[derive(Clone, Debug)]
pub(crate) struct IdentityDiag {
    pub(crate) namespace: String,
    pub(crate) tokens: Vec<i32>,
    /// Lookup side: acquire_recurrent matched. Stored side: still
    /// retained in the radix after the drain.
    pub(crate) found: Option<bool>,
    pub(crate) stored_token_count: Option<usize>,
    pub(crate) retained: Option<bool>,
}

static STORED_IDENTITIES: std::sync::Mutex<Vec<IdentityDiag>> = std::sync::Mutex::new(Vec::new());
static LOOKUP_CANDIDATES: std::sync::Mutex<Vec<IdentityDiag>> = std::sync::Mutex::new(Vec::new());

/// One capture decision by prefix: which checkpoint (shared prefill /
/// final prefill / post-decode), what outcome, and the position facts.
/// Lets tests account for EVERY expected capture of a request — including
/// the prefill-ladder boundary checkpoint — instead of only the
/// post-decode one.
#[derive(Clone, Debug)]
pub(crate) struct CaptureDecisionDiag {
    pub(crate) decision_prefix: String,
    pub(crate) outcome: String,
    pub(crate) checkpoint_token_count: u64,
    pub(crate) runtime_position: Option<u64>,
    pub(crate) namespace: Option<String>,
}

static CAPTURE_DECISIONS: std::sync::Mutex<Vec<CaptureDecisionDiag>> =
    std::sync::Mutex::new(Vec::new());

/// Records one capture decision. Called from the capture task at each
/// outcome: position_error, position_mismatch, recorded, record_none,
/// record_error, scheduler_rejected.
pub(crate) fn log_capture_decision(
    decision_prefix: &str,
    outcome: &str,
    checkpoint_token_count: u64,
    runtime_position: Option<u64>,
    namespace: Option<&str>,
) {
    CAPTURE_DECISIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(CaptureDecisionDiag {
            decision_prefix: decision_prefix.to_string(),
            outcome: outcome.to_string(),
            checkpoint_token_count,
            runtime_position,
            namespace: namespace.map(str::to_string),
        });
}

/// Drains the capture-decision log (snapshot-and-clear).
pub(crate) fn drain_capture_decisions() -> Vec<CaptureDecisionDiag> {
    std::mem::take(
        &mut *CAPTURE_DECISIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

/// Records the identity of a payload when the recorder has STORED it
/// (post-insert, post-eviction passes). Retention against later eviction
/// is verified separately via radix peeks.
pub(crate) fn log_stored_identity(namespace: &str, token_ids: &[i32]) {
    STORED_IDENTITIES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(IdentityDiag {
            namespace: namespace.to_string(),
            tokens: token_ids.to_vec(),
            found: None,
            stored_token_count: None,
            retained: None,
        });
}

/// Records one restore-side lookup candidate and its radix outcome.
pub(crate) fn log_lookup_candidate(
    namespace: &str,
    token_ids: &[i32],
    found: bool,
    stored_token_count: Option<usize>,
) {
    LOOKUP_CANDIDATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(IdentityDiag {
            namespace: namespace.to_string(),
            tokens: token_ids.to_vec(),
            found: Some(found),
            stored_token_count,
            retained: None,
        });
}

/// Drains the stored-identity log (snapshot-and-clear).
pub(crate) fn drain_stored_identities() -> Vec<IdentityDiag> {
    std::mem::take(
        &mut *STORED_IDENTITIES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

/// Drains the lookup-candidate log (snapshot-and-clear).
pub(crate) fn drain_lookup_candidates() -> Vec<IdentityDiag> {
    std::mem::take(
        &mut *LOOKUP_CANDIDATES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

pub(crate) fn render() -> String {
    format!(
        "entered={} gates_passed={} scheduler_rejected={} task_executed={} \
         skipped_position_error={} skipped_position_mismatch={} recorded={} \
         record_none={} record_error={} none_radix_busy={} none_already_recorded={} \
         none_begin_record={} none_min_tokens={} none_should_record={} \
         none_capacity={} none_payload_disabled={} admission_declined={} \
         any_admission_declined={} last_runtime_position={} \
         last_checkpoint_count={}",
        POST_DECODE_ENTERED.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_GATES_PASSED.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_SCHEDULER_REJECTED.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_TASK_EXECUTED.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_SKIPPED_POSITION_ERROR.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_SKIPPED_POSITION_MISMATCH.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_RECORDED.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_RECORD_NONE.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_RECORD_ERROR.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_NONE_RADIX_BUSY.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_NONE_ALREADY_RECORDED.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_NONE_BEGIN_RECORD.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_NONE_MIN_TOKENS.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_NONE_SHOULD_RECORD.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_NONE_CAPACITY.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_NONE_PAYLOAD_DISABLED.load(std::sync::atomic::Ordering::Acquire),
        POST_DECODE_ADMISSION_DECLINED.load(std::sync::atomic::Ordering::Acquire),
        ANY_ADMISSION_DECLINED.load(std::sync::atomic::Ordering::Acquire),
        LAST_RUNTIME_POSITION.load(std::sync::atomic::Ordering::Acquire),
        LAST_CHECKPOINT_COUNT.load(std::sync::atomic::Ordering::Acquire),
    )
}

pub(crate) fn reset() {
    POST_DECODE_ENTERED.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_GATES_PASSED.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_SCHEDULER_REJECTED.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_TASK_EXECUTED.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_SKIPPED_POSITION_ERROR.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_SKIPPED_POSITION_MISMATCH.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_RECORDED.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_RECORD_NONE.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_RECORD_ERROR.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_NONE_SHOULD_RECORD.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_NONE_MIN_TOKENS.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_NONE_BEGIN_RECORD.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_NONE_RADIX_BUSY.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_NONE_ALREADY_RECORDED.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_NONE_CAPACITY.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_NONE_PAYLOAD_DISABLED.store(0, std::sync::atomic::Ordering::Release);
    POST_DECODE_ADMISSION_DECLINED.store(0, std::sync::atomic::Ordering::Release);
    ANY_ADMISSION_DECLINED.store(0, std::sync::atomic::Ordering::Release);
    LAST_RUNTIME_POSITION.store(-1, std::sync::atomic::Ordering::Release);
    LAST_CHECKPOINT_COUNT.store(-1, std::sync::atomic::Ordering::Release);
    STORED_IDENTITIES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    CAPTURE_DECISIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    LOOKUP_CANDIDATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}
