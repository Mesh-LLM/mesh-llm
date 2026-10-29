#[cfg(test)]
mod export_admission_tests;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize},
        mpsc::{SyncSender, TrySendError},
    },
    thread::JoinHandle,
};

use anyhow::{Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use skippy_cache::{
    CacheBlobStore, ExactStatePayload, ResidentActivationCache, ResidentCacheConfig,
    SparseCheckpointPolicy, UnifiedRadixCache,
};
use skippy_metrics::attr as attr_key;
use skippy_runtime::{ActivationFrame, MemoryCacheCapabilities, ModelStateKind, RuntimeKvPageDesc};

use crate::kv_proto::{
    Checksum, ChecksumAlgorithm, KvPageManifest, MANIFEST_SCHEMA_VERSION, PageIdentity, PageState,
};

mod activation;
mod cache_affinity;
mod cache_payload;
mod config;
mod exact_state;
pub(crate) mod exact_state_diagnostics;
mod snapshot_validation;
pub(crate) use exact_state::CaptureAdmission;
mod identity;
mod kv_lease;
mod l2_serving;
pub mod lifecycle;
mod output_tokens;
mod records;
mod resident_prefix;
pub(crate) use kv_lease::{
    LeaseLedger, LeasedKvPage, LeasedRecordRefusal, LeasedRecordReservation,
};
mod resident_recurrent;
pub(crate) use resident_recurrent::{
    DemotionRefusal, InPlaceRecord, RecorderReceipt, ResidentDemotion, ResidentRecurrentPolicy,
};

pub use lifecycle::{KvLifecycleEvent, KvLifecycleObserver};
pub use records::{
    AttachedPage, ExactStateRecord, ExactStateRestore, LookupBatchOutcome, PrefillKvIdentity,
    RecordPageOutcome, ResidentActivationRecord, ResidentActivationRestore, ResidentPrefixRecord,
    ResidentPrefixRestore,
};
pub use resident_prefix::{ResidentCapacityDecision, ResidentPrefixEviction};

/// Return a bounded, stable telemetry class without exporting error text.
///
/// Detailed errors remain available to callers for local diagnostics, while metrics
/// only receive one of this fixed set of labels.
pub(crate) fn telemetry_error_class(error: &anyhow::Error) -> &'static str {
    telemetry_error_class_from_message(&error.to_string())
}

pub(crate) fn telemetry_error_class_from_message(message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    if message.contains("checksum") || message.contains("digest") {
        "integrity"
    } else if message.contains("not found") || message.contains("missing") {
        "not_found"
    } else if message.contains("unsupported") || message.contains("disabled") {
        "unsupported"
    } else if message.contains("invalid") || message.contains("mismatch") {
        "invalid_data"
    } else if message.contains("timeout") || message.contains("unavailable") {
        "unavailable"
    } else if message.contains("permission") || message.contains("denied") {
        "permission"
    } else if message.contains("io error")
        || message.contains("i/o error")
        || message.contains("failed to read")
        || message.contains("failed to write")
    {
        "io"
    } else if message.contains("native") || message.contains("runtime") {
        "runtime"
    } else {
        "internal"
    }
}

pub(crate) fn proactive_eviction_error_kind(error: &anyhow::Error) -> &'static str {
    let message = error.to_string();
    if message.contains("is not active") {
        "inactive_session"
    } else if message.contains("batch size") {
        "invalid_batch_size"
    } else {
        "native_drop_failed"
    }
}

pub(crate) fn proactive_eviction_attrs(
    status: &str,
    error_kind: Option<&str>,
    target_tokens: u64,
    evicted_entries: usize,
    evicted_tokens: u64,
) -> BTreeMap<String, Value> {
    let mut attrs = BTreeMap::from([
        (
            "skippy.kv.decision".to_string(),
            json!("proactive_eviction"),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTION_STATUS.to_string(),
            json!(status),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTION_TARGET_TOKENS.to_string(),
            json!(target_tokens),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTED_ENTRIES.to_string(),
            json!(evicted_entries),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTED_TOKENS.to_string(),
            json!(evicted_tokens),
        ),
    ]);
    if let Some(error_kind) = error_kind {
        attrs.insert(
            attr_key::KV_PROACTIVE_EVICTION_ERROR_KIND.to_string(),
            json!(error_kind),
        );
    }
    attrs
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKvMode {
    Disabled,
    Record,
    LookupRecord,
    Correctness,
}

/// Byte budget for the exact-state catalog.
///
/// `soft_bytes` is the stage cache budget, which is derived from attention KV
/// metadata and therefore cannot account for the recurrent and convolution
/// buffers an exact-state snapshot also carries. A single snapshot can
/// legitimately exceed it, so the soft budget only bounds the catalog once it
/// already holds a working set. `hard_bytes` bounds that allowance except that
/// the last indivisible snapshot is retained even when it exceeds the limit.
/// Zero means unbounded for either field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExactStateByteLimits {
    pub(crate) soft_bytes: u64,
    pub(crate) hard_bytes: u64,
}

#[derive(Clone)]
pub struct KvStageIntegration {
    pub(crate) mode: StageKvMode,
    /// The in-process cache representation. Dense models keep native resident
    /// KV here even when a durable tier is configured, so enabling disk does
    /// not replace the fast warm path with serialized state import.
    pub(crate) payload: StagePrefixCachePayload,
    pub(crate) payload_selection_reason: &'static str,
    pub(crate) loaded_state_kind: Option<ModelStateKind>,
    pub(crate) loaded_memory_cache: Option<MemoryCacheCapabilities>,
    /// Exportable exact-state representation retained in L1 and optionally
    /// written to L3. Resident KV remains the native serving representation.
    pub(crate) serialized_payload: Option<StagePrefixCachePayload>,
    pub(crate) correctness_mode: bool,
    pub(crate) trust_local_writes: bool,
    pub(crate) checkpoint_policy: SparseCheckpointPolicy,
    pub(crate) inflight_records: Arc<Mutex<BTreeSet<String>>>,
    pub(crate) resident_config: ResidentCacheConfig,
    pub(crate) resident_capacity_reservations: resident_prefix::ResidentCapacityReservations,
    pub(crate) resident_sequences: Arc<Mutex<ResidentSequencePool>>,
    pub(crate) activations: Arc<Mutex<ResidentActivationCache<ActivationFrame>>>,
    pub(crate) radix: Arc<Mutex<UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>>>,
    pub(crate) exact_blobs: Arc<Mutex<CacheBlobStore>>,
    pub(crate) exact_max_entries: usize,
    pub(crate) exact_byte_limits: ExactStateByteLimits,
    pub(crate) exact_state_record_worker: Arc<ExactStateRecordWorker>,
    pub(crate) snapshot_export_failures: Arc<AtomicU64>,
    pub(crate) exact_state_records_queued: Arc<AtomicU64>,
    pub(crate) exact_state_records_dropped: Arc<AtomicU64>,
    pub(crate) exact_state_records_pending: Arc<AtomicUsize>,
    /// Production admission accounting for the recorder pipeline: credits
    /// held by captures that are exporting, queued in the channel, or held by
    /// the worker. See `ExactStateAdmissionCredit`.
    pub(crate) admission_outstanding: Arc<AtomicUsize>,
    pub(crate) admission_best_effort_outstanding: Arc<AtomicUsize>,
    /// Test-only outstanding-capture tracking: incremented when a capture task
    /// is scheduled onto the iteration scheduler, released when that task
    /// reaches ANY terminal outcome (recorded, skipped, error, panic) via a
    /// drop guard inside the task. Lets tests establish a completion boundary
    /// for detached captures before draining the recorder.
    #[cfg(test)]
    pub(crate) exact_state_captures_outstanding: Arc<AtomicUsize>,
    /// Test-only: records received off the channel by the recorder worker
    /// (pre-storage), for the worker-receipt barrier in tests.
    #[cfg(test)]
    pub(crate) exact_state_worker_received: Arc<AtomicUsize>,
    /// Test-only recorder pause: when set, the recorder worker waits AFTER
    /// receiving a record and BEFORE storing it, holding no locks. Lets tests
    /// hold a record (and its credit) worker-side deterministically without
    /// radix contention.
    #[cfg(test)]
    pub(crate) exact_state_worker_pause: Arc<AtomicBool>,
    /// Test-only durable worker receipt/pause controls. They establish that
    /// L1 publication continues while an earlier L3 spill is blocked.
    #[cfg(test)]
    pub(crate) l3_spill_worker_received: Arc<AtomicUsize>,
    #[cfg(test)]
    pub(crate) l3_spill_worker_pause: Arc<AtomicBool>,
    pub(crate) exact_state_record_worker_healthy: Arc<AtomicBool>,
    pub(crate) exact_state_record_worker_panics: Arc<AtomicU64>,
    pub(crate) cache_healthy: Arc<AtomicBool>,
    pub(crate) output_tokens: Arc<Mutex<output_tokens::OutputTokenCache>>,
    pub(crate) split_prefill_tokens: Arc<Mutex<BTreeMap<String, Vec<i32>>>>,
    pub(crate) kv_lifecycle_observer: Option<Arc<dyn KvLifecycleObserver>>,
    /// Payload bytes held by records queued for the worker but not yet
    /// stored. The queue is bounded in bytes, not entries: an entry bound
    /// lets one multi-GiB export sit next to another and doubles the RAM the
    /// cache can pin behind a request.
    pub(crate) exact_state_record_queue_bytes: Arc<AtomicU64>,
    pub(crate) exact_state_record_queue_limit: u64,
    /// Optional bounded host-RAM tier. Qualified L3 fills enter L2; an L2 hit
    /// promotes back into L1 through the existing worker.
    pub(crate) l2: Option<l2_serving::StageL2>,
    /// Durable L3 floor under the radix cache: exact-state records write
    /// through to it on the worker, and radix misses fill back from it.
    pub(crate) l3: Option<Arc<skippy_cache::L3Tier>>,
    /// Whether this stage passed the backend and dtype gate for serving-path
    /// CacheGen writes. Exposed in status so an opt-in fallback is visible.
    pub(crate) cachegen_serving_enabled: bool,
    /// Manifest keys with an L3 fill in flight. Concurrent misses on one
    /// stored prefix must not each read it from disk: the loser prefills
    /// normally while the winner re-warms the radix for everyone.
    pub(crate) inflight_fills: Arc<Mutex<BTreeSet<String>>>,
    /// The model has attention KV but no recurrent memory, so an exact-state
    /// entry legitimately carries an empty recurrent snapshot. Restores set
    /// the position directly instead of importing one. Never true for a
    /// recurrent family, where an empty snapshot is corruption.
    pub(crate) dense_without_recurrent: bool,
    /// Resident entries pair lent attention cells with a recurrent snapshot
    /// taken at the same boundary. Records only happen at a session's
    /// current position and restores never slice.
    pub(crate) resident_recurrent_snapshots: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagePrefixCachePayload {
    Disabled,
    ResidentKv,
    KvRecurrent,
    FullState,
}

/// Entry backstop on the record queue. Byte admission is stage-specific;
/// this only caps bookkeeping. Credits take at most
/// `EXACT_STATE_ADMISSION_TOTAL` of these slots; the rest admit leased
/// records, whose owned recurrent snapshots are charged to the byte budget.
pub(crate) const EXACT_STATE_RECORD_CAPACITY: usize = 16;

/// Channel slots a leased record may occupy. Carved out above the credit
/// total so a credit-holding record always finds a slot: credits never
/// exceed `EXACT_STATE_ADMISSION_TOTAL`, leased records never exceed this,
/// and the two sum to the capacity.
pub(crate) const LEASED_RECORD_SLOTS: usize =
    EXACT_STATE_RECORD_CAPACITY - exact_state::EXACT_STATE_ADMISSION_TOTAL;
const _: () = assert!(
    LEASED_RECORD_SLOTS > 0,
    "the recorder channel must leave room for leased records beyond the credit total"
);

/// Minimum payload bytes the record queue may hold. A record that does not fit is
/// dropped, never delayed: recording is optional and inference is not. One
/// record larger than the whole bound is still admitted when the queue is
/// empty, or large models could never record at all.
pub(crate) const EXACT_STATE_RECORD_QUEUE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct PendingExactStateRecord {
    pub(crate) page_id: String,
    pub(crate) payload: ExactStatePayload,
    pub(crate) extra: ExactStateExtra,
    pub(crate) namespace: String,
    pub(crate) token_ids: Vec<i32>,
    /// When this record re-warms the radix after an L3 fill, the fill's
    /// claim key. The worker releases it only once the radix insert lands,
    /// so requests arriving during the asynchronous re-warm prefill normally
    /// instead of duplicating the disk read.
    pub(crate) l3_fill_claim: Option<String>,
    /// Only freshly exported request state writes through. Tier fills that
    /// merely re-warm L1 must not rewrite their existing durable entry.
    pub(crate) write_through_l3: bool,
    /// Durable manifest digest to mirror into L2 on this worker job. `None`
    /// leaves the payload out of L2.
    pub(crate) l2_promotion_digest: Option<String>,
    /// Measured cold-versus-restore cost for L3 benefit admission. Missing
    /// telemetry preserves the established LRU write-through behavior.
    pub(crate) l3_cost: Option<skippy_cache::policy::CostSample>,
    /// What bounds this record while it is in flight.
    pub(crate) admission: RecordAdmission,
}

/// How a record earned its place in the recorder.
#[derive(Debug)]
pub(crate) enum RecordAdmission {
    /// Owned payload: one of the stage's credits and its byte reservation.
    /// Released when the record is dropped after the worker stores (or
    /// fails) it; the credit's Drop impl performs the release.
    Credit(crate::kv_integration::exact_state::ExactStateAdmissionCredit),
    /// The lane's cells are the attention payload; the owned recurrent half
    /// has already reserved queue bytes and a leased slot before being cloned.
    /// The worker releases both reservations on completion.
    Leased(LeasedKvPage, LeasedRecordReservation),
    /// A leased page the worker has already copied out. Only the worker's
    /// own sequential execution bounds it now.
    Released,
}

impl PendingExactStateRecord {
    /// Bytes the record will occupy once stored, lease included.
    pub(crate) fn byte_len(&self) -> u64 {
        let leased = match &self.admission {
            RecordAdmission::Leased(lease, _) => lease.byte_len(),
            RecordAdmission::Credit(_) | RecordAdmission::Released => 0,
        };
        self.payload.byte_len().saturating_add(leased)
    }

    /// Bytes charged to the stage's record byte budget: a credit's whole
    /// payload or a lease's owned recurrent snapshot. The worker captures
    /// this before materialising the leased attention KV.
    pub(crate) fn budgeted_bytes(&self) -> u64 {
        match self.admission {
            RecordAdmission::Credit(_) => self.byte_len(),
            RecordAdmission::Leased(_, _) => self.payload.byte_len(),
            RecordAdmission::Released => 0,
        }
    }

    /// Copy a leased page out of the lane's cells into an owned payload and
    /// release the lane. A record without a lease is returned untouched.
    pub(crate) fn materialise_leased_kv(mut self) -> Result<Self> {
        let RecordAdmission::Leased(lease, _reservation) =
            std::mem::replace(&mut self.admission, RecordAdmission::Released)
        else {
            return Ok(self);
        };
        let kv = lease.materialise()?;
        drop(lease);
        let recurrent = self.payload.recurrent_state_bytes()?.into_owned();
        self.payload = ExactStatePayload::kv_recurrent(kv, recurrent);
        Ok(self)
    }
}

#[derive(Debug)]
pub(crate) struct ExactStateRecordWorker {
    sender: Mutex<Option<SyncSender<PendingExactStateRecord>>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl ExactStateRecordWorker {
    pub(crate) fn new(
        sender: SyncSender<PendingExactStateRecord>,
        record_task: JoinHandle<()>,
        durable_task: Option<JoinHandle<()>>,
    ) -> Self {
        let mut tasks = vec![record_task];
        tasks.extend(durable_task);
        Self {
            sender: Mutex::new(Some(sender)),
            tasks: Mutex::new(tasks),
        }
    }

    fn with_sender<T>(
        &self,
        use_sender: impl FnOnce(Option<&SyncSender<PendingExactStateRecord>>) -> T,
    ) -> T {
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        use_sender(sender.as_ref())
    }
}

impl Drop for ExactStateRecordWorker {
    fn drop(&mut self) {
        self.sender
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        for task in self
            .tasks
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
        {
            let _ = task.join();
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RadixResidentEntry {
    pub(crate) page_id: String,
    pub(crate) seq_id: i32,
    pub(crate) token_count: u64,
    /// Deterministic first-order estimate of work needed to recreate this
    /// entry: cached tokens multiplied by stage-local layer count.
    pub(crate) recompute_cost: u64,
    /// The recurrent state at exactly `token_count`, for a hybrid stage. The
    /// attention cells are the cache sequence's; this half cannot be shared
    /// by tagging and is only valid at its own boundary, so such an entry is
    /// never sliced on restore.
    pub(crate) recurrent: Option<Arc<[u8]>>,
}

#[derive(Debug, Clone)]
pub(crate) struct RadixExactEntry {
    pub(crate) page_id: String,
    pub(crate) payload: ExactStatePayload,
    pub(crate) extra: ExactStateExtra,
    /// Whether a later resident hit may promote this entry into durable L3.
    pub(crate) l3_promotion_eligible: bool,
}

impl RadixExactEntry {
    pub(crate) fn new(
        page_id: String,
        payload: ExactStatePayload,
        extra: ExactStateExtra,
        l3_promotion_eligible: bool,
    ) -> Self {
        Self {
            page_id,
            payload,
            extra,
            l3_promotion_eligible,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResidentSequencePool {
    reserved_seq_count: i32,
    next_seq_id: i32,
    free_seq_ids: Vec<i32>,
    allocated_seq_ids: BTreeSet<i32>,
    quarantined_seq_ids: BTreeSet<i32>,
}

impl ResidentSequencePool {
    fn new(reserved_seq_count: i32) -> Self {
        Self {
            reserved_seq_count,
            next_seq_id: reserved_seq_count,
            free_seq_ids: Vec::new(),
            allocated_seq_ids: BTreeSet::new(),
            quarantined_seq_ids: BTreeSet::new(),
        }
    }

    pub(crate) fn allocate(&mut self) -> Result<i32> {
        if let Some(seq_id) = self.free_seq_ids.pop() {
            if !self.allocated_seq_ids.insert(seq_id) {
                bail!("resident prefix sequence id {seq_id} is already allocated");
            }
            return Ok(seq_id);
        }
        let seq_id = self.next_seq_id;
        if seq_id < self.reserved_seq_count || seq_id >= skippy_cache::LLAMA_MAX_SEQ {
            bail!("resident prefix sequence id capacity exhausted");
        }
        self.next_seq_id = self
            .next_seq_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("resident prefix sequence id overflow"))?;
        if !self.allocated_seq_ids.insert(seq_id) {
            bail!("resident prefix sequence id {seq_id} is already allocated");
        }
        Ok(seq_id)
    }

    fn release(&mut self, seq_id: i32) -> Result<()> {
        self.validate_allocated(seq_id)?;
        self.allocated_seq_ids.remove(&seq_id);
        self.free_seq_ids.push(seq_id);
        Ok(())
    }

    fn quarantine(&mut self, seq_id: i32) -> Result<()> {
        self.validate_allocated(seq_id)?;
        self.allocated_seq_ids.remove(&seq_id);
        self.quarantined_seq_ids.insert(seq_id);
        Ok(())
    }

    fn force_quarantine(&mut self, seq_id: i32) {
        self.allocated_seq_ids.remove(&seq_id);
        self.free_seq_ids.retain(|candidate| *candidate != seq_id);
        self.quarantined_seq_ids.insert(seq_id);
    }

    fn validate_allocated(&self, seq_id: i32) -> Result<()> {
        if seq_id < self.reserved_seq_count || seq_id >= skippy_cache::LLAMA_MAX_SEQ {
            bail!("resident prefix sequence id {seq_id} is out of range");
        }
        if !self.allocated_seq_ids.contains(&seq_id) {
            bail!("resident prefix sequence id {seq_id} is not allocated");
        }
        Ok(())
    }

    fn stats(&self) -> (usize, usize, usize) {
        (
            self.allocated_seq_ids.len(),
            self.free_seq_ids.len(),
            self.quarantined_seq_ids.len(),
        )
    }
}

fn lock_resident_sequences(
    sequences: &Mutex<ResidentSequencePool>,
) -> std::sync::MutexGuard<'_, ResidentSequencePool> {
    sequences
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExactStateRecordAdmission {
    Queued,
    DroppedFull,
    WorkerStopped,
}

/// Whether `bytes` more may join a queue already holding `held` bytes under
/// `cap`. A single record over the cap is admitted only into an empty queue.
fn record_fits_queue(held: u64, bytes: u64, cap: u64) -> bool {
    held == 0 || held.saturating_add(bytes) <= cap
}
fn finish_exact_state_record(
    inflight_records: &Mutex<BTreeSet<String>>,
    pending_count: &AtomicUsize,
    queue_bytes: &AtomicU64,
    page_id: &str,
    bytes: u64,
) {
    inflight_records
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(page_id);
    pending_count.fetch_sub(1, std::sync::atomic::Ordering::Release);
    queue_bytes.fetch_sub(bytes, std::sync::atomic::Ordering::Release);
}

/// Shared bookkeeping handles for the exact-state-record background worker,
/// grouped so `run_exact_state_record_job` stays within clippy's
/// too-many-arguments limit. Every field is itself an already-shared
/// (`Arc`-backed at the call site) handle; this struct only borrows them for
/// the duration of one job.
struct ExactStateWorkerHandles<'a> {
    inflight_records: &'a Mutex<BTreeSet<String>>,
    dropped: &'a AtomicU64,
    pending_count: &'a AtomicUsize,
    queue_bytes: &'a AtomicU64,
    worker_healthy: &'a AtomicBool,
    worker_panics: &'a AtomicU64,
}

fn run_exact_state_record_job(
    handles: ExactStateWorkerHandles<'_>,
    observer: Option<&Arc<dyn KvLifecycleObserver>>,
    pending: PendingExactStateRecord,
    work: impl FnOnce(PendingExactStateRecord) -> Result<()>,
) {
    let notify = |event: KvLifecycleEvent| {
        if let Some(observer) = observer {
            observer.observe(event);
        }
    };
    let page_id = pending.page_id.clone();
    let bytes = pending.budgeted_bytes();
    // Leave the lane as early as possible: the copy is the only thing that
    // needs the cells, and it needs neither the runtime lock nor any tier.
    let pending = match pending.materialise_leased_kv() {
        Ok(pending) => pending,
        Err(error) => {
            tracing::debug!(error = %error, "leased KV page could not be read");
            handles
                .dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            notify(KvLifecycleEvent::ExactStateRecordFailed);
            finish_exact_state_record(
                handles.inflight_records,
                handles.pending_count,
                handles.queue_bytes,
                &page_id,
                bytes,
            );
            return;
        }
    };
    if !handles
        .worker_healthy
        .load(std::sync::atomic::Ordering::Acquire)
    {
        handles
            .dropped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        notify(KvLifecycleEvent::ExactStateRecordFailed);
        finish_exact_state_record(
            handles.inflight_records,
            handles.pending_count,
            handles.queue_bytes,
            &page_id,
            bytes,
        );
        return;
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(pending))) {
        Ok(Ok(())) => notify(KvLifecycleEvent::ExactStateRecordCompleted),
        Ok(Err(_)) => {
            handles
                .dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            notify(KvLifecycleEvent::ExactStateRecordFailed);
        }
        Err(_) => {
            handles
                .dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            handles
                .worker_panics
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            handles
                .worker_healthy
                .store(false, std::sync::atomic::Ordering::Release);
            notify(KvLifecycleEvent::ExactStateRecordFailed);
        }
    }
    finish_exact_state_record(
        handles.inflight_records,
        handles.pending_count,
        handles.queue_bytes,
        &page_id,
        bytes,
    );
}

#[allow(clippy::too_many_arguments)]
fn enqueue_exact_state_record(
    sender: &SyncSender<PendingExactStateRecord>,
    inflight_records: &Mutex<BTreeSet<String>>,
    queued: &AtomicU64,
    dropped: &AtomicU64,
    pending_count: &AtomicUsize,
    queue_bytes: &AtomicU64,
    queue_bytes_cap: u64,
    worker_healthy: &AtomicBool,
    mut pending: PendingExactStateRecord,
) -> ExactStateRecordAdmission {
    if !worker_healthy.load(std::sync::atomic::Ordering::Acquire) {
        inflight_records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&pending.page_id);
        dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return ExactStateRecordAdmission::WorkerStopped;
    }
    let bytes = pending.budgeted_bytes();
    let admitted = match &mut pending.admission {
        RecordAdmission::Credit(credit) => {
            // Claim the bytes before the send so two producers cannot both
            // see room. Released on every non-queued path below, and by the
            // worker on finish.
            let reserved = credit.transfer_reserved_bytes(queue_bytes, bytes);
            let held = if reserved {
                queue_bytes
                    .load(std::sync::atomic::Ordering::Acquire)
                    .saturating_sub(bytes)
            } else {
                queue_bytes.fetch_add(bytes, std::sync::atomic::Ordering::AcqRel)
            };
            let fits = reserved || record_fits_queue(held, bytes, queue_bytes_cap);
            if fits {
                pending_count.fetch_add(1, std::sync::atomic::Ordering::Release);
            } else {
                queue_bytes.fetch_sub(bytes, std::sync::atomic::Ordering::Release);
            }
            fits
        }
        // A lease claimed its slot and recurrent-byte budget before cloning.
        RecordAdmission::Leased(_, reservation) => {
            reservation.transfer(pending_count, queue_bytes, bytes);
            true
        }
        RecordAdmission::Released => false,
    };
    if !admitted {
        inflight_records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&pending.page_id);
        dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return ExactStateRecordAdmission::DroppedFull;
    }
    match sender.try_send(pending) {
        Ok(()) => {
            queued.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ExactStateRecordAdmission::Queued
        }
        Err(TrySendError::Full(pending)) => {
            pending_count.fetch_sub(1, std::sync::atomic::Ordering::Release);
            queue_bytes.fetch_sub(bytes, std::sync::atomic::Ordering::Release);
            inflight_records
                .lock()
                .expect("kv inflight record lock poisoned")
                .remove(&pending.page_id);
            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ExactStateRecordAdmission::DroppedFull
        }
        Err(TrySendError::Disconnected(pending)) => {
            pending_count.fetch_sub(1, std::sync::atomic::Ordering::Release);
            queue_bytes.fetch_sub(bytes, std::sync::atomic::Ordering::Release);
            inflight_records
                .lock()
                .expect("kv inflight record lock poisoned")
                .remove(&pending.page_id);
            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ExactStateRecordAdmission::WorkerStopped
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct ExactStateExtra {
    pub(crate) kv_desc: Option<RuntimeKvPageDesc>,
}

fn verify_resident_ownership(
    cache_healthy: &AtomicBool,
    resident_entries: usize,
    allocated_sequences: usize,
) -> Result<()> {
    if resident_entries == allocated_sequences {
        return Ok(());
    }
    if cache_healthy.swap(false, std::sync::atomic::Ordering::AcqRel) {
        let _ = skippy_events::diagnostics::emit(
            skippy_events::diagnostics::ServingDiagnostic::Warning {
                message: "Skippy KV cache disabled after resident ownership mismatch".to_string(),
                context: Some(format!(
                    "radix_entries={resident_entries} allocated_sequences={allocated_sequences}"
                )),
            },
        );
    }
    bail!(
        "resident cache ownership mismatch: radix_entries={resident_entries} allocated_sequences={allocated_sequences}"
    )
}

impl KvStageIntegration {
    /// Attaches an optional resident-prefix cache lifecycle observer.
    /// Never required; a stage with no observer behaves identically.
    #[must_use]
    pub fn with_kv_lifecycle_observer(mut self, observer: Arc<dyn KvLifecycleObserver>) -> Self {
        self.kv_lifecycle_observer = Some(observer);
        self
    }

    pub(crate) fn notify_kv_lifecycle(&self, event: KvLifecycleEvent) {
        if let Some(observer) = self.kv_lifecycle_observer.as_ref() {
            observer.observe(event);
        }
    }

    pub fn mode(&self) -> StageKvMode {
        self.mode
    }

    pub(crate) fn payload_is_exact_state(&self) -> bool {
        self.exact_state_payload().is_some()
    }

    /// Whether a recurrent exact checkpoint belongs to the active native
    /// lane and can be recorded without exporting its KV under the scheduler
    /// lock. Resident hybrids use this path for their serialized fallback.
    pub(crate) fn records_lane_owned_recurrent(&self) -> bool {
        self.exact_state_payload() == Some(StagePrefixCachePayload::KvRecurrent)
            && (self.payload == StagePrefixCachePayload::KvRecurrent
                || self.resident_recurrent_snapshots)
    }

    /// Whether a shared boundary must be captured while prefilling, because
    /// the state at the end of the request cannot stand in for it: exact
    /// state payloads, and resident entries that carry a recurrent snapshot.
    pub(crate) fn captures_boundaries_during_prefill(&self) -> bool {
        self.payload_is_exact_state() || self.resident_recurrent_snapshots
    }

    /// Whether the serving tier retains native resident prefixes independently
    /// of any serialized exact-state payload retained alongside it.
    pub(crate) fn records_resident_prefixes(&self) -> bool {
        self.payload == StagePrefixCachePayload::ResidentKv
    }

    pub(crate) fn exact_state_payload(&self) -> Option<StagePrefixCachePayload> {
        self.payload
            .is_exact_state()
            .then_some(self.payload)
            .or(self
                .serialized_payload
                .filter(|payload| payload.is_exact_state()))
    }

    pub fn should_lookup(&self) -> bool {
        self.cache_healthy
            .load(std::sync::atomic::Ordering::Acquire)
            && matches!(
                self.mode,
                StageKvMode::LookupRecord | StageKvMode::Correctness
            )
    }

    pub fn should_record(&self) -> bool {
        self.cache_healthy
            .load(std::sync::atomic::Ordering::Acquire)
            && matches!(
                self.mode,
                StageKvMode::Record | StageKvMode::LookupRecord | StageKvMode::Correctness
            )
    }

    pub(crate) fn verify_resident_ownership(
        &self,
        resident_entries: usize,
        allocated_sequences: usize,
    ) -> Result<()> {
        verify_resident_ownership(&self.cache_healthy, resident_entries, allocated_sequences)
    }

    pub(crate) fn meets_shared_prefix_min_tokens(&self, matched_tokens: usize) -> bool {
        u64::try_from(matched_tokens).unwrap_or(u64::MAX) >= self.checkpoint_policy.min_tokens
    }

    pub fn try_begin_record(&self, page_id: &str) -> bool {
        if !self
            .exact_state_record_worker_healthy
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return false;
        }
        self.inflight_records
            .lock()
            .expect("kv inflight record lock poisoned")
            .insert(page_id.to_string())
    }

    pub fn finish_record(&self, page_id: &str) {
        self.inflight_records
            .lock()
            .expect("kv inflight record lock poisoned")
            .remove(page_id);
    }

    /// Test-only completion boundary for detached exact-state capture tasks:
    /// waits until every capture scheduled so far has reached a TERMINAL
    /// outcome. Terminal means finished, not successful: a skipped capture
    /// releases its outstanding unit without touching the recorder's
    /// dropped/panics counters, so this boundary deliberately does not turn
    /// skips into errors. Combined with
    /// [`Self::wait_for_exact_state_recording`] it gives tests an exact
    /// ordering — all capture tasks for a request are finished before the
    /// recorder is drained, so no still-detached capture can be missed —
    /// while the unchanged cache-hit assertion remains the proof that the
    /// needed record was actually captured and stored.
    #[cfg(test)]
    pub(crate) fn wait_for_exact_state_captures_idle(
        &self,
        timeout: std::time::Duration,
    ) -> Result<(), String> {
        let deadline = std::time::Instant::now() + timeout;
        while self
            .exact_state_captures_outstanding
            .load(std::sync::atomic::Ordering::Acquire)
            != 0
        {
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "exact-state capture tasks still outstanding after {timeout:?}: \
                     outstanding={}",
                    self.exact_state_captures_outstanding
                        .load(std::sync::atomic::Ordering::Acquire)
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Ok(())
    }

    /// Test-only: number of records the recorder worker has RECEIVED off the
    /// channel (before storing). Lets the saturation test prove worker
    /// receipt, which pending/queued counters alone do not distinguish.
    #[cfg(test)]
    pub(crate) fn wait_for_exact_state_worker_received(
        &self,
        minimum: usize,
        timeout: std::time::Duration,
    ) -> Result<(), String> {
        let deadline = std::time::Instant::now() + timeout;
        while self
            .exact_state_worker_received
            .load(std::sync::atomic::Ordering::Acquire)
            < minimum
        {
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "recorder worker never received {minimum} record(s) within {timeout:?}; received={}",
                    self.exact_state_worker_received
                        .load(std::sync::atomic::Ordering::Acquire)
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Ok(())
    }

    /// Test-only wait for the recorder worker to drain everything enqueued and
    /// finish in a completed state. Call only after
    /// [`Self::wait_for_exact_state_captures_idle`]; fails with the recorder
    /// counters so a drop and a worker panic are distinguishable instead of
    /// surfacing as a downstream cache miss.
    #[cfg(test)]
    pub(crate) fn wait_for_exact_state_recording(
        &self,
        timeout: std::time::Duration,
    ) -> Result<(), String> {
        let state = || {
            format!(
                "pending={} queued={} dropped={} worker_panics={} healthy={}",
                self.exact_state_records_pending
                    .load(std::sync::atomic::Ordering::Acquire),
                self.exact_state_records_queued
                    .load(std::sync::atomic::Ordering::Acquire),
                self.exact_state_records_dropped
                    .load(std::sync::atomic::Ordering::Acquire),
                self.exact_state_record_worker_panics
                    .load(std::sync::atomic::Ordering::Acquire),
                self.exact_state_record_worker_healthy
                    .load(std::sync::atomic::Ordering::Acquire),
            )
        };
        let deadline = std::time::Instant::now() + timeout;
        // The recorder worker must drain everything enqueued.
        while self
            .exact_state_records_pending
            .load(std::sync::atomic::Ordering::Acquire)
            != 0
        {
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "exact-state recording did not drain within {timeout:?}; {}",
                    state()
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        // Drained records must have completed, not failed.
        let dropped = self
            .exact_state_records_dropped
            .load(std::sync::atomic::Ordering::Acquire);
        let panics = self
            .exact_state_record_worker_panics
            .load(std::sync::atomic::Ordering::Acquire);
        if dropped != 0 || panics != 0 {
            return Err(format!(
                "exact-state recording finished in a failure state: dropped={dropped}, worker_panics={panics}"
            ));
        }
        Ok(())
    }

    /// Test-only: whether the EXACT identity is RETAINED in the radix right
    /// now. Read-only peek — no LRU or refcount side effects. `peek_recurrent`
    /// is a LONGEST-PREFIX lookup, so a retained shorter entry would make a
    /// longer identity look retained; requiring `matched_tokens` to equal the
    /// queried length and the stored path to equal the queried tokens closes
    /// that false-certification. Storage-time logs name candidate keys; this
    /// checks actual retention after the recorder drains.
    #[cfg(test)]
    pub(crate) fn retained_exact_identity(&self, namespace: &str, token_ids: &[i32]) -> bool {
        match self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .peek_recurrent(namespace, token_ids)
        {
            Some(matched) => {
                matched.matched_tokens == token_ids.len() && matched.stored_tokens == token_ids
            }
            None => false,
        }
    }

    /// Test-only: how many tokens of `query_tokens` a retained exact entry
    /// under `namespace` would serve (read-only peek), i.e. the restore
    /// eligibility of the stored state against a later prompt.
    #[cfg(test)]
    pub(crate) fn eligible_exact_match_tokens(
        &self,
        namespace: &str,
        query_tokens: &[i32],
    ) -> Option<usize> {
        self.radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .peek_recurrent(namespace, query_tokens)
            .map(|matched| matched.matched_tokens)
    }

    /// Payload bytes waiting for the worker: the status contract's
    /// `write_queue_bytes`.
    pub fn exact_state_record_queue_bytes(&self) -> u64 {
        self.exact_state_record_queue_bytes
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The durable tier, when one is open.
    pub fn l3(&self) -> Option<&Arc<skippy_cache::L3Tier>> {
        self.l3.as_ref()
    }

    pub(crate) fn enqueue_exact_state_record(
        &self,
        pending: PendingExactStateRecord,
    ) -> ExactStateRecordAdmission {
        let admission = self.exact_state_record_worker.with_sender(|sender| {
            let Some(sender) = sender else {
                self.finish_record(&pending.page_id);
                self.exact_state_records_dropped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return ExactStateRecordAdmission::WorkerStopped;
            };
            enqueue_exact_state_record(
                sender,
                &self.inflight_records,
                &self.exact_state_records_queued,
                &self.exact_state_records_dropped,
                &self.exact_state_records_pending,
                &self.exact_state_record_queue_bytes,
                self.exact_state_record_queue_limit,
                &self.exact_state_record_worker_healthy,
                pending,
            )
        });
        if matches!(
            admission,
            ExactStateRecordAdmission::DroppedFull | ExactStateRecordAdmission::WorkerStopped
        ) {
            self.notify_kv_lifecycle(KvLifecycleEvent::ExactStateRecordFailed);
        }
        admission
    }

    pub async fn hello(&self) -> Result<()> {
        Ok(())
    }

    pub async fn lookup_prefixes(
        &self,
        _identities: Vec<PageIdentity>,
    ) -> Result<LookupBatchOutcome> {
        Ok(LookupBatchOutcome {
            pages: Vec::new(),
            errors: Vec::new(),
        })
    }

    pub async fn record_page(
        &self,
        page_id: String,
        identity: PageIdentity,
        bytes: &[u8],
        annotations: BTreeMap<String, String>,
    ) -> Result<KvPageManifest> {
        Ok(self
            .record_page_into(page_id, identity, bytes.len(), annotations, |output| {
                output.copy_from_slice(bytes);
                Ok(())
            })
            .await?
            .manifest)
    }

    pub async fn record_page_into(
        &self,
        page_id: String,
        identity: PageIdentity,
        byte_size: usize,
        mut annotations: BTreeMap<String, String>,
        write_page: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<RecordPageOutcome> {
        let mut bytes = vec![0; byte_size];
        write_page(&mut bytes)?;
        let checksum = local_trust_checksum(&page_id, byte_size as u64);
        annotations.insert(
            "mesh.skippy.prefix-cache-disabled".to_string(),
            "true".to_string(),
        );
        Ok(RecordPageOutcome {
            manifest: KvPageManifest {
                schema_version: MANIFEST_SCHEMA_VERSION,
                page_id,
                identity: Some(identity),
                state: PageState::Empty as i32,
                byte_size: byte_size as u64,
                shm_offset: 0,
                shm_len: byte_size as u64,
                checksum: Some(checksum),
                lease: None,
                annotations,
            },
            write_ms: 0.0,
            checksum_ms: 0.0,
        })
    }

    pub async fn attach_page(&self, _page_id: &str) -> Result<AttachedPage> {
        bail!("prefix cache integration is not included in mesh skippy-serving")
    }

    pub async fn drop_session(&self, _session_id: &str) -> Result<u64> {
        Ok(0)
    }

    pub fn attrs(&self) -> Vec<(&'static str, Value)> {
        let radix_stats = match self.radix.try_lock() {
            Ok(radix) => Some(radix.stats()),
            Err(std::sync::TryLockError::Poisoned(error)) => Some(error.into_inner().stats()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
        let radix = radix_stats.unwrap_or_default();
        let l2 = self
            .l2
            .as_ref()
            .map(|tier| tier.stats())
            .unwrap_or_default();
        let activations = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned");
        let activations = activations.stats();
        let exact_blob_stats = self.exact_blobs.try_lock().ok().map(|blobs| {
            (
                blobs.physical_bytes(),
                blobs.block_count(),
                blobs.logical_ref_count(),
            )
        });
        let exact_state_stats_busy = radix_stats.is_none() || exact_blob_stats.is_none();
        let (exact_physical_bytes, exact_blocks, exact_block_refs) =
            exact_blob_stats.unwrap_or_default();
        let (resident_allocated_sequences, resident_free_sequences, resident_quarantined_sequences) =
            lock_resident_sequences(&self.resident_sequences).stats();
        let resident_sequence_drift = radix
            .resident_entries
            .abs_diff(resident_allocated_sequences);
        let (resident_capacity_reservations, resident_capacity_reserved_tokens) =
            self.resident_capacity_reservations.stats();
        let output_token_entries = self
            .output_tokens
            .try_lock()
            .ok()
            .map(|tokens| tokens.len())
            .unwrap_or_default();
        let (split_prefill_sessions, split_prefill_tokens) = self
            .split_prefill_tokens
            .try_lock()
            .ok()
            .map(|sessions| {
                (
                    sessions.len(),
                    sessions
                        .values()
                        .map(Vec::len)
                        .fold(0, usize::saturating_add),
                )
            })
            .unwrap_or_default();
        vec![
            ("skippy.kv.mode", json!(format!("{:?}", self.mode))),
            ("skippy.kv.payload", json!(format!("{:?}", self.payload))),
            (
                "skippy.kv.payload_selection_reason",
                json!(self.payload_selection_reason),
            ),
            (
                "skippy.kv.loaded_state_kind",
                json!(self.loaded_state_kind.map(|kind| format!("{kind:?}"))),
            ),
            (
                "skippy.kv.loaded_memory_cache_resident",
                json!(self.loaded_memory_cache.map(|cache| cache.resident)),
            ),
            (
                "skippy.kv.loaded_memory_cache_kv_recurrent",
                json!(self.loaded_memory_cache.map(|cache| cache.kv_recurrent)),
            ),
            (
                "skippy.kv.page_size_tokens",
                json!(self.checkpoint_policy.page_size_tokens),
            ),
            ("skippy.kv.resident_entries", json!(radix.resident_entries)),
            ("skippy.kv.resident_tokens", json!(radix.resident_tokens)),
            ("skippy.kv.radix.namespaces", json!(radix.namespaces)),
            ("skippy.kv.radix.nodes", json!(radix.nodes)),
            ("skippy.kv.radix.token_edges", json!(radix.token_edges)),
            ("skippy.kv.radix.splits", json!(radix.splits)),
            (
                "skippy.kv.radix.resident_entries",
                json!(radix.resident_entries),
            ),
            (
                "skippy.kv.radix.resident_active_refs",
                json!(radix.resident_active_refs),
            ),
            (
                "skippy.kv.radix.recurrent_entries",
                json!(radix.recurrent_entries),
            ),
            (
                "skippy.kv.radix.recurrent_active_refs",
                json!(radix.recurrent_active_refs),
            ),
            (
                "skippy.kv.radix.resident_evictions",
                json!(radix.resident_evictions),
            ),
            (
                "skippy.kv.radix.recurrent_evictions",
                json!(radix.recurrent_evictions),
            ),
            (
                "skippy.kv.resident_estimated_bytes",
                json!(radix.resident_logical_bytes),
            ),
            (
                "skippy.kv.max_entries",
                json!(self.resident_config.max_entries),
            ),
            ("skippy.kv.max_bytes", json!(self.resident_config.max_bytes)),
            (
                "skippy.activation_cache.entries",
                json!(activations.entries),
            ),
            (
                "skippy.activation_cache.resident_bytes",
                json!(activations.resident_bytes),
            ),
            ("skippy.exact_cache.entries", json!(radix.recurrent_entries)),
            (
                "skippy.exact_cache.logical_bytes",
                json!(radix.recurrent_logical_bytes),
            ),
            (
                "skippy.exact_cache.physical_bytes",
                json!(exact_physical_bytes),
            ),
            ("skippy.exact_cache.blocks", json!(exact_blocks)),
            ("skippy.exact_cache.block_refs", json!(exact_block_refs)),
            (
                "skippy.exact_cache.stats_busy",
                json!(exact_state_stats_busy),
            ),
            (
                "skippy.exact_cache.export_failures",
                json!(
                    self.snapshot_export_failures
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.records_queued",
                json!(
                    self.exact_state_records_queued
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.records_dropped",
                json!(
                    self.exact_state_records_dropped
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.records_pending",
                json!(
                    self.exact_state_records_pending
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.worker_healthy",
                json!(
                    self.exact_state_record_worker_healthy
                        .load(std::sync::atomic::Ordering::Acquire)
                ),
            ),
            (
                "skippy.exact_cache.worker_panics",
                json!(
                    self.exact_state_record_worker_panics
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.max_bytes",
                json!(self.exact_byte_limits.soft_bytes),
            ),
            (
                "skippy.exact_cache.hard_max_bytes",
                json!(self.exact_byte_limits.hard_bytes),
            ),
            (
                "skippy.exact_cache.max_entries",
                json!(self.exact_max_entries),
            ),
            ("skippy.kv.l2.enabled", json!(self.l2.is_some())),
            ("skippy.kv.l2.budget_bytes", json!(l2.budget_bytes)),
            ("skippy.kv.l2.bytes", json!(l2.bytes)),
            ("skippy.kv.l2.logical_bytes", json!(l2.logical_bytes)),
            ("skippy.kv.l2.entries", json!(l2.entries)),
            ("skippy.kv.l2.segments", json!(l2.segments)),
            ("skippy.kv.l2.hits", json!(l2.hits)),
            ("skippy.kv.l2.misses", json!(l2.misses)),
            ("skippy.kv.l2.inserts", json!(l2.inserts)),
            ("skippy.kv.l2.evictions", json!(l2.evictions)),
            (
                "skippy.kv.l2.admission_rejects",
                json!(l2.admission_rejects),
            ),
            ("skippy.kv.l2.refused_bytes", json!(l2.refused_bytes)),
            (
                "skippy.kv.l3.cachegen_enabled",
                json!(self.cachegen_serving_enabled),
            ),
            (
                "skippy.kv.output_token_entries",
                json!(output_token_entries),
            ),
            (
                "skippy.kv.split_prefill_sessions",
                json!(split_prefill_sessions),
            ),
            (
                "skippy.kv.split_prefill_tokens",
                json!(split_prefill_tokens),
            ),
            (
                "skippy.kv.split_prefill_bytes",
                json!(split_prefill_tokens.saturating_mul(std::mem::size_of::<i32>())),
            ),
            (
                "skippy.kv.resident_allocated_sequences",
                json!(resident_allocated_sequences),
            ),
            (
                "skippy.kv.resident_free_sequences",
                json!(resident_free_sequences),
            ),
            (
                "skippy.kv.resident_quarantined_sequences",
                json!(resident_quarantined_sequences),
            ),
            (
                "skippy.kv.resident_sequence_drift",
                json!(resident_sequence_drift),
            ),
            (
                "skippy.kv.capacity_reservations",
                json!(resident_capacity_reservations),
            ),
            (
                "skippy.kv.capacity_reserved_tokens",
                json!(resident_capacity_reserved_tokens),
            ),
            ("skippy.kv.correctness_mode", json!(self.correctness_mode)),
            (
                "skippy.kv.cache_healthy",
                json!(
                    self.cache_healthy
                        .load(std::sync::atomic::Ordering::Acquire)
                ),
            ),
            (
                "skippy.kv.trust_local_writes",
                json!(self.trust_local_writes),
            ),
            (
                "skippy.kv.shared_prefix_min_tokens",
                json!(self.checkpoint_policy.min_tokens),
            ),
            (
                "skippy.kv.shared_prefix_stride_tokens",
                json!(self.checkpoint_policy.stride_tokens),
            ),
            (
                "skippy.kv.shared_prefix_record_limit",
                json!(self.checkpoint_policy.record_limit),
            ),
        ]
        .into_iter()
        .collect()
    }

    /// Test-only compatibility helper for exercising the side-cache capacity
    /// accounting without constructing a frontend sampling fingerprint.
    #[cfg(test)]
    pub fn record_cached_first_token(&self, identity: &PrefillKvIdentity, predicted: i32) -> bool {
        self.record_cached_first_token_with_key(&identity.page_id, identity, predicted)
    }

    /// Records a first token under a key that includes the request's sampling
    /// semantics. Sampled-output paths must use this method.
    pub(crate) fn record_cached_first_token_with_key(
        &self,
        cache_key: &str,
        identity: &PrefillKvIdentity,
        predicted: i32,
    ) -> bool {
        if !self.should_record()
            || identity.identity.token_count < self.checkpoint_policy.min_tokens
        {
            return false;
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .record_first(cache_key, predicted)
    }

    /// Test-only compatibility helper for the identity-only cache probe.
    #[cfg(test)]
    pub fn lookup_cached_first_token(&self, identity: &PrefillKvIdentity) -> Option<i32> {
        self.lookup_cached_first_token_with_key(&identity.page_id)
    }

    /// Looks up a first token using the sampling-aware cache key generated by
    /// the frontend. Callers should gate this operation on replay-safe
    /// sampling before attempting the lookup.
    pub(crate) fn lookup_cached_first_token_with_key(&self, cache_key: &str) -> Option<i32> {
        if !self.should_lookup() {
            return None;
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .lookup_first(cache_key)
    }

    pub fn record_cached_replay_tokens(
        &self,
        cache_key: &str,
        identity: &PrefillKvIdentity,
        previous: &[i32],
        predicted: i32,
        max_replay_tokens: usize,
    ) -> Option<usize> {
        if !self.should_record()
            || max_replay_tokens == 0
            || previous.len() >= max_replay_tokens
            || identity.identity.token_count < self.checkpoint_policy.min_tokens
        {
            return None;
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .record_replay(cache_key, previous, predicted, max_replay_tokens)
    }

    pub fn lookup_cached_replay_tokens(&self, cache_key: &str, max_tokens: usize) -> Vec<i32> {
        if !self.should_lookup() || max_tokens == 0 {
            return Vec::new();
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .lookup_replay(cache_key, max_tokens)
    }
}

fn local_trust_checksum(page_id: &str, byte_size: u64) -> Checksum {
    let mut digest = Sha256::new();
    digest.update(b"skippy-local-trust-v1");
    digest.update(page_id.as_bytes());
    digest.update(byte_size.to_le_bytes());
    Checksum {
        algorithm: ChecksumAlgorithm::Sha256 as i32,
        digest: digest.finalize().to_vec(),
    }
}

#[cfg(test)]
mod exact_state_record_queue_tests {
    use super::{
        LEASED_RECORD_SLOTS, LeaseLedger, LeasedKvPage, LeasedRecordRefusal,
        LeasedRecordReservation, RecordAdmission,
    };
    use skippy_cache::ExactStatePayload;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::sync_channel,
    };

    use super::{
        BTreeSet, EXACT_STATE_RECORD_CAPACITY, ExactStateExtra, ExactStateRecordAdmission,
        ExactStateRecordWorker, ExactStateWorkerHandles, KvLifecycleEvent, KvLifecycleObserver,
        PendingExactStateRecord, enqueue_exact_state_record, run_exact_state_record_job,
    };
    use crate::kv_integration::exact_state::{CaptureAdmission, ExactStateAdmissionCredit};

    /// A fresh, observable budget: the SAME counters back every credit a test
    /// creates, so release paths are asserted on real accounting, not on
    /// throwaway stubs.
    fn budget() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)))
    }

    fn credit(
        total: &Arc<AtomicUsize>,
        best_effort: &Arc<AtomicUsize>,
        class: CaptureAdmission,
    ) -> ExactStateAdmissionCredit {
        ExactStateAdmissionCredit::acquire(total, best_effort, class)
            .expect("a fresh budget must admit")
    }

    struct RecordingObserver(Arc<Mutex<Vec<KvLifecycleEvent>>>);

    impl KvLifecycleObserver for RecordingObserver {
        fn observe(&self, event: KvLifecycleEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    fn pending(
        page_id: &str,
        admission_credit: ExactStateAdmissionCredit,
    ) -> PendingExactStateRecord {
        PendingExactStateRecord {
            page_id: page_id.to_string(),
            payload: ExactStatePayload::full_state(vec![1]),
            extra: ExactStateExtra::default(),
            namespace: "test".to_string(),
            token_ids: vec![1],
            l3_fill_claim: None,
            write_through_l3: true,
            l2_promotion_digest: None,
            l3_cost: None,
            admission: RecordAdmission::Credit(admission_credit),
        }
    }

    fn pending_with_bytes(page_id: &str, bytes: usize) -> PendingExactStateRecord {
        let (total, best_effort) = budget();
        PendingExactStateRecord {
            payload: ExactStatePayload::full_state(vec![1; bytes]),
            ..pending(
                page_id,
                credit(&total, &best_effort, CaptureAdmission::BestEffort),
            )
        }
    }

    const CAP: u64 = 1024;

    #[test]
    fn final_worker_owner_drains_queued_records_before_drop_returns() {
        let (sender, receiver) = sync_channel(2);
        let completed = Arc::new(AtomicUsize::new(0));
        let worker_completed = completed.clone();
        let task = std::thread::spawn(move || {
            while receiver.recv().is_ok() {
                worker_completed.fetch_add(1, Ordering::Release);
            }
        });
        let worker = Arc::new(ExactStateRecordWorker::new(sender, task, None));
        let (total, best_effort) = budget();
        worker.with_sender(|sender| {
            sender
                .unwrap()
                .send(pending(
                    "latest",
                    credit(&total, &best_effort, CaptureAdmission::BestEffort),
                ))
                .unwrap()
        });

        drop(worker);

        assert_eq!(completed.load(Ordering::Acquire), 1);
    }

    #[test]
    fn queue_is_bounded_in_bytes_not_entries() {
        let (sender, _receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
        let inflight = Mutex::new(BTreeSet::from(["a".to_string(), "b".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(0);
        let queue_bytes = AtomicU64::new(0);
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                pending_with_bytes("a", 700),
            ),
            ExactStateRecordAdmission::Queued
        );
        assert_eq!(queue_bytes.load(Ordering::Relaxed), 700);

        // Plenty of entry slots left; the bytes are what is full.
        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                pending_with_bytes("b", 700),
            ),
            ExactStateRecordAdmission::DroppedFull
        );
        assert_eq!(
            queue_bytes.load(Ordering::Relaxed),
            700,
            "a dropped record left bytes claimed"
        );
        assert!(!inflight.lock().unwrap().contains("b"));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(pending_count.load(Ordering::Relaxed), 1);
    }

    /// A leased page over `bytes` of test-owned memory. The `Vec` stands in
    /// for the lane's cells; the ledger plays the lane.
    fn leased(
        page_id: &str,
        cells: &[u8],
        ledger: &LeaseLedger,
        reservation: LeasedRecordReservation,
    ) -> PendingExactStateRecord {
        let spans = skippy_runtime::KvPageSpans {
            desc: skippy_runtime::RuntimeKvPageDesc {
                version: 1,
                layer_start: 0,
                layer_end: 1,
                token_start: 0,
                token_count: 1,
                layer_count: 1,
                k_type: 0,
                v_type: 0,
                k_row_bytes: 0,
                v_row_bytes: 0,
                v_element_bytes: 0,
                k_idx_row_bytes: 0,
                payload_bytes: cells.len() as u64,
                flags: 0,
                codec: 0,
                component_count: 0,
                components: Box::default(),
            },
            spans: vec![skippy_runtime::KvPageSpan {
                data: cells.as_ptr(),
                bytes: cells.len(),
            }],
        };
        PendingExactStateRecord {
            page_id: page_id.to_string(),
            payload: ExactStatePayload::kv_recurrent(Vec::new(), vec![7; 4]),
            extra: ExactStateExtra::default(),
            namespace: "test".to_string(),
            token_ids: vec![1],
            l3_fill_claim: None,
            write_through_l3: true,
            l2_promotion_digest: None,
            l3_cost: None,
            admission: RecordAdmission::Leased(
                LeasedKvPage {
                    spans,
                    pin: ledger.pin(1),
                },
                reservation,
            ),
        }
    }

    #[test]
    fn leased_records_charge_recurrent_bytes_and_stop_at_their_slots() {
        let (sender, _receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
        let inflight = Mutex::new(BTreeSet::new());
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = Arc::new(AtomicUsize::new(0));
        let queue_bytes = Arc::new(AtomicU64::new(0));
        let worker_healthy = AtomicBool::new(true);
        let ledger = LeaseLedger::default();
        let cells = vec![9_u8; 64];

        for index in 0..LEASED_RECORD_SLOTS {
            let page_id = format!("lease-{index}");
            let reservation =
                LeasedRecordReservation::acquire(&pending_count, &queue_bytes, 4, CAP)
                    .expect("a leased slot and recurrent bytes fit");
            inflight.lock().unwrap().insert(page_id.clone());
            assert_eq!(
                enqueue_exact_state_record(
                    &sender,
                    &inflight,
                    &queued,
                    &dropped,
                    &pending_count,
                    &queue_bytes,
                    CAP,
                    &worker_healthy,
                    leased(&page_id, &cells, &ledger, reservation),
                ),
                ExactStateRecordAdmission::Queued,
                "lease {index} is admitted without a copying credit"
            );
        }
        assert_eq!(
            queue_bytes.load(Ordering::Relaxed),
            (LEASED_RECORD_SLOTS * 4) as u64,
            "queued recurrent snapshots count against the byte budget"
        );
        assert_eq!(pending_count.load(Ordering::Relaxed), LEASED_RECORD_SLOTS);

        // The slot after the carve-out is refused before a recurrent clone
        // or a lane pin is constructed.
        assert_eq!(
            LeasedRecordReservation::acquire(&pending_count, &queue_bytes, 4, CAP).unwrap_err(),
            LeasedRecordRefusal::QueueFull
        );
        assert_eq!(pending_count.load(Ordering::Relaxed), LEASED_RECORD_SLOTS);

        // Every admitted lease still pins the lane; a credit record still
        // fits in the slots the carve-out leaves it.
        assert!(ledger.in_flight());
        let (total, best_effort) = budget();
        inflight.lock().unwrap().insert("credit".to_string());
        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &AtomicU64::new(0),
                CAP,
                &worker_healthy,
                pending(
                    "credit",
                    credit(&total, &best_effort, CaptureAdmission::Continuation)
                ),
            ),
            ExactStateRecordAdmission::Queued,
            "a credit never competes with leases for a slot"
        );
    }

    #[test]
    fn materialising_a_lease_copies_the_cells_once_and_releases_the_lane() {
        let ledger = LeaseLedger::default();
        let cells = (0..=255_u8).collect::<Vec<_>>();
        let pending_count = Arc::new(AtomicUsize::new(0));
        let queue_bytes = Arc::new(AtomicU64::new(0));
        let reservation = LeasedRecordReservation::acquire(&pending_count, &queue_bytes, 4, CAP)
            .expect("test snapshot fits");
        let pending = leased("page", &cells, &ledger, reservation);
        assert!(ledger.in_flight());
        assert_eq!(pending.byte_len(), 256 + 4);
        assert_eq!(pending.budgeted_bytes(), 4, "recurrent bytes are budgeted");

        let pending = pending.materialise_leased_kv().expect("readable cells");
        assert!(!ledger.in_flight(), "the pin goes with the lease");
        assert!(matches!(pending.admission, RecordAdmission::Released));
        assert_eq!(pending.byte_len(), 256 + 4);
        assert_eq!(pending.budgeted_bytes(), 0);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
        assert_eq!(queue_bytes.load(Ordering::Acquire), 0);
        let ExactStatePayload::KvRecurrent { kv, recurrent } = &pending.payload else {
            panic!("materialised payload keeps its kind");
        };
        assert_eq!(kv.as_cow().unwrap().as_ref(), cells.as_slice());
        assert_eq!(recurrent.as_cow().unwrap().as_ref(), &[7; 4]);
    }

    #[test]
    fn one_oversized_record_is_admitted_into_an_empty_queue() {
        let (sender, _receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
        let inflight = Mutex::new(BTreeSet::from(["huge".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(0);
        let queue_bytes = AtomicU64::new(0);
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                pending_with_bytes("huge", 4096),
            ),
            ExactStateRecordAdmission::Queued,
            "a large model could never record if one export over the cap were refused"
        );
        assert_eq!(queue_bytes.load(Ordering::Relaxed), 4096);
        assert!(queue_bytes.load(Ordering::Acquire) >= CAP);
    }

    #[test]
    fn finishing_a_record_releases_its_bytes() {
        let inflight = Mutex::new(BTreeSet::from(["done".to_string()]));
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(1);
        let queue_bytes = AtomicU64::new(300);
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                queue_bytes: &queue_bytes,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            None,
            pending_with_bytes("done", 300),
            |_| Ok(()),
        );

        assert_eq!(queue_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(pending_count.load(Ordering::Relaxed), 0);
        assert!(inflight.lock().unwrap().is_empty());
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn leased_worker_completion_releases_recurrent_bytes_and_lane_pin() {
        let (sender, receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
        let inflight = Mutex::new(BTreeSet::from(["lease".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = Arc::new(AtomicUsize::new(0));
        let queue_bytes = Arc::new(AtomicU64::new(0));
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);
        let ledger = LeaseLedger::default();
        let cells = vec![9_u8; 64];
        let reservation = LeasedRecordReservation::acquire(&pending_count, &queue_bytes, 4, CAP)
            .expect("test snapshot fits");
        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                leased("lease", &cells, &ledger, reservation),
            ),
            ExactStateRecordAdmission::Queued
        );
        assert_eq!(queue_bytes.load(Ordering::Acquire), 4);
        assert!(ledger.in_flight());

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                queue_bytes: &queue_bytes,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            None,
            receiver.recv().unwrap(),
            |record| {
                assert_eq!(record.payload.byte_len(), 68);
                Ok(())
            },
        );
        assert!(!ledger.in_flight());
        assert_eq!(queue_bytes.load(Ordering::Acquire), 0);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
        assert!(inflight.lock().unwrap().is_empty());
        assert_eq!(dropped.load(Ordering::Acquire), 0);
    }

    #[test]
    fn stopped_worker_releases_preclaimed_leased_admission() {
        let (sender, _receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
        let inflight = Mutex::new(BTreeSet::from(["stopped".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = Arc::new(AtomicUsize::new(0));
        let queue_bytes = Arc::new(AtomicU64::new(0));
        let worker_healthy = AtomicBool::new(false);
        let ledger = LeaseLedger::default();
        let cells = vec![9_u8; 64];
        let reservation = LeasedRecordReservation::acquire(&pending_count, &queue_bytes, 4, CAP)
            .expect("test snapshot fits");
        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                leased("stopped", &cells, &ledger, reservation),
            ),
            ExactStateRecordAdmission::WorkerStopped
        );
        assert!(!ledger.in_flight());
        assert_eq!(queue_bytes.load(Ordering::Acquire), 0);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
        assert!(inflight.lock().unwrap().is_empty());
    }

    #[test]
    fn disconnected_channel_releases_transferred_leased_admission() {
        let (sender, receiver) = sync_channel(EXACT_STATE_RECORD_CAPACITY);
        drop(receiver);
        let inflight = Mutex::new(BTreeSet::from(["disconnected".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = Arc::new(AtomicUsize::new(0));
        let queue_bytes = Arc::new(AtomicU64::new(0));
        let worker_healthy = AtomicBool::new(true);
        let ledger = LeaseLedger::default();
        let cells = vec![9_u8; 64];
        let reservation = LeasedRecordReservation::acquire(&pending_count, &queue_bytes, 4, CAP)
            .expect("test snapshot fits");
        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                leased("disconnected", &cells, &ledger, reservation),
            ),
            ExactStateRecordAdmission::WorkerStopped
        );
        assert!(!ledger.in_flight());
        assert_eq!(queue_bytes.load(Ordering::Acquire), 0);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
        assert!(inflight.lock().unwrap().is_empty());
    }

    #[test]
    fn full_queue_drops_optional_record_and_releases_inflight_page() {
        let (total, best_effort) = budget();
        let (sender, receiver) = sync_channel(1);
        // One BestEffort record occupies the channel buffer; its credit stays
        // observable in `total`.
        sender
            .send(pending(
                "queued",
                credit(&total, &best_effort, CaptureAdmission::BestEffort),
            ))
            .unwrap();
        let inflight = Mutex::new(BTreeSet::from(["dropped".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(0);
        let queue_bytes = AtomicU64::new(0);
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                pending(
                    "dropped",
                    credit(&total, &best_effort, CaptureAdmission::Continuation)
                ),
            ),
            ExactStateRecordAdmission::DroppedFull
        );
        assert!(!inflight.lock().unwrap().contains("dropped"));
        assert_eq!(queued.load(Ordering::Relaxed), 0);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(pending_count.load(Ordering::Relaxed), 0);
        assert_eq!(
            total.load(Ordering::Acquire),
            1,
            "the Full rejection must release the dropped record's credit; the buffered one stays"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            1,
            "the buffered BestEffort credit is still outstanding"
        );

        // Receiver drop: discarding the channel drops the buffered record and
        // releases its credit too.
        drop(sender);
        drop(receiver);
        assert_eq!(
            total.load(Ordering::Acquire),
            0,
            "receiver drop released the buffered credit"
        );
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
    }

    #[test]
    fn disconnected_worker_releases_inflight_page() {
        let (total, best_effort) = budget();
        let (sender, receiver) = sync_channel(1);
        drop(receiver);
        let inflight = Mutex::new(BTreeSet::from(["orphaned".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(0);
        let queue_bytes = AtomicU64::new(0);
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                pending(
                    "orphaned",
                    credit(&total, &best_effort, CaptureAdmission::Continuation)
                ),
            ),
            ExactStateRecordAdmission::WorkerStopped
        );
        assert!(!inflight.lock().unwrap().contains("orphaned"));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(pending_count.load(Ordering::Relaxed), 0);
        assert_eq!(
            total.load(Ordering::Acquire),
            0,
            "the disconnected channel must release the orphaned credit"
        );
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
    }

    #[test]
    fn worker_keeps_page_inflight_until_record_finishes() {
        let (total, best_effort) = budget();
        let (sender, receiver) = sync_channel(1);
        let inflight = Arc::new(Mutex::new(BTreeSet::from(["page".to_string()])));
        let worker_inflight = inflight.clone();
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = Arc::new(AtomicUsize::new(0));
        let worker_pending_count = pending_count.clone();
        let queue_bytes = AtomicU64::new(0);
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                pending(
                    "page",
                    credit(&total, &best_effort, CaptureAdmission::Continuation)
                ),
            ),
            ExactStateRecordAdmission::Queued
        );
        assert!(inflight.lock().unwrap().contains("page"));
        assert_eq!(
            total.load(Ordering::Acquire),
            1,
            "the credit is outstanding while the record sits in the channel"
        );

        let worker = std::thread::spawn(move || {
            let pending = receiver.recv().unwrap();
            assert!(worker_inflight.lock().unwrap().contains(&pending.page_id));
            worker_inflight.lock().unwrap().remove(&pending.page_id);
            worker_pending_count.fetch_sub(1, Ordering::Relaxed);
            // `pending` (and its credit) drops when this closure returns.
        });
        worker.join().unwrap();

        assert!(!inflight.lock().unwrap().contains("page"));
        assert_eq!(queued.load(Ordering::Relaxed), 1);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        assert_eq!(pending_count.load(Ordering::Relaxed), 0);
        assert_eq!(
            total.load(Ordering::Acquire),
            0,
            "the credit released when the record dropped after processing"
        );
    }

    #[test]
    fn worker_success_notifies_completed_not_mere_admission() {
        let inflight = Mutex::new(BTreeSet::from(["written".to_string()]));
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(1);
        let queue_bytes = AtomicU64::new(1);
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);
        let events: Arc<Mutex<Vec<KvLifecycleEvent>>> = Arc::default();
        let observer: Arc<dyn KvLifecycleObserver> = Arc::new(RecordingObserver(events.clone()));
        let (total, best_effort) = budget();

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                queue_bytes: &queue_bytes,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            Some(&observer),
            pending(
                "written",
                credit(&total, &best_effort, CaptureAdmission::Continuation),
            ),
            |_| Ok(()),
        );

        assert_eq!(
            *events.lock().unwrap(),
            vec![KvLifecycleEvent::ExactStateRecordCompleted]
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        assert_eq!(
            total.load(Ordering::Acquire),
            0,
            "successful completion released the real credit"
        );
    }

    #[test]
    fn worker_write_error_notifies_failed_not_completed() {
        let inflight = Mutex::new(BTreeSet::from(["broken".to_string()]));
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(1);
        let queue_bytes = AtomicU64::new(1);
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);
        let events: Arc<Mutex<Vec<KvLifecycleEvent>>> = Arc::default();
        let observer: Arc<dyn KvLifecycleObserver> = Arc::new(RecordingObserver(events.clone()));
        let (total, best_effort) = budget();

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                queue_bytes: &queue_bytes,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            Some(&observer),
            pending(
                "broken",
                credit(&total, &best_effort, CaptureAdmission::Continuation),
            ),
            |_| anyhow::bail!("simulated write failure"),
        );

        assert_eq!(
            *events.lock().unwrap(),
            vec![KvLifecycleEvent::ExactStateRecordFailed]
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(
            total.load(Ordering::Acquire),
            0,
            "the worker error path released the real credit"
        );
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
    }

    #[test]
    fn worker_panic_fails_closed_and_releases_all_record_bookkeeping() {
        let (sender, _receiver) = sync_channel(1);
        let inflight = Mutex::new(BTreeSet::from(["panicked".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(1);
        let queue_bytes = AtomicU64::new(1);
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);
        let (total, best_effort) = budget();

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                queue_bytes: &queue_bytes,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            None,
            pending(
                "panicked",
                credit(&total, &best_effort, CaptureAdmission::Continuation),
            ),
            |_| panic!("injected exact-record worker failure"),
        );

        assert!(!worker_healthy.load(Ordering::Acquire));
        assert_eq!(worker_panics.load(Ordering::Relaxed), 1);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
        assert!(inflight.lock().unwrap().is_empty());
        assert_eq!(
            total.load(Ordering::Acquire),
            0,
            "the unwind dropped the record and released its real credit"
        );
        assert_eq!(best_effort.load(Ordering::Acquire), 0);

        inflight.lock().unwrap().insert("later".to_string());
        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &queue_bytes,
                CAP,
                &worker_healthy,
                pending(
                    "later",
                    credit(&total, &best_effort, CaptureAdmission::Continuation)
                ),
            ),
            ExactStateRecordAdmission::WorkerStopped
        );
        assert!(inflight.lock().unwrap().is_empty());
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
        assert_eq!(
            total.load(Ordering::Acquire),
            0,
            "the WorkerStopped follow-up also released its real credit"
        );
    }
}

#[cfg(test)]
mod telemetry_error_class_tests {
    use super::telemetry_error_class_from_message;

    #[test]
    fn maps_detailed_errors_to_bounded_classes() {
        assert_eq!(
            telemetry_error_class_from_message("checksum mismatch for page abc"),
            "integrity"
        );
        assert_eq!(
            telemetry_error_class_from_message("permission denied: /secret/path"),
            "permission"
        );
        assert_eq!(
            telemetry_error_class_from_message("arbitrary secret detail 123"),
            "internal"
        );
    }
}

#[cfg(test)]
mod resident_ownership_reconciliation_tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    use super::{ResidentSequencePool, lock_resident_sequences, verify_resident_ownership};

    #[test]
    fn ownership_mismatch_permanently_disables_cache_operations() {
        let healthy = AtomicBool::new(true);

        let error = verify_resident_ownership(&healthy, 2, 1).unwrap_err();

        assert_eq!(
            error.to_string(),
            "resident cache ownership mismatch: radix_entries=2 allocated_sequences=1"
        );
        assert!(!healthy.load(Ordering::Acquire));
        verify_resident_ownership(&healthy, 1, 1).unwrap();
        assert!(!healthy.load(Ordering::Acquire));
    }

    #[test]
    fn poisoned_resident_sequence_lock_recovers_without_reusing_state() {
        let sequences = Arc::new(Mutex::new(ResidentSequencePool::new(4)));
        let poisoned = sequences.clone();
        assert!(
            std::thread::spawn(move || {
                let mut guard = poisoned.lock().unwrap();
                guard.allocate().unwrap();
                panic!("poison resident sequence pool for test");
            })
            .join()
            .is_err()
        );

        let mut guard = lock_resident_sequences(&sequences);
        assert_eq!(guard.stats(), (1, 0, 0));
        assert_eq!(guard.allocate().unwrap(), 5);
    }

    #[test]
    fn forced_quarantine_removes_a_sequence_from_every_reusable_set() {
        let mut sequences = ResidentSequencePool::new(4);
        let seq_id = sequences.allocate().unwrap();
        sequences.release(seq_id).unwrap();

        sequences.force_quarantine(seq_id);

        assert_eq!(sequences.stats(), (0, 0, 1));
        assert_eq!(sequences.allocate().unwrap(), seq_id + 1);
    }
}
