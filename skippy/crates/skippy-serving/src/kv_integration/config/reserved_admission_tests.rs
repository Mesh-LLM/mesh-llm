use super::*;
use crate::kv_integration::exact_state::{CaptureAdmission, ExactStateAdmissionCredit};
use crate::kv_integration::{ExactStateExtra, ExactStateRecordAdmission};
use crate::runtime_state::RuntimeState;
use skippy_cache::ExactStatePayload;
use skippy_protocol::MessageBase;
use std::sync::atomic::Ordering as AtomicOrdering;

const NAMESPACE: &str = "reserved-admission-test";
const SESSION: &str = "harness-session";
const WAIT: std::time::Duration = std::time::Duration::from_secs(30);

fn harness_config() -> StageConfig {
    StageConfig {
        stage_id: "stage-0".to_string(),
        model_id: NAMESPACE.to_string(),
        kv_cache: Some(StageKvCacheConfig {
            exact_max_bytes: None,
            mode: StageKvCacheMode::LookupRecord,
            max_entries: 8,
            max_bytes: 0,
            l2_max_bytes: 0,
            codec: skippy_protocol::StageKvCacheCodec::Native,
            min_tokens: 1,
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 0,
        }),
        ..StageConfig::default()
    }
}

fn harness() -> (Arc<KvStageIntegration>, StageConfig) {
    let config = harness_config();
    let kv = KvStageIntegration::from_test_state(&config, ModelStateKind::Recurrent)
        .expect("kv integration init")
        .expect("kv integration must be enabled for LookupRecord/KvRecurrent");
    (Arc::new(kv), config)
}

fn message_base() -> MessageBase {
    MessageBase {
        schema_version: skippy_protocol::SCHEMA_VERSION,
        run_id: NAMESPACE.to_string(),
        request_id: "harness-request".to_string(),
        session_id: SESSION.to_string(),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        topology_id: NAMESPACE.to_string(),
        model_id: Some(NAMESPACE.to_string()),
        tokenizer_id: None,
        chat_template_id: None,
        seq: None,
    }
}

fn identity(
    kv: &KvStageIntegration,
    config: &StageConfig,
    token_ids: &[i32],
) -> crate::kv_integration::PrefillKvIdentity {
    kv.prefill_identity(config, &message_base(), 0, token_ids)
}

fn record(
    page_id: &str,
    token_ids: Vec<i32>,
    admission_credit: ExactStateAdmissionCredit,
) -> PendingExactStateRecord {
    PendingExactStateRecord {
        page_id: page_id.to_string(),
        payload: ExactStatePayload::full_state(vec![1, 2, 3]),
        extra: ExactStateExtra::default(),
        namespace: NAMESPACE.to_string(),
        token_ids,
        l3_fill_claim: None,
        write_through_l3: true,
        l2_promotion_digest: None,
        l3_cost: None,
        admission: crate::kv_integration::RecordAdmission::Credit(admission_credit),
    }
}

fn acquire(kv: &KvStageIntegration, class: CaptureAdmission) -> Option<ExactStateAdmissionCredit> {
    ExactStateAdmissionCredit::acquire(
        &kv.admission_outstanding,
        &kv.admission_best_effort_outstanding,
        class,
    )
}

fn outstanding(kv: &KvStageIntegration) -> usize {
    kv.admission_outstanding.load(AtomicOrdering::Acquire)
}

fn best_effort_outstanding(kv: &KvStageIntegration) -> usize {
    kv.admission_best_effort_outstanding
        .load(AtomicOrdering::Acquire)
}

fn stored(kv: &KvStageIntegration, token_ids: &[i32]) -> bool {
    kv.radix
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .recurrent_exact(NAMESPACE, token_ids)
        .is_some()
}

/// Reserved-credit delivery end-to-end under an EXPLICITLY paused
/// recorder worker (no radix contention): the BestEffort payload is
/// received worker-side and paused BEFORE storage, four Continuation
/// payloads are ENQUEUED behind it, both classes decline overflow, and
/// unpausing stores every identity and drains every real credit; the budget
/// then admits and stores a recovery record.
#[test]
fn reserved_credits_deliver_continuation_payloads_and_recover_after_drain() {
    let (kv, _config) = harness();
    kv.exact_state_worker_pause
        .store(true, AtomicOrdering::Release);

    let be1 = acquire(&kv, CaptureAdmission::BestEffort).expect("first BestEffort admitted");
    assert_eq!(
        kv.enqueue_exact_state_record(record("be1", vec![1, 2, 3], be1)),
        ExactStateRecordAdmission::Queued,
    );
    kv.wait_for_exact_state_worker_received(1, WAIT)
        .expect("worker received the BestEffort record before storage");
    assert_eq!(outstanding(&kv), 1, "credit held by the paused worker");
    assert!(
        !stored(&kv, &[1, 2, 3]),
        "paused worker must not have stored yet"
    );

    assert!(
        acquire(&kv, CaptureAdmission::BestEffort).is_none(),
        "second BestEffort declined at the reserved cap"
    );
    assert_eq!(best_effort_outstanding(&kv), 1);

    let continuations = [
        vec![4, 5, 6],
        vec![7, 8, 9],
        vec![10, 11, 12],
        vec![13, 14, 15],
    ];
    for (index, tokens) in continuations.iter().enumerate() {
        let credit = acquire(&kv, CaptureAdmission::Continuation)
            .expect("Continuation admitted into the reserved share");
        assert_eq!(
            kv.enqueue_exact_state_record(record(&format!("pd{index}"), tokens.clone(), credit,)),
            ExactStateRecordAdmission::Queued,
            "the Continuation payload must be enqueued, never dropped by the harness"
        );
    }
    assert_eq!(outstanding(&kv), 5);
    assert!(
        acquire(&kv, CaptureAdmission::Continuation).is_none(),
        "total budget exhausted"
    );

    kv.exact_state_worker_pause
        .store(false, AtomicOrdering::Release);
    kv.wait_for_exact_state_recording(WAIT)
        .expect("recorder drained all records without drops or panics");
    assert_eq!(
        outstanding(&kv),
        0,
        "credits released by real worker completion"
    );
    assert_eq!(best_effort_outstanding(&kv), 0);
    assert!(stored(&kv, &[1, 2, 3]), "BestEffort identity stored");
    for tokens in &continuations {
        assert!(stored(&kv, tokens), "Continuation identity stored");
    }

    let recovered =
        acquire(&kv, CaptureAdmission::Continuation).expect("budget recovers after the drain");
    assert_eq!(
        kv.enqueue_exact_state_record(record("recovered", vec![16, 17], recovered)),
        ExactStateRecordAdmission::Queued,
    );
    kv.wait_for_exact_state_recording(WAIT)
        .expect("recovered record stored");
    assert_eq!(outstanding(&kv), 0);
    assert!(stored(&kv, &[16, 17]), "recovered identity stored");
}

/// Unhealthy recorder releases real credits on BOTH release paths: the
/// enqueue-level WorkerStopped drop, and the record_exact_state
/// try_begin_record decline that happens after acquisition.
#[test]
fn unhealthy_worker_releases_credits_on_both_release_paths() {
    let (kv, config) = harness();
    kv.exact_state_record_worker_healthy
        .store(false, AtomicOrdering::Release);

    let credit =
        acquire(&kv, CaptureAdmission::BestEffort).expect("budget admits before the health gate");
    assert_eq!(
        kv.enqueue_exact_state_record(record("stopped", vec![1], credit)),
        ExactStateRecordAdmission::WorkerStopped,
    );
    assert_eq!(
        outstanding(&kv),
        0,
        "enqueue-level WorkerStopped released the real credit"
    );

    let mut runtime = RuntimeState::new_modelless_for_test(1);
    let chat_identity = identity(&kv, &config, &[1, 2, 3]);
    let begin_record =
        crate::frontend::capture_trace::POST_DECODE_NONE_BEGIN_RECORD.load(AtomicOrdering::Acquire);
    let result = kv.record_exact_state(
        &mut runtime,
        SESSION,
        &chat_identity,
        CaptureAdmission::BestEffort,
    );
    assert!(
        matches!(result, Ok(None)),
        "unhealthy worker must decline recording"
    );
    assert_eq!(
        crate::frontend::capture_trace::POST_DECODE_NONE_BEGIN_RECORD.load(AtomicOrdering::Acquire),
        begin_record + 1,
        "the decline happened at try_begin_record, i.e. after acquisition"
    );
    assert_eq!(
        outstanding(&kv),
        0,
        "record_exact_state released the credit on the try_begin_record decline"
    );
}

/// record_exact_state releases its real credit when the radix lock is
/// busy: the capture is skipped without waiting and the inflight page is
/// released.
#[test]
fn record_exact_state_releases_credit_when_radix_is_busy() {
    let (kv, config) = harness();
    let mut runtime = RuntimeState::new_modelless_for_test(1);
    let chat_identity = identity(&kv, &config, &[1, 2, 3]);
    let radix_guard = kv.radix.lock().unwrap();
    let result = kv.record_exact_state(
        &mut runtime,
        SESSION,
        &chat_identity,
        CaptureAdmission::BestEffort,
    );
    drop(radix_guard);
    assert!(
        matches!(result, Ok(None)),
        "busy radix skips without waiting"
    );
    assert_eq!(outstanding(&kv), 0, "credit released on the busy skip");
    assert!(
        kv.inflight_records.lock().unwrap().is_empty(),
        "inflight page released on the busy skip"
    );
}

/// retained_exact_identity requires the EXACT radix node: a retained
/// shorter entry must not certify a longer identity, and a longer entry
/// must not certify its own prefix.
#[test]
fn retained_exact_identity_requires_the_exact_radix_node() {
    let (kv, config) = harness();
    let chat_identity = identity(&kv, &config, &[1, 2, 3]);
    kv.radix
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert_recurrent(
            chat_identity.namespace.clone(),
            &chat_identity.token_ids,
            32,
            RadixExactEntry::new(
                "exact".to_string(),
                ExactStatePayload::full_state(vec![9]),
                ExactStateExtra::default(),
                true,
            ),
        )
        .expect("seeding the radix must succeed");

    assert!(
        kv.retained_exact_identity(&chat_identity.namespace, &[1, 2, 3]),
        "the exact node is retained"
    );
    assert!(
        !kv.retained_exact_identity(&chat_identity.namespace, &[1, 2]),
        "a shorter query must not be certified by the longer retained entry"
    );
    assert!(
        !kv.retained_exact_identity(&chat_identity.namespace, &[1, 2, 3, 4]),
        "a longer identity must not be certified by a retained shorter prefix"
    );
    assert!(
        !kv.retained_exact_identity(&chat_identity.namespace, &[9, 9, 9]),
        "an unrelated identity is absent"
    );
}

/// record_exact_state releases its real credit when the identity is
/// already recorded.
#[test]
fn record_exact_state_releases_credit_when_already_recorded() {
    let (kv, config) = harness();
    let mut runtime = RuntimeState::new_modelless_for_test(1);
    let chat_identity = identity(&kv, &config, &[1, 2, 3]);
    kv.radix
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert_recurrent(
            chat_identity.namespace.clone(),
            &chat_identity.token_ids,
            32,
            RadixExactEntry::new(
                "pre-existing".to_string(),
                ExactStatePayload::full_state(vec![9]),
                ExactStateExtra::default(),
                true,
            ),
        )
        .expect("pre-seeding the radix must succeed");
    let result = kv.record_exact_state(
        &mut runtime,
        SESSION,
        &chat_identity,
        CaptureAdmission::BestEffort,
    );
    assert!(
        matches!(result, Ok(None)),
        "already-recorded identity skips"
    );
    assert_eq!(
        outstanding(&kv),
        0,
        "credit released on the already-recorded skip"
    );
    assert!(kv.inflight_records.lock().unwrap().is_empty());
}

/// record_exact_state releases its real credit when the payload export
/// fails (unknown session here), and nothing is enqueued.
#[test]
fn record_exact_state_releases_credit_when_export_fails() {
    let (kv, config) = harness();
    let mut runtime = RuntimeState::new_modelless_for_test(1);
    let chat_identity = identity(&kv, &config, &[1, 2, 3]);
    let result = kv.record_exact_state(
        &mut runtime,
        "missing-session",
        &chat_identity,
        CaptureAdmission::BestEffort,
    );
    assert!(result.is_err(), "exporting an unknown session must fail");
    assert_eq!(outstanding(&kv), 0, "credit released when the export fails");
    assert!(kv.inflight_records.lock().unwrap().is_empty());
}
