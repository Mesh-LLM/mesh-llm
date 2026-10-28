//! Converts a lane-owned hybrid checkpoint to the existing serialized tiers.
use super::exact_state::{CaptureAdmission, ExactStateAdmissionCredit};
use super::exact_state_diagnostics::emit_exact_state_diagnostic;
use super::kv_lease::{LeaseLedger, LeasedKvPage};
use super::*;

#[derive(Clone, Copy)]
pub(crate) struct ResidentRecurrentPolicy {
    pub(crate) write_through_l3: bool,
    pub(crate) l3_cost: Option<skippy_cache::policy::CostSample>,
}

/// Proof that a checkpoint's bytes reached the recorder path: queued, or
/// consumed and lost on the way. Only this module mints one, so a lane
/// cannot claim a checkpoint is gone without having offered it.
#[derive(Debug)]
#[must_use = "a receipt is the only thing that lets a checkpoint leave its lane"]
pub(crate) struct RecorderReceipt(());

#[cfg(test)]
impl RecorderReceipt {
    /// Tests of the lane layer stand in for the recorder.
    pub(crate) fn for_test() -> Self {
        Self(())
    }
}

/// Why the recorder would not take an export. Nothing left the lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DemotionRefusal {
    /// Copying path: every credit is held.
    CreditFull,
    /// The byte budget cannot take the copying export or a leased record's
    /// owned recurrent snapshot.
    ByteFull,
    /// In-place path: every leased channel slot is taken.
    QueueFull,
    WorkerStopped,
}

/// What became of an in-place record of a resident checkpoint. The lane
/// keeps the checkpoint in every case; only `Queued` means the recorder has
/// it too.
#[derive(Debug)]
pub(crate) enum InPlaceRecord {
    Queued,
    Refused(DemotionRefusal),
    /// The loaded memory is not host visible after all; copy at recycle time.
    Unsupported,
    /// Exported and dropped after the fact, or the export failed. Counted
    /// and logged; the resident copy is intact.
    Lost,
}

/// What became of one lane-owned checkpoint offered to the recorder.
#[derive(Debug)]
pub(crate) enum ResidentDemotion {
    Queued(RecorderReceipt),
    /// The recurrent snapshot goes back to the caller; the native cells were
    /// never touched.
    Refused {
        recurrent: Vec<u8>,
        reason: DemotionRefusal,
    },
    /// Consumed and not retained: an export error, no native prefix to pair
    /// with, or a recorder that dropped it after the export. Already counted
    /// and logged.
    Lost(RecorderReceipt),
}

impl KvStageIntegration {
    pub(crate) fn notify_demotion_error(&self) {
        self.exact_state_records_dropped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.notify_kv_lifecycle(KvLifecycleEvent::ExactStateRecordFailed);
    }
    pub(crate) fn has_durable_recurrent_tier(&self) -> bool {
        self.l3.is_some()
    }
    pub(crate) fn resident_recurrent_limits(&self) -> (usize, u64) {
        (
            self.exact_max_entries.min(4),
            self.exact_byte_limits.hard_bytes,
        )
    }

    /// Offer one checkpoint to the recorder. Never fails: a checkpoint that
    /// cannot be recorded is either handed back untouched or reported lost.
    /// Losses affect optional cache retention, never generation output.
    pub(crate) fn demote_resident_recurrent(
        &self,
        session: &mut skippy_runtime::StageSession,
        layers: std::ops::Range<u32>,
        identity: &PrefillKvIdentity,
        recurrent: Vec<u8>,
        policy: ResidentRecurrentPolicy,
    ) -> ResidentDemotion {
        let token_count = identity.token_ids.len() as u64;
        if !session.has_resident_recurrent_prefix(token_count) {
            return self.lose("demotion_no_native_prefix", token_count);
        }
        let (Ok(layer_start), Ok(layer_end)) =
            (i32::try_from(layers.start), i32::try_from(layers.end))
        else {
            return self.lose("demotion_error", token_count);
        };
        let Some(mut admission_credit) = ExactStateAdmissionCredit::acquire(
            &self.admission_outstanding,
            &self.admission_best_effort_outstanding,
            CaptureAdmission::Continuation,
        ) else {
            emit_exact_state_diagnostic("demotion_credit_full", &[("position", token_count)]);
            return ResidentDemotion::Refused {
                recurrent,
                reason: DemotionRefusal::CreditFull,
            };
        };
        let recurrent_bytes = recurrent.len() as u64;
        let kv_bytes = match session.kv_page_export_size(layer_start, layer_end, 0, token_count) {
            Ok(bytes) => bytes as u64,
            Err(error) if super::exact_state::is_native_kv_unavailable(&error) => 0,
            Err(error) => return self.lose_with_error("demotion_error", token_count, &error),
        };
        if !admission_credit.reserve_bytes(
            &self.exact_state_record_queue_bytes,
            kv_bytes.saturating_add(recurrent_bytes),
            self.exact_state_record_queue_limit,
        ) {
            emit_exact_state_diagnostic(
                "demotion_byte_full",
                &[
                    ("position", token_count),
                    ("kv_bytes", kv_bytes),
                    ("recurrent_bytes", recurrent_bytes),
                    (
                        "pending_bytes",
                        self.exact_state_record_queue_bytes
                            .load(std::sync::atomic::Ordering::Acquire),
                    ),
                ],
            );
            return ResidentDemotion::Refused {
                recurrent,
                reason: DemotionRefusal::ByteFull,
            };
        }
        if !self.try_begin_record(&identity.page_id) {
            emit_exact_state_diagnostic(
                "demotion_duplicate_or_stopped",
                &[("position", token_count)],
            );
            // Another producer owns this identity, so its record covers ours.
            return if self
                .exact_state_record_worker_healthy
                .load(std::sync::atomic::Ordering::Acquire)
            {
                ResidentDemotion::Queued(RecorderReceipt(()))
            } else {
                ResidentDemotion::Refused {
                    recurrent,
                    reason: DemotionRefusal::WorkerStopped,
                }
            };
        }

        let exported =
            Self::export_kv_page_copy(session, layer_start, layer_end, token_count, recurrent);
        let (payload, extra) = match self.validate_snapshot_export(exported) {
            Ok(result) => result,
            Err(error) => {
                self.finish_record(&identity.page_id);
                return self.lose_with_error("demotion_error", token_count, &error);
            }
        };
        let outcome = self.enqueue_exact_state_record(PendingExactStateRecord {
            page_id: identity.page_id.clone(),
            payload,
            extra,
            namespace: identity.namespace.clone(),
            token_ids: identity.token_ids.clone(),
            l3_fill_claim: None,
            write_through_l3: policy.write_through_l3,
            l2_promotion_digest: None,
            l3_cost: policy.l3_cost,
            admission: RecordAdmission::Credit(admission_credit),
        });
        // The recorder already counted a drop in `enqueue_exact_state_record`.
        let (event, demotion) = match outcome {
            ExactStateRecordAdmission::Queued => (
                "demotion_queued",
                ResidentDemotion::Queued(RecorderReceipt(())),
            ),
            ExactStateRecordAdmission::DroppedFull => (
                "demotion_queue_full",
                ResidentDemotion::Lost(RecorderReceipt(())),
            ),
            ExactStateRecordAdmission::WorkerStopped => (
                "demotion_worker_stopped",
                ResidentDemotion::Lost(RecorderReceipt(())),
            ),
        };
        emit_exact_state_diagnostic(
            event,
            &[
                ("position", token_count),
                ("kv_bytes", kv_bytes),
                ("recurrent_bytes", recurrent_bytes),
            ],
        );
        demotion
    }

    /// Record a resident checkpoint from the lane's own cells. The attention
    /// KV has no copy or credit; the recurrent snapshot reserves its owned
    /// bytes and a queue slot before cloning. The lane keeps its copy.
    pub(crate) fn record_resident_recurrent_in_place(
        &self,
        session: &mut skippy_runtime::StageSession,
        layers: std::ops::Range<u32>,
        identity: &PrefillKvIdentity,
        recurrent: &[u8],
        policy: ResidentRecurrentPolicy,
        ledger: &LeaseLedger,
    ) -> InPlaceRecord {
        let token_count = identity.token_ids.len() as u64;
        if !session.has_resident_recurrent_prefix(token_count) {
            self.notify_demotion_error();
            emit_exact_state_diagnostic("demotion_no_native_prefix", &[("position", token_count)]);
            return InPlaceRecord::Lost;
        }
        let (Ok(layer_start), Ok(layer_end)) =
            (i32::try_from(layers.start), i32::try_from(layers.end))
        else {
            self.notify_demotion_error();
            emit_exact_state_diagnostic("demotion_error", &[("position", token_count)]);
            return InPlaceRecord::Lost;
        };
        let spans = match session.kv_page_spans(layer_start, layer_end, 0, token_count) {
            Ok(Some(spans)) => spans,
            Ok(None) => return InPlaceRecord::Unsupported,
            Err(error) => {
                self.notify_demotion_error();
                tracing::debug!(error = %error, "in-place KV page description failed");
                emit_exact_state_diagnostic("demotion_error", &[("position", token_count)]);
                return InPlaceRecord::Lost;
            }
        };
        if !self.try_begin_record(&identity.page_id) {
            emit_exact_state_diagnostic(
                "demotion_duplicate_or_stopped",
                &[("position", token_count)],
            );
            // Another producer owns this identity, so its record covers ours.
            return if self
                .exact_state_record_worker_healthy
                .load(std::sync::atomic::Ordering::Acquire)
            {
                InPlaceRecord::Queued
            } else {
                InPlaceRecord::Refused(DemotionRefusal::WorkerStopped)
            };
        }
        let recurrent_bytes = recurrent.len() as u64;
        let reservation = match LeasedRecordReservation::acquire(
            &self.exact_state_records_pending,
            &self.exact_state_record_queue_bytes,
            recurrent_bytes,
            self.exact_state_record_queue_limit,
        ) {
            Ok(reservation) => reservation,
            Err(refusal) => {
                self.finish_record(&identity.page_id);
                let (event, reason) = match refusal {
                    LeasedRecordRefusal::QueueFull => {
                        ("demotion_queue_full_in_place", DemotionRefusal::QueueFull)
                    }
                    LeasedRecordRefusal::ByteFull => {
                        ("demotion_byte_full_in_place", DemotionRefusal::ByteFull)
                    }
                };
                emit_exact_state_diagnostic(event, &[("position", token_count)]);
                return InPlaceRecord::Refused(reason);
            }
        };
        let kv_bytes = spans.byte_len() as u64;
        let extra = ExactStateExtra {
            kv_desc: Some(spans.desc.clone()),
        };
        let lease = LeasedKvPage {
            spans,
            pin: ledger.pin(token_count),
        };
        let outcome = self.enqueue_exact_state_record(PendingExactStateRecord {
            page_id: identity.page_id.clone(),
            payload: ExactStatePayload::kv_recurrent(Vec::new(), recurrent.to_vec()),
            extra,
            namespace: identity.namespace.clone(),
            token_ids: identity.token_ids.clone(),
            l3_fill_claim: None,
            write_through_l3: policy.write_through_l3,
            l2_promotion_digest: None,
            l3_cost: policy.l3_cost,
            admission: RecordAdmission::Leased(lease, reservation),
        });
        let (event, record) = match outcome {
            ExactStateRecordAdmission::Queued => {
                ("demotion_queued_in_place", InPlaceRecord::Queued)
            }
            // No leased slot: the record never left the lane. The recorder
            // counted a drop, but nothing was lost; correct that here.
            ExactStateRecordAdmission::DroppedFull => {
                self.exact_state_records_dropped
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                (
                    "demotion_queue_full_in_place",
                    InPlaceRecord::Refused(DemotionRefusal::QueueFull),
                )
            }
            ExactStateRecordAdmission::WorkerStopped => (
                "demotion_worker_stopped",
                InPlaceRecord::Refused(DemotionRefusal::WorkerStopped),
            ),
        };
        emit_exact_state_diagnostic(
            event,
            &[
                ("position", token_count),
                ("kv_bytes", kv_bytes),
                ("recurrent_bytes", recurrent_bytes),
            ],
        );
        record
    }

    fn export_kv_page_copy(
        session: &mut skippy_runtime::StageSession,
        layer_start: i32,
        layer_end: i32,
        token_count: u64,
        recurrent: Vec<u8>,
    ) -> Result<(ExactStatePayload, ExactStateExtra)> {
        match session.export_kv_page(layer_start, layer_end, 0, token_count) {
            Ok(page) => Ok((
                ExactStatePayload::kv_recurrent(page.payload, recurrent),
                ExactStateExtra {
                    kv_desc: Some(page.desc),
                },
            )),
            // Pure recurrent stages have no attention page to export.
            Err(error) if super::exact_state::is_native_kv_unavailable(&error) => Ok((
                ExactStatePayload::recurrent_only(recurrent),
                ExactStateExtra::default(),
            )),
            Err(error) => Err(error),
        }
    }

    fn lose(&self, event: &'static str, position: u64) -> ResidentDemotion {
        self.notify_demotion_error();
        emit_exact_state_diagnostic(event, &[("position", position)]);
        ResidentDemotion::Lost(RecorderReceipt(()))
    }

    fn lose_with_error(
        &self,
        event: &'static str,
        position: u64,
        error: &anyhow::Error,
    ) -> ResidentDemotion {
        tracing::debug!(error = %error, "resident recurrent checkpoint demotion failed");
        self.lose(event, position)
    }
}
