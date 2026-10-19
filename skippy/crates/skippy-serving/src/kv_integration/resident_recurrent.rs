//! Converts a lane-owned hybrid checkpoint to the existing serialized tiers.
use super::exact_state::{CaptureAdmission, ExactStateAdmissionCredit};
use super::*;

#[derive(Clone, Copy)]
pub(crate) struct ResidentRecurrentPolicy {
    pub(crate) write_through_l3: bool,
    pub(crate) l3_cost: Option<skippy_cache::policy::CostSample>,
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

    pub(crate) fn demote_resident_recurrent(
        &self,
        session: &mut skippy_runtime::StageSession,
        layer_start: u32,
        layer_end: u32,
        identity: &PrefillKvIdentity,
        recurrent: Vec<u8>,
        policy: ResidentRecurrentPolicy,
    ) -> Result<ExactStateRecordAdmission> {
        let layer_start = i32::try_from(layer_start)?;
        let layer_end = i32::try_from(layer_end)?;
        let Some(mut admission_credit) = ExactStateAdmissionCredit::acquire(
            &self.admission_outstanding,
            &self.admission_best_effort_outstanding,
            CaptureAdmission::Continuation,
        ) else {
            self.notify_demotion_error();
            return Ok(ExactStateRecordAdmission::DroppedFull);
        };
        let token_count = identity.token_ids.len() as u64;
        let kv_bytes = match session.kv_page_export_size(layer_start, layer_end, 0, token_count) {
            Ok(bytes) => bytes as u64,
            Err(error) if super::exact_state::is_native_kv_unavailable(&error) => 0,
            Err(error) => return Err(error),
        };
        if !admission_credit.reserve_bytes(
            &self.exact_state_record_queue_bytes,
            kv_bytes.saturating_add(recurrent.len() as u64),
            EXACT_STATE_RECORD_QUEUE_BYTES,
        ) {
            self.notify_demotion_error();
            return Ok(ExactStateRecordAdmission::DroppedFull);
        }
        if !self.try_begin_record(&identity.page_id) {
            return Ok(
                if self
                    .exact_state_record_worker_healthy
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    ExactStateRecordAdmission::Queued // Another producer owns this identity.
                } else {
                    ExactStateRecordAdmission::WorkerStopped
                },
            );
        }

        let exported =
            session.export_kv_page(layer_start, layer_end, 0, identity.token_ids.len() as u64);
        let exported = match exported {
            Ok(page) => Ok((
                ExactStatePayload::kv_recurrent(page.payload, recurrent),
                ExactStateExtra {
                    kv_desc: Some(page.desc),
                },
            )),
            Err(error) if super::exact_state::is_native_kv_unavailable(&error) => Ok((
                ExactStatePayload::kv_recurrent(Vec::new(), recurrent),
                ExactStateExtra::default(),
            )),
            Err(error) => Err(error),
        };
        let (payload, extra) = match self.validate_snapshot_export(exported) {
            Ok(result) => result,
            Err(error) => {
                self.finish_record(&identity.page_id);
                return Err(error);
            }
        };
        Ok(self.enqueue_exact_state_record(PendingExactStateRecord {
            page_id: identity.page_id.clone(),
            payload,
            extra,
            namespace: identity.namespace.clone(),
            token_ids: identity.token_ids.clone(),
            l3_fill_claim: None,
            write_through_l3: policy.write_through_l3,
            l2_promotion_digest: None,
            l3_cost: policy.l3_cost,
            admission_credit,
        }))
    }
}
