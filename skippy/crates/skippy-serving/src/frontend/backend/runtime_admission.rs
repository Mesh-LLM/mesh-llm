//! Bound frontend work by the execution lanes shared with binary stage clients.
use super::{StageOpenAiBackend, generation_queue_timeout_error, request_cancelled_error};
use crate::runtime_state::lane_admission::RuntimeLanePermit;
use crate::runtime_state::panic_recovery::lock_runtime;
use skippy_inference_api::{CancellationToken, InferenceError, InferenceResult};
use std::time::Instant;

impl StageOpenAiBackend {
    pub(super) async fn acquire_runtime_lane(
        &self,
        admission_started: Instant,
        cancellation: &CancellationToken,
    ) -> InferenceResult<RuntimeLanePermit> {
        let lanes = lock_runtime(&self.runtime).lane_admission();
        let deadline = if self.generation_admission_timeout.is_zero() {
            None
        } else {
            Some(
                admission_started
                    .checked_add(self.generation_admission_timeout)
                    .ok_or_else(|| {
                        InferenceError::backend("runtime lane admission deadline overflow")
                    })?,
            )
        };
        let timeout = async {
            if let Some(deadline) = deadline {
                tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! {
            biased;
            () = cancellation.cancelled() => Err(request_cancelled_error()),
            () = timeout => Err(generation_queue_timeout_error(self.generation_admission_timeout)),
            permit = lanes.acquire() => permit.map_err(|error| InferenceError::backend(format!("{error:#}"))),
        }
    }
}
