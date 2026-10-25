//! The reply side of one direct iteration request.
//!
//! A caller waits on its result channel while also holding a sender for it,
//! so the channel never reports disconnection. If the worker dropped a
//! request without answering, for example while unwinding from a panic in
//! the batch that held it, the caller would wait forever. A [`DirectReply`]
//! that is dropped unanswered sends an error instead.

use std::sync::mpsc as std_mpsc;

use skippy_inference_api::{InferenceError, InferenceResult};

use super::SchedulerIterationOutcome;

type ReplySender = std_mpsc::SyncSender<InferenceResult<SchedulerIterationOutcome>>;

pub(super) struct DirectReply(Option<ReplySender>);

impl DirectReply {
    /// Answers the request.
    pub(super) fn send(mut self, result: InferenceResult<SchedulerIterationOutcome>) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(result);
        }
    }
}

impl From<ReplySender> for DirectReply {
    fn from(sender: ReplySender) -> Self {
        Self(Some(sender))
    }
}

impl Drop for DirectReply {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            // Each request gets one answer, so the slot is free; never block,
            // since this can run while unwinding.
            let _ = sender.try_send(Err(InferenceError::backend(
                "iteration scheduler dropped the request without answering",
            )));
        }
    }
}
