//! Exact token-prefix reuse, using the existing model/topology cache identity.

use super::sequence::DecisionSequence;
use super::*;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::generation::InferenceBackendMode;
use crate::frontend::util::openai_io_error;
use crate::frontend::wire_messages::embedded_prefix_cache_message;
use skippy_protocol::binary::{WireMessageKind, WireReplyKind, recv_reply};

impl DecisionSequence<'_> {
    /// Restore the longest usable exact prefix for a local or split readout sequence.
    pub(super) fn restore_prefix(&mut self, tokens: &[i32]) -> InferenceResult<()> {
        let Some(kv) = self.backend.kv.as_ref().filter(|kv| kv.should_lookup()) else {
            return Ok(());
        };
        if tokens.is_empty() {
            return Ok(());
        }
        let base = self.backend.local_kv_message_base(&self.session, self.ids);
        let identities = kv.lookup_identities(&self.backend.config, &base, 0, tokens);
        let cache = kv.clone();
        let session = self.session.clone();
        let token_ids = tokens.to_vec();
        let restored = self.backend.iteration_scheduler.execute_runtime(
            "decision-prefix-restore",
            move |runtime| {
                if let Some(hit) = cache
                    .restore_exact_state_with_shared_durable_fallback(
                        runtime,
                        &session,
                        &identities,
                    )
                    .map_err(openai_backend_error)?
                {
                    return Ok(hit.token_count);
                }
                cache
                    .restore_resident_prefix(runtime, &session, &identities, &token_ids)
                    .map(|hit| hit.map_or(0, |hit| hit.token_count))
                    .map_err(openai_backend_error)
            },
        )?;
        if restored > tokens.len() {
            return Err(InferenceError::backend(
                "decision cache restored past requested prefix",
            ));
        }
        if restored > 0 && !self.restore_downstream(&tokens[..restored])? {
            let session = self.session.clone();
            self.backend.iteration_scheduler.execute_runtime(
                "decision-prefix-miss",
                move |runtime| {
                    runtime
                        .drop_session_timed(&session)
                        .map(|_| ())
                        .map_err(openai_backend_error)
                },
            )?;
            return Ok(());
        }
        self.position = restored;
        self.cached = restored;
        Ok(())
    }

    /// Restore matching downstream prefix state or fall back to recomputing the prompt.
    fn restore_downstream(&mut self, tokens: &[i32]) -> InferenceResult<bool> {
        let Some((_, lane)) = self.lane.as_mut() else {
            return Ok(true);
        };
        let InferenceBackendMode::EmbeddedStageZero {
            downstream_wire_condition,
            ..
        } = &self.backend.mode
        else {
            return Ok(false);
        };
        let message = embedded_prefix_cache_message(
            WireMessageKind::TryRestorePrefill,
            tokens,
            self.ids.request_id,
            self.ids.session_id,
        )?;
        write_stage_message_conditioned(&mut lane.stream, &message, *downstream_wire_condition)
            .map_err(openai_io_error)?;
        let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
        if reply.kind != WireReplyKind::Ack {
            return Err(InferenceError::backend(
                "expected decision prefix restore ACK",
            ));
        }
        let restored = reply.stats.kv_lookup_hits > 0
            && reply.stats.kv_lookup_misses == 0
            && reply.stats.kv_lookup_errors == 0;
        if !restored {
            // Reset every stage, including recurrent state that cannot safely
            // be truncated after a partial chain restore.
            let stop = skippy_protocol::binary::StageWireMessage::stop_with_identity(
                self.ids.request_id,
                self.ids.session_id,
            );
            write_stage_message_conditioned(&mut lane.stream, &stop, *downstream_wire_condition)
                .map_err(openai_io_error)?;
            if recv_reply(&mut lane.stream).map_err(openai_io_error)?.kind != WireReplyKind::Ack {
                return Err(InferenceError::backend(
                    "expected partial decision restore cleanup ACK",
                ));
            }
        }
        Ok(restored)
    }
}
