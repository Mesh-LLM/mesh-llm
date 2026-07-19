use crate::binary_transport::BinaryStageExecutionOptions;
use crate::binary_transport::PredictionReturnReceiver;
use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::run_binary_stage_message;
use crate::binary_transport::stage_output_activation_capacity;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::generation::EmbeddedExecutionStats;
use crate::frontend::generation::EmbeddedLocalOutput;
use crate::frontend::generation::EmbeddedSessionControl;
use crate::frontend::generation::EmbeddedStageExecution;
use crate::frontend::generation::EmbeddedStageZeroGeneration;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::util::ms_to_us;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::openai_io_error;
use crate::frontend::wire_messages::embedded_session_control_message;
use crate::frontend::wire_messages::embedded_trim_session_message;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireMessageKind;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;
use skippy_protocol::binary::state_flags;
use std::net::TcpStream;
use std::time::Duration;
use std::time::Instant;

const DIRECT_RETURN_FALLBACK_POLL: Duration = Duration::from_millis(10);
const DIRECT_RETURN_FALLBACK_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) struct DispatchedEmbeddedStage {
    started: Instant,
    stats: StageReplyStats,
    execution: EmbeddedExecutionStats,
    message_kind: WireMessageKind,
    token_count: i32,
}

impl StageOpenAiBackend {
    pub(super) fn execute_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let dispatched = self.dispatch_embedded_stage_message(
            request,
            downstream,
            session_key,
            message,
            token_ids,
        )?;
        self.complete_dispatched_stage_message(request, downstream, dispatched, expected_reply)
    }

    pub(super) fn dispatch_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
    ) -> OpenAiResult<DispatchedEmbeddedStage> {
        let started = Instant::now();
        let mut stats = StageReplyStats::default();
        let stage0_timer = PhaseTimer::start();
        let output = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            let hold_timer = PhaseTimer::start();
            if message.kind == WireMessageKind::VerifyWindow
                && (message.state.flags & state_flags::SKIP_VERIFY_CHECKPOINT) == 0
            {
                let checkpoint_timer = PhaseTimer::start();
                runtime
                    .checkpoint_session(session_key)
                    .map_err(openai_backend_error)?;
                let checkpoint_us = ms_to_us(checkpoint_timer.elapsed_ms());
                stats.checkpoint_local_us += checkpoint_us;
                stats.checkpoint_total_us += checkpoint_us;
                stats.verify_window_checkpointed_requests += 1;
            } else if message.kind == WireMessageKind::VerifyWindow {
                stats.verify_window_skip_checkpoint_requests += 1;
            }
            let output = run_binary_stage_message(
                &mut runtime,
                session_key,
                message,
                token_ids,
                None,
                BinaryStageExecutionOptions::new(
                    false,
                    stage_output_activation_capacity(
                        request.config,
                        message.token_count,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?,
                    request.native_mtp_enabled,
                )
                .with_native_mtp_max_tokens(request.speculative.native_mtp.max_draft_tokens),
            )
            .map_err(openai_backend_error)?
            .2;
            let hold_ms = hold_timer.elapsed_ms();
            EmbeddedLocalOutput {
                output,
                runtime_lock_wait_ms: lock_wait_ms,
                runtime_lock_hold_ms: hold_ms,
            }
        };
        let stage0_compute_ms = stage0_timer.elapsed_ms();
        let forwarded = forwarded_stage_message_timed(
            request.config,
            message,
            &output.output,
            request.wire_dtype,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &forwarded.message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let forward_write_ms = write_timer.elapsed_ms();
        Ok(DispatchedEmbeddedStage {
            started,
            stats,
            execution: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: output.runtime_lock_wait_ms,
                runtime_lock_hold_ms: output.runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.output.payload.len(),
                forward_activation_bytes: forwarded.message.activation.len(),
                forward_write_ms,
                downstream_wait_ms: 0.0,
            },
            message_kind: message.kind,
            token_count: message.token_count,
        })
    }

    pub(super) fn complete_dispatched_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        dispatched: DispatchedEmbeddedStage,
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        self.complete_dispatched_stage_message_with_return(
            request,
            downstream,
            dispatched,
            expected_reply,
            false,
        )
    }

    pub(super) fn complete_dispatched_stage_message_direct(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        dispatched: DispatchedEmbeddedStage,
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        self.complete_dispatched_stage_message_with_return(
            request,
            downstream,
            dispatched,
            expected_reply,
            true,
        )
    }

    fn complete_dispatched_stage_message_with_return(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        mut dispatched: DispatchedEmbeddedStage,
        expected_reply: WireReplyKind,
        require_direct_return: bool,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let wait_timer = PhaseTimer::start();
        let reply = if require_direct_return {
            receive_direct_prediction_return(request.prediction_return.as_ref(), expected_reply)?
        } else {
            receive_embedded_stage_reply(
                downstream,
                request.prediction_return.as_ref(),
                expected_reply,
            )?
        };
        dispatched.execution.downstream_wait_ms = wait_timer.elapsed_ms();
        dispatched.stats.merge(reply.stats);
        if dispatched.message_kind == WireMessageKind::VerifyWindow {
            dispatched.stats.verify_window_compute_us +=
                ms_to_us(dispatched.execution.stage0_compute_ms);
            dispatched.stats.verify_window_forward_write_us +=
                ms_to_us(dispatched.execution.forward_write_ms);
            dispatched.stats.verify_window_downstream_wait_us +=
                ms_to_us(dispatched.execution.downstream_wait_ms);
            dispatched.stats.verify_window_total_us +=
                ms_to_us(dispatched.started.elapsed().as_secs_f64() * 1000.0);
            dispatched.stats.verify_window_stage_count += 1;
            dispatched.stats.verify_window_request_count += 1;
            dispatched.stats.verify_window_token_count += i64::from(dispatched.token_count.max(0));
            dispatched.stats.verify_window_max_tokens = dispatched
                .stats
                .verify_window_max_tokens
                .max(i64::from(dispatched.token_count.max(0)));
        }
        Ok(EmbeddedStageExecution {
            reply: StageReply {
                stats: dispatched.stats,
                ..reply
            },
            stats: dispatched.execution,
            elapsed_ms: dispatched.started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub(super) fn restore_embedded_stage_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .restore_session(session_key)
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message = embedded_session_control_message(
            request.wire_dtype,
            WireMessageKind::RestoreSession,
            request_id,
            session_id,
        );
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "restore expected ACK from downstream, got {:?}",
                reply.kind
            )));
        }
        Ok(EmbeddedSessionControl {
            elapsed_ms: timer.elapsed_ms(),
            local_ms,
            downstream_write_ms,
            downstream_wait_ms,
        })
    }

    pub(super) fn trim_embedded_stage_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
        token_count: usize,
    ) -> OpenAiResult<EmbeddedSessionControl> {
        let timer = PhaseTimer::start();
        let local_timer = PhaseTimer::start();
        {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .trim_session(session_key, token_count as u64)
                .map_err(openai_backend_error)?;
        }
        let local_ms = local_timer.elapsed_ms();
        let message =
            embedded_trim_session_message(request.wire_dtype, request_id, session_id, token_count)?;
        let write_timer = PhaseTimer::start();
        write_stage_message_conditioned(
            &mut *downstream,
            &message,
            request.wire_dtype,
            request.downstream_wire_condition,
        )
        .map_err(openai_io_error)?;
        let downstream_write_ms = write_timer.elapsed_ms();
        let wait_timer = PhaseTimer::start();
        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
        let downstream_wait_ms = wait_timer.elapsed_ms();
        if reply.kind != WireReplyKind::Ack {
            return Err(OpenAiError::backend(format!(
                "trim expected ACK from downstream, got {:?}",
                reply.kind
            )));
        }
        Ok(EmbeddedSessionControl {
            elapsed_ms: timer.elapsed_ms(),
            local_ms,
            downstream_write_ms,
            downstream_wait_ms,
        })
    }
}

fn receive_direct_prediction_return(
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReply> {
    let prediction_return = prediction_return.ok_or_else(|| {
        OpenAiError::backend("direct prediction return was required but is not configured")
    })?;
    let started = Instant::now();
    loop {
        if let Some(reply) = prediction_return
            .try_recv_expected(expected_reply)
            .map_err(openai_backend_error)?
        {
            return Ok(reply);
        }
        if started.elapsed() >= DIRECT_RETURN_FALLBACK_TIMEOUT {
            return Err(OpenAiError::backend(format!(
                "timed out waiting for {expected_reply:?} reply from direct prediction return"
            )));
        }
        std::thread::sleep(DIRECT_RETURN_FALLBACK_POLL);
    }
}

pub(crate) fn receive_embedded_stage_reply(
    downstream: &mut TcpStream,
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReply> {
    receive_embedded_stage_reply_one_of(
        downstream,
        prediction_return,
        std::slice::from_ref(&expected_reply),
    )
}

pub(crate) fn receive_embedded_stage_reply_one_of(
    downstream: &mut TcpStream,
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_replies: &[WireReplyKind],
) -> OpenAiResult<StageReply> {
    if expected_replies.is_empty() {
        return Err(OpenAiError::backend(
            "at least one expected stage reply kind is required",
        ));
    }
    let Some(prediction_return) = prediction_return else {
        return receive_downstream_stage_reply_one_of(downstream, expected_replies);
    };
    poll_direct_or_downstream_reply(downstream, prediction_return, expected_replies)
}

fn poll_direct_or_downstream_reply(
    downstream: &mut TcpStream,
    prediction_return: &PredictionReturnReceiver,
    expected_replies: &[WireReplyKind],
) -> OpenAiResult<StageReply> {
    let mut timeout_restore = DirectReturnFallbackTimeout::install(downstream)?;
    let started = Instant::now();
    let result = loop {
        if let Some(reply) = prediction_return
            .try_recv_one_of(expected_replies)
            .map_err(openai_backend_error)?
        {
            break Ok(reply);
        }
        if downstream_reply_available(downstream)? {
            break receive_downstream_stage_reply_one_of(downstream, expected_replies);
        }
        if started.elapsed() >= DIRECT_RETURN_FALLBACK_TIMEOUT {
            break Err(OpenAiError::backend(format!(
                "timed out waiting for one of {expected_replies:?} from direct return or downstream"
            )));
        }
    };
    timeout_restore.restore()?;
    result
}

fn downstream_reply_available(downstream: &TcpStream) -> OpenAiResult<bool> {
    let mut byte = [0u8; 1];
    match downstream.peek(&mut byte) {
        Ok(0) => Err(OpenAiError::backend("downstream closed before stage reply")),
        Ok(_) => Ok(true),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) =>
        {
            Ok(false)
        }
        Err(error) => Err(openai_io_error(error)),
    }
}

struct DirectReturnFallbackTimeout {
    downstream: TcpStream,
    previous_timeout: Option<Duration>,
    restored: bool,
}

impl DirectReturnFallbackTimeout {
    fn install(downstream: &TcpStream) -> OpenAiResult<Self> {
        let previous_timeout = downstream.read_timeout().map_err(openai_io_error)?;
        let restore_stream = downstream.try_clone().map_err(openai_io_error)?;
        downstream
            .set_read_timeout(Some(DIRECT_RETURN_FALLBACK_POLL))
            .map_err(openai_io_error)?;
        Ok(Self {
            downstream: restore_stream,
            previous_timeout,
            restored: false,
        })
    }

    fn restore(&mut self) -> OpenAiResult<()> {
        self.downstream
            .set_read_timeout(self.previous_timeout)
            .map_err(openai_io_error)?;
        self.restored = true;
        Ok(())
    }
}

impl Drop for DirectReturnFallbackTimeout {
    fn drop(&mut self) {
        if !self.restored {
            let _ = self.downstream.set_read_timeout(self.previous_timeout);
        }
    }
}

fn receive_downstream_stage_reply_one_of(
    downstream: &mut TcpStream,
    expected_replies: &[WireReplyKind],
) -> OpenAiResult<StageReply> {
    let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
    if !expected_replies.contains(&reply.kind) {
        return Err(OpenAiError::backend(format!(
            "expected one of {expected_replies:?} from downstream, got {:?}",
            reply.kind
        )));
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_transport::PredictionReturnHub;
    use skippy_protocol::binary::{StageStateHeader, WireActivationDType};
    use std::net::TcpListener;
    use std::sync::Arc;

    #[test]
    fn embedded_stage_reply_accepts_fused_restore_hits_and_misses_from_direct_return() {
        assert_eq!(
            receive_direct_reply_one_of(
                WireReplyKind::PredictedToken,
                &[WireReplyKind::PredictedToken, WireReplyKind::Ack],
            ),
            WireReplyKind::PredictedToken
        );
        assert_eq!(
            receive_direct_reply_one_of(
                WireReplyKind::Ack,
                &[WireReplyKind::PredictedToken, WireReplyKind::Ack],
            ),
            WireReplyKind::Ack
        );
    }

    fn receive_direct_reply_one_of(
        reply_kind: WireReplyKind,
        expected_replies: &[WireReplyKind],
    ) -> WireReplyKind {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let (mut direct_client, direct_server) = tcp_pair();
        let hub_thread = {
            let hub = hub.clone();
            std::thread::spawn(move || {
                hub.handle_return_connection(
                    StageWireMessage {
                        kind: WireMessageKind::PredictionReturnOpen,
                        pos_start: 0,
                        token_count: 0,
                        state: StageStateHeader::new(
                            WireMessageKind::PredictionReturnOpen,
                            WireActivationDType::F32,
                        ),
                        request_id,
                        session_id,
                        sampling: None,
                        chat_sampling_metadata: None,
                        tokens: Vec::new(),
                        positions: Vec::new(),
                        activation: Vec::new(),
                        raw_bytes: Vec::new(),
                    },
                    direct_server,
                )
            })
        };
        skippy_protocol::binary::send_reply_message(
            &mut direct_client,
            &StageReply {
                kind: reply_kind,
                predicted: 0,
                predicted_tokens: Vec::new(),
                native_mtp_draft: None,
                window: Default::default(),
                stats: StageReplyStats::default(),
            },
        )
        .unwrap();
        let (mut downstream, _downstream_peer) = tcp_pair();
        let reply =
            receive_embedded_stage_reply_one_of(&mut downstream, Some(&receiver), expected_replies)
                .unwrap();
        drop(direct_client);
        hub_thread.join().unwrap().unwrap();
        reply.kind
    }

    fn tcp_pair() -> (TcpStream, TcpStream) {
        connected_stream_pair()
    }

    fn connected_stream_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        (client, server)
    }

    #[test]
    fn direct_return_fallback_timeout_restores_on_drop_after_early_exit() {
        let (stream, _peer) = connected_stream_pair();
        let original = Some(Duration::from_millis(123));
        stream.set_read_timeout(original).unwrap();

        {
            let _restore = DirectReturnFallbackTimeout::install(&stream).unwrap();
            assert_eq!(
                stream.read_timeout().unwrap(),
                Some(DIRECT_RETURN_FALLBACK_POLL)
            );
        }

        assert_eq!(stream.read_timeout().unwrap(), original);
    }
}
