use crate::binary_transport::AsyncForwardReceipt;
use crate::binary_transport::AsyncForwarder;
use crate::binary_transport::BinaryStageExecutionOptions;
use crate::binary_transport::PredictionReturnReceiver;
use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::run_binary_stage_message;
use crate::binary_transport::stage_output_activation_capacity;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::generation::EmbeddedExecutionStats;
use crate::frontend::generation::EmbeddedLocalOutput;
use crate::frontend::generation::EmbeddedStageExecution;
use crate::frontend::generation::EmbeddedStageZeroGeneration;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::util::ms_to_us;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::openai_io_error;
use crate::telemetry::now_unix_nanos;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireMessageKind;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;
use std::net::TcpStream;
use std::time::Duration;
use std::time::Instant;

const DIRECT_RETURN_FALLBACK_POLL: Duration = Duration::from_millis(10);
// A dead downstream tunnel can leave both the persistent lane and the direct
// return reader open without producing EOF. Bound every reply wait so the
// request reaches lane replacement and session teardown instead of occupying
// a generation permit indefinitely. This is deliberately much larger than a
// normal WAN verify traversal while remaining shorter than the HTTP client's
// request timeout.
const DIRECT_RETURN_FALLBACK_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) struct DispatchedEmbeddedStage {
    started: Instant,
    stats: StageReplyStats,
    execution: EmbeddedExecutionStats,
    message_kind: WireMessageKind,
    token_count: i32,
    forward_receipt: Option<AsyncForwardReceipt>,
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
            None,
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
        async_forwarder: Option<&mut AsyncForwarder>,
    ) -> OpenAiResult<DispatchedEmbeddedStage> {
        let started = Instant::now();
        let stats = StageReplyStats::default();
        let stage0_timer = PhaseTimer::start();
        let output = {
            let lock_timer = PhaseTimer::start();
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let lock_wait_ms = lock_timer.elapsed_ms();
            let hold_timer = PhaseTimer::start();
            if let Some(target_token_count) = message.authoritative_session_position()
                && let Some(align) = runtime
                    .align_session_to_token_count_if_ahead(session_key, target_token_count)
                    .map_err(openai_backend_error)?
            {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.session_auto_align_before_tokens".to_string(),
                    json!(align.before_token_count),
                );
                attrs.insert(
                    "llama_stage.session_auto_align_after_tokens".to_string(),
                    json!(align.after_token_count),
                );
                self.telemetry
                    .emit_debug("stage.openai_session_auto_align", attrs);
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
        if self.telemetry.is_debug_enabled() {
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "llama_stage.message_kind".to_string(),
                json!(format!("{:?}", message.kind)),
            );
            attrs.insert(
                "llama_stage.token_count".to_string(),
                json!(message.token_count),
            );
            if let Some(window_id) = message.verify_window_id() {
                attrs.insert("llama_stage.verify_window_id".to_string(), json!(window_id));
            }
            self.telemetry.emit_debug_span(
                "stage.openai_stage0_llama_decode",
                attrs,
                stage0_timer.start_unix_nanos,
                now_unix_nanos() as u64,
            );
        }
        let forwarded = forwarded_stage_message_timed(
            request.config,
            message,
            &output.output,
            request.wire_dtype,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let forward_activation_bytes = forwarded.message.activation.len();
        let write_timer = PhaseTimer::start();
        let forward_receipt = if let Some(forwarder) = async_forwarder {
            Some(
                forwarder
                    .send_tracked(
                        forwarded.message,
                        request.wire_dtype,
                        request.downstream_wire_condition,
                        self.openai_attrs(request.ids),
                    )
                    .map_err(openai_backend_error)?,
            )
        } else {
            write_stage_message_conditioned(
                &mut *downstream,
                &forwarded.message,
                request.wire_dtype,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
            None
        };
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
                forward_activation_bytes,
                forward_write_ms,
                downstream_wait_ms: 0.0,
            },
            message_kind: message.kind,
            token_count: message.token_count,
            forward_receipt,
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
        if let Some(receipt) = dispatched.forward_receipt.take() {
            dispatched.execution.forward_write_ms =
                receipt.finish().map_err(openai_backend_error)?;
        }
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
}

fn receive_direct_prediction_return(
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReply> {
    let prediction_return = prediction_return.ok_or_else(|| {
        OpenAiError::backend("direct prediction return was required but is not configured")
    })?;
    prediction_return
        .recv_expected_timeout(expected_reply, DIRECT_RETURN_FALLBACK_TIMEOUT)
        .map_err(openai_backend_error)?
        .ok_or_else(|| {
            OpenAiError::backend(format!(
                "timed out waiting for {expected_reply:?} reply from direct prediction return"
            ))
        })
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
            // `peek` only proves that the first byte has arrived. Tunnelled
            // replies may be fragmented, so retaining the short poll timeout
            // while decoding the complete frame turns an ordinary partial
            // arrival into EWOULDBLOCK. Once downstream wins the race, give
            // the frame the remainder of the bounded fallback deadline.
            let remaining = DIRECT_RETURN_FALLBACK_TIMEOUT.saturating_sub(started.elapsed());
            downstream
                .set_read_timeout(Some(remaining.max(DIRECT_RETURN_FALLBACK_POLL)))
                .map_err(openai_io_error)?;
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

    #[test]
    fn direct_return_fallback_accepts_fragmented_downstream_reply() {
        use std::io::Write;

        let request_id = 91;
        let session_id = 92;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let (mut downstream, mut downstream_peer) = connected_stream_pair();
        let mut bytes = Vec::new();
        skippy_protocol::binary::send_reply_message(
            &mut bytes,
            &StageReply {
                kind: WireReplyKind::PredictedTokens,
                predicted: 0,
                predicted_tokens: vec![17, 23],
                native_mtp_draft: None,
                window: Default::default(),
                stats: StageReplyStats::default(),
            },
        )
        .unwrap();
        let writer = std::thread::spawn(move || {
            downstream_peer.write_all(&bytes[..1]).unwrap();
            downstream_peer.flush().unwrap();
            std::thread::sleep(DIRECT_RETURN_FALLBACK_POLL * 3);
            downstream_peer.write_all(&bytes[1..]).unwrap();
        });

        let reply = receive_embedded_stage_reply_one_of(
            &mut downstream,
            Some(&receiver),
            &[WireReplyKind::PredictedTokens],
        )
        .unwrap();

        assert_eq!(reply.predicted_tokens, vec![17, 23]);
        assert_eq!(downstream.read_timeout().unwrap(), None);
        writer.join().unwrap();
    }
}
