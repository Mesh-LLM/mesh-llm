use crate::binary_transport::direct_return;
use crate::binary_transport::direct_return::PredictionReturnSinks;
use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use skippy_protocol::StageConfig;
use skippy_protocol::StageTopology;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageReplyWindow;
use skippy_protocol::binary::WireActivationDType;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;
use skippy_protocol::binary::send_reply_message;
use std::collections::BTreeMap;
use std::net::TcpStream;
use std::time::Duration;

pub(super) fn drain_deferred_prefill_replies(
    downstream: Option<&mut TcpStream>,
    pending_prefill_replies: &mut usize,
    pending_reply_stats: &mut StageReplyStats,
) -> Result<()> {
    let Some(downstream) = downstream else {
        return Ok(());
    };
    while *pending_prefill_replies > 0 {
        let reply =
            recv_reply(&mut *downstream).context("drain deferred downstream prefill ACK")?;
        if reply.kind != WireReplyKind::Ack {
            bail!("expected deferred downstream ACK");
        }
        pending_reply_stats.merge(reply.stats);
        *pending_prefill_replies -= 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(in crate::binary_transport) fn configure_prediction_return_stream(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
    downstream_connect_timeout_secs: u64,
    prediction_return_sinks: &PredictionReturnSinks,
    prediction_return_streams: &mut BTreeMap<(u64, u64), TcpStream>,
) {
    if prediction_return_streams.contains_key(&(request_id, session_id)) {
        return;
    }
    match prediction_return_sinks.take_wait(request_id, session_id, Duration::from_millis(250)) {
        Ok(Some(stream)) => {
            prediction_return_streams.insert((request_id, session_id), stream);
            eprintln!("direct prediction return using upstream-opened sink");
            return;
        }
        Ok(None) => {}
        Err(error) => {
            eprintln!("direct prediction return sink lookup failed: {error:#}");
        }
    }

    match direct_return::open_prediction_return_stream(
        config,
        topology,
        request_id,
        session_id,
        wire_dtype,
        downstream_connect_timeout_secs,
    ) {
        Ok(stream) => {
            prediction_return_streams.insert((request_id, session_id), stream);
        }
        Err(error) => {
            eprintln!(
                "direct prediction return unavailable; falling back to upstream reply: {error:#}"
            );
        }
    }
}
pub(super) fn send_stage_reply(stream: &mut TcpStream, reply: StageReply) -> Result<()> {
    send_reply_message(stream, &reply).context("send stage reply")
}

pub(super) fn reply_window_for_message(
    message: &skippy_protocol::binary::StageWireMessage,
) -> StageReplyWindow {
    if message.kind == skippy_protocol::binary::WireMessageKind::VerifyWindow {
        StageReplyWindow {
            window_id: message.state.seq_id,
        }
    } else {
        Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{StageStateHeader, StageWireMessage, WireMessageKind};

    #[test]
    fn verify_window_reply_reports_only_the_coordinator_window_id() {
        let kind = WireMessageKind::VerifyWindow;
        let mut state = StageStateHeader::new(kind, WireActivationDType::F16);
        state.seq_id = 42;
        let message = StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 3,
            state,
            request_id: 11,
            session_id: 13,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![10, 11, 12],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };

        let reply = reply_window_for_message(&message);

        assert_eq!(reply.window_id, 42);
    }
}
