use super::STAGE_BINARY_DOWNSTREAM_FORWARD_ERROR;
use super::downstream_forward_error_attrs;
use crate::binary_transport::WireCondition;
use crate::binary_transport::stage_execution::elapsed_ms;
use crate::binary_transport::write_stage_message_after_propagation;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use serde_json::Value;
use serde_json::json;
use skippy_protocol::PeerConfig;
use skippy_protocol::binary::StageWireMessage;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::net::TcpStream;
use std::sync::mpsc;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::mpsc::TryRecvError;
use std::thread;
use std::time::Duration;
use std::time::Instant;

const ASYNC_FORWARD_TERMINAL_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) struct AsyncForwarder {
    sender: Option<mpsc::SyncSender<AsyncForwardJob>>,
    pending: VecDeque<AsyncForwardReceipt>,
    writer: Option<thread::JoinHandle<()>>,
}

impl Drop for AsyncForwarder {
    /// Queued frames must not still be on the wire after the request that
    /// owns them returns: a persistent lane is handed back for reuse, and a
    /// teardown `Stop` written through another clone of the same socket
    /// would interleave with a frame this forwarder is still writing.
    /// Dropping the sender ends the writer loop once its queue drains, and
    /// the join makes that ordering observable to the caller.
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

pub(crate) struct AsyncForwardReceipt {
    receiver: mpsc::Receiver<Result<f64, String>>,
}

struct AsyncForwardJob {
    message: StageWireMessage,
    condition: WireCondition,
    attrs: BTreeMap<String, Value>,
    done: mpsc::Sender<Result<f64, String>>,
    enqueued_at: Instant,
    enqueued_unix_nanos: u64,
}

impl AsyncForwarder {
    pub(crate) fn new(
        downstream: &TcpStream,
        downstream_config: Option<PeerConfig>,
        telemetry: Telemetry,
        queue_capacity: usize,
    ) -> Result<Self> {
        let mut writer = downstream
            .try_clone()
            .context("clone downstream stream for async activation forwarding")?;
        writer
            .set_write_timeout(Some(ASYNC_FORWARD_TERMINAL_TIMEOUT))
            .context("set async activation forward write timeout")?;
        let (sender, receiver) = mpsc::sync_channel::<AsyncForwardJob>(queue_capacity.max(1));
        let writer_thread = thread::spawn(move || {
            run_forwarder(&mut writer, &receiver, &telemetry, &downstream_config)
        });
        Ok(Self {
            sender: Some(sender),
            pending: VecDeque::new(),
            writer: Some(writer_thread),
        })
    }

    pub(crate) fn send(
        &mut self,
        message: StageWireMessage,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<()> {
        let receipt = self.send_tracked(message, condition, attrs)?;
        self.pending.push_back(receipt);
        Ok(())
    }

    pub(crate) fn send_tracked(
        &mut self,
        message: StageWireMessage,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<AsyncForwardReceipt> {
        self.reap_completed()?;
        let (done, receiver) = mpsc::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow!("async activation forwarder stopped"))?
            .send(AsyncForwardJob {
                message,
                condition,
                attrs,
                done,
                enqueued_at: Instant::now(),
                enqueued_unix_nanos: now_unix_nanos() as u64,
            })
            .map_err(|_| anyhow!("async activation forwarder stopped"))?;
        Ok(AsyncForwardReceipt { receiver })
    }

    fn reap_completed(&mut self) -> Result<()> {
        loop {
            let Some(receiver) = self.pending.front() else {
                return Ok(());
            };
            match receiver.try_finish() {
                Ok(Some(_write_ms)) => {
                    self.pending.pop_front();
                }
                Ok(None) => return Ok(()),
                Err(error) => {
                    self.pending.pop_front();
                    return Err(error);
                }
            }
        }
    }

    pub(crate) fn flush(&mut self) -> Result<()> {
        while let Some(receiver) = self.pending.pop_front() {
            receiver.finish()?;
        }
        Ok(())
    }
}

fn run_forwarder(
    writer: &mut TcpStream,
    receiver: &mpsc::Receiver<AsyncForwardJob>,
    telemetry: &Telemetry,
    downstream_config: &Option<PeerConfig>,
) {
    while let Ok(job) = receiver.recv() {
        let wait = time_until_ready(&job);
        if !wait.is_zero() {
            thread::sleep(wait);
        }
        forward_job(writer, telemetry, job, downstream_config.as_ref());
    }
}

fn time_until_ready(job: &AsyncForwardJob) -> std::time::Duration {
    let ready_at = job.enqueued_at + job.condition.propagation_delay();
    ready_at.saturating_duration_since(Instant::now())
}

fn forward_job(
    writer: &mut TcpStream,
    telemetry: &Telemetry,
    job: AsyncForwardJob,
    downstream: Option<&PeerConfig>,
) {
    let result = write_stage_message_after_propagation(writer, &job.message, job.condition)
        .context("async forward activation frame downstream")
        .map(|()| elapsed_ms(job.enqueued_at))
        .map_err(|error| format!("{error:#}"));
    let write_end_unix_nanos = now_unix_nanos() as u64;
    // The write span keeps its original name and timing on both outcomes;
    // the failure event is additive and fires at normal telemetry level
    // (identity attrs are built at the enqueue site, outside any debug
    // guard) so failures are observable without debug telemetry.
    let mut span_attrs = job.attrs.clone();
    span_attrs.insert(
        "llama_stage.forward_write_ms".to_string(),
        json!(elapsed_ms(job.enqueued_at)),
    );
    telemetry.emit_debug_span(
        "stage.binary_downstream_write",
        span_attrs,
        job.enqueued_unix_nanos,
        write_end_unix_nanos,
    );
    if let Err(error) = &result {
        telemetry.emit_span(
            STAGE_BINARY_DOWNSTREAM_FORWARD_ERROR,
            downstream_forward_error_attrs(job.attrs, downstream, error),
            job.enqueued_unix_nanos,
            write_end_unix_nanos,
        );
    }
    let _ = job.done.send(result);
}

impl AsyncForwardReceipt {
    pub(crate) fn finish(self) -> Result<f64> {
        self.finish_with_timeout(ASYNC_FORWARD_TERMINAL_TIMEOUT)
    }

    fn finish_with_timeout(self, timeout: Duration) -> Result<f64> {
        match self.receiver.recv_timeout(timeout) {
            Ok(result) => result.map_err(|error| anyhow!(error)),
            Err(RecvTimeoutError::Timeout) => {
                Err(anyhow!("timed out waiting for async activation forward"))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(anyhow!("async activation forwarder dropped result"))
            }
        }
    }

    fn try_finish(&self) -> Result<Option<f64>> {
        match self.receiver.try_recv() {
            Ok(Ok(write_ms)) => Ok(Some(write_ms)),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(anyhow!("async activation forwarder dropped result"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use skippy_protocol::binary::{StageStateHeader, WireMessageKind, read_stage_message};

    use super::*;
    use crate::binary_transport::stage_execution::prefix_cache_test_config;
    use crate::binary_transport::stage_execution::{
        binary_message_attrs, binary_message_session_id,
    };
    use crate::telemetry::TelemetryLevel;

    fn message(kind: WireMessageKind, pos_start: i32) -> StageWireMessage {
        StageWireMessage {
            kind,
            pos_start,
            token_count: if kind == WireMessageKind::RetireVerifyWindow {
                4
            } else {
                0
            },
            state: StageStateHeader::new(kind),
            request_id: 1,
            session_id: 2,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    /// A delayed discard must be fully written before a teardown `Stop` that
    /// goes out through a different clone of the same socket, otherwise the
    /// two frames interleave and poison a lane that is handed back for reuse.
    #[test]
    fn a_delayed_discard_lands_before_a_teardown_stop_on_another_clone() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(address).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let telemetry = Telemetry::new(None, 1, prefix_cache_test_config(), TelemetryLevel::Off);
        let mut forwarder = AsyncForwarder::new(&client, None, telemetry, 8).unwrap();
        // 250ms of simulated propagation: without the drop-time join, the
        // teardown write below wins the race and the frames interleave.
        let condition = WireCondition::new(250.0, None).unwrap();

        forwarder
            .send(
                message(WireMessageKind::DiscardStaleWindows, 11),
                condition,
                BTreeMap::new(),
            )
            .unwrap();
        drop(forwarder);

        write_stage_message_after_propagation(
            &mut client,
            &message(WireMessageKind::Stop, 22),
            WireCondition::new(0.0, None).unwrap(),
        )
        .unwrap();

        let first = read_stage_message(&mut server, 4).unwrap();
        let second = read_stage_message(&mut server, 4).unwrap();

        assert_eq!(first.kind, WireMessageKind::DiscardStaleWindows);
        assert_eq!(first.pos_start, 11);
        assert_eq!(second.kind, WireMessageKind::Stop);
        assert_eq!(second.pos_start, 22);
    }

    /// The lane-reuse half of the same property: after a delayed discard and
    /// the teardown `Stop`, the socket must be clean enough to serve the next
    /// request. A frame left half-written by the torn-down forwarder would be
    /// read as the next request's header, so this asserts both frames arrive
    /// whole and in order and that the next request's traffic follows them
    /// undisturbed on the same socket.
    #[test]
    fn a_reused_lane_carries_the_next_request_after_a_delayed_teardown() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut lane = TcpStream::connect(address).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let telemetry = Telemetry::new(None, 1, prefix_cache_test_config(), TelemetryLevel::Off);
        let mut forwarder = AsyncForwarder::new(&lane, None, telemetry, 8).unwrap();
        let delayed = WireCondition::new(250.0, None).unwrap();
        let immediate = WireCondition::new(0.0, None).unwrap();

        // First request: a discard still in flight when the request ends.
        forwarder
            .send(
                message(WireMessageKind::DiscardStaleWindows, 31),
                delayed,
                BTreeMap::new(),
            )
            .unwrap();
        drop(forwarder);
        write_stage_message_after_propagation(
            &mut lane,
            &message(WireMessageKind::Stop, 32),
            immediate,
        )
        .unwrap();

        // The pool hands the same socket to the next request.
        write_stage_message_after_propagation(
            &mut lane,
            &message(WireMessageKind::VerifyWindow, 41),
            immediate,
        )
        .unwrap();

        let frames = (0..3)
            .map(|_| read_stage_message(&mut server, 4).unwrap())
            .collect::<Vec<_>>();
        let observed = frames
            .iter()
            .map(|frame| (frame.kind, frame.pos_start))
            .collect::<Vec<_>>();

        assert_eq!(
            observed,
            vec![
                (WireMessageKind::DiscardStaleWindows, 31),
                (WireMessageKind::Stop, 32),
                (WireMessageKind::VerifyWindow, 41),
            ],
            "a lane returned to the pool must carry the next request intact"
        );
    }

    #[test]
    fn retirement_receipt_orders_all_prior_verify_writes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect(address).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let telemetry = Telemetry::new(None, 1, prefix_cache_test_config(), TelemetryLevel::Off);
        let mut forwarder = AsyncForwarder::new(&client, None, telemetry, 3).unwrap();
        let condition = WireCondition::new(0.0, None).unwrap();

        forwarder
            .send(
                message(WireMessageKind::VerifyWindow, 10),
                condition,
                BTreeMap::new(),
            )
            .unwrap();
        forwarder
            .send(
                message(WireMessageKind::VerifyWindow, 14),
                condition,
                BTreeMap::new(),
            )
            .unwrap();
        forwarder
            .send_tracked(
                message(WireMessageKind::RetireVerifyWindow, 10),
                condition,
                BTreeMap::new(),
            )
            .unwrap()
            .finish()
            .unwrap();

        let first = read_stage_message(&mut server, 1).unwrap();
        let second = read_stage_message(&mut server, 1).unwrap();
        let retire = read_stage_message(&mut server, 1).unwrap();
        assert_eq!(first.kind, WireMessageKind::VerifyWindow);
        assert_eq!(first.pos_start, 10);
        assert_eq!(second.kind, WireMessageKind::VerifyWindow);
        assert_eq!(second.pos_start, 14);
        assert_eq!(retire.kind, WireMessageKind::RetireVerifyWindow);
        assert_eq!(retire.pos_start, 10);
    }

    #[test]
    fn forward_receipt_has_a_terminal_wait_bound() {
        let (_sender, receiver) = mpsc::channel();
        let receipt = AsyncForwardReceipt { receiver };

        let error = receipt
            .finish_with_timeout(Duration::from_millis(1))
            .unwrap_err();

        assert!(error.to_string().contains("timed out"));
    }

    fn failing_write_pair() -> TcpStream {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(address).unwrap();
        let (server, _) = listener.accept().unwrap();
        // Linger-0 close sends a TCP RST immediately. std `tcp_linger` is
        // unstable on the pinned 1.98 toolchain, so this goes through
        // socket2 (already a dependency).
        socket2::SockRef::from(&server)
            .set_linger(Some(Duration::from_secs(0)))
            .unwrap();
        drop(server);
        // Poke until the kernel surfaces the reset, then ASSERT the
        // precondition instead of assuming it: a write that still succeeds
        // would silently invalidate the test's premise.
        use std::io::Write as _;
        let mut reset_observed = false;
        for _ in 0..200 {
            if client.write_all(&[0]).is_err() {
                reset_observed = true;
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            reset_observed,
            "precondition: client socket must observe the peer reset before the forward under test"
        );
        client
    }

    fn forward_error_peer() -> PeerConfig {
        PeerConfig {
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            endpoint: "10.0.0.2:50051".to_string(),
        }
    }

    fn run_forward_job(
        client: &mut TcpStream,
        telemetry: &Telemetry,
    ) -> std::result::Result<f64, String> {
        let frame = message(WireMessageKind::VerifyWindow, 7);
        let attrs = binary_message_attrs(
            &prefix_cache_test_config(),
            binary_message_session_id(0, &frame),
            &frame,
        );
        let (done, receiver) = mpsc::channel();
        let job = AsyncForwardJob {
            message: frame,
            condition: WireCondition::new(0.0, None).unwrap(),
            attrs,
            done,
            enqueued_at: Instant::now(),
            enqueued_unix_nanos: now_unix_nanos() as u64,
        };
        forward_job(client, telemetry, job, Some(&forward_error_peer()));
        receiver.recv().unwrap()
    }

    /// A failed activation forward must preserve the original debug write
    /// span (same name and timing semantics) and additively emit the
    /// normal-level failure event carrying the frame's request/session
    /// identity, the intended downstream identity and the error chain.
    #[test]
    fn forward_failure_preserves_write_span_and_emits_identity_error_event() {
        let mut client = failing_write_pair();
        let config = prefix_cache_test_config();
        let (telemetry, rx) = Telemetry::captured(config.clone(), TelemetryLevel::Debug);

        let result = run_forward_job(&mut client, &telemetry);
        let error = result.unwrap_err();
        assert!(error.contains("async forward activation frame downstream"));

        let mut write_span = None;
        let mut error_event = None;
        while let Ok(event) = rx.try_recv() {
            match event.event.as_str() {
                "stage.binary_downstream_write" => write_span = Some(event),
                "stage.binary_downstream_forward_error" => error_event = Some(event),
                _ => {}
            }
        }

        let write_span = write_span.expect("write span preserved on error at debug level");
        assert!(
            write_span
                .attributes
                .contains_key("llama_stage.forward_write_ms")
        );
        assert_eq!(write_span.attributes["skippy.request_id"], json!("1"));

        let error_event = error_event.expect("failure event emitted");
        let attrs = &error_event.attributes;
        assert_eq!(attrs["llama_stage.downstream_stage_id"], json!("stage-1"));
        assert_eq!(attrs["llama_stage.downstream_stage_index"], json!(1));
        assert_eq!(
            attrs["llama_stage.downstream_endpoint"],
            json!("10.0.0.2:50051")
        );
        assert_eq!(attrs["skippy.request_id"], json!("1"));
        assert_eq!(attrs["skippy.session_id"], json!("2"));
        assert!(
            attrs["llama_stage.error"]
                .as_str()
                .unwrap()
                .contains("async forward")
        );
        assert_eq!(attrs["skippy.run_id"], json!("run"));
        assert_eq!(attrs["skippy.stage_id"], json!("stage-0"));
        assert!(error_event.end_time_unix_nanos >= error_event.start_time_unix_nanos);
    }

    /// At `Summary` telemetry the failure event still fires with full
    /// identity, while the debug-only write span stays suppressed — that is
    /// the documented level contract for downstream forward failures.
    #[test]
    fn forward_failure_event_fires_at_summary_level_without_debug_span() {
        let mut client = failing_write_pair();
        let (telemetry, rx) =
            Telemetry::captured(prefix_cache_test_config(), TelemetryLevel::Summary);

        run_forward_job(&mut client, &telemetry).unwrap_err();

        let mut names = Vec::new();
        while let Ok(event) = rx.try_recv() {
            names.push(event.event);
        }
        assert_eq!(
            names,
            vec!["stage.binary_downstream_forward_error"],
            "only the normal-level failure event fires at Summary"
        );
    }
}
