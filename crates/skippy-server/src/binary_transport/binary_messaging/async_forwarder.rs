use crate::binary_transport::WireCondition;
use crate::binary_transport::stage_execution::elapsed_ms;
use crate::binary_transport::write_stage_message_conditioned;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use serde_json::Value;
use serde_json::json;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireActivationDType;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

pub(super) struct AsyncForwarder {
    sender: mpsc::SyncSender<AsyncForwardJob>,
    pending: VecDeque<mpsc::Receiver<Result<()>>>,
}

struct AsyncForwardJob {
    message: StageWireMessage,
    wire_dtype: WireActivationDType,
    condition: WireCondition,
    attrs: BTreeMap<String, Value>,
    done: mpsc::Sender<Result<()>>,
}

impl AsyncForwarder {
    pub(super) fn new(downstream: &TcpStream, telemetry: Telemetry) -> Result<Self> {
        let mut writer = downstream
            .try_clone()
            .context("clone downstream stream for async activation forwarding")?;
        let (sender, receiver) = mpsc::sync_channel::<AsyncForwardJob>(1);
        thread::spawn(move || {
            while let Ok(job) = receiver.recv() {
                let write_start_unix_nanos = now_unix_nanos() as u64;
                let write_started = Instant::now();
                let result = write_stage_message_conditioned(
                    &mut writer,
                    &job.message,
                    job.wire_dtype,
                    job.condition,
                )
                .context("async forward activation frame downstream");
                let write_end_unix_nanos = now_unix_nanos() as u64;
                let mut attrs = job.attrs;
                attrs.insert(
                    "llama_stage.forward_write_ms".to_string(),
                    json!(elapsed_ms(write_started)),
                );
                telemetry.emit_debug_span(
                    "stage.binary_downstream_write",
                    attrs,
                    write_start_unix_nanos,
                    write_end_unix_nanos,
                );
                let _ = job.done.send(result);
            }
        });
        Ok(Self {
            sender,
            pending: VecDeque::new(),
        })
    }

    pub(super) fn send(
        &mut self,
        message: StageWireMessage,
        wire_dtype: WireActivationDType,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<()> {
        let (done, receiver) = mpsc::channel();
        self.sender
            .send(AsyncForwardJob {
                message,
                wire_dtype,
                condition,
                attrs,
                done,
            })
            .map_err(|_| anyhow!("async activation forwarder stopped"))?;
        self.pending.push_back(receiver);
        Ok(())
    }

    pub(super) fn flush(&mut self) -> Result<()> {
        while let Some(receiver) = self.pending.pop_front() {
            receiver
                .recv()
                .map_err(|_| anyhow!("async activation forwarder dropped result"))??;
        }
        Ok(())
    }
}
