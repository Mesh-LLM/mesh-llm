//! Persist only typed measurement facts, separate from filtered child diagnostics.
use serde::Serialize;
use serde_json::{Map, Value};
use std::{
    collections::VecDeque,
    fs::File,
    io::Write,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread::JoinHandle,
};

const MAX_PENDING: usize = 1024;
const MAX_BYTES: usize = 128 * 1024 * 1024;

#[derive(Serialize)]
struct Measurement {
    event: &'static str,
    attributes: Map<String, Value>,
}

fn measurement(bytes: &[u8]) -> Result<Option<Measurement>, String> {
    if bytes.len() > 8192 {
        return Err("telemetry observation exceeds its line bound".into());
    }
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
        return Ok(None);
    };
    let event = match value["event"].as_str() {
        Some("stage.openai_generation_summary") => "stage.openai_generation_summary",
        Some("stage.openai_kv_capacity_decision") => "stage.openai_kv_capacity_decision",
        Some("stage.openai_kv_record_decision") => "stage.openai_kv_record_decision",
        _ => return Ok(None),
    };
    let source = value["attributes"]
        .as_object()
        .ok_or("measurement event requires typed attributes")?;
    let mut attributes = Map::new();
    for key in [
        "skippy.kv.matched_prefix_tokens",
        "skippy.kv.suffix_prefill_tokens",
        "skippy.kv.capacity_predicted_recompute_cost",
        "skippy.kv.capacity_evicted_tokens",
        "skippy.kv.capacity_evicted_entries",
        "skippy.kv.proactive_evicted_tokens",
        "skippy.kv.proactive_evicted_entries",
        "llama_stage.completion_token_count",
    ] {
        if let Some(value) = source.get(key) {
            let numeric = value
                .as_f64()
                .ok_or("measurement requires numeric counters")?;
            if !numeric.is_finite() || numeric < 0.0 {
                return Err("invalid measurement counter".into());
            }
            attributes.insert(key.into(), value.clone());
        }
    }
    for (key, accepted) in [
        ("skippy.kv.status", &["hit", "miss"][..]),
        ("skippy.kv.capacity_status", &["rejected"][..]),
        ("skippy.kv.decision", &["proactive_eviction"][..]),
    ] {
        if let Some(status) = source.get(key).and_then(Value::as_str)
            && let Some(known) = accepted.iter().find(|known| **known == status)
        {
            attributes.insert(key.into(), serde_json::json!(known));
        }
    }
    if let Some(value) = source.get("skippy.request_id") {
        let id = value
            .as_str()
            .ok_or("server request identity must be a string")?;
        let numeric: u64 = id
            .parse()
            .map_err(|_| "server request identity must be decimal u64")?;
        attributes.insert(
            "skippy.request_id".into(),
            serde_json::json!(numeric.to_string()),
        );
    }
    Ok(Some(Measurement { event, attributes }))
}

pub(super) struct Sink {
    pending: VecDeque<Measurement>,
    sender: Option<SyncSender<Measurement>>,
    worker: Option<JoinHandle<std::io::Result<()>>>,
    failed: Arc<AtomicBool>,
}

impl Sink {
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let mut file = File::create_new(path)?;
        let (sender, receiver) = mpsc::sync_channel::<Measurement>(MAX_PENDING);
        let failed = Arc::new(AtomicBool::new(false));
        let flag = failed.clone();
        let worker = std::thread::spawn(move || {
            let result = (|| -> std::io::Result<()> {
                let mut retained = 0_usize;
                for event in receiver {
                    let mut bytes = serde_json::to_vec(&event)?;
                    bytes.push(b'\n');
                    if bytes.len() > MAX_BYTES.saturating_sub(retained) {
                        return Err(std::io::Error::other("measurement file exceeds 128 MiB"));
                    }
                    retained += bytes.len();
                    file.write_all(&bytes)?;
                    file.flush()?;
                }
                file.sync_all()
            })();
            if result.is_err() {
                flag.store(true, Ordering::SeqCst);
            }
            result
        });
        Ok(Self {
            pending: VecDeque::new(),
            sender: Some(sender),
            worker: Some(worker),
            failed,
        })
    }

    /// No I/O, blocking sends, raw-line retention, logging, or thread creation.
    pub fn observe(&mut self, bytes: &[u8]) -> Result<(), String> {
        if let Some(event) = measurement(bytes)? {
            if self.pending.len() == MAX_PENDING {
                return Err("measurement queue overflow".into());
            }
            self.pending.push_back(event);
        }
        Ok(())
    }

    /// The coordinator tick exchanges bounded typed messages with its writer.
    pub fn tick(&mut self) -> Result<(), String> {
        if self.failed.load(Ordering::SeqCst) {
            return Err("measurement writer failed".into());
        }
        let sender = self.sender.as_ref().ok_or("measurement writer closed")?;
        while let Some(event) = self.pending.pop_front() {
            match sender.try_send(event) {
                Ok(()) => {}
                Err(TrySendError::Full(event)) => {
                    self.pending.push_front(event);
                    break;
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err("measurement writer disconnected".into());
                }
            }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> std::io::Result<()> {
        let pending = !self.pending.is_empty();
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| std::io::Error::other("measurement writer panicked"))??;
        }
        if pending {
            return Err(std::io::Error::other(
                "measurement facts were not delivered before cleanup",
            ));
        }
        Ok(())
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
#[path = "telemetry_sink_tests.rs"]
mod tests;
