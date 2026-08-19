use super::*;
use crate::logging::{
    Clock, LoggingService, OpenAiLifecycleAttachment, PersistSink, RawMeshLifecycleOwners,
    RawMeshRequestLifecycle, RequestSummaryEntry,
};
use crate::network::target_health::TargetHealthOutcome;
use anyhow::Result;
use mesh_llm_events::logging::proxy::ProxyRecord;
use mesh_llm_events::logging::replay::ReplayChannel;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

/// Deterministic counter clock for tests that assert on serialized record
/// contents. `SystemClock` stamps nanosecond-precision wall-clock time, and a
/// redaction assertion checking for bare digit substrings (e.g. a port
/// fixture) can collide with those random nanosecond digits; this clock keeps
/// timestamps fixed-format and predictable so such assertions can't flake.
#[derive(Default)]
struct DeterministicClock {
    counter: AtomicU64,
}

impl Clock for DeterministicClock {
    fn now(&self) -> String {
        let n = self.counter.fetch_add(1, AtomicOrdering::Relaxed);
        format!("2025-01-01T00:00:00.{n:09}Z")
    }
}

#[derive(Default)]
struct TransportProxySink {
    proxy_records: Mutex<Vec<ProxyRecord>>,
    summaries: Mutex<HashMap<String, RequestSummaryEntry>>,
    artifact_pointers: Mutex<Vec<(String, serde_json::Value)>>,
}

impl TransportProxySink {
    fn proxy_records(&self) -> Vec<ProxyRecord> {
        self.proxy_records
            .lock()
            .expect("transport proxy records lock")
            .clone()
    }

    fn summary_count(&self) -> usize {
        self.summaries
            .lock()
            .expect("transport summaries lock")
            .len()
    }

    fn artifact_pointers(&self) -> Vec<(String, serde_json::Value)> {
        self.artifact_pointers
            .lock()
            .expect("transport artifact pointers lock")
            .clone()
    }
}

#[async_trait::async_trait]
impl PersistSink for TransportProxySink {
    async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String> {
        self.summaries
            .lock()
            .expect("transport summaries lock")
            .insert(entry.request_id.clone(), entry);
        Ok(())
    }

    async fn persist_event(
        &self,
        _request_id: String,
        _event_id: String,
        _channel: ReplayChannel,
        _sequence: u64,
        _occurred_at: String,
        _payload_json: String,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn persist_artifact_pointer(
        &self,
        request_id: String,
        artifact_data: serde_json::Value,
    ) -> Result<(), String> {
        self.artifact_pointers
            .lock()
            .expect("transport artifact pointers lock")
            .push((request_id, artifact_data));
        Ok(())
    }

    async fn persist_proxy_record(&self, proxy_json: String) -> Result<(), String> {
        self.proxy_records
            .lock()
            .expect("transport proxy records lock")
            .push(serde_json::from_str(&proxy_json).expect("bounded proxy record"));
        Ok(())
    }

    async fn persist_audit_entry(
        &self,
        _record: crate::logging::OperationalAuditRecord,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn persist_webhook_delivery(
        &self,
        _request_id: Option<String>,
        _status_code: u16,
        _error: Option<String>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn persist_cleanup_run(&self, _deleted_count: u64) -> Result<(), String> {
        Ok(())
    }
}

fn recorded_lifecycle_events(
    service: &LoggingService,
) -> Vec<mesh_llm_events::logging::events::LifecycleEvent> {
    service
        .bus_ref()
        .replay_window()
        .records
        .into_iter()
        .filter_map(|record| {
            let envelope = serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
            let payload = envelope.get("payload")?.as_str()?;
            serde_json::from_str(payload).ok()
        })
        .collect()
}

#[path = "transport_tests/durable_artifacts.rs"]
mod durable_artifacts;
#[path = "transport_tests/lifecycle.rs"]
mod lifecycle;
#[path = "transport_tests/routing.rs"]
mod routing;
