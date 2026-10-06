//! Local producer timestamps classify delayed stderr events, including owned shutdown-tail drain.
use super::mixed_counters::{Measured, Prefill, Projection, Scheduler};
use serde_json::Value;
#[derive(serde::Serialize)]
struct Stamped {
    start: u64,
    end: u64,
    scheduler: Vec<Scheduler>,
    prefills: Vec<Prefill>,
}
#[derive(Default, serde::Serialize)]
pub(super) struct Observation {
    rows: Vec<Stamped>,
    pub error: Option<String>,
}
impl Observation {
    pub fn observe(&mut self, bytes: &[u8]) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = self.line(bytes) {
            self.error = Some(error.into());
        }
    }
    fn line(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        if bytes.len() > 8192 {
            return Err("mixed phase line exceeds capture bound");
        }
        let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(());
        };
        if !matches!(
            value["event"].as_str(),
            Some(
                "stage.openai_prefill"
                    | "stage.scheduler_iteration"
                    | "stage.scheduler_feature_iteration"
            )
        ) {
            return Ok(());
        }
        if self.rows.len() >= 65536 {
            return Err("mixed phase rows exceed bound");
        }
        let attrs = &value["attributes"];
        for key in ["skippy.otel_dropped_events", "skippy.otel_export_errors"] {
            if attrs[key].as_u64() != Some(0) {
                return Err("mixed phase telemetry dropped/export errors absent or nonzero");
            }
        }
        let start = value["start_time_unix_nanos"]
            .as_u64()
            .filter(|v| *v > 0)
            .ok_or("mixed producer start timestamp absent")?;
        let end = value["end_time_unix_nanos"]
            .as_u64()
            .filter(|v| *v > start)
            .ok_or("mixed producer end timestamp absent or inverted")?;
        let mut projection = Projection::default();
        projection.observe(bytes);
        let (scheduler, prefills) = projection.into_parts()?;
        self.rows.push(Stamped {
            start,
            end,
            scheduler,
            prefills,
        });
        Ok(())
    }
    pub fn measured(
        &self,
        worker: &Value,
        requests: usize,
        complete: bool,
    ) -> Result<Measured, &'static str> {
        if self.error.is_some() || !complete || !(1..=16).contains(&requests) {
            return Err("mixed complete owned phase capture unavailable");
        }
        let start = worker["measured_start_unix_nanos"]
            .as_u64()
            .ok_or("mixed measured start absent")?;
        let end = worker["measured_end_unix_nanos"]
            .as_u64()
            .filter(|v| *v > start)
            .ok_or("mixed measured end absent or inverted")?;
        if worker["local_clock_consistent"] != true || !worker["error"].is_null() {
            return Err("mixed local clock or worker completion unqualified");
        }
        let request_sha256 = worker["input_sha256"]
            .as_str()
            .filter(|v| {
                v.len() == 64
                    && v.bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            })
            .ok_or("mixed worker SHA absent")?;
        let mut warmup = 0;
        let mut scheduler = Vec::new();
        let mut prefills = Vec::new();
        for row in &self.rows {
            if row.end <= start {
                warmup += row.prefills.len();
                continue;
            }
            if row.start >= end {
                continue;
            }
            if row.start < start || row.end > end {
                return Err("mixed producer span crosses measured phase boundary");
            }
            scheduler.extend(row.scheduler.clone());
            prefills.extend(row.prefills.clone());
        }
        if warmup != 1 || prefills.len() != requests || scheduler.is_empty() {
            return Err("mixed warmup/measured numeric roster incomplete");
        }
        for row in &scheduler {
            row.tokens()?;
        }
        Ok(Measured {
            scheduler,
            prefills,
            request_sha256: request_sha256.into(),
            phase_provenance: "owned-local-producer-timestamps-with-complete-shutdown-tail",
            prefill_role_provenance: "token-count-ranked-original-heuristic-not-request-ID-correlation",
        })
    }
}

#[cfg(test)]
#[path = "mixed_phase_tests.rs"]
mod tests;
