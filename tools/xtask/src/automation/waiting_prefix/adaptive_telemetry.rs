//! Required bounded typed prefill observations before diagnostic sanitization.
use serde::Serialize;
use serde_json::Value;
#[derive(Serialize)]
pub(super) struct Prefill {
    chunks: u64,
    minimum: u64,
    maximum: u64,
    elapsed_ms: f64,
}
#[derive(Default)]
pub(super) struct Observation {
    pub rows: Vec<Prefill>,
    pub error: Option<String>,
    pub latest_calibration: Option<std::collections::BTreeMap<String, f64>>,
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
        let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(());
        };
        if value["event"] == "stage.openai_prefill_calibration" {
            let mut projection = std::collections::BTreeMap::new();
            for key in [
                "llama_stage.prefill_bottleneck_compute_ms",
                "llama_stage.prefill_bottleneck_compute_ms_per_token",
                "llama_stage.prefill_bottleneck_stage_index",
                "llama_stage.prefill_bottleneck_token_count",
                "llama_stage.prefill_transport_write_to_compute",
                "llama_stage.prefill_transport_wait_to_compute",
                "llama_stage.prefill_calibration_observations",
            ] {
                if let Some(v) = value["attributes"].get(key) {
                    let number = v.as_f64().ok_or("calibration metric is not numeric")?;
                    if !number.is_finite() || number < 0.0 {
                        return Err("calibration metric invalid");
                    }
                    projection.insert(key.into(), number);
                }
            }
            self.latest_calibration = Some(projection);
            return Ok(());
        }
        if value["event"] != "stage.openai_prefill" {
            return Ok(());
        }
        if self.rows.len() >= 1001 {
            return Err("prefill observation count exceeds bound");
        }
        let attrs = &value["attributes"];
        let count = attrs["llama_stage.prefill_chunk_count"]
            .as_u64()
            .ok_or("prefill chunk count absent")?;
        let min = attrs["llama_stage.prefill_min_chunk_size"]
            .as_u64()
            .ok_or("prefill minimum absent")?;
        let max = attrs["llama_stage.prefill_max_chunk_size"]
            .as_u64()
            .ok_or("prefill maximum absent")?;
        let elapsed = attrs["llama_stage.elapsed_ms"]
            .as_f64()
            .ok_or("prefill elapsed absent")?;
        if count == 0 || min == 0 || min > max || !elapsed.is_finite() || elapsed < 0.0 {
            return Err("prefill telemetry invalid");
        }
        for key in [
            "skippy.kv.chain_cache_errors",
            "skippy.kv.stage0_cache_errors",
        ] {
            if attrs[key].as_u64() != Some(0) {
                return Err("prefill cache error evidence missing or nonzero");
            }
        }
        self.rows.push(Prefill {
            chunks: count,
            minimum: min,
            maximum: max,
            elapsed_ms: elapsed,
        });
        Ok(())
    }
    pub fn measured(&self, requests: usize, complete: bool) -> Result<&[Prefill], String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if !complete || self.rows.len() != requests + 1 {
            return Err(
                "required calibration plus measured prefill telemetry incomplete or ambiguous"
                    .into(),
            );
        }
        Ok(&self.rows[1..])
    }
}
