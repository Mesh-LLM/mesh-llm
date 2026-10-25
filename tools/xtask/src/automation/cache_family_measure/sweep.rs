//! Ordered scaling stages on one retained parent's unchanged host.
use super::{contract::Input, measurement};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InputSweep {
    pub schema_version: u64,
    pub stages: Vec<Input>,
    pub execution_timeout_ms: u64,
}
impl InputSweep {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.stages.is_empty()
            || self.stages.len() > 32
            || !(1..=3_600_000).contains(&self.execution_timeout_ms)
        {
            return Err("invalid bounded persistent cache sweep".into());
        }
        let first = &self.stages[0];
        let mut previous = 0;
        for stage in &self.stages {
            stage.validate()?;
            if stage.cohort == super::contract::Cohort::NativeSerial
                || stage.cohort != first.cohort
                || stage.base_url != first.base_url
                || stage.prompt != first.prompt
                || stage.model_id != first.model_id
                || stage.output_tokens != first.output_tokens
                || stage.concurrency <= previous
            {
                return Err(
                    "cache sweep must preserve host/prompt/protocol and ordered concurrency".into(),
                );
            }
            previous = stage.concurrency;
        }
        Ok(())
    }
}
pub(super) async fn execute(
    input: &InputSweep,
    request_sha256: String,
    cancel: &Cancellation,
) -> Value {
    let until = Instant::now() + Duration::from_millis(input.execution_timeout_ms);
    let mut rows = Vec::new();
    for (index, stage) in input.stages.iter().enumerate() {
        let remaining = until.saturating_duration_since(Instant::now()).as_millis();
        if remaining == 0 || cancel.is_cancelled() {
            break;
        }
        let mut bounded = stage.clone();
        bounded.execution_timeout_ms = bounded
            .execution_timeout_ms
            .min(u64::try_from(remaining).unwrap_or(u64::MAX));
        let hash =
            super::hash(&serde_json::to_vec(&bounded).expect("typed sweep stage serialization"));
        let receipt = measurement::execute(&bounded, hash, cancel).await;
        let complete = receipt["status"] == "completed";
        rows.push(json!({"index":index,"concurrency":stage.concurrency,"measurement":receipt}));
        if !complete {
            break;
        }
    }
    let complete = rows.len() == input.stages.len()
        && rows
            .iter()
            .all(|r| r["measurement"]["status"] == "completed")
        && !cancel.is_cancelled()
        && Instant::now() < until;
    json!({"schema_version":1,"request_sha256":request_sha256,"status":if complete{"completed"}else{"incomplete"},
        "host_custody":"single retained parent; host identity assessed by owning cell","planned_stages":input.stages.len(),"sweep":rows})
}
