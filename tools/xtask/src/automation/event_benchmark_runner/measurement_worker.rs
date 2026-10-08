//! Readiness, one excluded warmup, and one measured request for an owned server.
use super::{http_measurement, stream_metrics};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    future::Future,
    time::{Duration, Instant},
};

const MAX_MODEL_BYTES: usize = 4096;

fn bounded_error(error: &dyn std::fmt::Display) -> String {
    error.to_string().chars().take(1024).collect()
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub port: u16,
    pub prompt: String,
    pub prompt_sha256: String,
    pub max_tokens: u64,
    pub readiness_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub readiness_poll_ms: u64,
}

impl Input {
    pub fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.port == 0
            || self.prompt.is_empty()
            || self.prompt.len() > 16 * 1024
            || !(1..=4096).contains(&self.max_tokens)
            || !(1..=86_400_000).contains(&self.readiness_timeout_ms)
            || !(1..=86_400_000).contains(&self.request_timeout_ms)
            || !(1..=5000).contains(&self.readiness_poll_ms)
            || self.prompt_sha256 != hex::encode(Sha256::digest(self.prompt.as_bytes()))
            || self.readiness_timeout_ms + 2 * self.request_timeout_ms > 86_400_000
        {
            return Err(
                "invalid benchmark measurement worker identity, prompt or bounded budgets".into(),
            );
        }
        Ok(())
    }
    pub fn budget(&self) -> Duration {
        Duration::from_millis(self.readiness_timeout_ms + 2 * self.request_timeout_ms)
    }
}

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Evidence {
    pub schema_version: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_sha256: Option<String>,
    pub model: Option<String>,
    pub prompt_sha256: String,
    pub readiness_ms: Option<f64>,
    pub warmup_ms: Option<f64>,
    pub warmup_error: Option<String>,
    pub measurement: Option<stream_metrics::Measurement>,
    pub error: Option<String>,
}

pub(super) async fn bounded<T>(
    timeout: Duration,
    cancellation: &Cancellation,
    future: impl Future<Output = DynResult<T>>,
) -> DynResult<T> {
    let mut watch = tokio::time::interval(Duration::from_millis(25));
    let limited = tokio::time::timeout(timeout, future);
    tokio::pin!(limited);
    loop {
        if cancellation.is_cancelled() {
            return Err("benchmark measurement interrupted".into());
        }
        tokio::select! {
            result = &mut limited => return result.map_err(|_| "benchmark measurement deadline expired")?,
            _ = watch.tick() => {}
        }
    }
}

async fn ready(input: &Input, cancellation: &Cancellation) -> DynResult<String> {
    let deadline = Instant::now() + Duration::from_millis(input.readiness_timeout_ms);
    let url = format!("http://127.0.0.1:{}/v1/models", input.port);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("benchmark model readiness deadline expired".into());
        }
        if let Ok(bytes) = bounded(
            remaining
                .min(Duration::from_secs(2))
                .min(Duration::from_millis(input.request_timeout_ms)),
            cancellation,
            crate::automation::openai_exchange::get(&url),
        )
        .await
            && let Ok(value) = serde_json::from_slice(&bytes)
            && let Some(model) = stream_metrics::first_model(&value)
            && !model.trim().is_empty()
            && model.len() <= MAX_MODEL_BYTES
        {
            return Ok(model.into());
        }
        let pause = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(input.readiness_poll_ms));
        bounded(
            deadline.saturating_duration_since(Instant::now()),
            cancellation,
            async {
                tokio::time::sleep(pause).await;
                Ok(())
            },
        )
        .await?;
    }
}

async fn measure(
    input: &Input,
    cancellation: &Cancellation,
    output: &mut Evidence,
) -> DynResult<()> {
    let readiness = Instant::now();
    let resolved = ready(input, cancellation).await;
    output.readiness_ms = Some(readiness.elapsed().as_secs_f64() * 1000.0);
    let model = resolved?;
    output.model = Some(model.clone());
    let body = stream_metrics::chat_body(&input.prompt, input.max_tokens, &model)?;
    let timeout = Duration::from_millis(input.request_timeout_ms);
    let warmup = Instant::now();
    if let Err(error) = bounded(
        timeout,
        cancellation,
        http_measurement::request(input.port, &body),
    )
    .await
    {
        output.warmup_error = Some(bounded_error(&error));
    }
    output.warmup_ms = Some(warmup.elapsed().as_secs_f64() * 1000.0);
    let measurement = bounded(
        timeout,
        cancellation,
        http_measurement::request(input.port, &body),
    )
    .await?;
    let malformed = measurement.malformed;
    output.measurement = Some(measurement);
    if malformed {
        Err("benchmark measured response lacks valid completion metrics".into())
    } else {
        Ok(())
    }
}

pub(super) async fn execute(input: &Input, cancellation: &Cancellation) -> DynResult<Evidence> {
    input.validate()?;
    let mut output = Evidence {
        schema_version: 1,
        prompt_sha256: input.prompt_sha256.clone(),
        ..Evidence::default()
    };
    if let Err(error) = bounded(
        input.budget(),
        cancellation,
        measure(input, cancellation, &mut output),
    )
    .await
    {
        output.error = Some(bounded_error(&error));
    }
    Ok(output)
}

#[cfg(test)]
#[path = "measurement_worker_tests.rs"]
mod tests;
