//! Seed and measure one already-owned local A/B server with one signal scope.
use super::{options, publish, requests, synthetic_prompts, telemetry, telemetry_log};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Seed {
    families: u64,
    prefix_blocks: u64,
    output_tokens: u64,
    stagger_ms: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u64,
    phase: requests::Input,
    cache_seed: Option<Seed>,
    server_log: PathBuf,
    startup_timeout_secs: u64,
    telemetry_timeout_secs: u64,
}

impl Input {
    fn seed_phase(&self) -> DynResult<Option<requests::Input>> {
        self.cache_seed
            .as_ref()
            .map(|seed| {
                let mut phase = self.phase.clone();
                phase.prompts =
                    synthetic_prompts::interleaved(seed.families, 1, seed.prefix_blocks)?;
                phase.output_tokens = seed.output_tokens;
                phase.stagger_ms = seed.stagger_ms;
                phase.validate()?;
                Ok(phase)
            })
            .transpose()
    }

    fn validate(&self, seed: Option<&requests::Input>) -> DynResult<()> {
        self.phase.validate()?;
        if self.schema_version != 1
            || !self.server_log.is_absolute()
            || !(1..=86400).contains(&self.startup_timeout_secs)
            || !(1..=86400).contains(&self.telemetry_timeout_secs)
        {
            return Err("invalid A/B cell schema, log path or deadlines".into());
        }
        let budget = |phase: &requests::Input| {
            phase.request_timeout_secs
                + phase.prompts.len().saturating_sub(1) as f64 * phase.stagger_ms / 1000.0
        };
        let total = self.startup_timeout_secs as f64
            + 2.0 * self.telemetry_timeout_secs as f64
            + budget(&self.phase)
            + seed.map_or(0.0, budget);
        if !total.is_finite() || total > 86400.0 {
            return Err("A/B cell exceeds the one-day deadline budget".into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Evidence {
    schema_version: u64,
    round: u64,
    version: super::acceptance::Version,
    model: String,
    server_log: PathBuf,
    cache_seed: Option<requests::Phase>,
    measurement: Option<requests::Phase>,
    telemetry: Option<telemetry_log::Events>,
    summary: Option<telemetry::Cell>,
    error: Option<String>,
}

impl Evidence {
    fn new(input: &Input) -> Self {
        Self {
            schema_version: 1,
            round: input.phase.round,
            version: input.phase.version,
            model: input.phase.model.clone(),
            server_log: input.server_log.clone(),
            cache_seed: None,
            measurement: None,
            telemetry: None,
            summary: None,
            error: None,
        }
    }
}

fn check(cancellation: &Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        return Err("A/B cell interrupted".into());
    }
    Ok(())
}

#[derive(Deserialize)]
struct Models {
    data: Vec<Model>,
}
#[derive(Deserialize)]
struct Model {
    id: String,
}

async fn ready(input: &Input, cancellation: &Cancellation) -> DynResult<()> {
    let deadline = Instant::now() + Duration::from_secs(input.startup_timeout_secs);
    loop {
        check(cancellation)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("A/B model startup deadline expired".into());
        }
        let response = tokio::time::timeout(
            remaining.min(Duration::from_secs(2)),
            crate::automation::openai_exchange::get(&format!("{}/models", input.phase.base_url)),
        )
        .await;
        if let Ok(Ok(bytes)) = response
            && let Ok(models) = serde_json::from_slice::<Models>(&bytes)
        {
            if let [model] = models.data.as_slice() {
                if model.id == input.phase.model {
                    return Ok(());
                }
                return Err("A/B server model differs from the admitted stage identity".into());
            }
            if !models.data.is_empty() {
                return Err("A/B server advertises more than the one admitted model".into());
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn events(
    input: &Input,
    cursor: &telemetry_log::Cursor,
    expected: usize,
    cancellation: &Cancellation,
) -> DynResult<telemetry_log::Collection> {
    let deadline = Instant::now() + Duration::from_secs(input.telemetry_timeout_secs);
    loop {
        check(cancellation)?;
        if let Some(collection) = telemetry_log::collect(&input.server_log, cursor, expected)? {
            return Ok(collection);
        }
        if Instant::now() >= deadline {
            return Err("A/B telemetry deadline expired".into());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn summarize(
    phase: &requests::Phase,
    events: &telemetry_log::Events,
) -> DynResult<telemetry::Cell> {
    let mut document = serde_json::to_value(phase)?;
    let attributes = serde_json::to_value(events)?;
    let object = document
        .as_object_mut()
        .ok_or("request phase must serialize as an object")?;
    for (name, value) in attributes
        .as_object()
        .ok_or("telemetry must serialize as an object")?
    {
        object.insert(name.clone(), value.clone());
    }
    telemetry::summarize(serde_json::from_value(document)?)
}

async fn measure(
    input: &Input,
    seed: Option<requests::Input>,
    cancellation: &Cancellation,
    output: &mut Evidence,
) -> DynResult<()> {
    ready(input, cancellation).await?;
    let mut cursor = telemetry_log::snapshot(&input.server_log)?;
    if let Some(seed) = seed {
        let phase = requests::execute(seed, cancellation.clone()).await?;
        let successful = phase.successful();
        let passed = phase.passed();
        output.cache_seed = Some(phase);
        if !passed {
            return Err("A/B cache seed failed; seed request evidence retained".into());
        }
        cursor = events(input, &cursor, successful, cancellation)
            .await?
            .cursor;
    }
    let phase = requests::execute(input.phase.clone(), cancellation.clone()).await?;
    let successful = phase.successful();
    let passed = phase.passed();
    output.measurement = Some(phase);
    let collection = events(input, &cursor, successful, cancellation).await?;
    let phase = output
        .measurement
        .as_ref()
        .ok_or("missing measured phase")?;
    output.telemetry = Some(collection.events);
    output.summary = Some(summarize(
        phase,
        output
            .telemetry
            .as_ref()
            .ok_or("missing measured telemetry")?,
    )?);
    if !passed {
        return Err("A/B measured requests failed; cell evidence retained".into());
    }
    Ok(())
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let input: Input = serde_json::from_slice(&std::fs::read(opts["--input"])?)?;
    let seed = input.seed_phase()?;
    input.validate(seed.as_ref())?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let mut output = Evidence::new(&input);
    if let Err(error) = runtime.block_on(measure(
        &input,
        seed,
        &interrupt.cancellation(),
        &mut output,
    )) {
        output.error = Some(error.to_string());
    }
    if let Err(error) = interrupt.finish() {
        output.error = Some(match output.error.take() {
            Some(prior) => format!("{prior}; signal finalization failed: {error}"),
            None => error.to_string(),
        });
    }
    let mut bytes = serde_json::to_vec_pretty(&output)?;
    bytes.push(b'\n');
    publish(Path::new(opts["--output"]), &bytes)?;
    if output.error.is_some() {
        Err("A/B cell failed; partial evidence retained".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "cell_worker_tests.rs"]
mod tests;
