//! Concurrent staggered prompt requests against an already-owned local server.
use super::{Prompt, acceptance::Version, options, publish};
use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub(super) schema_version: u64,
    pub(super) round: u64,
    pub(super) version: Version,
    pub(super) base_url: String,
    pub(super) model: String,
    pub(super) output_tokens: u64,
    pub(super) request_timeout_secs: f64,
    pub(super) stagger_ms: f64,
    pub(super) prompts: Vec<Prompt>,
}

impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        let endpoint: hyper::Uri = self.base_url.parse()?;
        if endpoint.scheme_str() != Some("http")
            || endpoint.host() != Some("127.0.0.1")
            || endpoint.port_u16().is_none_or(|port| port == 0)
            || endpoint.path() != "/v1"
            || endpoint.query().is_some()
        {
            return Err("A/B requests require http://127.0.0.1:<port>/v1".into());
        }
        if self.schema_version != 1
            || self.round == 0
            || self.model.trim().is_empty()
            || self.output_tokens == 0
            || self.prompts.is_empty()
            || self.prompts.len() > 10000
            || !(0.0..=86400.0).contains(&self.request_timeout_secs)
            || self.request_timeout_secs == 0.0
            || !self.stagger_ms.is_finite()
            || self.stagger_ms < 0.0
            || self
                .prompts
                .iter()
                .any(|prompt| prompt.family.trim().is_empty() || prompt.prompt.trim().is_empty())
        {
            return Err("invalid A/B request identity, counts, prompts or timing".into());
        }
        let last = self.prompts.len().saturating_sub(1) as f64 * self.stagger_ms / 1000.0;
        if !last.is_finite() || last + self.request_timeout_secs > 86400.0 {
            return Err("A/B request phase exceeds the one-day deadline".into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum Outcome {
    Completed {
        submitted_ms: f64,
        first_token_ms: f64,
        completed_ms: f64,
        ttft_ms: f64,
        elapsed_ms: f64,
        tokens_predicted: u64,
        cached_tokens: u64,
        content_sha256: String,
    },
    Failed {
        error: String,
    },
}

#[derive(Serialize)]
struct Row {
    request_id: usize,
    family: String,
    prompt_sha256: String,
    #[serde(flatten)]
    outcome: Outcome,
}

#[derive(Serialize)]
pub(super) struct Phase {
    schema_version: u64,
    round: u64,
    version: Version,
    requests: Vec<Row>,
    makespan_ms: f64,
}

impl Phase {
    pub(super) fn successful(&self) -> usize {
        self.requests
            .iter()
            .filter(|row| matches!(row.outcome, Outcome::Completed { .. }))
            .count()
    }
    pub(super) fn passed(&self) -> bool {
        self.successful() == self.requests.len()
    }
}

async fn cancelled(cancellation: &Cancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

struct Endpoint {
    base: String,
    model: String,
    output_tokens: u64,
    timeout: Duration,
    stagger_ms: f64,
    epoch: Instant,
    cancellation: Cancellation,
}

async fn request(endpoint: &Endpoint, request_id: usize, prompt: Prompt) -> Row {
    let prompt_sha256 = hex::encode(Sha256::digest(prompt.prompt.as_bytes()));
    let body = serde_json::json!({"model":endpoint.model,
        "messages":[{"role":"user","content":prompt.prompt}],
        "max_tokens":endpoint.output_tokens,"temperature":0,"seed":0,"stream":true,
        "stream_options":{"include_usage":true}});
    let exchange = async {
        let target = endpoint.epoch
            + Duration::from_secs_f64(request_id as f64 * endpoint.stagger_ms / 1000.0);
        tokio::time::sleep_until(tokio::time::Instant::from_std(target)).await;
        let submitted_ms = endpoint.epoch.elapsed().as_secs_f64() * 1000.0;
        let evidence = tokio::time::timeout(
            endpoint.timeout,
            crate::automation::openai_exchange::request(&endpoint.base, &body, false),
        )
        .await
        .map_err(|_| "A/B request deadline expired".to_owned())?
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(Outcome::Completed {
            submitted_ms,
            first_token_ms: submitted_ms + evidence.ttft_seconds * 1000.0,
            completed_ms: endpoint.epoch.elapsed().as_secs_f64() * 1000.0,
            ttft_ms: evidence.ttft_seconds * 1000.0,
            elapsed_ms: evidence.elapsed_seconds * 1000.0,
            tokens_predicted: evidence.completion_tokens,
            cached_tokens: evidence.cached_tokens,
            content_sha256: evidence.content_sha256,
        })
    };
    let result = tokio::select! {
        biased;
        () = cancelled(&endpoint.cancellation) => Err("A/B request phase interrupted".into()),
        result = exchange => result,
    };
    Row {
        request_id,
        family: prompt.family,
        prompt_sha256,
        outcome: result.unwrap_or_else(|error| Outcome::Failed { error }),
    }
}

pub(super) async fn execute(input: Input, cancellation: Cancellation) -> DynResult<Phase> {
    let epoch = Instant::now();
    let endpoint = std::sync::Arc::new(Endpoint {
        base: input.base_url,
        model: input.model,
        output_tokens: input.output_tokens,
        timeout: Duration::from_secs_f64(input.request_timeout_secs),
        stagger_ms: input.stagger_ms,
        epoch,
        cancellation,
    });
    let mut pending = tokio::task::JoinSet::new();
    for (id, prompt) in input.prompts.into_iter().enumerate() {
        let endpoint = endpoint.clone();
        pending.spawn(async move { request(&endpoint, id, prompt).await });
    }
    let mut rows = Vec::new();
    while let Some(row) = pending.join_next().await {
        rows.push(row?);
    }
    rows.sort_by_key(|row| row.request_id);
    Ok(Phase {
        schema_version: 1,
        round: input.round,
        version: input.version,
        requests: rows,
        makespan_ms: epoch.elapsed().as_secs_f64() * 1000.0,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let input: Input = serde_json::from_slice(&std::fs::read(opts["--input"])?)?;
    input.validate()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = runtime.block_on(execute(input, interrupt.cancellation()))?;
    let failed = result
        .requests
        .iter()
        .any(|row| matches!(row.outcome, Outcome::Failed { .. }));
    let mut bytes = serde_json::to_vec_pretty(&result)?;
    bytes.push(b'\n');
    publish(Path::new(opts["--output"]), &bytes)?;
    interrupt.finish()?;
    if failed {
        Err("A/B request phase failed; request evidence retained".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "requests_tests.rs"]
mod tests;
