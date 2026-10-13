use super::contract::{Cohort, Input};
use crate::{automation::openai_exchange, process::Cancellation};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

async fn cancelled(cancellation: &Cancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn observed(input: &Input) -> Result<Value, &'static str> {
    match input.cohort {
        Cohort::NativeSerial | Cohort::NativeConcurrent => {
            let evidence = openai_exchange::native_completion::request(
                &input.base_url,
                &input.prompt,
                input.output_tokens,
                input.cohort == Cohort::NativeConcurrent,
            )
            .await
            .map_err(|_| "native HTTP/protocol failure")?;
            serde_json::to_value(evidence).map_err(|_| "native evidence serialization failure")
        }
        Cohort::OpenaiConcurrent => {
            let body = json!({"model":input.model_id,"messages":[{"role":"user","content":input.prompt}],
                "max_tokens":input.output_tokens,"temperature":0,"seed":0,"stream":true,
                "stream_options":{"include_usage":true}});
            let evidence = openai_exchange::cache_request(&input.base_url, &body)
                .await
                .map_err(|_| "OpenAI HTTP/protocol failure")?;
            if evidence.completion_tokens > input.output_tokens
                || evidence.finish_reason.is_none()
                || evidence.first_generated_sha256.is_none()
                || evidence.finish_reason.as_ref().is_some_and(|reason| {
                    reason.len() > 128 || reason.chars().any(char::is_control)
                })
            {
                return Err("OpenAI completion lacks matched final usage/output evidence");
            }
            let mut projection = serde_json::to_value(evidence)
                .map_err(|_| "OpenAI evidence serialization failure")?;
            // Cache throughput consumes client TTFT/TPOT, not replay inter-event samples.
            projection
                .as_object_mut()
                .ok_or("OpenAI evidence shape")?
                .remove("decode_inter_token_seconds");
            Ok(projection)
        }
    }
}

async fn request(input: Input, id: usize, deadline: Instant, cancellation: Cancellation) -> Value {
    let started = Instant::now();
    let cap = Duration::from_millis(input.request_timeout_ms)
        .min(deadline.saturating_duration_since(started));
    let result = if cancellation.is_cancelled() {
        Err("cancelled")
    } else if cap.is_zero() {
        Err("execution deadline")
    } else {
        tokio::select! {
            biased;
            () = cancelled(&cancellation) => Err("cancelled"),
            result = tokio::time::timeout(cap, observed(&input)) => result.unwrap_or(Err("request/execution deadline")),
        }
    };
    let excluded = input.cohort == Cohort::NativeSerial && id == 0;
    match result {
        Ok(evidence) => {
            let elapsed = evidence["elapsed_seconds"].as_f64().unwrap_or(0.0);
            let ttft = evidence["ttft_seconds"].as_f64();
            let tokens = evidence["tokens_predicted"]
                .as_u64()
                .or_else(|| evidence["completion_tokens"].as_u64())
                .unwrap_or(0);
            let intervals = openai_exchange::stream::number(tokens.saturating_sub(1).max(1));
            let tpot = ttft.map(|first| 1000.0 * (elapsed - first).max(0.0) / intervals);
            json!({"request_id":id,"status":"completed","excluded_warmup":excluded,
                "elapsed_ms":1000.0*elapsed,"ttft_ms":ttft.map(|value|1000.0*value),
                "tpot_ms":tpot,"tokens_predicted":tokens,"evidence":evidence})
        }
        Err(error) => json!({"request_id":id,"status":"failed","excluded_warmup":excluded,
            "elapsed_ms":1000.0*started.elapsed().as_secs_f64(),"error":error}),
    }
}

fn spawn(
    tasks: &mut JoinSet<Value>,
    input: &Input,
    id: usize,
    deadline: Instant,
    cancellation: &Cancellation,
) {
    tasks.spawn(request(input.clone(), id, deadline, cancellation.clone()));
}

fn summary(input: &Input, rows: &[Value], makespan: f64, complete: bool) -> Value {
    if !complete {
        return Value::Null;
    }
    let measured: Vec<_> = rows
        .iter()
        .filter(|row| row["excluded_warmup"] == false)
        .collect();
    if input.cohort == Cohort::NativeSerial {
        let mut elapsed: Vec<_> = measured
            .iter()
            .filter_map(|row| row["elapsed_ms"].as_f64())
            .collect();
        if elapsed.is_empty() {
            return json!({"warm_mean_ms":null,"warm_median_ms":null,"excluded_runs":1});
        }
        elapsed.sort_by(f64::total_cmp);
        let mean = elapsed.iter().sum::<f64>()
            / f64::from(u32::try_from(elapsed.len()).unwrap_or(u32::MAX));
        let middle = elapsed.len() / 2;
        let median = if elapsed.len().is_multiple_of(2) {
            (elapsed[middle - 1] + elapsed[middle]) / 2.0
        } else {
            elapsed[middle]
        };
        return json!({"warm_mean_ms":mean,"warm_median_ms":median,"excluded_runs":1});
    }
    let tokens = measured
        .iter()
        .filter_map(|row| row["tokens_predicted"].as_u64())
        .sum::<u64>();
    let count = f64::from(u32::try_from(measured.len()).unwrap_or(u32::MAX));
    json!({"throughput_requests_per_sec":count/makespan,"output_tokens_per_sec":openai_exchange::stream::number(tokens)/makespan,
        "observed_completion_tokens":tokens,"measured_requests":measured.len(),"makespan_seconds":makespan})
}

pub(super) async fn execute(
    input: &Input,
    request_sha256: String,
    cancellation: &Cancellation,
) -> Value {
    let started = Instant::now();
    let deadline = started + Duration::from_millis(input.execution_timeout_ms);
    let mut rows = Vec::new();
    let mut tasks = JoinSet::new();
    let mut next = 0;
    if !cancellation.is_cancelled() {
        for id in 0..input.concurrency {
            spawn(&mut tasks, input, id, deadline, cancellation);
            next += 1;
        }
    }
    let mut interrupted = None;
    while !tasks.is_empty() {
        let result = tokio::select! {
            biased;
            () = cancelled(cancellation) => { interrupted = Some("cancelled"); break; },
            () = tokio::time::sleep(deadline.saturating_duration_since(Instant::now())) => { interrupted = Some("execution deadline"); break; },
            row = tasks.join_next() => row,
        };
        match result {
            Some(Ok(row)) => rows.push(row),
            Some(Err(_)) => {
                interrupted = Some("owned request task failure");
                break;
            }
            None => break,
        }
        if next < input.requests {
            spawn(&mut tasks, input, next, deadline, cancellation);
            next += 1;
        }
    }
    let makespan = started.elapsed().as_secs_f64();
    // Drain already completed tasks before aborting pending IO. Aborted tasks have no
    // final token observations; their rows are honestly incomplete, never zero samples.
    while let Some(result) = tasks.try_join_next() {
        if let Ok(row) = result {
            rows.push(row);
        }
    }
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    rows.sort_by_key(|row| row["request_id"].as_u64().unwrap_or(u64::MAX));
    let mut by_id = rows
        .into_iter()
        .map(|row| (row["request_id"].as_u64().unwrap_or(u64::MAX), row))
        .collect::<std::collections::BTreeMap<_, _>>();
    let rows: Vec<_> = (0..input.requests).map(|id| by_id.remove(&(id as u64)).unwrap_or_else(|| json!({
        "request_id":id,"status":if id < next {"incomplete"} else {"not-launched"},
        "excluded_warmup":input.cohort == Cohort::NativeSerial && id == 0,
        "error":interrupted.unwrap_or(if cancellation.is_cancelled(){"cancelled"}else{"missing owned task receipt"})}))).collect();
    let complete = interrupted.is_none()
        && !cancellation.is_cancelled()
        && Instant::now() < deadline
        && rows.iter().all(|row| row["status"] == "completed");
    let summary = summary(input, &rows, makespan, complete);
    json!({"schema_version":1,"request_sha256":request_sha256,"cohort":input.cohort,
        "status":if complete {"completed"} else {"incomplete"},"endpoint_custody":"parent-must-bind-owned-host",
        "prompt_sha256":hex::encode(sha2::Sha256::digest(input.prompt.as_bytes())),
        "concurrency":input.concurrency,"requested_requests":input.requests,"output_tokens":input.output_tokens,
        "makespan_seconds":makespan,"rows":rows,"summary":summary})
}
use sha2::Digest as _;
