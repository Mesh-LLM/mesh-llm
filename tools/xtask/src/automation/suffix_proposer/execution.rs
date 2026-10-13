use super::{
    contract::{Arm, Input, Workload},
    sample::Sample,
};
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::{
    io::Write,
    path::Path,
    time::{Duration, Instant},
};
async fn cancelled(c: &Cancellation) {
    while !c.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await
    }
}
async fn exchange(
    input: &Input,
    arm: &Arm,
    path: &str,
    body: Option<&Value>,
    deadline: Instant,
    c: &Cancellation,
) -> DynResult<Value> {
    if c.is_cancelled() {
        return Err("suffix cancelled".into());
    }
    let cap = deadline
        .saturating_duration_since(Instant::now())
        .min(Duration::from_millis(input.request_timeout_ms));
    if cap.is_zero() {
        return Err("suffix deadline".into());
    }
    let limit = Instant::now() + cap;
    let request = crate::automation::guardrail_corpus::transport::exchange_with_authorization(
        &arm.base_url,
        path,
        body,
        limit,
        c,
        None,
    );
    let response = tokio::select! {biased;()=cancelled(c)=>return Err("suffix cancelled".into()),result=tokio::time::timeout(cap,request)=>result.map_err(|_|"suffix request/overall deadline")??};
    if c.is_cancelled() || Instant::now() >= deadline || response.status != 200 {
        return Err("suffix HTTP/status/cancellation/deadline refusal".into());
    }
    Ok(serde_json::from_slice(&response.body)?)
}
pub(super) fn models_identity(value: &Value, model: &str) -> DynResult<String> {
    let rows = value["data"].as_array().ok_or("invalid model listing")?;
    if rows.len() > 1024 {
        return Err("too many advertised models".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    for row in rows {
        let id = row["id"].as_str().ok_or("invalid advertised model id")?;
        if id.is_empty() || id.len() > 4096 || id.chars().any(char::is_control) || !ids.insert(id) {
            return Err("invalid duplicate/bounded model id".into());
        }
    }
    if !ids.contains(model) {
        return Err("declared suffix model not advertised".into());
    }
    Ok(super::evidence::digest(&serde_json::to_vec(&ids)?))
}
async fn identity(input: &Input, deadline: Instant, c: &Cancellation) -> DynResult<Value> {
    let mut identities = Vec::new();
    for arm in &input.arms {
        let v = exchange(input, arm, "v1/models", None, deadline, c).await?;
        let observed = models_identity(&v, &input.model)?;
        identities.push(json!({"arm":arm.name,"advertised_model_ids_sha256":observed,"declared_stages":arm.declared_stages,"declared_mtp_capable":arm.declared_mtp_capable}));
    }
    Ok(json!(identities))
}
fn shuffle(indices: &mut [usize], state: &mut u64) {
    for i in (1..indices.len()).rev() {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = *state;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        indices.swap(i, (x % ((i + 1) as u64)) as usize);
    }
}
async fn once(
    input: &Input,
    arm: &Arm,
    w: &Workload,
    run: u32,
    deadline: Instant,
    c: &Cancellation,
) -> DynResult<Sample> {
    let body = json!({"model":input.model,"messages":[{"role":"user","content":w.prompt}],"max_tokens":input.max_tokens,"temperature":0.0});
    let started = Instant::now();
    let v = exchange(input, arm, "v1/chat/completions", Some(&body), deadline, c).await?;
    super::sample::decode(&v, arm, w, run, input, started.elapsed().as_secs_f64())
}
pub(super) async fn execute(
    input: &Input,
    workloads: &[Workload],
    root: &Path,
    deadline: Instant,
    c: &Cancellation,
    report: &mut Value,
) -> DynResult<()> {
    let before = identity(input, deadline, c).await?;
    report["identity_before"] = before.clone();
    let mut state = input.seed;
    let mut arms: Vec<usize> = (0..input.arms.len()).collect();
    let mut samples = Vec::new();
    let mut results = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("results.jsonl"))?;
    for round in 0..input.warmups + input.runs {
        shuffle(&mut arms, &mut state);
        for w in workloads {
            for &a in &arms {
                let measured = round >= input.warmups;
                let run = round.saturating_sub(input.warmups);
                report["failed_cell"] = json!({"arm":input.arms[a].name,"workload":w.name,"run":run,"warmup":!measured});
                let row = once(input, &input.arms[a], w, run, deadline, c).await?;
                report["failed_cell"] = Value::Null;
                if measured {
                    let bytes = serde_json::to_vec(&row)?;
                    results.write_all(&bytes)?;
                    results.write_all(b"\n")?;
                    results.flush()?;
                    samples.push(row);
                    report["samples"] = serde_json::to_value(&samples)?;
                } else {
                    report["completed_warmups"] =
                        json!(report["completed_warmups"].as_u64().unwrap_or(0) + 1);
                }
            }
        }
    }
    let after = identity(input, deadline, c).await?;
    report["identity_after"] = after.clone();
    if before != after {
        return Err("suffix advertised identity drift".into());
    }
    let summary = super::summary::summarize(&samples, &input.baseline_arm)?;
    super::evidence::publish(&root.join("summary.json"), &summary)?;
    std::fs::write(root.join("summary.md"), super::summary::markdown(&summary))?;
    report["summary"] = summary;
    super::summary::activation(&samples, input.require_drafts_arm.as_deref())?;
    Ok(())
}
