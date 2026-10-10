//! All concurrency rungs share one admitted retained host per arm.
use super::{
    child,
    contract::{Input, Profile},
    metrics, pipeline,
};
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
fn request(
    input: &Input,
    cell: &Value,
    template: &Value,
    mode: &str,
    rungs: &[Value],
    prompt: &str,
) -> DynResult<Value> {
    let first = rungs.first().ok_or("cache sweep empty ladder")?;
    let mut value = pipeline::serving_input(input, cell, template, mode, first, prompt)?;
    let base = value["worker"]["base_url"].clone();
    let mut stages = Vec::new();
    for rung in rungs {
        let stage = pipeline::serving_input(input, cell, template, mode, rung, prompt)?;
        let mut worker = stage["worker"].clone();
        worker["base_url"] = base.clone();
        stages.push(worker);
    }
    value["worker_sweep"] = json!(stages);
    Ok(value)
}
pub(super) fn execute(
    input: &Input,
    cell: &Value,
    profile: &Profile,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
    prompt: &str,
) -> DynResult<(Vec<Value>, bool)> {
    let tasks = &cell["tasks"];
    let rungs = tasks["concurrent_baseline"]["rungs"]
        .as_array()
        .ok_or("cache sweep ladder absent")?;
    let paired = tasks["paired_serving"]["planned"] == true;
    let native = if tasks["concurrent_baseline"]["planned"] == true {
        profile.native.as_ref()
    } else {
        None
    };
    let mut arms = Vec::new();
    let mut complete = true;
    for (mode, template) in [
        ("native-concurrent", native),
        (
            "skippy-old",
            if paired { profile.old.as_ref() } else { None },
        ),
        (
            "skippy-new",
            if paired { profile.new.as_ref() } else { None },
        ),
    ] {
        let Some(template) = template else {
            if paired && mode != "native-concurrent" {
                complete = false;
            }
            continue;
        };
        if cancel.is_cancelled() || Instant::now() >= until {
            complete = false;
            break;
        }
        let launched = child::run(
            "cache-family-cell",
            &request(input, cell, template, mode, rungs, prompt)?,
            &directory.join(format!("sweep-{mode}")),
            until,
            cancel,
        );
        let (receipt, clean) = launched.unwrap_or_else(|_| {
            (
                json!({"status":"refused","reason":"owned_child_admission_or_execution_refused"}),
                false,
            )
        });
        let owned = clean && receipt["status"] == "completed";
        let stages = receipt["measurement"]["sweep"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        complete &= owned && stages.len() == rungs.len();
        arms.push((mode, owned, stages));
    }
    let (rows, accepted) = project(input, rungs, paired, &arms)?;
    Ok((rows, complete && accepted))
}
fn project(
    input: &Input,
    rungs: &[Value],
    paired: bool,
    arms: &[(&str, bool, Vec<Value>)],
) -> DynResult<(Vec<Value>, bool)> {
    let mut complete = true;
    let mut observations = Vec::new();
    for (index, rung) in rungs.iter().enumerate() {
        let mut old = None;
        let mut new = None;
        for (mode, owned, stages) in arms {
            let Some(stage) = stages.get(index) else {
                pipeline::child_refused(&mut observations, mode, rung, false);
                complete = false;
                continue;
            };
            let measurement = &stage["measurement"];
            let correlated = stage["index"] == index && stage["concurrency"] == rung["concurrency"];
            let success = *owned && correlated && measurement["status"] == "completed";
            let metrics = metrics::goodput(
                measurement,
                input.plan["ttft_slo_ms"]
                    .as_u64()
                    .ok_or("TTFT SLO")?
                    .try_into()?,
                input.plan["tpot_slo_ms"]
                    .as_u64()
                    .ok_or("TPOT SLO")?
                    .try_into()?,
            )
            .ok();
            complete &= success && metrics.is_some();
            observations.push(json!({"cohort":mode,"concurrency":rung["concurrency"],"status":if success{"completed"}else{"incomplete"},"client_metrics":metrics,"evidence":format!("sweep-{mode}/output/cell.json"),"measurement_index":index,"persistent_host":true}));
            if *mode == "skippy-old" && success {
                old = Some(measurement);
            }
            if *mode == "skippy-new" && success {
                new = Some(measurement);
            }
        }
        if paired {
            let parity = match (old, new) {
                (Some(l), Some(r)) => metrics::accepted_parity(l, r, true, true).ok(),
                _ => Some(
                    json!({"matches":false,"complete":false,"unavailable_reason":"custody_or_lifecycle_failure"}),
                ),
            };
            complete &= parity.as_ref().is_some_and(|p| p["matches"] == true);
            observations.push(json!({"cohort":"old-new-parity","concurrency":rung["concurrency"],"parity":parity}));
        }
    }
    Ok((observations, complete))
}
