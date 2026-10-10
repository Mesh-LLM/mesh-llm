use super::{
    child,
    contract::{Input, Profile},
};
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn port() -> DynResult<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port())
}
fn pinned(cell: &Value, input: &Input, profile: &Profile) -> DynResult<()> {
    let case = &cell["case"];
    let model = if cell["model_observation"]["runtime_entrypoint"].is_null() {
        &cell["model_observation"]["canonical"]
    } else {
        &cell["model_observation"]["runtime_entrypoint"]
    };
    let key = case["key"].as_str().ok_or("cache case key")?;
    let expected = &input.plan["model_sha256"][key];
    if expected.as_str().is_none()
        || profile.correctness["case_key"] != key
        || profile.correctness["model_sha256"] != *expected
        || profile.correctness["model"] != *model
    {
        return Err(
            "cache supplied correctness profile does not bind planned model/key/pin".into(),
        );
    }
    for (side, template) in [("old_server", &profile.old), ("new_server", &profile.new)] {
        if let Some(t) = template
            && (t["binary"] != input.plan[side]["path"]
                || t["binary_sha256"] != input.plan[side]["sha256"]
                || t["model"] != *model
                || t["model_sha256"] != *expected
                || t["model_id"] != case["model_id"]
                || t["artifact"] != profile.correctness["artifact"])
        {
            return Err("cache supplied serving profile does not bind plan binary/model".into());
        }
    }
    if let (Some(old), Some(new)) = (&profile.old, &profile.new)
        && (old["environment"] != new["environment"]
            || old["toolkit_directories"] != new["toolkit_directories"])
    {
        return Err("cache paired inherited tuning profiles differ".into());
    }
    for template in [&profile.native, &profile.old, &profile.new]
        .into_iter()
        .flatten()
    {
        let tuning = profile
            .correctness
            .get("settings")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let toolkit = profile
            .correctness
            .get("toolkit_directories")
            .cloned()
            .unwrap_or_else(|| json!({}));
        if template
            .get("environment")
            .cloned()
            .unwrap_or_else(|| json!({}))
            != tuning
            || template
                .get("toolkit_directories")
                .cloned()
                .unwrap_or_else(|| json!({}))
                != toolkit
        {
            return Err("cache correctness and serving effective tuning profiles differ".into());
        }
    }
    if let Some(native) = &profile.native
        && (native["model"] != *model
            || native["model_sha256"] != *expected
            || native["artifact"] != profile.correctness["artifact"])
    {
        return Err("cache native baseline model/pin differs".into());
    }
    Ok(())
}
fn correctness_input(input: &Input, cell: &Value, profile: &Profile) -> DynResult<Value> {
    let mut value = profile.correctness.clone();
    let case = &cell["case"];
    for field in [
        "model_id",
        "ctx_size",
        "prefix_tokens",
        "cache_hit_repeats",
        "n_gpu_layers",
    ] {
        value[field] = case[field].clone();
    }
    value["topologies"] = if case["key"] == "deepseek3" {
        json!(["package-stage1"])
    } else {
        json!(["one-stage", "split-stage0", "split-middle", "split-final"])
    };
    value["prompt"] = cell["use_case"]["prompt"].clone();
    value["execution_seconds"] = json!(input.cell_seconds);
    value["cell_seconds"] = json!(input.cell_seconds.saturating_sub(9).max(1));
    let reservation = std::net::TcpListener::bind("127.0.0.1:0")?;
    let source = reservation.local_addr()?.port();
    let restore = port()?;
    drop(reservation);
    value["source_port"] = json!(source);
    value["restore_port"] = json!(restore);
    Ok(value)
}
fn prompt(receipt: &Value, expected: usize) -> DynResult<&str> {
    if receipt["status"] != "completed" {
        return Err("cache correctness stage incomplete".into());
    }
    let rows = receipt["rows"]
        .as_array()
        .ok_or("cache correctness roster absent")?;
    if rows.len() != expected || receipt["planned_topologies"] != expected {
        return Err("cache correctness topology roster incomplete".into());
    }
    let first = rows[0]["evidence"]["skippy"]["benchmark_prompt_text"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("accepted cache benchmark prompt absent")?;
    if rows.iter().any(|r| {
        r["evidence"]["status"] != "pass"
            || r["evidence"]["skippy"]["benchmark_prompt_text"] != first
    }) {
        return Err("cache correctness reports disagree about benchmark prompt".into());
    }
    Ok(first)
}
pub(super) fn serving_input(
    input: &Input,
    cell: &Value,
    template: &Value,
    mode: &str,
    rung: &Value,
    prompt: &str,
) -> DynResult<Value> {
    let mut value = template.clone();
    let case = &cell["case"];
    let tasks = &cell["tasks"];
    let serial = mode == "native-serial";
    let native = mode.starts_with("native");
    value["host"] = json!(if native { "native-baseline" } else { mode });
    for field in ["model_id", "layer_end", "n_gpu_layers"] {
        value[field] = case[field].clone();
    }
    value["ctx_size"] = if native {
        case["ctx_size"].clone()
    } else {
        tasks["paired_serving"]["shared_ctx_size"].clone()
    };
    value["lane_count"] = if native {
        input.plan["llama_parallel"].clone()
    } else {
        tasks["paired_serving"]["runtime_lanes"].clone()
    };
    let p = port()?;
    value["port"] = json!(p);
    value["execution_timeout_secs"] = json!(input.cell_seconds);
    let cohort = if serial {
        "native-serial"
    } else if native {
        "native-concurrent"
    } else {
        "openai-concurrent"
    };
    value["worker"] = json!({"schema_version":1,"cohort":cohort,"base_url":format!("http://127.0.0.1:{p}{}",if native{"/"}else{"/v1"}),"model_id":if native{Value::Null}else{case["model_id"].clone()},"prompt":prompt,"requests":if serial{input.plan["llama_repeats"].clone()}else{rung["requests"].clone()},"concurrency":if serial{json!(1)}else{rung["concurrency"].clone()},"output_tokens":if serial{json!(1)}else if native{json!(128)}else{input.plan["concurrent_output_tokens"].clone()},"request_timeout_ms":input.request_timeout_ms,"execution_timeout_ms":input.cell_seconds*1000});
    Ok(value)
}
fn observe(
    input: &Input,
    cell: &Value,
    profile: &Profile,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    pinned(cell, input, profile)?;
    let (correctness, clean) = child::run(
        "cache-family-correctness",
        &correctness_input(input, cell, profile)?,
        &directory.join("correctness"),
        until,
        cancel,
    )?;
    if !clean {
        return Ok(
            json!({"status":"incomplete","reason":"correctness_process_refused","correctness":"correctness/output/cache-correctness-stage.json"}),
        );
    }
    let package = cell["case"]["key"] == "deepseek3";
    let prompt = prompt(&correctness, if package { 1 } else { 4 })?;
    let tasks = &cell["tasks"];
    let mut observations = Vec::new();
    let mut complete = true;
    if tasks["serial_baseline"]["planned"] == true {
        let native = profile
            .native
            .as_ref()
            .ok_or("planned native baseline profile missing")?;
        let request = serving_input(input, cell, native, "native-serial", &Value::Null, prompt)?;
        let (receipt, clean) = match child::run(
            "cache-family-cell",
            &request,
            &directory.join("native-serial"),
            until,
            cancel,
        ) {
            Ok(value) => value,
            Err(_) => (
                json!({"status":if directory.join("native-serial").exists(){"refused"}else{"not-launched"},"reason":"owned_child_admission_or_execution_refused","launch_observation":if directory.join("native-serial").exists(){"unknown"}else{"not-launched"}}),
                false,
            ),
        };
        complete &= clean && receipt["status"] == "completed";
        observations.push(json!({"cohort":"native-serial","status":receipt["status"],"warm_statistics":receipt["measurement"]["summary"],"evidence":"native-serial/output/cell.json"}));
    } else {
        observations.push(json!({"cohort":"native-serial","status":if package{"unavailable"}else{"skipped"},"reason":tasks["serial_baseline"]["skip_reason"]}));
    }
    let (mut curve, curve_complete) =
        super::sweeps::execute(input, cell, profile, directory, until, cancel, prompt)?;
    observations.append(&mut curve);
    complete &= curve_complete;
    let report_row = super::reporting::row(cell, Some(&correctness), &observations, complete);
    Ok(
        json!({"status":if complete{"completed"}else{"incomplete"},"correctness":"correctness/output/cache-correctness-stage.json","observations":observations,"producer_report_row":report_row,"promotion":null}),
    )
}
pub(super) fn execute(input: &Input, output: &Path, cancel: &Cancellation) -> DynResult<Value> {
    let until = Instant::now() + Duration::from_secs(input.execution_seconds);
    let (plan, clean) = child::run(
        "cache-family-plan",
        &input.plan,
        &output.join("plan"),
        until,
        cancel,
    )?;
    if !clean
        || !input.plan.as_object().is_some_and(|provided| {
            provided
                .iter()
                .all(|(key, value)| plan["input"].get(key) == Some(value))
        })
    {
        return Err("cache source-owned plan refused/correlation mismatch".into());
    }
    let cells = plan["cells"].as_array().ok_or("cache plan cells absent")?;
    let mut rows = Vec::new();
    let mut report_rows = Vec::new();
    let mut complete = true;
    for (index, cell) in cells.iter().enumerate() {
        let mut report_row = super::reporting::row(cell, None, &[], false);
        let status = if cell["model_observation"]["status"] == "missing-model" {
            json!({"status":"missing-model","launched":false})
        } else if ![
            "single-gguf",
            "split-gguf-first-shard",
            "layer-package-tree",
        ]
        .iter()
        .any(|k| cell["model_observation"]["kind"] == *k)
        {
            complete = false;
            json!({"status":"unsupported-artifact-adapter","launched":false})
        } else if cancel.is_cancelled() || Instant::now() >= until {
            complete = false;
            json!({"status":"not-launched","reason":"cancellation_or_deadline"})
        } else {
            let directory = output.join(format!("cell-{index:04}"));
            std::fs::create_dir(&directory)?;
            let key = cell["key"].as_str().ok_or("cache planned key")?;
            let result = match input.profiles.get(key) {
                Some(profile) => observe(input, cell, profile, &directory, until, cancel),
                None => Err("cache supplied profile missing".into()),
            };
            let value=result.unwrap_or_else(|_|json!({"status":"refused","reason":"profile_identity_correctness_or_owned_execution_failed"}));
            complete &= value["status"] == "completed";
            if !value["producer_report_row"].is_null() {
                report_row = value["producer_report_row"].clone();
            }
            super::publish(&directory.join("row.json"), &value)?;
            json!({"status":value["status"],"evidence":format!("cell-{index:04}/row.json")})
        };
        report_rows.push(report_row);
        rows.push(json!({"case":cell["key"],"prefix_tokens":cell["case"]["prefix_tokens"],"use_case":cell["use_case"]["key"],"result":status}));
    }
    let report = super::reporting::publish_report(&report_rows, cells, output);
    complete &= report.is_ok();
    Ok(
        json!({"schema_version":1,"status":if complete&&!cancel.is_cancelled()&&Instant::now()<until{"completed"}else{"incomplete"},"scope":"supplied_pinned_GGUF_shard_or_package_cache_producer_no_build_download_or_promotion","report_error":report.err().map(|_|"typed_native_report_refused"),"report":"production-cache-bench.md","report_input":"production-cache-bench.json","catalog_sha256":plan["catalog_sha256"],"planned_cells":cells.len(),"rows":rows,"promotion":null}),
    )
}
/// Preserve prior observations when the next owned child cannot be admitted.
pub(super) fn child_refused(
    observations: &mut Vec<Value>,
    mode: &str,
    rung: &Value,
    not_launched: bool,
) {
    observations.push(json!({"cohort":mode,"concurrency":rung["concurrency"],"status":if not_launched{"not-launched"}else{"not-run"},"launch_observation":if not_launched{"not-launched"}else{"unknown"},"reason":"owned_child_admission_or_execution_refused","client_metrics":null}));
}
