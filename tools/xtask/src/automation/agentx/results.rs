use super::{Prepared, Profile, write};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::path::Path;

fn metric(export: &Value, name: &str) -> DynResult<f64> {
    export[name]["avg"]
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("missing finite AgentX metric: {name}").into())
}
fn count(export: &Value, name: &str) -> DynResult<u64> {
    let value = metric(export, name)?;
    if value < 0.0 || value.fract() != 0.0 || value > (1_u64 << 53) as f64 {
        return Err(format!("invalid AgentX count: {name}").into());
    }
    Ok(value as u64)
}

fn distribution(export: &Value, name: &str) -> DynResult<Value> {
    if export[name]["unit"] != "ms" || metric(export, name)? < 0.0 {
        return Err(format!("expected nonnegative milliseconds: {name}").into());
    }
    let mut percentiles = serde_json::Map::new();
    for (key, value) in export[name].as_object().ok_or("metric is not an object")? {
        if let Some(percentile) = key.strip_prefix('p')
            && percentile
                .parse::<f64>()
                .is_ok_and(|p| (0.0..=100.0).contains(&p))
        {
            if !value.as_f64().is_some_and(|v| v.is_finite() && v >= 0.0) {
                return Err(format!("invalid percentile: {name}.{key}").into());
            }
            percentiles.insert(percentile.into(), value.clone());
        }
    }
    Ok(json!({"mean":metric(export,name)?,"p":percentiles}))
}

pub(super) fn point(input: &Prepared, concurrency: u32) -> DynResult<Value> {
    let export: Value = serde_json::from_slice(&std::fs::read(
        input.output.join("upstream/profile_export_aiperf.json"),
    )?)?;
    validate(&export, input.profile)?;
    metric_point(&export, concurrency)
}

fn metric_point(export: &Value, concurrency: u32) -> DynResult<Value> {
    let duration = metric(export, "benchmark_duration")?;
    let throughput = metric(export, "output_token_throughput")?;
    if export["benchmark_duration"]["unit"] != "sec"
        || export["output_token_throughput"]["unit"] != "tokens/sec"
        || duration <= 0.0
        || throughput <= 0.0
    {
        return Err("invalid AgentX duration or throughput".into());
    }
    Ok(json!({
        "concurrency":concurrency,
        "requests_completed":count(export,"request_count")?,"requests_failed":0,
        "duration_s":duration,
        "throughput":{"output_tok_s":throughput},
        "tokens":{"input":count(export,"total_usage_prompt_tokens")?,
                  "output":count(export,"total_output_tokens")?},
        "ttft_ms":distribution(export,"time_to_first_token")?,
        "tpot_ms":distribution(export,"inter_token_latency")?,
        "e2e_ms":distribution(export,"request_latency")?
    }))
}
pub(super) fn validate(export: &Value, profile: Profile) -> DynResult<()> {
    if export["was_cancelled"] == true || export["metadata"]["submission_valid"] != false {
        return Err("expected uncancelled, explicitly non-submission local-corpus export".into());
    }
    if metric(export, "request_count")? < 1.0
        || metric(export, "output_token_count")? < 1.0
        || metric(export, "usage_prompt_tokens")? < 1.0
        || metric(export, "usage_completion_tokens")? < 1.0
        || metric(export, "time_to_first_token")? < 0.0
        || metric(export, "inter_token_latency")? < 0.0
    {
        return Err(
            "AgentX requires completed measured requests, token usage and streaming metrics".into(),
        );
    }
    let errors = export["error_request_count"]["avg"].as_f64().unwrap_or(0.0);
    let overflow = export["context_overflow_count"]["avg"]
        .as_f64()
        .unwrap_or(0.0);
    if errors != 0.0 || overflow != 0.0 {
        return Err("AgentX cell contains failed or context-overflow requests".into());
    }
    if profile == Profile::Smoke && export["warmup_metrics"].is_null() {
        return Err("smoke warmup evidence missing".into());
    }
    Ok(())
}
pub(super) fn read(input: &Prepared, engine: &str, concurrency: u32) -> DynResult<Value> {
    let path = input.output.join("upstream/profile_export_aiperf.json");
    let export: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    validate(&export, input.profile)?;
    let memory: Value =
        serde_json::from_slice(&std::fs::read(input.output.join("memory-summary.json"))?)?;
    let summary = json!({
        "name":format!("agentx/{}/{}/{}/c{}",input.model.key,engine,input.profile.label(),concurrency),
        "engine":engine,"offered_sessions":concurrency,"generation_slots":1,
        "submission_valid":false,"request_count":metric(&export,"request_count")?,
        "output_tok_s":metric(&export,"output_token_throughput")?,
        "ttft":export["time_to_first_token"],"itl":export["inter_token_latency"],"e2e":export["request_latency"],
        "errors":export["error_request_count"],"context_overflows":export["context_overflow_count"],
        "usage_prompt_tokens":export["usage_prompt_tokens"],"usage_completion_tokens":export["usage_completion_tokens"],
        "cache_read_tokens":export["usage_prompt_cache_read_tokens"],
        "memory":memory,
        "upstream_metadata":export["metadata"],
        "export_sha256":crate::product::digest::file_sha256(&path).map_err(|failure|failure.error)?
    });
    write(&input.output.join("summary.json"), &summary)?;
    Ok(summary)
}
pub(super) fn archive(output: &Path) -> DynResult<()> {
    if let Some(directory) = std::env::var_os("JIANYANG_ARTIFACT_DIR") {
        let path = std::path::PathBuf::from(directory).join("agentx.tar.gz");
        let encoder = flate2::write::GzEncoder::new(
            std::fs::File::create(path)?,
            flate2::Compression::fast(),
        );
        let mut archive = tar::Builder::new(encoder);
        archive.append_dir_all("agentx", output)?;
        archive.into_inner()?.finish()?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projects_real_export_without_averaging_token_totals_or_rescaling_ms() {
        let export: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/agentx/smoke-metrics.json"
        ))
        .unwrap();
        validate(&export, Profile::Smoke).unwrap();
        let point = metric_point(&export, 1).unwrap();
        assert_eq!(point["requests_completed"], 4);
        assert_eq!(point["tokens"]["input"], 263976);
        assert_eq!(point["tokens"]["output"], 1236);
        assert_eq!(point["ttft_ms"]["p"]["50"], 873.2040969999999);
        assert_eq!(point["throughput"]["output_tok_s"], 20.599353189236243);
        for (name, value) in [
            ("total_output_tokens", json!({"avg":1.5})),
            ("total_usage_prompt_tokens", Value::Null),
            ("time_to_first_token", json!({"avg":1,"unit":"sec"})),
            ("benchmark_duration", json!({"avg":-1,"unit":"sec"})),
        ] {
            let mut invalid = export.clone();
            invalid[name] = value;
            assert!(metric_point(&invalid, 1).is_err(), "{name}");
        }
    }
    fn export() -> Value {
        json!({"metadata":{"submission_valid":false},"request_count":{"avg":2},"output_token_count":{"avg":8},"usage_prompt_tokens":{"avg":32},"usage_completion_tokens":{"avg":4},"time_to_first_token":{"avg":2},"inter_token_latency":{"avg":1},"warmup_metrics":{}})
    }
    #[test]
    fn rejects_empty_failed_and_cancelled_runs() {
        assert!(validate(&export(), Profile::Smoke).is_ok());
        for (name, value) in [
            ("request_count", json!({"avg":0})),
            ("error_request_count", json!({"avg":1})),
            ("context_overflow_count", json!({"avg":1})),
            ("was_cancelled", json!(true)),
            ("inter_token_latency", Value::Null),
            ("usage_completion_tokens", Value::Null),
        ] {
            let mut failed = export();
            failed[name] = value;
            assert!(validate(&failed, Profile::Smoke).is_err(), "{name}");
        }
    }
}
