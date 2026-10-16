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
    let summary = json!({
        "name":format!("agentx/{}/{}/{}/c{}",input.model.key,engine,input.profile.label(),concurrency),
        "engine":engine,"offered_sessions":concurrency,"generation_slots":1,
        "submission_valid":false,"request_count":metric(&export,"request_count")?,
        "output_tok_s":metric(&export,"output_token_throughput")?,
        "ttft":export["time_to_first_token"],"itl":export["inter_token_latency"],"e2e":export["request_latency"],
        "errors":export["error_request_count"],"context_overflows":export["context_overflow_count"],
        "usage_prompt_tokens":export["usage_prompt_tokens"],"usage_completion_tokens":export["usage_completion_tokens"],
        "cache_read_tokens":export["usage_prompt_cache_read_tokens"],
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
