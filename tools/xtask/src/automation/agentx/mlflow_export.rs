//! Publication plans for the existing AgentX experiment; never create an experiment.
use super::{final_sweep, prepared_tools::File};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

pub(super) const EXPERIMENT: &str = "6";
pub(super) const EXPERIMENT_NAME: &str = "agentx-skippy-vs-llama";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub report: File,
    pub archive: File,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    pub schema_version: u32,
    pub experiment_id: String,
    pub expected_experiment_name: String,
    pub campaign_root: CampaignRoot,
    pub run_key: String,
    pub name: String,
    pub tags: BTreeMap<String, String>,
    pub children: Vec<Child>,
    pub report: File,
    pub archive: File,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CampaignRoot {
    pub run_key: String,
    pub name: String,
    pub tags: BTreeMap<String, String>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Child {
    pub run_key: String,
    pub name: String,
    pub tags: BTreeMap<String, String>,
    pub metrics: BTreeMap<String, f64>,
}
pub(super) fn run(input: Input, output: &Path) -> DynResult<()> {
    input.report.verify()?;
    input.archive.verify()?;
    let report: Value = serde_json::from_slice(&crate::automation::receipt_files::bounded(
        &input.report.path,
        4 * 1024 * 1024,
    )?)?;
    let plan = build(&report, input)?;
    crate::automation::receipt_files::fresh(output, &serde_json::to_vec_pretty(&plan)?)
}
fn build(report: &Value, input: Input) -> DynResult<Plan> {
    if report["status"] != "passed" || report["scope"] != "fixed_complete_AgentX_comparison" {
        return Err("MLflow performance publication requires a complete non-CPU campaign".into());
    }
    let arms = report["arms"].as_array().ok_or("arms missing")?;
    final_sweep::validate_work(arms)?;
    let campaign = report["campaign"].as_str().ok_or("campaign missing")?;
    let model = report["model_key"].as_str().ok_or("model key missing")?;
    let backend = report["backend"].as_str().ok_or("backend missing")?;
    let source = report["source_attribution"]["product"]
        .as_str()
        .ok_or("source missing")?;
    let mut tags = BTreeMap::from([
        ("ax.campaign".into(), campaign.into()),
        ("ax.model_key".into(), model.into()),
        ("ax.backend".into(), backend.into()),
        ("ax.source_commit".into(), source.into()),
        ("ax.audit_state".into(), "verified_work_accounting".into()),
        ("ax.report_sha256".into(), input.report.sha256.clone()),
        ("ax.archive_sha256".into(), input.archive.sha256.clone()),
    ]);
    let catalog: Vec<super::Model> = serde_json::from_str(include_str!(
        "../../../../../skippy/evals/kv-validation/models.json"
    ))?;
    let selected = catalog
        .iter()
        .find(|row| row.key == model)
        .ok_or("unknown publication model")?;
    tags.insert("ax.model_ref".into(), selected.registry_ref());
    tags.insert("ax.model_sha256".into(), selected.files[0].sha256.clone());
    tags.insert("ax.category".into(), selected.family.clone());
    tags.insert("ax.stage".into(), "final-comparison".into());
    for (key, pointer) in [
        (
            "ax.native_source_commit",
            "/request/tools/native_source_commit",
        ),
        (
            "ax.native_patch_digest",
            "/request/tools/native_patch_digest",
        ),
        ("ax.runtime_sha256", "/request/tools/runtime/sha256"),
        ("ax.skippy_sha256", "/request/tools/skippy/sha256"),
        ("ax.llama_sha256", "/request/tools/llama/sha256"),
        (
            "ax.standard_runtime_sha256",
            "/request/tools/standard_runtime/sha256",
        ),
        ("ax.client_sha256", "/request/tools/client/sha256"),
        (
            "ax.client_environment_sha256",
            "/request/tools/client_environment/sha256",
        ),
        ("ax.corpus_sha256", "/request/corpus/sha256"),
        (
            "ax.automation_sha256",
            "/source_attribution/automation_binary_sha256",
        ),
    ] {
        tags.insert(
            key.into(),
            report
                .pointer(pointer)
                .and_then(Value::as_str)
                .ok_or("publication provenance missing")?
                .into(),
        );
    }
    tags.insert("ax.cohort_sha256".into(), final_sweep::COHORT_SHA.into());
    tags.insert(
        "ax.work_sha256".into(),
        arms[0]["cells"][0]["completed_work"]["source_turn_output_sha256"]
            .as_str()
            .ok_or("work identity missing")?
            .into(),
    );
    for (field, key) in [("jy_job_id", "ax.job_id"), ("jy_run_id", "ax.run_id")] {
        if let Some(value) = report[field].as_str() {
            tags.insert(key.into(), value.into());
        }
    }
    let key = format!("{campaign}:{source}:{backend}:{model}");
    let mut children = Vec::new();
    for (index, arm) in arms.iter().enumerate() {
        let engine = arm["engine"].as_str().ok_or("engine missing")?;
        let mut metrics = BTreeMap::new();
        for cell in arm["cells"].as_array().ok_or("cells missing")? {
            let concurrency = cell["offered_sessions"]
                .as_u64()
                .ok_or("concurrency missing")?;
            let prefix = format!("c{concurrency}");
            for (name, pointer) in [
                ("output_tok_s", "/output_tok_s"),
                ("ttft_p50_ms", "/ttft/p50"),
                ("ttft_p95_ms", "/ttft/p95"),
                ("itl_p50_ms", "/itl/p50"),
                ("cache_read_tokens_avg", "/cache_read_tokens/avg"),
                (
                    "peak_process_rss_bytes",
                    "/memory/sampled_peak_process_rss_bytes",
                ),
                (
                    "peak_process_footprint_bytes",
                    "/memory/sampled_peak_process_physical_footprint_bytes",
                ),
                (
                    "peak_cuda_process_accounted_bytes",
                    "/memory/sampled_peak_cuda_process_accounted_memory_bytes",
                ),
            ] {
                if let Some(value) = cell.pointer(pointer).and_then(Value::as_f64) {
                    if !value.is_finite() || value < 0.0 {
                        return Err("nonfinite/negative publication metric".into());
                    }
                    metrics.insert(format!("{prefix}.{name}"), value);
                }
            }
            metrics.insert(format!("{prefix}.requests"), 69.0);
            metrics.insert(format!("{prefix}.output_tokens"), 18713.0);
        }
        let mut child_tags = tags.clone();
        child_tags.insert("ax.engine".into(), engine.into());
        child_tags.insert("ax.arm".into(), (index + 1).to_string());
        children.push(Child {
            run_key: format!("{key}:arm{}", index + 1),
            name: format!("{backend} {model} arm{} {engine}", index + 1),
            tags: child_tags,
            metrics,
        });
    }
    Ok(Plan {
        schema_version: 1,
        experiment_id: EXPERIMENT.into(),
        expected_experiment_name: EXPERIMENT_NAME.into(),
        campaign_root: CampaignRoot {
            run_key: format!("{campaign}:{source}:root"),
            name: campaign.into(),
            tags: BTreeMap::from([
                ("ax.campaign".into(), campaign.into()),
                ("ax.source_commit".into(), source.into()),
                ("ax.stage".into(), "four-model-two-backend-campaign".into()),
                (
                    "ax.completion_policy".into(),
                    "pending_external_eight_group_audit".into(),
                ),
            ]),
        },
        run_key: format!("{key}:parent"),
        name: format!("{campaign} {backend} {model}"),
        tags,
        children,
        report: input.report,
        archive: input.archive,
    })
}
pub(super) fn validate_experiment(value: &Value) -> DynResult<()> {
    let experiment = &value["experiment"];
    if experiment["experiment_id"] != EXPERIMENT
        || experiment["name"] != EXPERIMENT_NAME
        || experiment["lifecycle_stage"] != "active"
    {
        return Err(
            "existing AgentX experiment6/name/active state required; creation is forbidden".into(),
        );
    }
    Ok(())
}
pub(super) fn validate_plan(plan: &Plan) -> DynResult<()> {
    if plan.schema_version != 1
        || plan.experiment_id != EXPERIMENT
        || plan.expected_experiment_name != EXPERIMENT_NAME
    {
        return Err("publication plan cannot select another experiment".into());
    }
    plan.report.verify()?;
    plan.archive.verify()?;
    let report: Value = serde_json::from_slice(&crate::automation::receipt_files::bounded(
        &plan.report.path,
        4 * 1024 * 1024,
    )?)?;
    let expected = build(
        &report,
        Input {
            report: plan.report.clone(),
            archive: plan.archive.clone(),
        },
    )?;
    if serde_json::to_value(expected)? != serde_json::to_value(plan)? {
        return Err("MLflow plan differs from verified report".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn existing_experiment_must_not_be_missing_deleted_or_replaced() {
        validate_experiment(&json!({"experiment":{"experiment_id":"6","name":EXPERIMENT_NAME,"lifecycle_stage":"active"}})).unwrap();
        for value in [
            json!({}),
            json!({"experiment":{"experiment_id":"7","name":EXPERIMENT_NAME,"lifecycle_stage":"active"}}),
            json!({"experiment":{"experiment_id":"6","name":"replacement","lifecycle_stage":"active"}}),
            json!({"experiment":{"experiment_id":"6","name":EXPERIMENT_NAME,"lifecycle_stage":"deleted"}}),
        ] {
            assert!(validate_experiment(&value).is_err());
        }
    }
    #[test]
    fn publication_keeps_shared_campaign_and_complete_work_provenance() {
        let tools: Value = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/agentx/tools.example.json"
        ))
        .unwrap();
        let mut report = json!({"status":"passed","scope":"fixed_complete_AgentX_comparison",
            "campaign":"agentx-final-fixture","backend":"cuda","model_key":"qwen3.5-4b-q4-k-m",
            "source_attribution":{"product":"382c6c62a5e6b7dc9b624f7e64041e39913759d5","automation_binary_sha256":"automation"},
            "request":{"tools":tools,"corpus":{"sha256":"corpus"}},"arms":super::super::final_sweep_tests::arms()});
        report["arms"][0]["cells"][0]["cache_read_tokens"] = json!({"avg":12.5});
        let input = || Input {
            report: File {
                path: "/report.json".into(),
                sha256: "report".into(),
            },
            archive: File {
                path: "/archive.tar.gz".into(),
                sha256: "archive".into(),
            },
        };
        let plan = build(&report, input()).unwrap();
        assert_eq!(plan.experiment_id, "6");
        assert_eq!(
            plan.campaign_root.run_key,
            "agentx-final-fixture:382c6c62a5e6b7dc9b624f7e64041e39913759d5:root"
        );
        assert_eq!(plan.children.len(), 6);
        assert_eq!(plan.children[0].metrics["c1.cache_read_tokens_avg"], 12.5);
        for tag in [
            "ax.native_source_commit",
            "ax.native_patch_digest",
            "ax.runtime_sha256",
            "ax.client_environment_sha256",
            "ax.cohort_sha256",
            "ax.work_sha256",
        ] {
            assert!(plan.tags.contains_key(tag), "{tag}");
        }
        report["request"]["tools"]["runtime"]["sha256"] = Value::Null;
        assert!(build(&report, input()).is_err());
        report["scope"] = json!("cpu_diagnostic_not_gpu_performance");
        assert!(build(&report, input()).is_err());
    }
}
