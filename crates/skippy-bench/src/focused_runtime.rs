use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use serde::Serialize;

use super::deployment::{
    parse_hosts, parse_stage_ranges, validate_balanced_stage_ranges, validate_distinct_stage_hosts,
    validate_topology_plan,
};
use super::{
    DistributedRunOutcome, PromptDriverReport, effective_run_max_new_tokens, generate_bench_run_id,
    run_distributed_collect, write_json_file,
};
use crate::cli::{DEFAULT_RUN_MAX_NEW_TOKENS, FocusedRuntimeArgs, FocusedRuntimeScenario, RunArgs};

#[derive(Debug, Serialize)]
struct FocusedRuntimeReport {
    schema_version: u32,
    scenario: String,
    mode: String,
    run_id: String,
    topology_id: String,
    model_id: String,
    model_identity: ModelIdentity,
    stage_count: usize,
    hosts: Vec<String>,
    topology: FocusedRuntimeTopology,
    model: FocusedRuntimeModel,
    latency_ms: FocusedRuntimeLatency,
    throughput_tokens_per_second: FocusedRuntimeThroughput,
    token_counts: FocusedRuntimeTokenCounts,
    preset: FocusedRuntimePreset,
    summary: FocusedRuntimeSummary,
    outputs: FocusedRuntimeOutputs,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeSummary {
    startup_elapsed_ms: Option<u128>,
    run_elapsed_ms: u128,
    prompt_count: usize,
    max_new_tokens: usize,
    prompt_tokens_total: usize,
    generated_tokens_total: usize,
    elapsed_ms_p50: u128,
    elapsed_ms_p95: u128,
    ttft_ms_p50: u128,
    ttft_ms_p95: u128,
    decode_elapsed_ms_p50: u128,
    decode_elapsed_ms_p95: u128,
    total_tokens_per_second: f64,
    generated_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeTopology {
    topology_id: String,
    stage_count: usize,
    hosts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeModel {
    model_id: String,
    model_identity: ModelIdentity,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeLatency {
    startup_elapsed_ms: Option<u128>,
    run_elapsed_ms: u128,
    elapsed_ms_p50: u128,
    elapsed_ms_p95: u128,
    ttft_ms_p50: u128,
    ttft_ms_p95: u128,
    decode_elapsed_ms_p50: u128,
    decode_elapsed_ms_p95: u128,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeThroughput {
    total: f64,
    generated: f64,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeTokenCounts {
    prompt_total: usize,
    generated_total: usize,
    prompt_count: usize,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimePreset {
    scenario: String,
    description: &'static str,
    prompt_limit: Option<usize>,
    max_new_tokens: usize,
    generated_prompt_corpus: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct FocusedRuntimeOutputs {
    report: PathBuf,
    driver_result: Option<PathBuf>,
    deployment_plan: PathBuf,
    remote_status: Option<PathBuf>,
}

pub(super) fn focused_runtime(args: FocusedRuntimeArgs) -> Result<()> {
    let mut args = apply_focused_runtime_preset(args);
    validate_focused_runtime_args(&args)?;
    if args.schema_smoke {
        let report = focused_runtime_schema_smoke_report(&args)?;
        write_or_print_focused_runtime_report(&report, args.focused_output.as_deref())?;
        return Ok(());
    }

    let scenario = args.scenario;
    let focused_output = args.focused_output.clone();
    let preset = prepare_focused_runtime_inputs(&mut args)?;
    let outcome = run_distributed_collect(args.run)?;
    let report = focused_runtime_report_from_outcome(scenario, preset, &outcome)?;
    let output =
        focused_output.unwrap_or_else(|| outcome.run_dir.join("focused-runtime-report.json"));
    write_json_file(&output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn validate_focused_runtime_args(args: &FocusedRuntimeArgs) -> Result<()> {
    validate_focused_runtime_topology(&args.run)?;
    if args.schema_smoke {
        return Ok(());
    }
    if !args.run.execute_remote {
        bail!(
            "focused-runtime requires --execute-remote so driver timing fields are produced; use --schema-smoke for CI-only schema validation"
        );
    }
    Ok(())
}

fn apply_focused_runtime_preset(mut args: FocusedRuntimeArgs) -> FocusedRuntimeArgs {
    match args.scenario {
        FocusedRuntimeScenario::ColdStartup | FocusedRuntimeScenario::FirstToken => {
            if args.run.prompt_limit.is_none() {
                args.run.prompt_limit = Some(1);
            }
            if args.run.max_new_tokens.is_none() {
                args.run.max_new_tokens = Some(DEFAULT_RUN_MAX_NEW_TOKENS);
            }
        }
        FocusedRuntimeScenario::SteadyDecode => {
            if args.run.prompt_limit.is_none() {
                args.run.prompt_limit = Some(1);
            }
            if args.run.max_new_tokens.is_none() {
                args.run.max_new_tokens = Some(128_usize);
            }
        }
        FocusedRuntimeScenario::KvWarmReuse => {
            if args.run.prompt_limit.is_none() {
                args.run.prompt_limit = Some(2);
            }
            if args.run.max_new_tokens.is_none() {
                args.run.max_new_tokens = Some(16_usize);
            }
        }
    }
    args
}

fn validate_focused_runtime_topology(run: &RunArgs) -> Result<()> {
    let hosts = parse_hosts(&run.hosts)?;
    let ranges = parse_stage_ranges(&run.splits, run.layer_end)?;
    validate_distinct_stage_hosts(&hosts, ranges.len())?;
    validate_topology_plan(run, &hosts, &ranges)?;
    validate_balanced_stage_ranges(&ranges)?;
    Ok(())
}

fn prepare_focused_runtime_inputs(args: &mut FocusedRuntimeArgs) -> Result<FocusedRuntimePreset> {
    let mut generated_prompt_corpus = None;
    if matches!(args.scenario, FocusedRuntimeScenario::KvWarmReuse)
        && args.run.prompt_corpus.is_none()
        && args.run.prompt_token_ids.is_none()
    {
        let run_id = args
            .run
            .run_id
            .clone()
            .unwrap_or_else(generate_bench_run_id);
        args.run.run_id = Some(run_id.clone());
        let path = args
            .run
            .work_dir
            .join(&run_id)
            .join("focused-kv-warm-reuse-corpus.jsonl");
        let escaped_prompt = serde_json::to_string(&args.run.prompt)?;
        let corpus = format!(
            "{{\"id\":\"kv-warm-reuse-1\",\"category\":\"kv_warm_reuse\",\"prompt\":{escaped_prompt}}}\n{{\"id\":\"kv-warm-reuse-2\",\"category\":\"kv_warm_reuse\",\"prompt\":{escaped_prompt}}}\n"
        );
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "create focused KV warm-reuse corpus dir {}",
                    parent.display()
                )
            })?;
        }
        fs::write(&path, corpus)
            .with_context(|| format!("write focused KV warm-reuse corpus {}", path.display()))?;
        args.run.prompt_corpus = Some(path.clone());
        generated_prompt_corpus = Some(path);
    }

    Ok(FocusedRuntimePreset {
        scenario: args.scenario.as_str().to_string(),
        description: focused_runtime_preset_description(args.scenario),
        prompt_limit: args.run.prompt_limit,
        max_new_tokens: effective_run_max_new_tokens(&args.run),
        generated_prompt_corpus,
    })
}

fn focused_runtime_preset_description(scenario: FocusedRuntimeScenario) -> &'static str {
    match scenario {
        FocusedRuntimeScenario::ColdStartup => {
            "one-prompt run with a default one-token decode budget; report startup readiness separately from driver latency"
        }
        FocusedRuntimeScenario::FirstToken => {
            "one-prompt run focused on existing prompt-driver TTFT percentiles"
        }
        FocusedRuntimeScenario::SteadyDecode => {
            "one-prompt run with a larger default decode budget when max-new-tokens is otherwise left at the run default"
        }
        FocusedRuntimeScenario::KvWarmReuse => {
            "two identical prompts by default so the second request can exercise warm-prefix reuse where supported"
        }
    }
}

fn focused_runtime_report_from_outcome(
    scenario: FocusedRuntimeScenario,
    preset: FocusedRuntimePreset,
    outcome: &DistributedRunOutcome,
) -> Result<FocusedRuntimeReport> {
    let driver = outcome
        .driver_report
        .as_ref()
        .context("focused-runtime requires driver-result output from an executed staged run")?;
    let summary = focused_runtime_summary(driver, outcome.startup_elapsed, outcome.run_elapsed);
    Ok(FocusedRuntimeReport {
        schema_version: 1,
        scenario: scenario.as_str().to_string(),
        mode: "executed".to_string(),
        run_id: outcome.run_id.clone(),
        topology_id: outcome.topology_id.clone(),
        model_id: outcome.model_id.clone(),
        model_identity: outcome.model_identity.clone(),
        stage_count: outcome.stage_count,
        hosts: outcome.hosts.clone(),
        topology: focused_runtime_topology(
            &outcome.topology_id,
            outcome.stage_count,
            &outcome.hosts,
        ),
        model: focused_runtime_model(&outcome.model_id, &outcome.model_identity),
        latency_ms: focused_runtime_latency(&summary),
        throughput_tokens_per_second: focused_runtime_throughput(&summary),
        token_counts: focused_runtime_token_counts(&summary),
        preset,
        summary,
        outputs: FocusedRuntimeOutputs {
            report: outcome.report_path.clone(),
            driver_result: outcome.driver_result_path.clone(),
            deployment_plan: outcome.plan_path.clone(),
            remote_status: outcome.remote_status_path.clone(),
        },
    })
}

fn focused_runtime_summary(
    driver: &PromptDriverReport,
    startup_elapsed: Option<Duration>,
    run_elapsed: Duration,
) -> FocusedRuntimeSummary {
    FocusedRuntimeSummary {
        startup_elapsed_ms: startup_elapsed.map(|elapsed| elapsed.as_millis()),
        run_elapsed_ms: run_elapsed.as_millis(),
        prompt_count: driver.prompt_count,
        max_new_tokens: driver.max_new_tokens,
        prompt_tokens_total: driver.summary.prompt_tokens_total,
        generated_tokens_total: driver.summary.generated_tokens_total,
        elapsed_ms_p50: driver.summary.elapsed_ms_p50,
        elapsed_ms_p95: driver.summary.elapsed_ms_p95,
        ttft_ms_p50: driver.summary.ttft_ms_p50,
        ttft_ms_p95: driver.summary.ttft_ms_p95,
        decode_elapsed_ms_p50: driver.summary.decode_elapsed_ms_p50,
        decode_elapsed_ms_p95: driver.summary.decode_elapsed_ms_p95,
        total_tokens_per_second: driver.summary.total_tokens_per_second,
        generated_tokens_per_second: driver.summary.generated_tokens_per_second,
    }
}

fn focused_runtime_schema_smoke_report(args: &FocusedRuntimeArgs) -> Result<FocusedRuntimeReport> {
    let hosts = parse_hosts(&args.run.hosts)?;
    let ranges = parse_stage_ranges(&args.run.splits, args.run.layer_end)?;
    validate_distinct_stage_hosts(&hosts, ranges.len())?;
    validate_topology_plan(&args.run, &hosts, &ranges)?;
    validate_balanced_stage_ranges(&ranges)?;
    let stage_count = ranges.len();
    let prompt_count = args.run.prompt_limit.unwrap_or(1);
    let model_identity = ModelIdentity::from_model_id(args.run.model_id.clone());
    let summary = FocusedRuntimeSummary {
        startup_elapsed_ms: Some(0),
        run_elapsed_ms: 0,
        prompt_count,
        max_new_tokens: effective_run_max_new_tokens(&args.run),
        prompt_tokens_total: 8 * prompt_count,
        generated_tokens_total: effective_run_max_new_tokens(&args.run) * prompt_count,
        elapsed_ms_p50: 10,
        elapsed_ms_p95: 10,
        ttft_ms_p50: 5,
        ttft_ms_p95: 5,
        decode_elapsed_ms_p50: 5,
        decode_elapsed_ms_p95: 5,
        total_tokens_per_second: 900.0,
        generated_tokens_per_second: 100.0,
    };
    Ok(FocusedRuntimeReport {
        schema_version: 1,
        scenario: args.scenario.as_str().to_string(),
        mode: "schema-smoke".to_string(),
        run_id: args
            .run
            .run_id
            .clone()
            .unwrap_or_else(|| "focused-runtime-schema-smoke".to_string()),
        topology_id: args.run.topology_id.clone(),
        model_id: args.run.model_id.clone(),
        model_identity: model_identity.clone(),
        stage_count,
        hosts: hosts.clone(),
        topology: focused_runtime_topology(&args.run.topology_id, stage_count, &hosts),
        model: focused_runtime_model(&args.run.model_id, &model_identity),
        latency_ms: focused_runtime_latency(&summary),
        throughput_tokens_per_second: focused_runtime_throughput(&summary),
        token_counts: focused_runtime_token_counts(&summary),
        preset: FocusedRuntimePreset {
            scenario: args.scenario.as_str().to_string(),
            description: focused_runtime_preset_description(args.scenario),
            prompt_limit: args.run.prompt_limit,
            max_new_tokens: effective_run_max_new_tokens(&args.run),
            generated_prompt_corpus: None,
        },
        summary,
        outputs: FocusedRuntimeOutputs {
            report: PathBuf::from("schema-smoke-report.json"),
            driver_result: Some(PathBuf::from("schema-smoke-driver-result.json")),
            deployment_plan: PathBuf::from("schema-smoke-deployment-plan.json"),
            remote_status: None,
        },
    })
}

fn focused_runtime_topology(
    topology_id: &str,
    stage_count: usize,
    hosts: &[String],
) -> FocusedRuntimeTopology {
    FocusedRuntimeTopology {
        topology_id: topology_id.to_string(),
        stage_count,
        hosts: hosts.to_vec(),
    }
}

fn focused_runtime_model(model_id: &str, model_identity: &ModelIdentity) -> FocusedRuntimeModel {
    FocusedRuntimeModel {
        model_id: model_id.to_string(),
        model_identity: model_identity.clone(),
    }
}

fn focused_runtime_latency(summary: &FocusedRuntimeSummary) -> FocusedRuntimeLatency {
    FocusedRuntimeLatency {
        startup_elapsed_ms: summary.startup_elapsed_ms,
        run_elapsed_ms: summary.run_elapsed_ms,
        elapsed_ms_p50: summary.elapsed_ms_p50,
        elapsed_ms_p95: summary.elapsed_ms_p95,
        ttft_ms_p50: summary.ttft_ms_p50,
        ttft_ms_p95: summary.ttft_ms_p95,
        decode_elapsed_ms_p50: summary.decode_elapsed_ms_p50,
        decode_elapsed_ms_p95: summary.decode_elapsed_ms_p95,
    }
}

fn focused_runtime_throughput(summary: &FocusedRuntimeSummary) -> FocusedRuntimeThroughput {
    FocusedRuntimeThroughput {
        total: summary.total_tokens_per_second,
        generated: summary.generated_tokens_per_second,
    }
}

fn focused_runtime_token_counts(summary: &FocusedRuntimeSummary) -> FocusedRuntimeTokenCounts {
    FocusedRuntimeTokenCounts {
        prompt_total: summary.prompt_tokens_total,
        generated_total: summary.generated_tokens_total,
        prompt_count: summary.prompt_count,
        max_new_tokens: summary.max_new_tokens,
    }
}

fn write_or_print_focused_runtime_report(
    report: &FocusedRuntimeReport,
    output: Option<&Path>,
) -> Result<()> {
    if let Some(output) = output {
        write_json_file(output, report)?;
    }
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{PromptDriverResult, prompt_driver_summary};
    use serde_json::json;

    #[test]
    fn focused_runtime_schema_smoke_uses_compact_output_shape() {
        let mut run = test_run_args();
        run.hosts = "host-a,host-b".to_string();
        run.splits = "1".to_string();
        run.layer_end = 2;
        run.prompt_limit = Some(3);
        run.max_new_tokens = Some(7);
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::FirstToken,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let report = focused_runtime_schema_smoke_report(&args).unwrap();
        let value = serde_json::to_value(&report).unwrap();

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["scenario"], "first-token");
        assert_eq!(value["mode"], "schema-smoke");
        assert_eq!(value["stage_count"], 2);
        assert_eq!(value["hosts"], json!(["host-a", "host-b"]));
        assert_eq!(value["summary"]["prompt_count"], 3);
        assert_eq!(value["summary"]["max_new_tokens"], 7);
        assert_eq!(value["summary"]["elapsed_ms_p95"], 10);
        assert_eq!(value["summary"]["ttft_ms_p50"], 5);
        assert_eq!(value["summary"]["generated_tokens_per_second"], 100.0);
        assert_eq!(value["topology"]["topology_id"], "topology");
        assert_eq!(value["topology"]["stage_count"], 2);
        assert_eq!(
            value["model"]["model_id"],
            "test-org/bench-model-GGUF:Q4_K_M"
        );
        assert_eq!(value["latency_ms"]["elapsed_ms_p95"], 10);
        assert_eq!(value["latency_ms"]["startup_elapsed_ms"], 0);
        assert_eq!(value["throughput_tokens_per_second"]["generated"], 100.0);
        assert_eq!(value["token_counts"]["prompt_total"], 24);
        assert_eq!(value["token_counts"]["generated_total"], 21);
        assert_eq!(
            value["model_identity"]["model_id"],
            "test-org/bench-model-GGUF:Q4_K_M"
        );
        assert_eq!(
            value["outputs"]["deployment_plan"],
            "schema-smoke-deployment-plan.json"
        );
    }

    #[test]
    fn focused_runtime_preset_only_rewrites_omitted_decode_budget() {
        let mut run = test_run_args();
        run.max_new_tokens = None;
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::SteadyDecode,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let args = apply_focused_runtime_preset(args);
        assert_eq!(args.run.prompt_limit, Some(1));
        assert_eq!(args.run.max_new_tokens, Some(128));

        let mut run = test_run_args();
        run.max_new_tokens = Some(1);
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::SteadyDecode,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let args = apply_focused_runtime_preset(args);
        assert_eq!(args.run.max_new_tokens, Some(1));
    }

    #[test]
    fn focused_runtime_kv_warm_reuse_preserves_explicit_one_token_budget() {
        let mut run = test_run_args();
        run.max_new_tokens = Some(1);
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::KvWarmReuse,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let args = apply_focused_runtime_preset(args);
        assert_eq!(args.run.prompt_limit, Some(2));
        assert_eq!(args.run.max_new_tokens, Some(1));
    }

    #[test]
    fn focused_runtime_cold_startup_description_matches_default_decode_budget() {
        let description = focused_runtime_preset_description(FocusedRuntimeScenario::ColdStartup);

        assert!(description.contains("default one-token decode budget"));
        assert!(!description.starts_with("one-prompt, one-token run"));
    }

    #[test]
    fn run_args_default_to_one_generated_token_when_omitted() {
        let run = test_run_args();

        assert_eq!(run.max_new_tokens, None);
        assert_eq!(effective_run_max_new_tokens(&run), 1);
    }

    #[test]
    fn focused_runtime_schema_smoke_rejects_invalid_topology() {
        let mut run = test_run_args();
        run.hosts = "host-a".to_string();
        run.splits = "1".to_string();
        run.layer_end = 2;
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::FirstToken,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let err = focused_runtime_schema_smoke_report(&args).unwrap_err();
        assert!(
            err.to_string()
                .contains("provide one separate node per stage")
        );

        let mut run = test_run_args();
        run.hosts = "host-a,host-b,host-c,host-d".to_string();
        run.splits = "1,4,7".to_string();
        run.layer_end = 40;
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::FirstToken,
            focused_output: None,
            schema_smoke: true,
            run,
        };

        let err = focused_runtime_schema_smoke_report(&args).unwrap_err();
        assert!(err.to_string().contains("evenly balanced"));
    }

    #[test]
    fn focused_runtime_requires_executed_run_unless_schema_smoke() {
        let mut run = test_run_args();
        run.hosts = "host-a,host-b".to_string();
        let args = FocusedRuntimeArgs {
            scenario: FocusedRuntimeScenario::SteadyDecode,
            focused_output: None,
            schema_smoke: false,
            run,
        };

        let err = validate_focused_runtime_args(&args).unwrap_err();
        assert!(err.to_string().contains("requires --execute-remote"));

        let smoke_args = FocusedRuntimeArgs {
            schema_smoke: true,
            ..args
        };
        validate_focused_runtime_args(&smoke_args).unwrap();
    }

    #[test]
    fn focused_runtime_summary_reuses_prompt_driver_fields() {
        let results = [120_u128, 240, 360]
            .into_iter()
            .map(|elapsed_ms| PromptDriverResult {
                prompt_id: None,
                category: None,
                prompt: "hello".to_string(),
                token_ids: vec![1, 2, 3],
                prefill_token_count: 2,
                prefill_chunk_count: 1,
                effective_prefill_chunk_size: Some(2),
                predicted_tokens: vec![4, 5],
                max_new_tokens: 2,
                elapsed_ms,
                wire_elapsed_ms: elapsed_ms - 5,
                prefill_elapsed_ms: elapsed_ms - 10,
                ttft_ms: elapsed_ms - 20,
                decode_elapsed_ms: 20,
            })
            .collect::<Vec<_>>();
        let summary = prompt_driver_summary(&results);
        let driver = PromptDriverReport {
            first_stage_endpoint: "tcp://host-a:19031".to_string(),
            prompt_count: 3,
            max_new_tokens: 2,
            prefill_chunk_size: Some(2),
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            corpus: None,
            summary,
            results,
        };

        let focused = focused_runtime_summary(
            &driver,
            Some(Duration::from_millis(1234)),
            Duration::from_millis(5678),
        );

        assert_eq!(focused.startup_elapsed_ms, Some(1234));
        assert_eq!(focused.run_elapsed_ms, 5678);
        assert_eq!(focused.prompt_count, 3);
        assert_eq!(focused.max_new_tokens, 2);
        assert_eq!(focused.prompt_tokens_total, 9);
        assert_eq!(focused.generated_tokens_total, 6);
        assert_eq!(focused.elapsed_ms_p50, 240);
        assert_eq!(focused.elapsed_ms_p95, 360);
        assert_eq!(focused.ttft_ms_p50, 220);
        assert_eq!(focused.decode_elapsed_ms_p95, 20);
        assert!(focused.total_tokens_per_second > 0.0);
        assert!(focused.generated_tokens_per_second > 0.0);
    }

    fn test_run_args() -> RunArgs {
        RunArgs {
            metrics_server_bin: PathBuf::from("metrics-server"),
            stage_server_bin: PathBuf::from("skippy-server"),
            hosts: "host.local".to_string(),
            run_id: Some("run-1".to_string()),
            topology_id: "topology".to_string(),
            model_id: "test-org/bench-model-GGUF:Q4_K_M".to_string(),
            model_path: Some(PathBuf::from("model.gguf")),
            stage_model: None,
            stage_load_mode: "runtime-slice".to_string(),
            splits: "1".to_string(),
            layer_end: 2,
            ctx_size: 128,
            n_gpu_layers: 0,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            activation_width: 2048,
            activation_wire_dtype: "f32".to_string(),
            prompt: "Hello".to_string(),
            prompt_corpus: None,
            prompt_limit: None,
            prompt_token_ids: None,
            max_new_tokens: None,
            prefill_chunk_size: None,
            prefill_chunk_threshold: None,
            prefill_chunk_schedule: None,
            metrics_http_addr: "127.0.0.1:18080".parse().unwrap(),
            metrics_otlp_grpc_addr: "127.0.0.1:14317".parse().unwrap(),
            metrics_otlp_grpc_url: None,
            db: None,
            output: None,
            work_dir: PathBuf::from("/tmp/work"),
            remote_root: "/tmp/remote".to_string(),
            remote_root_map: None,
            remote_shared_root_map: None,
            endpoint_host_map: None,
            remote_bind_host: "0.0.0.0".to_string(),
            first_stage_port: 19031,
            execute_remote: false,
            keep_remote: false,
            rsync_model_artifacts: false,
            child_logs: false,
            startup_timeout_secs: 60,
            stage_max_inflight: 4,
            stage_reply_credit_limit: None,
            stage_async_prefill_forward: false,
            stage_downstream_wire_delay_ms: 0.0,
            stage_downstream_wire_mbps: None,
            stage_telemetry_queue_capacity: 8192,
            stage_telemetry_level: "summary".to_string(),
        }
    }
}
