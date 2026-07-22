use super::{adapters::run_command, doctor::preflight_eval_run, registry::definition, *};

pub(super) fn run_eval(args: EvalRunArgs) -> Result<()> {
    if args.endpoint_concurrency == 0 {
        bail!("--endpoint-concurrency must be greater than zero");
    }
    let root = absolute_path(cache_root(args.cache_root.clone())?)?;
    let definition = definition(args.eval);
    let run_dir = absolute_path(output_dir(args.output_dir.clone(), args.eval)?)?;
    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| eval_run_id(args.eval));
    let metrics_run_id = args
        .metrics_run_id
        .clone()
        .unwrap_or_else(|| run_id.clone());
    let metrics_http = args.metrics_http.trim_end_matches('/').to_string();

    if !args.dry_run {
        let harness = harness_dir(&root, definition);
        if !harness.exists() {
            bail!(
                "{} is not installed at {}; run `skippy-bench eval sync {}` first",
                definition.id.as_str(),
                harness.display(),
                definition.id.as_str()
            );
        }
        preflight_eval_run(definition)?;
    }

    fs::create_dir_all(run_dir.join("raw"))
        .with_context(|| format!("create eval run dir {}", run_dir.display()))?;

    let harness_commit = resolved_harness_commit(&root, definition)?;
    let command = run_command(definition, &args, &root, &run_dir)?;
    let display = command.display();
    println!("{display}");

    let mut report = RunReport {
        run_id: run_id.clone(),
        eval_id: definition.id.as_str(),
        model: args.model.clone(),
        base_url: args.base_url.clone(),
        endpoint_concurrency: args.endpoint_concurrency,
        run_dir: run_dir.display().to_string(),
        harness_commit,
        dry_run: args.dry_run,
        command: display,
        exit_status: None,
        success: args.dry_run,
        timed_out: false,
        timeout_secs: args.timeout_secs,
        harness_timeout_secs: args.harness_timeout_secs,
        stdout_path: None,
        stderr_path: None,
        metrics: EvalMetrics::default(),
        telemetry: telemetry_report::pending(&metrics_http, &metrics_run_id),
        artifacts: run_artifacts(definition, &run_dir),
    };

    if !args.dry_run {
        create_metrics_run(&args, &run_id, &metrics_run_id)?;
        let started = Instant::now();
        let stdout_path = run_dir.join("raw").join("stdout.log");
        let stderr_path = run_dir.join("raw").join("stderr.log");
        let outcome = run_command_with_timeout(
            &command,
            args.harness_timeout_secs.map(Duration::from_secs),
            &stdout_path,
            &stderr_path,
        )
        .with_context(|| format!("run {}", definition.id.as_str()))?;
        let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
        report.exit_status = outcome.exit_status;
        report.success = outcome.success;
        report.timed_out = outcome.timed_out;
        report.stdout_path = Some(stdout_path.display().to_string());
        report.stderr_path = Some(stderr_path.display().to_string());
        report.metrics = collect_metrics(definition, &run_dir, duration_ms);
        let telemetry = if args.metrics_finalize_only {
            telemetry_report::finalize_only(&metrics_http, &metrics_run_id)
        } else {
            collect_telemetry(&metrics_http, &metrics_run_id, &run_dir)
        };
        report.telemetry = telemetry_or_unavailable(&metrics_http, &metrics_run_id, telemetry);
    }

    let report_path = run_dir.join("run.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    if !report.success {
        bail!(
            "{} failed; see {}",
            definition.id.as_str(),
            report_path.display()
        );
    }
    Ok(())
}

pub(super) fn resolved_harness_commit(
    root: &Path,
    definition: EvalDefinition,
) -> Result<Option<String>> {
    let harness = harness_dir(root, definition);
    if !harness.exists() {
        return Ok(None);
    }
    let output = Command::new("git")
        .args(["-C", &harness.display().to_string(), "rev-parse", "HEAD"])
        .output()
        .with_context(|| format!("read harness revision from {}", harness.display()))?;
    if !output.status.success() {
        bail!(
            "git rev-parse failed for {}: {}",
            harness.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let commit = String::from_utf8(output.stdout)
        .context("harness revision was not UTF-8")?
        .trim()
        .to_string();
    if commit.is_empty() {
        bail!("git returned an empty revision for {}", harness.display());
    }
    Ok(Some(commit))
}

pub(super) fn run_artifacts(definition: EvalDefinition, run_dir: &Path) -> Vec<RunArtifact> {
    let mut artifacts = vec![
        RunArtifact {
            kind: "stdout",
            path: run_dir.join("raw/stdout.log").display().to_string(),
        },
        RunArtifact {
            kind: "stderr",
            path: run_dir.join("raw/stderr.log").display().to_string(),
        },
    ];
    match definition.id {
        EvalId::SpeedBench => artifacts.extend([
            RunArtifact {
                kind: "speed-bench-json",
                path: speed_bench_output_path(run_dir).display().to_string(),
            },
            RunArtifact {
                kind: "speed-bench-response-timings",
                path: speed_bench_response_timings_path(run_dir)
                    .display()
                    .to_string(),
            },
        ]),
        EvalId::SweBenchPro => artifacts.extend([
            RunArtifact {
                kind: "swe-bench-pro-sweagent-results",
                path: swe_bench_pro_sweagent_output_path(run_dir)
                    .display()
                    .to_string(),
            },
            RunArtifact {
                kind: "swe-bench-pro-patches-json",
                path: swe_bench_pro_patches_path(run_dir).display().to_string(),
            },
            RunArtifact {
                kind: "swe-bench-pro-eval-json",
                path: swe_bench_pro_output_path(run_dir).display().to_string(),
            },
        ]),
        EvalId::McpAtlas => artifacts.extend([
            RunArtifact {
                kind: "mcp-atlas-completion-results-csv",
                path: mcp_atlas_output_path(run_dir).display().to_string(),
            },
            RunArtifact {
                kind: "mcp-atlas-score-dir",
                path: mcp_atlas_score_dir(run_dir).display().to_string(),
            },
        ]),
        EvalId::TerminalBench => artifacts.push(RunArtifact {
            kind: "terminal-bench-results",
            path: terminal_bench_output_path(run_dir).display().to_string(),
        }),
    }
    artifacts
}

fn collect_metrics(definition: EvalDefinition, run_dir: &Path, duration_ms: f64) -> EvalMetrics {
    let mut metrics = match definition.id {
        EvalId::SpeedBench => speed_bench_metrics(run_dir).unwrap_or_default(),
        EvalId::SweBenchPro => swe_bench_pro_metrics(run_dir).unwrap_or_default(),
        EvalId::McpAtlas => mcp_atlas_metrics(run_dir).unwrap_or_default(),
        EvalId::TerminalBench => terminal_bench_metrics(run_dir).unwrap_or_default(),
    };
    metrics.duration_ms = Some(duration_ms);
    fill_client_rates(&mut metrics, duration_ms);
    metrics
}

fn create_metrics_run(args: &EvalRunArgs, eval_run_id: &str, metrics_run_id: &str) -> Result<()> {
    let config = json!({
        "eval_run_id": eval_run_id,
        "mode": "skippy-bench-external-eval",
        "eval_id": args.eval.as_str(),
        "model": args.model,
        "base_url": args.base_url,
    });
    telemetry_report::create_run(&args.metrics_http, metrics_run_id, &config)
}

fn collect_telemetry(
    metrics_http: &str,
    metrics_run_id: &str,
    run_dir: &Path,
) -> Result<BenchTelemetry> {
    telemetry_report::finalize_and_collect(
        metrics_http,
        metrics_run_id,
        &metrics_report_path(run_dir),
    )
}

pub(super) fn telemetry_or_unavailable(
    metrics_http: &str,
    metrics_run_id: &str,
    result: Result<BenchTelemetry>,
) -> BenchTelemetry {
    result
        .unwrap_or_else(|error| telemetry_report::unavailable(metrics_http, metrics_run_id, &error))
}

pub(super) fn speed_bench_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let value = read_json(&speed_bench_output_path(run_dir))?;
    let overall = value
        .get("summary")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| string_field(row, "category") == Some("overall"))
        });
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut metrics = EvalMetrics {
        request_count: value.get("completed_samples").and_then(Value::as_u64),
        failed_count: value.get("failed_samples").and_then(Value::as_u64),
        ..EvalMetrics::default()
    };
    metrics.prompt_tokens = sum_u64_field(results, "prompt_tokens");
    metrics.completion_tokens = sum_u64_field(results, "completion_tokens");
    metrics.total_tokens = sum_u64_field(results, "total_tokens");
    metrics.draft_tokens = sum_u64_field(results, "draft_n");
    metrics.draft_accepted_tokens = sum_u64_field(results, "draft_n_accepted");
    if let Some(overall) = overall {
        metrics.prompt_tok_s = numeric_field(overall, "avg_prompt_t_s");
        metrics.completion_tok_s = numeric_field(overall, "avg_pred_t_s");
        metrics.avg_latency_ms =
            numeric_field(overall, "avg_latency").map(|seconds| seconds * 1000.0);
        metrics.draft_accept_rate = numeric_field(overall, "accept_rate");
    }
    Ok(metrics)
}

pub(super) fn swe_bench_pro_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let value = read_json(&swe_bench_pro_output_path(run_dir))?;
    let rows = value
        .get("rows")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut metrics = EvalMetrics {
        request_count: swe_bench_pro_request_count(&value, rows),
        failed_count: swe_bench_pro_failed_count(&value),
        pass_rate: swe_bench_pro_pass_rate(&value),
        ..EvalMetrics::default()
    };
    metrics.prompt_tokens = sum_nested_usage(rows, "prompt_tokens");
    metrics.completion_tokens = sum_nested_usage(rows, "completion_tokens");
    metrics.total_tokens = sum_nested_usage(rows, "total_tokens");
    Ok(metrics)
}

fn swe_bench_pro_request_count(value: &Value, rows: &[Value]) -> Option<u64> {
    value
        .get("total_instances")
        .or_else(|| value.get("total"))
        .or_else(|| value.get("n_total"))
        .and_then(Value::as_u64)
        .or_else(|| (!rows.is_empty()).then_some(rows.len() as u64))
}

fn swe_bench_pro_failed_count(value: &Value) -> Option<u64> {
    value
        .get("failed_instances")
        .or_else(|| value.get("unresolved_instances"))
        .or_else(|| value.get("n_unresolved"))
        .and_then(Value::as_u64)
}

fn swe_bench_pro_pass_rate(value: &Value) -> Option<f64> {
    if let Some(rate) = value
        .get("pass_rate")
        .or_else(|| value.get("resolved_rate"))
        .or_else(|| value.get("accuracy"))
        .and_then(Value::as_f64)
    {
        return Some(rate);
    }
    let resolved = value
        .get("resolved_instances")
        .or_else(|| value.get("n_resolved"))
        .and_then(Value::as_u64)?;
    let total = value
        .get("total_instances")
        .or_else(|| value.get("total"))
        .or_else(|| value.get("n_total"))
        .and_then(Value::as_u64)?;
    (total > 0).then_some(resolved as f64 / total as f64)
}

pub(super) fn terminal_bench_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let value = read_json(&terminal_bench_results_path(run_dir)?)?;
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut metrics = EvalMetrics {
        request_count: Some(results.len() as u64),
        failed_count: value.get("n_unresolved").and_then(Value::as_u64),
        pass_rate: value.get("accuracy").and_then(Value::as_f64),
        ..EvalMetrics::default()
    };
    metrics.prompt_tokens = sum_u64_field(results, "total_input_tokens");
    metrics.completion_tokens = sum_u64_field(results, "total_output_tokens");
    metrics.total_tokens = match (metrics.prompt_tokens, metrics.completion_tokens) {
        (Some(prompt), Some(completion)) => Some(prompt + completion),
        _ => None,
    };
    Ok(metrics)
}

fn terminal_bench_results_path(run_dir: &Path) -> Result<PathBuf> {
    let root = terminal_bench_output_path(run_dir);
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path().join("results.json");
        if path.is_file() {
            return Ok(path);
        }
    }
    bail!("no Terminal-Bench results.json under {}", root.display())
}

fn mcp_atlas_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let mut reader = csv::Reader::from_path(mcp_atlas_output_path(run_dir))?;
    let data_rows = reader.records().filter(|record| record.is_ok()).count();
    Ok(EvalMetrics {
        request_count: Some(data_rows as u64),
        ..EvalMetrics::default()
    })
}

pub(super) fn fill_client_rates(metrics: &mut EvalMetrics, duration_ms: f64) {
    if duration_ms <= 0.0 {
        return;
    }
    let seconds = duration_ms / 1000.0;
    if metrics.prompt_tok_s.is_none() {
        metrics.prompt_tok_s = metrics.prompt_tokens.map(|tokens| tokens as f64 / seconds);
    }
    if metrics.completion_tok_s.is_none() {
        metrics.completion_tok_s = metrics
            .completion_tokens
            .map(|tokens| tokens as f64 / seconds);
    }
    metrics.total_tok_s = metrics.total_tokens.map(|tokens| tokens as f64 / seconds);
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn sum_u64_field(rows: &[Value], key: &str) -> Option<u64> {
    let mut total = 0;
    let mut found = false;
    for row in rows {
        if let Some(value) = row.get(key).and_then(Value::as_u64) {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}

fn sum_nested_usage(rows: &[Value], key: &str) -> Option<u64> {
    let mut total = 0;
    let mut found = false;
    for row in rows {
        if let Some(value) = row
            .get("usage")
            .and_then(|usage| usage.get(key))
            .and_then(Value::as_u64)
        {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}

fn numeric_field(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

pub(super) fn speed_bench_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/speed-bench.json")
}

pub(super) fn speed_bench_response_timings_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/speed-bench-response-timings.jsonl")
}

pub(super) fn swe_bench_pro_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/swe-bench-pro/eval/eval_results.json")
}

pub(super) fn swe_bench_pro_sweagent_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/swe-bench-pro/sweagent-results")
}

pub(super) fn swe_bench_pro_patches_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/swe-bench-pro/patches.json")
}

pub(super) fn mcp_atlas_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/mcp-atlas-completion-results.csv")
}

pub(super) fn mcp_atlas_score_dir(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/mcp-atlas-evaluation-results")
}

pub(super) fn terminal_bench_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/terminal-bench")
}

fn metrics_report_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/metrics-report.json")
}

fn eval_run_id(eval: EvalId) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("skippy-eval-{}-{millis}", eval.as_str())
}
