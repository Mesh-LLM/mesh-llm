pub(crate) struct TuneBenchmarkRunRequest<'a> {
    pub(crate) config: &'a mesh_llm_config::MeshConfig,
    pub(crate) prepared: &'a [crate::gpus::tune_apply::PreparedTunePlan],
    pub(crate) ctx_sizes: &'a [u32],
    pub(crate) batch_sizes: &'a [u32],
    pub(crate) ubatch_sizes: &'a [u32],
    pub(crate) mmap_values: &'a [mesh_llm_cli::benchmark::BenchmarkBoolOrAuto],
    pub(crate) mlock_values: &'a [mesh_llm_cli::benchmark::BenchmarkBool],
    pub(crate) throughput_tolerance_pct: f64,
    pub(crate) max_tokens: u32,
    pub(crate) startup_timeout_secs: u64,
    pub(crate) request_timeout_secs: u64,
    pub(crate) prompt: &'a str,
}

pub(crate) fn run_benchmark_plans(
    request: TuneBenchmarkRunRequest<'_>,
) -> Vec<TuneBenchmarkTargetReport> {
    debug_assert!(
        request.throughput_tolerance_pct.is_finite() && request.throughput_tolerance_pct >= 0.0
    );
    request
        .prepared
        .iter()
        .filter(|prepared| !plan_has_errors(&prepared.plan))
        .map(|prepared| run_target_benchmarks(&request, prepared))
        .collect()
}

fn run_target_benchmarks(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> TuneBenchmarkTargetReport {
    let candidates = benchmark_candidates(request, prepared);
    eprintln!(
        "benchmark tune: target `{}` running {} trials (throughput tolerance {:.2}%)",
        prepared.target.requested_input,
        candidates.len(),
        request.throughput_tolerance_pct,
    );
    let total = candidates.len();
    let trials = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| run_trial_with_progress(request, prepared, index, total, candidate))
        .collect::<Vec<_>>();
    let selection = select_benchmark_trials(&trials, request.throughput_tolerance_pct);
    log_target_selection(&prepared.target.requested_input, &selection);

    TuneBenchmarkTargetReport {
        requested: prepared.target.requested_input.clone(),
        throughput_tolerance_pct: request.throughput_tolerance_pct,
        best: selection.recommended,
        raw_best: selection.raw_best,
        pareto_frontier: selection.pareto_frontier,
        selection_reason: selection.reason,
        trials,
    }
}

fn benchmark_candidates(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Vec<TuneBenchmarkCandidate> {
    let default_ctx =
        default_model_fit_u32(request, prepared, TuneField::CtxSize).unwrap_or(8192);
    let contexts = if request.ctx_sizes.is_empty() {
        default_context_sizes(default_ctx)
    } else {
        unique_positive(request.ctx_sizes)
    };
    let batches = if request.batch_sizes.is_empty() {
        vec![default_model_fit_u32(request, prepared, TuneField::Batch).unwrap_or(512)]
    } else {
        unique_positive(request.batch_sizes)
    };
    let ubatches = if request.ubatch_sizes.is_empty() {
        vec![default_model_fit_u32(request, prepared, TuneField::Ubatch).unwrap_or(128)]
    } else {
        unique_positive(request.ubatch_sizes)
    };
    let cache_type_k =
        recommended_cache_type(&prepared.plan, TuneField::CacheTypeK).unwrap_or(TuneKvCacheType::Q8_0);
    let cache_type_v =
        recommended_cache_type(&prepared.plan, TuneField::CacheTypeV).unwrap_or(cache_type_k);
    let mmap_values = benchmark_mmap_values(request.mmap_values, &prepared.plan);
    let mlock_values = benchmark_mlock_values(request.mlock_values, &prepared.plan);

    let mut candidates = Vec::new();
    for ctx_size in contexts {
        for batch in &batches {
            for ubatch in &ubatches {
                if *ubatch > *batch {
                    continue;
                }
                for mmap in &mmap_values {
                    for mlock in &mlock_values {
                        candidates.push(TuneBenchmarkCandidate {
                            ctx_size,
                            batch: *batch,
                            ubatch: *ubatch,
                            cache_type_k,
                            cache_type_v,
                            mmap: *mmap,
                            mlock: *mlock,
                        });
                    }
                }
            }
        }
    }
    candidates
}

fn default_model_fit_u32(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    field: TuneField,
) -> Option<u32> {
    recommended_u32(&prepared.plan, field).or_else(|| {
        preserved_model_fit_u32(
            benchmark_model_entry(request.config, prepared),
            request.config.defaults.as_ref(),
            field,
        )
    })
}

fn benchmark_model_entry<'a>(
    config: &'a mesh_llm_config::MeshConfig,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Option<&'a mesh_llm_config::ModelConfigEntry> {
    config
        .models
        .get(prepared.target.config_matches.first()?.row_index)
}

fn run_trial(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    index: usize,
    candidate: TuneBenchmarkCandidate,
) -> TuneBenchmarkTrial {
    match run_trial_inner(request, prepared, index, &candidate) {
        Ok(success) => success,
        Err(error) => TuneBenchmarkTrial {
            candidate,
            status: TuneBenchmarkTrialStatus::Failed,
            completion_tokens: None,
            elapsed_ms: None,
            decode_tok_s: None,
            timings: None,
            log_path: None,
            error: Some(error.to_string()),
        },
    }
}

fn run_trial_inner(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    index: usize,
    candidate: &TuneBenchmarkCandidate,
) -> anyhow::Result<TuneBenchmarkTrial> {
    anyhow::ensure!(request.max_tokens > 0, "--max-tokens must be greater than zero");
    let mut timings = TrialTimingRecorder::new();
    let setup_started = std::time::Instant::now();
    let trial_dir = create_trial_dir(prepared, index)?;
    let config_path = trial_dir.join("config.toml");
    let log_path = trial_dir.join("serve.log");
    std::fs::write(&config_path, trial_config(prepared, candidate)?)?;

    let port = reserve_local_port()?;
    let console = reserve_local_port()?;
    let mut child = TrialChild::spawn(&config_path, &log_path, port, console)?;
    let request_timeout = std::time::Duration::from_secs(request.request_timeout_secs.max(1));
    let client = reqwest::blocking::Client::builder()
        .timeout(request_timeout)
        .build()?;
    timings.setup_ms = elapsed_ms_since(setup_started);

    let readiness_started = std::time::Instant::now();
    let readiness_result = wait_for_trial_ready(
        &client,
        &mut child,
        port,
        request.prompt,
        request.startup_timeout_secs,
        request_timeout,
        &mut timings.readiness_attempts,
    );
    timings.readiness_ms = elapsed_ms_since(readiness_started);
    if let Err(error) = readiness_result {
        return Ok(finish_failed_trial(
            candidate,
            &log_path,
            &mut timings,
            &mut child,
            error,
        ));
    }

    let started = std::time::Instant::now();
    let response_result = send_chat_request_with_watchdog(
        &client,
        &mut child,
        port,
        request.prompt,
        request.max_tokens,
        request_timeout,
    );
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    timings.request_ms = Some(elapsed_ms);
    let response = match response_result {
        Ok(response) => response,
        Err(error) => {
            return Ok(finish_failed_trial(
                candidate,
                &log_path,
                &mut timings,
                &mut child,
                error,
            ));
        }
    };
    let completion_tokens = match response_completion_tokens(&response) {
        Some(tokens) => tokens,
        None => {
            return Ok(finish_failed_trial(
                candidate,
                &log_path,
                &mut timings,
                &mut child,
                anyhow::anyhow!("chat completion response did not include completion_tokens"),
            ));
        }
    };
    if completion_tokens == 0 {
        return Ok(finish_failed_trial(
            candidate,
            &log_path,
            &mut timings,
            &mut child,
            anyhow::anyhow!("chat completion returned zero completion tokens"),
        ));
    }
    let decode_tok_s = completion_tokens as f64 / (elapsed_ms / 1000.0);
    record_shutdown(&mut child, &mut timings);

    Ok(TuneBenchmarkTrial {
        candidate: candidate.clone(),
        status: TuneBenchmarkTrialStatus::Succeeded,
        completion_tokens: Some(completion_tokens),
        elapsed_ms: Some(elapsed_ms),
        decode_tok_s: Some(decode_tok_s),
        timings: Some(timings.snapshot()),
        log_path: Some(log_path.display().to_string()),
        error: None,
    })
}

struct TrialTimingRecorder {
    trial_started: std::time::Instant,
    setup_ms: f64,
    readiness_ms: f64,
    request_ms: Option<f64>,
    shutdown_ms: Option<f64>,
    readiness_attempts: u32,
}

impl TrialTimingRecorder {
    fn new() -> Self {
        Self {
            trial_started: std::time::Instant::now(),
            setup_ms: 0.0,
            readiness_ms: 0.0,
            request_ms: None,
            shutdown_ms: None,
            readiness_attempts: 0,
        }
    }

    fn snapshot(&self) -> TuneBenchmarkTimingStats {
        TuneBenchmarkTimingStats {
            total_ms: elapsed_ms_since(self.trial_started),
            setup_ms: self.setup_ms,
            readiness_ms: self.readiness_ms,
            request_ms: self.request_ms,
            shutdown_ms: self.shutdown_ms,
            readiness_attempts: self.readiness_attempts,
        }
    }
}

fn elapsed_ms_since(started: std::time::Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn record_shutdown(child: &mut TrialChild, timings: &mut TrialTimingRecorder) {
    let shutdown_started = std::time::Instant::now();
    child.shutdown();
    timings.shutdown_ms = Some(elapsed_ms_since(shutdown_started));
}

fn finish_failed_trial(
    candidate: &TuneBenchmarkCandidate,
    log_path: &std::path::Path,
    timings: &mut TrialTimingRecorder,
    child: &mut TrialChild,
    error: impl std::fmt::Display,
) -> TuneBenchmarkTrial {
    record_shutdown(child, timings);
    failed_trial_with_evidence(candidate, log_path, timings.snapshot(), error)
}

fn failed_trial_with_evidence(
    candidate: &TuneBenchmarkCandidate,
    log_path: &std::path::Path,
    timings: TuneBenchmarkTimingStats,
    error: impl std::fmt::Display,
) -> TuneBenchmarkTrial {
    TuneBenchmarkTrial {
        candidate: candidate.clone(),
        status: TuneBenchmarkTrialStatus::Failed,
        completion_tokens: None,
        elapsed_ms: timings.request_ms,
        decode_tok_s: None,
        timings: Some(timings),
        log_path: Some(log_path.display().to_string()),
        error: Some(error.to_string()),
    }
}

struct TrialChild {
    child: std::process::Child,
}

impl TrialChild {
    fn spawn(
        config_path: &std::path::Path,
        log_path: &std::path::Path,
        port: u16,
        console: u16,
    ) -> anyhow::Result<Self> {
        let exe = std::env::current_exe()?;
        let log = std::fs::File::create(log_path)?;
        let stderr = log.try_clone()?;
        let child = std::process::Command::new(exe)
            .arg("--config")
            .arg(config_path)
            .arg("--port")
            .arg(port.to_string())
            .arg("--console")
            .arg(console.to_string())
            .arg("--log-format")
            .arg("json")
            .arg("--headless")
            .arg("serve")
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(stderr))
            .spawn()?;
        Ok(Self { child })
    }

    fn shutdown(&mut self) {
        terminate_child(&mut self.child);
    }
}

impl Drop for TrialChild {
    fn drop(&mut self) {
        terminate_child(&mut self.child);
    }
}

fn terminate_child(child: &mut std::process::Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    #[cfg(unix)]
    {
        let _ = std::process::Command::new("kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .status();
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while std::time::Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_for_trial_ready(
    client: &reqwest::blocking::Client,
    child: &mut TrialChild,
    port: u16,
    prompt: &str,
    startup_timeout_secs: u64,
    request_timeout: std::time::Duration,
    readiness_attempts: &mut u32,
) -> anyhow::Result<()> {
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(startup_timeout_secs.max(1));
    let mut last_error = String::new();
    while std::time::Instant::now() < deadline {
        if let Some(status) = child.child.try_wait()? {
            anyhow::bail!("trial server exited before readiness: {status}");
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let attempt_timeout = std::cmp::min(request_timeout, remaining);
        *readiness_attempts += 1;
        match send_chat_request_with_watchdog(client, child, port, prompt, 1, attempt_timeout) {
            Ok(_) => return Ok(()),
            Err(error) => last_error = error.to_string(),
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    anyhow::bail!("trial server did not become ready: {last_error}");
}

fn send_chat_request_with_watchdog(
    client: &reqwest::blocking::Client,
    child: &mut TrialChild,
    port: u16,
    prompt: &str,
    max_tokens: u32,
    timeout: std::time::Duration,
) -> anyhow::Result<serde_json::Value> {
    let client = client.clone();
    let prompt = prompt.to_string();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(send_chat_request(&client, port, &prompt, max_tokens));
    });

    match receiver.recv_timeout(timeout.max(std::time::Duration::from_secs(1))) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            child.shutdown();
            anyhow::bail!(
                "chat completion exceeded request timeout of {}s",
                timeout.as_secs().max(1)
            );
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("chat completion worker exited without a response")
        }
    }
}

fn send_chat_request(
    client: &reqwest::blocking::Client,
    port: u16,
    prompt: &str,
    max_tokens: u32,
) -> anyhow::Result<serde_json::Value> {
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": false
        }))
        .send()?;
    let status = response.status();
    let body: serde_json::Value = response.json()?;
    if !status.is_success() {
        anyhow::bail!("chat completion failed with HTTP {status}: {body}");
    }
    Ok(body)
}

fn response_completion_tokens(response: &serde_json::Value) -> Option<u64> {
    response
        .get("usage")?
        .get("completion_tokens")?
        .as_u64()
}

fn trial_config(
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    candidate: &TuneBenchmarkCandidate,
) -> anyhow::Result<String> {
    let mut doc = toml_edit::DocumentMut::new();
    doc["version"] = toml_edit::value(1);

    let mut table = toml_edit::Table::new();
    table["model"] = toml_edit::value(prepared.target.resolved_path.display().to_string());
    crate::gpus::tune_apply::apply_config_edits(&mut table, &prepared.plan.config_edits())?;
    apply_candidate_overrides(&mut table, candidate)?;

    let mut models = toml_edit::ArrayOfTables::new();
    models.push(table);
    doc["models"] = toml_edit::Item::ArrayOfTables(models);
    Ok(doc.to_string())
}

fn apply_candidate_overrides(
    table: &mut toml_edit::Table,
    candidate: &TuneBenchmarkCandidate,
) -> anyhow::Result<()> {
    let model_fit = ensure_trial_subtable(table, "model_fit")?;
    model_fit["ctx_size"] = toml_edit::value(i64::from(candidate.ctx_size));
    model_fit["batch"] = toml_edit::value(i64::from(candidate.batch));
    model_fit["ubatch"] = toml_edit::value(i64::from(candidate.ubatch));
    model_fit["cache_type_k"] = toml_edit::value(render_cache_type(candidate.cache_type_k));
    model_fit["cache_type_v"] = toml_edit::value(render_cache_type(candidate.cache_type_v));
    let hardware = ensure_trial_subtable(table, "hardware")?;
    hardware["mmap"] = toml_edit::value(render_bool_or_auto(candidate.mmap));
    hardware["mlock"] = toml_edit::value(candidate.mlock);
    Ok(())
}

fn ensure_trial_subtable<'a>(
    table: &'a mut toml_edit::Table,
    key: &str,
) -> anyhow::Result<&'a mut toml_edit::Table> {
    if !table.contains_key(key) {
        table[key] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    table[key]
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("config key `models[].{key}` is not a TOML table"))
}

fn create_trial_dir(
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    index: usize,
) -> anyhow::Result<std::path::PathBuf> {
    let mut dir = std::env::current_dir()?;
    dir.push("target");
    dir.push("gpu-tune");
    dir.push(sanitize_path_component(&prepared.target.canonical_model_ref));
    dir.push(format!(
        "{}-{index}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn reserve_local_port() -> anyhow::Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn default_context_sizes(planned: u32) -> Vec<u32> {
    let mut values = [4096, 8192, 16_384, 32_768, 65_536, planned]
        .into_iter()
        .filter(|value| *value > 0 && *value <= planned.max(4096))
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn unique_positive(values: &[u32]) -> Vec<u32> {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn benchmark_mmap_values(
    requested: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto],
    _plan: &TunePlan,
) -> Vec<TuneBoolOrAutoValue> {
    if requested.is_empty() {
        return vec![
            TuneBoolOrAutoValue::Auto,
            TuneBoolOrAutoValue::Enabled,
            TuneBoolOrAutoValue::Disabled,
        ];
    }
    let mut values = requested
        .iter()
        .copied()
        .map(|value| match value {
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Auto => TuneBoolOrAutoValue::Auto,
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Enabled => {
                TuneBoolOrAutoValue::Enabled
            }
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled => {
                TuneBoolOrAutoValue::Disabled
            }
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|value| match value {
        TuneBoolOrAutoValue::Auto => 0,
        TuneBoolOrAutoValue::Enabled => 1,
        TuneBoolOrAutoValue::Disabled => 2,
    });
    values.dedup();
    values
}

fn benchmark_mlock_values(
    requested: &[mesh_llm_cli::benchmark::BenchmarkBool],
    plan: &TunePlan,
) -> Vec<bool> {
    if requested.is_empty() {
        return if recommended_bool(plan, TuneField::Mlock).unwrap_or(false) {
            vec![false, true]
        } else {
            vec![false]
        };
    }
    let mut values = requested
        .iter()
        .copied()
        .map(|value| match value {
            mesh_llm_cli::benchmark::BenchmarkBool::Enabled => true,
            mesh_llm_cli::benchmark::BenchmarkBool::Disabled => false,
        })
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values
}

fn recommended_u32(plan: &TunePlan, field: TuneField) -> Option<u32> {
    plan.field_statuses.iter().find_map(|status| match status {
        TuneFieldStatus::Applied { recommendation, .. }
        | TuneFieldStatus::ReportOnly { recommendation, .. }
            if recommendation.field == field =>
        {
            match recommendation.value {
                TuneRecommendedValue::ContextSize(value)
                | TuneRecommendedValue::Batch(value)
                | TuneRecommendedValue::Ubatch(value) => Some(value),
                _ => None,
            }
        }
        _ => None,
    })
}

fn recommended_bool(plan: &TunePlan, field: TuneField) -> Option<bool> {
    plan.field_statuses.iter().find_map(|status| match status {
        TuneFieldStatus::Applied { recommendation, .. }
        | TuneFieldStatus::ReportOnly { recommendation, .. }
            if recommendation.field == field =>
        {
            match recommendation.value {
                TuneRecommendedValue::Bool(value) => Some(value),
                _ => None,
            }
        }
        _ => None,
    })
}

fn recommended_cache_type(plan: &TunePlan, field: TuneField) -> Option<TuneKvCacheType> {
    plan.field_statuses.iter().find_map(|status| match status {
        TuneFieldStatus::Applied { recommendation, .. }
        | TuneFieldStatus::ReportOnly { recommendation, .. }
            if recommendation.field == field =>
        {
            match recommendation.value {
                TuneRecommendedValue::KvCacheType(value) => Some(value),
                _ => None,
            }
        }
        _ => None,
    })
}

fn render_bool_or_auto(value: TuneBoolOrAutoValue) -> toml_edit::Value {
    match value {
        TuneBoolOrAutoValue::Enabled => toml_edit::Value::from(true),
        TuneBoolOrAutoValue::Disabled => toml_edit::Value::from(false),
        TuneBoolOrAutoValue::Auto => toml_edit::Value::from("auto"),
    }
}

fn render_cache_type(value: TuneKvCacheType) -> &'static str {
    match value {
        TuneKvCacheType::F16 => "f16",
        TuneKvCacheType::Q8_0 => "q8_0",
        TuneKvCacheType::Q4_0 => "q4_0",
    }
}

fn plan_has_errors(plan: &TunePlan) -> bool {
    plan.field_statuses
        .iter()
        .any(|status| matches!(status, TuneFieldStatus::Error { .. }))
        || plan
            .diagnostics
            .iter()
            .any(|diagnostic| matches!(diagnostic.severity, TuneDiagnosticSeverity::Error))
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    use crate::gpus::tune_apply::PreparedTunePlan;
    use crate::gpus::tune_resolver::{
        ConfigModelMatch, LocalTargetSource, ResolvedTuneTarget, TuneTargetSelection,
    };

    #[test]
    fn trial_config_renders_string_paths_and_hardware_edits() {
        let prepared = PreparedTunePlan::new(
            ResolvedTuneTarget {
                requested_input: "model".to_string(),
                canonical_model_ref: "model".to_string(),
                resolved_path: std::path::PathBuf::from("/tmp/model with spaces.gguf"),
                local_source: LocalTargetSource::FilesystemPath {
                    synthetic_model_ref: "model".to_string(),
                },
                config_matches: Vec::new(),
                selection: TuneTargetSelection::Explicit { configured: false },
            },
            TunePlan {
                target: TuneTarget {
                    requested: "model".to_string(),
                    resolved: Some("/tmp/model with spaces.gguf".to_string()),
                    config_model_ref: None,
                    derived_profile: None,
                },
                apply_mode: TuneApplyMode::Review,
                field_statuses: vec![
                    TuneFieldStatus::Applied {
                        recommendation: TuneRecommendation {
                            field: TuneField::GpuLayers,
                            value: TuneRecommendedValue::GpuLayers(TuneGpuLayersValue::All),
                            rationale: "test".to_string(),
                        },
                        edit: TuneConfigEdit::SetHardwareGpuLayers(TuneGpuLayersValue::All),
                    },
                    TuneFieldStatus::Applied {
                        recommendation: TuneRecommendation {
                            field: TuneField::FitTargetMib,
                            value: TuneRecommendedValue::FitTargetMib(60_000),
                            rationale: "test".to_string(),
                        },
                        edit: TuneConfigEdit::SetHardwareFitTargetMib(60_000),
                    },
                ],
                diagnostics: Vec::new(),
            },
        );
        let candidate = TuneBenchmarkCandidate {
            ctx_size: 4096,
            batch: 2048,
            ubatch: 1024,
            cache_type_k: TuneKvCacheType::Q8_0,
            cache_type_v: TuneKvCacheType::Q8_0,
            mmap: TuneBoolOrAutoValue::Disabled,
            mlock: true,
        };

        let rendered = trial_config(&prepared, &candidate).expect("trial config renders");
        let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
        let model = parsed.models.first().expect("model row exists");

        assert_eq!(model.model, "/tmp/model with spaces.gguf");
        assert_eq!(
            model.model_fit
                .as_ref()
                .and_then(|model_fit| model_fit.ctx_size),
            Some(4096)
        );
        assert!(matches!(
            model.hardware.as_ref().and_then(|hardware| hardware.gpu_layers.as_ref()),
            Some(mesh_llm_config::IntegerOrString::Integer(-1))
        ));
        assert_eq!(
            model
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.fit_target_mib),
            Some(60_000)
        );
        assert_eq!(
            model.hardware.as_ref().and_then(|hardware| hardware.mmap.as_ref()),
            Some(&mesh_llm_config::BoolOrAuto::Bool(false))
        );
        assert_eq!(
            model.hardware.as_ref().and_then(|hardware| hardware.mlock),
            Some(true)
        );
    }

    #[test]
    fn benchmark_candidates_sweep_mmap_and_available_mlock_independently() {
        let prepared = PreparedTunePlan::new(
            ResolvedTuneTarget {
                requested_input: "model".to_string(),
                canonical_model_ref: "model".to_string(),
                resolved_path: std::path::PathBuf::from("/tmp/model.gguf"),
                local_source: LocalTargetSource::FilesystemPath {
                    synthetic_model_ref: "model".to_string(),
                },
                config_matches: Vec::new(),
                selection: TuneTargetSelection::Explicit { configured: false },
            },
            TunePlan {
                target: TuneTarget {
                    requested: "model".to_string(),
                    resolved: Some("/tmp/model.gguf".to_string()),
                    config_model_ref: None,
                    derived_profile: None,
                },
                apply_mode: TuneApplyMode::Review,
                field_statuses: vec![TuneFieldStatus::Applied {
                    recommendation: TuneRecommendation {
                        field: TuneField::Mlock,
                        value: TuneRecommendedValue::Bool(true),
                        rationale: "test".to_string(),
                    },
                    edit: TuneConfigEdit::SetHardwareMlock(true),
                }],
                diagnostics: Vec::new(),
            },
        );
        let prepared = [prepared];
        let config = mesh_llm_config::MeshConfig::default();
        let request = TuneBenchmarkRunRequest {
            config: &config,
            prepared: &prepared,
            ctx_sizes: &[4096],
            batch_sizes: &[1024],
            ubatch_sizes: &[256],
            mmap_values: &[],
            mlock_values: &[],
            throughput_tolerance_pct: 3.0,
            max_tokens: 32,
            startup_timeout_secs: 5,
            request_timeout_secs: 5,
            prompt: "hello",
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);

        assert_eq!(candidates.len(), 6);
        assert!(candidates.iter().any(|candidate| {
            candidate.mmap == TuneBoolOrAutoValue::Auto && !candidate.mlock
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.mmap == TuneBoolOrAutoValue::Enabled && candidate.mlock
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.mmap == TuneBoolOrAutoValue::Disabled && candidate.mlock
        }));
    }

    #[test]
    fn benchmark_candidates_default_to_preserved_config_model_fit() {
        let config = mesh_llm_config::MeshConfig {
            models: vec![mesh_llm_config::ModelConfigEntry {
                model: "model".to_string(),
                model_fit: Some(mesh_llm_config::ModelFitConfig {
                    ctx_size: Some(131_072),
                    batch: Some(2048),
                    ubatch: Some(1024),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let prepared = PreparedTunePlan::new(
            ResolvedTuneTarget {
                requested_input: "model".to_string(),
                canonical_model_ref: "model".to_string(),
                resolved_path: std::path::PathBuf::from("/tmp/model.gguf"),
                local_source: LocalTargetSource::FilesystemPath {
                    synthetic_model_ref: "model".to_string(),
                },
                config_matches: vec![ConfigModelMatch {
                    row_index: 0,
                    configured_model: "model".to_string(),
                }],
                selection: TuneTargetSelection::Configured,
            },
            TunePlan {
                target: TuneTarget {
                    requested: "model".to_string(),
                    resolved: Some("/tmp/model.gguf".to_string()),
                    config_model_ref: Some("model".to_string()),
                    derived_profile: None,
                },
                apply_mode: TuneApplyMode::Review,
                field_statuses: Vec::new(),
                diagnostics: Vec::new(),
            },
        );
        let prepared = [prepared];
        let request = TuneBenchmarkRunRequest {
            config: &config,
            prepared: &prepared,
            ctx_sizes: &[],
            batch_sizes: &[],
            ubatch_sizes: &[],
            mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
            mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
            throughput_tolerance_pct: 10.0,
            max_tokens: 32,
            startup_timeout_secs: 5,
            request_timeout_secs: 5,
            prompt: "hello",
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);

        assert_eq!(candidates.len(), 6);
        assert!(
            candidates.iter().all(|candidate| candidate.batch == 2048),
            "configured batch should be used when --batch-sizes is omitted"
        );
        assert!(
            candidates.iter().all(|candidate| candidate.ubatch == 1024),
            "configured ubatch should be used when --ubatch-sizes is omitted"
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.ctx_size == 131_072),
            "configured context should anchor the default context ladder"
        );
    }

    #[test]
    fn selection_prefers_larger_context_within_throughput_tolerance() {
        let trials = vec![
            succeeded_trial(8192, 18.65, 2000.0),
            succeeded_trial(262_144, 18.23, 2100.0),
            succeeded_trial(65_536, 16.0, 2200.0),
        ];

        let selection = select_benchmark_trials(&trials, 3.0);

        assert_eq!(
            selection
                .raw_best
                .as_ref()
                .expect("raw best")
                .candidate
                .ctx_size,
            8192
        );
        assert_eq!(
            selection
                .recommended
                .as_ref()
                .expect("recommended")
                .candidate
                .ctx_size,
            262_144
        );
        assert!(
            selection
                .reason
                .as_deref()
                .expect("selection reason")
                .contains("within 3.00%")
        );
    }

    #[test]
    fn selection_keeps_pareto_frontier_tradeoffs() {
        let trials = vec![
            succeeded_trial(4096, 20.0, 2000.0),
            succeeded_trial(8192, 19.0, 2000.0),
            succeeded_trial(4096, 18.0, 1900.0),
            succeeded_trial(16_384, 16.0, 2000.0),
        ];

        let selection = select_benchmark_trials(&trials, 1.0);
        let frontier_contexts = selection
            .pareto_frontier
            .iter()
            .map(|trial| trial.candidate.ctx_size)
            .collect::<Vec<_>>();

        assert_eq!(frontier_contexts, vec![16_384, 8192, 4096]);
        assert!(
            !selection
                .pareto_frontier
                .iter()
                .any(|trial| trial.decode_tok_s == Some(18.0)),
            "dominated lower-throughput 4096 ctx trial should be excluded"
        );
    }

    fn succeeded_trial(ctx_size: u32, decode_tok_s: f64, request_ms: f64) -> TuneBenchmarkTrial {
        TuneBenchmarkTrial {
            candidate: TuneBenchmarkCandidate {
                ctx_size,
                batch: 2048,
                ubatch: 1024,
                cache_type_k: TuneKvCacheType::Q8_0,
                cache_type_v: TuneKvCacheType::Q8_0,
                mmap: TuneBoolOrAutoValue::Disabled,
                mlock: false,
            },
            status: TuneBenchmarkTrialStatus::Succeeded,
            completion_tokens: Some(128),
            elapsed_ms: Some(request_ms),
            decode_tok_s: Some(decode_tok_s),
            timings: Some(TuneBenchmarkTimingStats {
                total_ms: request_ms + 1000.0,
                setup_ms: 10.0,
                readiness_ms: 900.0,
                request_ms: Some(request_ms),
                shutdown_ms: Some(90.0),
                readiness_attempts: 3,
            }),
            log_path: None,
            error: None,
        }
    }
}
