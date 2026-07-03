pub(crate) struct TuneBenchmarkRunRequest<'a> {
    pub(crate) config: &'a mesh_llm_config::MeshConfig,
    pub(crate) prepared: &'a [crate::gpus::tune_apply::PreparedTunePlan],
    pub(crate) ctx_sizes: &'a [u32],
    pub(crate) batch_sizes: &'a [u32],
    pub(crate) ubatch_sizes: &'a [u32],
    pub(crate) mmap_values: &'a [mesh_llm_cli::benchmark::BenchmarkBoolOrAuto],
    pub(crate) mlock_values: &'a [mesh_llm_cli::benchmark::BenchmarkBool],
    pub(crate) speculative_types: &'a [mesh_llm_cli::benchmark::BenchmarkSpeculativeType],
    pub(crate) no_speculative_tune: bool,
    pub(crate) spec_draft_models: &'a [std::path::PathBuf],
    pub(crate) spec_draft_max_tokens: &'a [u32],
    pub(crate) spec_draft_min_tokens: &'a [u32],
    pub(crate) spec_ngram_min: &'a [u32],
    pub(crate) spec_ngram_max: &'a [u32],
    pub(crate) throughput_tolerance_pct: f64,
    pub(crate) max_tokens: u32,
    pub(crate) startup_timeout_secs: u64,
    pub(crate) request_timeout_secs: u64,
    pub(crate) debug_telemetry: bool,
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
        .map(|(index, candidate)| {
            run_trial_with_progress(request, prepared, index, total, candidate)
        })
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
    let default_ctx = default_model_fit_u32(request, prepared, TuneField::CtxSize).unwrap_or(8192);
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
    let cache_type_k = recommended_cache_type(&prepared.plan, TuneField::CacheTypeK)
        .unwrap_or(TuneKvCacheType::Q8_0);
    let cache_type_v =
        recommended_cache_type(&prepared.plan, TuneField::CacheTypeV).unwrap_or(cache_type_k);
    let mmap_values = benchmark_mmap_values(request.mmap_values, &prepared.plan);
    let mlock_values = benchmark_mlock_values(request.mlock_values, &prepared.plan);
    let speculative_values = benchmark_speculative_values(request, prepared);

    let mut candidates = Vec::new();
    for ctx_size in contexts {
        for batch in &batches {
            for ubatch in &ubatches {
                if *ubatch > *batch {
                    continue;
                }
                for mmap in &mmap_values {
                    for mlock in &mlock_values {
                        for speculative in &speculative_values {
                            candidates.push(TuneBenchmarkCandidate {
                                ctx_size,
                                batch: *batch,
                                ubatch: *ubatch,
                                cache_type_k,
                                cache_type_v,
                                mmap: *mmap,
                                mlock: *mlock,
                                speculative: speculative.clone(),
                            });
                        }
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
    anyhow::ensure!(
        request.max_tokens > 0,
        "--max-tokens must be greater than zero"
    );
    let mut timings = TrialTimingRecorder::new();
    let setup_started = std::time::Instant::now();
    let trial_dir = create_trial_dir(prepared, index)?;
    let config_path = trial_dir.join("config.toml");
    let log_path = trial_dir.join("serve.log");
    std::fs::write(&config_path, trial_config(prepared, candidate)?)?;

    let port = reserve_local_port()?;
    let console = reserve_local_port()?;
    let mut child = TrialChild::spawn(
        &config_path,
        &log_path,
        port,
        console,
        request.debug_telemetry,
    )?;
    let request_timeout = std::time::Duration::from_secs(request.request_timeout_secs.max(1));
    let client = reqwest::blocking::Client::builder()
        .timeout(request_timeout)
        .build()?;
    timings.setup_ms = elapsed_ms_since(setup_started);

    let readiness_started = std::time::Instant::now();
    let readiness_result = wait_for_trial_ready(TrialReadinessWait {
        client: &client,
        child: &mut child,
        log_path: &log_path,
        port,
        prompt: request.prompt,
        startup_timeout_secs: request.startup_timeout_secs,
        request_timeout,
        readiness_attempts: &mut timings.readiness_attempts,
    });
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
        debug_telemetry: bool,
    ) -> anyhow::Result<Self> {
        let exe = std::env::current_exe()?;
        let log = std::fs::File::create(log_path)?;
        let stderr = log.try_clone()?;
        let child = build_trial_child_command(&exe, config_path, port, console, debug_telemetry)
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

fn build_trial_child_command(
    exe: &std::path::Path,
    config_path: &std::path::Path,
    port: u16,
    console: u16,
    debug_telemetry: bool,
) -> std::process::Command {
    let mut command = std::process::Command::new(exe);
    if debug_telemetry {
        command.arg("--debug").env("SKIPPY_TELEMETRY_STDERR", "1");
    }
    command
        .arg("--config")
        .arg(config_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--console")
        .arg(console.to_string())
        .arg("--log-format")
        .arg("json")
        .arg("--headless")
        .arg("serve");
    command
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

struct TrialReadinessWait<'a> {
    client: &'a reqwest::blocking::Client,
    child: &'a mut TrialChild,
    log_path: &'a std::path::Path,
    port: u16,
    prompt: &'a str,
    startup_timeout_secs: u64,
    request_timeout: std::time::Duration,
    readiness_attempts: &'a mut u32,
}

fn wait_for_trial_ready(wait: TrialReadinessWait<'_>) -> anyhow::Result<()> {
    let deadline = std::time::Instant::now()
        + std::time::Duration::from_secs(wait.startup_timeout_secs.max(1));
    let mut last_error = String::new();
    while std::time::Instant::now() < deadline {
        if let Some(status) = wait.child.child.try_wait()? {
            anyhow::bail!("trial server exited before readiness: {status}");
        }
        if let Some(error) = trial_startup_failure_from_log(wait.log_path) {
            anyhow::bail!("trial startup failed: {error}");
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let attempt_timeout = std::cmp::min(wait.request_timeout, remaining);
        *wait.readiness_attempts += 1;
        match send_chat_request_with_watchdog(
            wait.client,
            wait.child,
            wait.port,
            wait.prompt,
            1,
            attempt_timeout,
        ) {
            Ok(_) => return Ok(()),
            Err(error) => last_error = error.to_string(),
        }
        if let Some(error) = trial_startup_failure_from_log(wait.log_path) {
            anyhow::bail!("trial startup failed: {error}");
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    anyhow::bail!("trial server did not become ready: {last_error}");
}

fn trial_startup_failure_from_log(log_path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(log_path).ok()?;
    contents
        .lines()
        .rev()
        .take(200)
        .find_map(trial_startup_failure_from_log_line)
}

fn trial_startup_failure_from_log_line(line: &str) -> Option<String> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(line)
        && let Some(message) = value.get("message").and_then(|value| value.as_str())
        && message.contains("Failed to start model")
    {
        return Some(message.to_string());
    }
    line.contains("Failed to start model")
        .then(|| line.trim().to_string())
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
    response.get("usage")?.get("completion_tokens")?.as_u64()
}

fn trial_config(
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    candidate: &TuneBenchmarkCandidate,
) -> anyhow::Result<String> {
    let mut doc = toml_edit::DocumentMut::new();
    doc["version"] = toml_edit::value(1);

    let mut table = toml_edit::Table::new();
    table["model"] = toml_edit::value(trial_model_ref(prepared));
    crate::gpus::tune_apply::apply_config_edits(&mut table, &prepared.plan.config_edits())?;
    apply_resolved_model_path(&mut table, prepared)?;
    apply_candidate_overrides(&mut table, candidate)?;

    let mut models = toml_edit::ArrayOfTables::new();
    models.push(table);
    doc["models"] = toml_edit::Item::ArrayOfTables(models);
    Ok(doc.to_string())
}

fn trial_model_ref(prepared: &crate::gpus::tune_apply::PreparedTunePlan) -> String {
    match &prepared.target.local_source {
        crate::gpus::tune_resolver::LocalTargetSource::HuggingFaceCache { canonical_ref } => {
            canonical_ref.clone()
        }
        crate::gpus::tune_resolver::LocalTargetSource::FilesystemPath { .. } => {
            prepared.target.resolved_path.display().to_string()
        }
    }
}

fn apply_resolved_model_path(
    table: &mut toml_edit::Table,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> anyhow::Result<()> {
    let hardware = ensure_trial_subtable(table, "hardware")?;
    hardware["model_path"] = toml_edit::value(prepared.target.resolved_path.display().to_string());
    Ok(())
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
    apply_speculative_overrides(table, &candidate.speculative)?;
    Ok(())
}

fn apply_speculative_overrides(
    table: &mut toml_edit::Table,
    speculative: &TuneBenchmarkSpeculativeCandidate,
) -> anyhow::Result<()> {
    let spec_table = ensure_trial_subtable(table, "speculative")?;
    match speculative {
        TuneBenchmarkSpeculativeCandidate::Disabled => {
            spec_table["strategy"] = toml_edit::value("disabled");
            spec_table["mode"] = toml_edit::value("disabled");
        }
        TuneBenchmarkSpeculativeCandidate::Mtp {
            draft_model_path,
            draft_max_tokens,
            draft_min_tokens,
        } => {
            spec_table["strategy"] = toml_edit::value("mtp");
            spec_table["mode"] = toml_edit::value("auto");
            if let Some(draft_model_path) = draft_model_path {
                spec_table["draft_model_path"] = toml_edit::value(draft_model_path.as_str());
                spec_table["draft_selection_policy"] = toml_edit::value("manual");
                spec_table["pairing_fault"] = toml_edit::value("fail_closed");
            }
            spec_table["draft_max_tokens"] = toml_edit::value(i64::from(*draft_max_tokens));
            spec_table["draft_min_tokens"] = toml_edit::value(i64::from(*draft_min_tokens));
        }
        TuneBenchmarkSpeculativeCandidate::Draft {
            draft_model_path,
            draft_max_tokens,
            draft_min_tokens,
        } => {
            spec_table["strategy"] = toml_edit::value("disabled");
            spec_table["mode"] = toml_edit::value("draft");
            spec_table["draft_model_path"] = toml_edit::value(draft_model_path.as_str());
            spec_table["draft_selection_policy"] = toml_edit::value("manual");
            spec_table["pairing_fault"] = toml_edit::value("fail_closed");
            spec_table["draft_max_tokens"] = toml_edit::value(i64::from(*draft_max_tokens));
            if let Some(draft_min_tokens) = draft_min_tokens {
                spec_table["draft_min_tokens"] = toml_edit::value(i64::from(*draft_min_tokens));
            }
        }
        TuneBenchmarkSpeculativeCandidate::Ngram {
            ngram_min,
            ngram_max,
        } => {
            spec_table["strategy"] = toml_edit::value("disabled");
            spec_table["mode"] = toml_edit::value("ngram");
            spec_table["ngram_min"] = toml_edit::value(i64::from(*ngram_min));
            spec_table["ngram_max"] = toml_edit::value(i64::from(*ngram_max));
        }
    }
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
    dir.push(sanitize_path_component(
        &prepared.target.canonical_model_ref,
    ));
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
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Enabled => TuneBoolOrAutoValue::Enabled,
            mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled => TuneBoolOrAutoValue::Disabled,
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

fn benchmark_speculative_values(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Vec<TuneBenchmarkSpeculativeCandidate> {
    if request.no_speculative_tune {
        return vec![TuneBenchmarkSpeculativeCandidate::Disabled];
    }
    let requested = requested_speculative_types(request.speculative_types);
    let mut candidates = Vec::new();
    for requested_type in requested {
        match requested_type {
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Auto => {
                push_auto_speculative_candidates(&mut candidates, request, prepared);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Disabled => {
                candidates.push(TuneBenchmarkSpeculativeCandidate::Disabled);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Mtp => {
                push_mtp_speculative_candidates(&mut candidates, request, prepared);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Draft => {
                push_draft_speculative_candidates(&mut candidates, request, prepared);
            }
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Ngram => {
                push_ngram_speculative_candidates(&mut candidates, request);
            }
        }
    }
    dedup_speculative_candidates(candidates)
}

fn requested_speculative_types(
    requested: &[mesh_llm_cli::benchmark::BenchmarkSpeculativeType],
) -> Vec<mesh_llm_cli::benchmark::BenchmarkSpeculativeType> {
    if requested.is_empty() {
        return vec![mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Auto];
    }
    let mut values = requested.to_vec();
    values.sort_by_key(|value| speculative_type_priority(*value));
    values.dedup();
    values
}

fn speculative_type_priority(value: mesh_llm_cli::benchmark::BenchmarkSpeculativeType) -> u8 {
    match value {
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Auto => 0,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Mtp => 1,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Draft => 2,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Ngram => 3,
        mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Disabled => 4,
    }
}

fn push_auto_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) {
    if looks_like_mtp_target(prepared) {
        push_mtp_speculative_candidates(candidates, request, prepared);
    }
    push_draft_speculative_candidates(candidates, request, prepared);
    push_ngram_speculative_candidates(candidates, request);
    candidates.push(TuneBenchmarkSpeculativeCandidate::Disabled);
}

fn push_mtp_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) {
    let draft_models = discover_draft_model_candidates(request, prepared);
    let draft_models = if draft_models.is_empty() {
        vec![None]
    } else {
        draft_models.into_iter().map(Some).collect()
    };
    let max_tokens = positive_or_default(request.spec_draft_max_tokens, &[2, 3, 4]);
    let min_tokens = values_or_default_allow_zero(request.spec_draft_min_tokens, &[0]);
    for draft_model_path in draft_models {
        for draft_max_tokens in &max_tokens {
            for draft_min_tokens in &min_tokens {
                if *draft_min_tokens <= *draft_max_tokens {
                    candidates.push(TuneBenchmarkSpeculativeCandidate::Mtp {
                        draft_model_path: draft_model_path.clone(),
                        draft_max_tokens: *draft_max_tokens,
                        draft_min_tokens: *draft_min_tokens,
                    });
                }
            }
        }
    }
}

fn push_draft_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) {
    let draft_models = discover_draft_model_candidates(request, prepared);
    if draft_models.is_empty() {
        return;
    }
    let max_tokens = positive_or_default(request.spec_draft_max_tokens, &[4, 8, 16]);
    let min_tokens = optional_positive_values(request.spec_draft_min_tokens);
    for draft_model_path in draft_models {
        for draft_max_tokens in &max_tokens {
            if min_tokens.is_empty() {
                candidates.push(TuneBenchmarkSpeculativeCandidate::Draft {
                    draft_model_path: draft_model_path.clone(),
                    draft_max_tokens: *draft_max_tokens,
                    draft_min_tokens: None,
                });
                continue;
            }
            for draft_min_tokens in &min_tokens {
                if draft_min_tokens <= draft_max_tokens {
                    candidates.push(TuneBenchmarkSpeculativeCandidate::Draft {
                        draft_model_path: draft_model_path.clone(),
                        draft_max_tokens: *draft_max_tokens,
                        draft_min_tokens: Some(*draft_min_tokens),
                    });
                }
            }
        }
    }
}

fn push_ngram_speculative_candidates(
    candidates: &mut Vec<TuneBenchmarkSpeculativeCandidate>,
    request: &TuneBenchmarkRunRequest<'_>,
) {
    let ngram_min_values = positive_or_default(request.spec_ngram_min, &[12, 24]);
    let ngram_max_values = positive_or_default(request.spec_ngram_max, &[48, 64]);
    for ngram_min in &ngram_min_values {
        for ngram_max in &ngram_max_values {
            if ngram_min <= ngram_max {
                candidates.push(TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: *ngram_min,
                    ngram_max: *ngram_max,
                });
            }
        }
    }
}

fn positive_or_default(requested: &[u32], defaults: &[u32]) -> Vec<u32> {
    if requested.is_empty() {
        return defaults.to_vec();
    }
    unique_positive(requested)
}

fn optional_positive_values(requested: &[u32]) -> Vec<u32> {
    if requested.is_empty() {
        return Vec::new();
    }
    unique_positive(requested)
}

fn values_or_default_allow_zero(requested: &[u32], defaults: &[u32]) -> Vec<u32> {
    if requested.is_empty() {
        return defaults.to_vec();
    }
    let mut values = requested.to_vec();
    values.sort_unstable();
    values.dedup();
    values
}

fn discover_draft_model_candidates(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Vec<String> {
    let mut candidates = request
        .spec_draft_models
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    if let Some(model_entry) = benchmark_model_entry(request.config, prepared)
        && let Some(path) = model_entry
            .speculative
            .as_ref()
            .and_then(|speculative| speculative.draft_model_path.as_ref())
    {
        candidates.push(path.clone());
    }
    if let Some(path) = request
        .config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.speculative.as_ref())
        .and_then(|speculative| speculative.draft_model_path.as_ref())
    {
        candidates.push(path.clone());
    }
    candidates.extend(discover_sibling_draft_models(
        &prepared.target.resolved_path,
    ));
    candidates.sort();
    candidates.dedup();
    candidates
}

fn discover_sibling_draft_models(model_path: &std::path::Path) -> Vec<String> {
    let Some(parent) = model_path.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };
    let model_file_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != model_path)
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| looks_like_draft_model_name(name, model_file_name))
        })
        .map(|path| path.display().to_string())
        .collect()
}

fn looks_like_draft_model_name(name: &str, target_name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let target_name = target_name.to_ascii_lowercase();
    (name.contains("draft") || name.contains("eagle"))
        && !target_name.is_empty()
        && shares_model_family_token(&name, &target_name)
}

fn shares_model_family_token(left: &str, right: &str) -> bool {
    left.split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 4)
        .any(|token| right.contains(token))
}

fn looks_like_mtp_target(prepared: &crate::gpus::tune_apply::PreparedTunePlan) -> bool {
    [
        &prepared.target.requested_input,
        &prepared.target.canonical_model_ref,
    ]
    .into_iter()
    .any(|value| contains_mtp_marker(value))
        || prepared
            .target
            .resolved_path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(contains_mtp_marker)
}

fn contains_mtp_marker(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized.contains("-mtp")
        || normalized.contains("_mtp")
        || normalized.contains("/mtp")
        || normalized.contains("mtp-gguf")
        || normalized.contains("mtp_gguf")
}

fn dedup_speculative_candidates(
    mut candidates: Vec<TuneBenchmarkSpeculativeCandidate>,
) -> Vec<TuneBenchmarkSpeculativeCandidate> {
    candidates.sort_by_key(speculative_candidate_sort_key);
    candidates.dedup();
    candidates
}

fn speculative_candidate_sort_key(candidate: &TuneBenchmarkSpeculativeCandidate) -> String {
    match candidate {
        TuneBenchmarkSpeculativeCandidate::Mtp {
            draft_model_path,
            draft_max_tokens,
            draft_min_tokens,
        } => format!(
            "0:mtp:{}:{draft_max_tokens}:{draft_min_tokens}",
            draft_model_path.as_deref().unwrap_or("")
        ),
        TuneBenchmarkSpeculativeCandidate::Draft {
            draft_model_path,
            draft_max_tokens,
            draft_min_tokens,
        } => format!(
            "1:draft:{draft_model_path}:{draft_max_tokens}:{}",
            draft_min_tokens.unwrap_or(0)
        ),
        TuneBenchmarkSpeculativeCandidate::Ngram {
            ngram_min,
            ngram_max,
        } => format!("2:ngram:{ngram_min}:{ngram_max}"),
        TuneBenchmarkSpeculativeCandidate::Disabled => "9:disabled".to_string(),
    }
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
        let prepared = prepared_plan_fixture(
            "/tmp/model with spaces.gguf",
            Vec::new(),
            vec![
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
        );
        let candidate = TuneBenchmarkCandidate {
            ctx_size: 4096,
            batch: 2048,
            ubatch: 1024,
            cache_type_k: TuneKvCacheType::Q8_0,
            cache_type_v: TuneKvCacheType::Q8_0,
            mmap: TuneBoolOrAutoValue::Disabled,
            mlock: true,
            speculative: TuneBenchmarkSpeculativeCandidate::Mtp {
                draft_model_path: None,
                draft_max_tokens: 3,
                draft_min_tokens: 0,
            },
        };

        let rendered = trial_config(&prepared, &candidate).expect("trial config renders");
        let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
        let model = parsed.models.first().expect("model row exists");

        assert_eq!(model.model, "/tmp/model with spaces.gguf");
        assert_eq!(
            model
                .model_fit
                .as_ref()
                .and_then(|model_fit| model_fit.ctx_size),
            Some(4096)
        );
        assert!(matches!(
            model
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.gpu_layers.as_ref()),
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
            model
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.model_path.as_deref()),
            Some("/tmp/model with spaces.gguf")
        );
        assert_eq!(
            model
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.mmap.as_ref()),
            Some(&mesh_llm_config::BoolOrAuto::Bool(false))
        );
        assert_eq!(
            model.hardware.as_ref().and_then(|hardware| hardware.mlock),
            Some(true)
        );
        assert_eq!(
            model
                .speculative
                .as_ref()
                .and_then(|speculative| speculative.strategy.as_deref()),
            Some("mtp")
        );
        let speculative = model.speculative.as_ref().expect("speculative config");
        assert_eq!(speculative.draft_max_tokens, Some(3));
        assert_eq!(speculative.draft_min_tokens, Some(0));
        assert_eq!(
            model
                .speculative
                .as_ref()
                .and_then(|speculative| speculative.mode.as_deref()),
            Some("auto")
        );
    }

    #[test]
    fn trial_config_renders_draft_speculative_candidate() {
        let prepared = prepared_plan_fixture("/tmp/model.gguf", Vec::new(), Vec::new());
        let candidate = TuneBenchmarkCandidate {
            ctx_size: 4096,
            batch: 2048,
            ubatch: 1024,
            cache_type_k: TuneKvCacheType::Q8_0,
            cache_type_v: TuneKvCacheType::Q8_0,
            mmap: TuneBoolOrAutoValue::Disabled,
            mlock: false,
            speculative: TuneBenchmarkSpeculativeCandidate::Draft {
                draft_model_path: "/tmp/model-draft.gguf".to_string(),
                draft_max_tokens: 8,
                draft_min_tokens: Some(2),
            },
        };

        let rendered = trial_config(&prepared, &candidate).expect("trial config renders");
        let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
        let speculative = parsed
            .models
            .first()
            .and_then(|model| model.speculative.as_ref())
            .expect("speculative config exists");

        assert_eq!(speculative.strategy.as_deref(), Some("disabled"));
        assert_eq!(speculative.mode.as_deref(), Some("draft"));
        assert_eq!(
            speculative.draft_model_path.as_deref(),
            Some("/tmp/model-draft.gguf")
        );
        assert_eq!(speculative.pairing_fault.as_deref(), Some("fail_closed"));
        assert_eq!(speculative.draft_max_tokens, Some(8));
        assert_eq!(speculative.draft_min_tokens, Some(2));
    }

    #[test]
    fn trial_config_renders_mtp_speculative_sidecar_candidate() {
        let prepared = prepared_plan_fixture("/tmp/model.gguf", Vec::new(), Vec::new());
        let candidate = TuneBenchmarkCandidate {
            ctx_size: 4096,
            batch: 2048,
            ubatch: 1024,
            cache_type_k: TuneKvCacheType::Q8_0,
            cache_type_v: TuneKvCacheType::Q8_0,
            mmap: TuneBoolOrAutoValue::Enabled,
            mlock: false,
            speculative: TuneBenchmarkSpeculativeCandidate::Mtp {
                draft_model_path: Some("/tmp/mtp-gemma.gguf".to_string()),
                draft_max_tokens: 3,
                draft_min_tokens: 0,
            },
        };

        let rendered = trial_config(&prepared, &candidate).expect("trial config renders");
        let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
        let speculative = parsed
            .models
            .first()
            .and_then(|model| model.speculative.as_ref())
            .expect("speculative config exists");

        assert_eq!(speculative.strategy.as_deref(), Some("mtp"));
        assert_eq!(speculative.mode.as_deref(), Some("auto"));
        assert_eq!(
            speculative.draft_model_path.as_deref(),
            Some("/tmp/mtp-gemma.gguf")
        );
        assert_eq!(speculative.pairing_fault.as_deref(), Some("fail_closed"));
        assert_eq!(speculative.draft_max_tokens, Some(3));
        assert_eq!(speculative.draft_min_tokens, Some(0));
    }

    #[test]
    fn trial_config_pins_resolved_model_path_for_huggingface_cache_targets() {
        let prepared = PreparedTunePlan::new(
            ResolvedTuneTarget {
                requested_input: "/cache/snapshot/model.gguf".to_string(),
                canonical_model_ref: "unsloth/example-GGUF:Q4_K_M".to_string(),
                resolved_path: std::path::PathBuf::from("/cache/blobs/model"),
                local_source: LocalTargetSource::HuggingFaceCache {
                    canonical_ref: "unsloth/example-GGUF@sha/model.gguf".to_string(),
                },
                config_matches: Vec::new(),
                selection: TuneTargetSelection::Explicit { configured: false },
            },
            TunePlan {
                target: TuneTarget {
                    requested: "/cache/snapshot/model.gguf".to_string(),
                    resolved: Some("/cache/blobs/model".to_string()),
                    config_model_ref: None,
                    derived_profile: None,
                },
                apply_mode: TuneApplyMode::Review,
                field_statuses: Vec::new(),
                diagnostics: Vec::new(),
            },
        );
        let candidate = TuneBenchmarkCandidate {
            ctx_size: 4096,
            batch: 2048,
            ubatch: 1024,
            cache_type_k: TuneKvCacheType::Q8_0,
            cache_type_v: TuneKvCacheType::Q8_0,
            mmap: TuneBoolOrAutoValue::Enabled,
            mlock: false,
            speculative: TuneBenchmarkSpeculativeCandidate::Disabled,
        };

        let rendered = trial_config(&prepared, &candidate).expect("trial config renders");
        let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
        let model = parsed.models.first().expect("model row exists");

        assert_eq!(model.model, "unsloth/example-GGUF@sha/model.gguf");
        assert_eq!(
            model
                .hardware
                .as_ref()
                .and_then(|hardware| hardware.model_path.as_deref()),
            Some("/cache/blobs/model")
        );
    }

    #[test]
    fn trial_config_renders_ngram_speculative_candidate() {
        let prepared = prepared_plan_fixture("/tmp/model.gguf", Vec::new(), Vec::new());
        let candidate = TuneBenchmarkCandidate {
            ctx_size: 4096,
            batch: 2048,
            ubatch: 1024,
            cache_type_k: TuneKvCacheType::Q8_0,
            cache_type_v: TuneKvCacheType::Q8_0,
            mmap: TuneBoolOrAutoValue::Disabled,
            mlock: false,
            speculative: TuneBenchmarkSpeculativeCandidate::Ngram {
                ngram_min: 12,
                ngram_max: 48,
            },
        };

        let rendered = trial_config(&prepared, &candidate).expect("trial config renders");
        let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
        let speculative = parsed
            .models
            .first()
            .and_then(|model| model.speculative.as_ref())
            .expect("speculative config exists");

        assert_eq!(speculative.strategy.as_deref(), Some("disabled"));
        assert_eq!(speculative.mode.as_deref(), Some("ngram"));
        assert_eq!(speculative.ngram_min, Some(12));
        assert_eq!(speculative.ngram_max, Some(48));
    }

    #[test]
    fn benchmark_candidates_sweep_mmap_and_available_mlock_independently() {
        let prepared = prepared_plan_fixture(
            "/tmp/model.gguf",
            Vec::new(),
            vec![TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::Mlock,
                    value: TuneRecommendedValue::Bool(true),
                    rationale: "test".to_string(),
                },
                edit: TuneConfigEdit::SetHardwareMlock(true),
            }],
        );
        let prepared = [prepared];
        let config = mesh_llm_config::MeshConfig::default();
        let request = TuneBenchmarkRunRequest {
            ctx_sizes: &[4096],
            batch_sizes: &[1024],
            ubatch_sizes: &[256],
            no_speculative_tune: true,
            ..benchmark_request_fixture(&config, &prepared)
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);

        assert_eq!(candidates.len(), 6);
        assert!(
            candidates.iter().any(|candidate| {
                candidate.mmap == TuneBoolOrAutoValue::Auto && !candidate.mlock
            })
        );
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
        let prepared = prepared_plan_fixture(
            "/tmp/model.gguf",
            vec![ConfigModelMatch {
                row_index: 0,
                configured_model: "model".to_string(),
            }],
            Vec::new(),
        );
        let prepared = [prepared];
        let request = TuneBenchmarkRunRequest {
            mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
            mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
            throughput_tolerance_pct: 10.0,
            no_speculative_tune: true,
            ..benchmark_request_fixture(&config, &prepared)
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
    fn benchmark_candidates_auto_prioritizes_native_mtp_for_mtp_targets() {
        let prepared =
            prepared_plan_fixture("/tmp/Qwen3.6-27B-MTP-GGUF.gguf", Vec::new(), Vec::new());
        let prepared = [prepared];
        let config = mesh_llm_config::MeshConfig::default();
        let request = TuneBenchmarkRunRequest {
            ctx_sizes: &[4096],
            batch_sizes: &[1024],
            ubatch_sizes: &[256],
            mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
            mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
            ..benchmark_request_fixture(&config, &prepared)
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);
        let speculation = candidates
            .iter()
            .map(|candidate| candidate.speculative.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            speculation,
            vec![
                TuneBenchmarkSpeculativeCandidate::Mtp {
                    draft_model_path: None,
                    draft_max_tokens: 2,
                    draft_min_tokens: 0,
                },
                TuneBenchmarkSpeculativeCandidate::Mtp {
                    draft_model_path: None,
                    draft_max_tokens: 3,
                    draft_min_tokens: 0,
                },
                TuneBenchmarkSpeculativeCandidate::Mtp {
                    draft_model_path: None,
                    draft_max_tokens: 4,
                    draft_min_tokens: 0,
                },
                TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: 12,
                    ngram_max: 48,
                },
                TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: 12,
                    ngram_max: 64,
                },
                TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: 24,
                    ngram_max: 48,
                },
                TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: 24,
                    ngram_max: 64,
                },
                TuneBenchmarkSpeculativeCandidate::Disabled,
            ]
        );
    }

    #[test]
    fn benchmark_candidates_no_speculative_tune_uses_disabled_baseline_only() {
        let prepared =
            prepared_plan_fixture("/tmp/Qwen3.6-27B-MTP-GGUF.gguf", Vec::new(), Vec::new());
        let prepared = [prepared];
        let config = mesh_llm_config::MeshConfig::default();
        let request = TuneBenchmarkRunRequest {
            ctx_sizes: &[4096],
            batch_sizes: &[1024],
            ubatch_sizes: &[256],
            mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
            mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
            no_speculative_tune: true,
            ..benchmark_request_fixture(&config, &prepared)
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);
        let speculation = candidates
            .iter()
            .map(|candidate| candidate.speculative.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            speculation,
            vec![TuneBenchmarkSpeculativeCandidate::Disabled]
        );
    }

    #[test]
    fn benchmark_candidates_auto_includes_ngram_fallback_for_plain_targets() {
        let prepared = prepared_plan_fixture("/tmp/qwen-target.gguf", Vec::new(), Vec::new());
        let prepared = [prepared];
        let config = mesh_llm_config::MeshConfig::default();
        let request = TuneBenchmarkRunRequest {
            ctx_sizes: &[4096],
            batch_sizes: &[1024],
            ubatch_sizes: &[256],
            mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
            mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
            spec_ngram_min: &[2],
            spec_ngram_max: &[4],
            ..benchmark_request_fixture(&config, &prepared)
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);
        let speculation = candidates
            .iter()
            .map(|candidate| candidate.speculative.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            speculation,
            vec![
                TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: 2,
                    ngram_max: 4,
                },
                TuneBenchmarkSpeculativeCandidate::Disabled,
            ]
        );
    }

    #[test]
    fn benchmark_candidates_auto_orders_draft_before_ngram_when_discovered() {
        let prepared = prepared_plan_fixture("/tmp/qwen-target.gguf", Vec::new(), Vec::new());
        let prepared = [prepared];
        let config = mesh_llm_config::MeshConfig::default();
        let draft_model = std::path::PathBuf::from("/tmp/qwen-draft.gguf");
        let request = TuneBenchmarkRunRequest {
            ctx_sizes: &[4096],
            batch_sizes: &[1024],
            ubatch_sizes: &[256],
            mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
            mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
            spec_draft_models: std::slice::from_ref(&draft_model),
            spec_draft_max_tokens: &[4],
            spec_ngram_min: &[2],
            spec_ngram_max: &[4],
            ..benchmark_request_fixture(&config, &prepared)
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);
        let speculation = candidates
            .iter()
            .map(|candidate| candidate.speculative.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            speculation,
            vec![
                TuneBenchmarkSpeculativeCandidate::Draft {
                    draft_model_path: "/tmp/qwen-draft.gguf".to_string(),
                    draft_max_tokens: 4,
                    draft_min_tokens: None,
                },
                TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: 2,
                    ngram_max: 4,
                },
                TuneBenchmarkSpeculativeCandidate::Disabled,
            ]
        );
    }

    #[test]
    fn benchmark_candidates_explicit_speculative_sweeps_draft_and_ngram_settings() {
        let prepared = prepared_plan_fixture("/tmp/qwen-target.gguf", Vec::new(), Vec::new());
        let prepared = [prepared];
        let config = mesh_llm_config::MeshConfig::default();
        let draft_model = std::path::PathBuf::from("/tmp/qwen-draft.gguf");
        let request = TuneBenchmarkRunRequest {
            ctx_sizes: &[4096],
            batch_sizes: &[1024],
            ubatch_sizes: &[256],
            mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
            mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
            speculative_types: &[
                mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Draft,
                mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Ngram,
            ],
            spec_draft_models: std::slice::from_ref(&draft_model),
            spec_draft_max_tokens: &[4],
            spec_draft_min_tokens: &[2],
            spec_ngram_min: &[12],
            spec_ngram_max: &[48],
            ..benchmark_request_fixture(&config, &prepared)
        };

        let candidates = benchmark_candidates(&request, &prepared[0]);
        let speculation = candidates
            .iter()
            .map(|candidate| candidate.speculative.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            speculation,
            vec![
                TuneBenchmarkSpeculativeCandidate::Draft {
                    draft_model_path: "/tmp/qwen-draft.gguf".to_string(),
                    draft_max_tokens: 4,
                    draft_min_tokens: Some(2),
                },
                TuneBenchmarkSpeculativeCandidate::Ngram {
                    ngram_min: 12,
                    ngram_max: 48,
                },
            ]
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

    #[test]
    fn selection_tie_breaks_toward_unlocked_auto_mmap() {
        let trials = vec![
            succeeded_trial_with_memory(8192, 20.0, 2000.0, TuneBoolOrAutoValue::Enabled, true),
            succeeded_trial_with_memory(8192, 20.0, 2000.0, TuneBoolOrAutoValue::Disabled, false),
            succeeded_trial_with_memory(8192, 20.0, 2000.0, TuneBoolOrAutoValue::Auto, false),
        ];

        let selection = select_benchmark_trials(&trials, 0.0);
        let recommended = selection.recommended.expect("recommended trial");

        assert_eq!(recommended.candidate.mmap, TuneBoolOrAutoValue::Auto);
        assert!(!recommended.candidate.mlock);
    }

    fn prepared_plan_fixture(
        resolved_path: &str,
        config_matches: Vec<ConfigModelMatch>,
        field_statuses: Vec<TuneFieldStatus>,
    ) -> PreparedTunePlan {
        let config_model_ref = config_matches
            .first()
            .map(|config_match| config_match.configured_model.clone());
        let selection = if config_matches.is_empty() {
            TuneTargetSelection::Explicit { configured: false }
        } else {
            TuneTargetSelection::Configured
        };
        PreparedTunePlan::new(
            ResolvedTuneTarget {
                requested_input: "model".to_string(),
                canonical_model_ref: "model".to_string(),
                resolved_path: std::path::PathBuf::from(resolved_path),
                local_source: LocalTargetSource::FilesystemPath {
                    synthetic_model_ref: "model".to_string(),
                },
                config_matches,
                selection,
            },
            TunePlan {
                target: TuneTarget {
                    requested: "model".to_string(),
                    resolved: Some(resolved_path.to_string()),
                    config_model_ref,
                    derived_profile: None,
                },
                apply_mode: TuneApplyMode::Review,
                field_statuses,
                diagnostics: Vec::new(),
            },
        )
    }

    fn benchmark_request_fixture<'a>(
        config: &'a mesh_llm_config::MeshConfig,
        prepared: &'a [PreparedTunePlan],
    ) -> TuneBenchmarkRunRequest<'a> {
        TuneBenchmarkRunRequest {
            config,
            prepared,
            ctx_sizes: &[],
            batch_sizes: &[],
            ubatch_sizes: &[],
            mmap_values: &[],
            mlock_values: &[],
            speculative_types: &[],
            no_speculative_tune: false,
            spec_draft_models: &[],
            spec_draft_max_tokens: &[],
            spec_draft_min_tokens: &[],
            spec_ngram_min: &[],
            spec_ngram_max: &[],
            throughput_tolerance_pct: 3.0,
            max_tokens: 32,
            startup_timeout_secs: 5,
            request_timeout_secs: 5,
            debug_telemetry: false,
            prompt: "hello",
        }
    }

    #[test]
    fn debug_telemetry_enables_child_debug_and_stderr_spans() {
        let command = build_trial_child_command(
            std::path::Path::new("/bin/mesh-llm"),
            std::path::Path::new("/tmp/config.toml"),
            9337,
            3131,
            true,
        );
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert!(args.contains(&"--debug".to_string()));
        assert_eq!(args.last().map(String::as_str), Some("serve"));
        assert_eq!(
            command
                .get_envs()
                .find(|(key, _)| *key == "SKIPPY_TELEMETRY_STDERR")
                .and_then(|(_, value)| value)
                .map(|value| value.to_string_lossy()),
            Some(std::borrow::Cow::Borrowed("1"))
        );
    }

    #[test]
    fn child_debug_telemetry_is_opt_in() {
        let command = build_trial_child_command(
            std::path::Path::new("/bin/mesh-llm"),
            std::path::Path::new("/tmp/config.toml"),
            9337,
            3131,
            false,
        );
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert!(!args.contains(&"--debug".to_string()));
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != "SKIPPY_TELEMETRY_STDERR")
        );
    }

    fn succeeded_trial(ctx_size: u32, decode_tok_s: f64, request_ms: f64) -> TuneBenchmarkTrial {
        succeeded_trial_with_memory(
            ctx_size,
            decode_tok_s,
            request_ms,
            TuneBoolOrAutoValue::Disabled,
            false,
        )
    }

    fn succeeded_trial_with_memory(
        ctx_size: u32,
        decode_tok_s: f64,
        request_ms: f64,
        mmap: TuneBoolOrAutoValue,
        mlock: bool,
    ) -> TuneBenchmarkTrial {
        TuneBenchmarkTrial {
            candidate: TuneBenchmarkCandidate {
                ctx_size,
                batch: 2048,
                ubatch: 1024,
                cache_type_k: TuneKvCacheType::Q8_0,
                cache_type_v: TuneKvCacheType::Q8_0,
                mmap,
                mlock,
                speculative: TuneBenchmarkSpeculativeCandidate::Disabled,
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

    #[test]
    fn trial_startup_failure_scans_json_serve_logs() {
        let log = tempfile::NamedTempFile::new().expect("temp log");
        std::fs::write(
            log.path(),
            r#"{"level":"INFO","message":"API ready"}
{"level":"ERROR","message":"Failed to start model unsloth/Qwen3.6-MTP-GGUF: skippy speculative.strategy = \"mtp\" requires proven native MTP support"}
"#,
        )
        .expect("write log");

        let error = trial_startup_failure_from_log(log.path()).expect("startup error");
        assert!(error.contains("requires proven native MTP support"));
    }

    #[test]
    fn trial_startup_failure_scans_plain_serve_logs() {
        let line = "2026-07-02 Failed to start model qwen: bad draft pair";

        let error = trial_startup_failure_from_log_line(line).expect("startup error");
        assert_eq!(error, line);
    }
}
