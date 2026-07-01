pub(crate) struct TuneBenchmarkRunRequest<'a> {
    pub(crate) prepared: &'a [crate::gpus::tune_apply::PreparedTunePlan],
    pub(crate) ctx_sizes: &'a [u32],
    pub(crate) batch_sizes: &'a [u32],
    pub(crate) ubatch_sizes: &'a [u32],
    pub(crate) max_tokens: u32,
    pub(crate) startup_timeout_secs: u64,
    pub(crate) request_timeout_secs: u64,
    pub(crate) prompt: &'a str,
}

pub(crate) fn run_benchmark_plans(
    request: TuneBenchmarkRunRequest<'_>,
) -> Vec<TuneBenchmarkTargetReport> {
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
    let trials = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| run_trial(request, prepared, index, candidate))
        .collect::<Vec<_>>();
    let best = trials
        .iter()
        .filter(|trial| matches!(trial.status, TuneBenchmarkTrialStatus::Succeeded))
        .filter_map(|trial| trial.decode_tok_s.map(|rate| (trial, rate)))
        .max_by(|(left_trial, left_rate), (right_trial, right_rate)| {
            left_rate
                .partial_cmp(right_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_trial.candidate.ctx_size.cmp(&right_trial.candidate.ctx_size))
        })
        .map(|(trial, _)| trial.clone());

    TuneBenchmarkTargetReport {
        requested: prepared.target.requested_input.clone(),
        best,
        trials,
    }
}

fn benchmark_candidates(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> Vec<TuneBenchmarkCandidate> {
    let default_ctx = recommended_u32(&prepared.plan, TuneField::CtxSize).unwrap_or(8192);
    let contexts = if request.ctx_sizes.is_empty() {
        default_context_sizes(default_ctx)
    } else {
        unique_positive(request.ctx_sizes)
    };
    let batches = if request.batch_sizes.is_empty() {
        vec![recommended_u32(&prepared.plan, TuneField::Batch).unwrap_or(512)]
    } else {
        unique_positive(request.batch_sizes)
    };
    let ubatches = if request.ubatch_sizes.is_empty() {
        vec![recommended_u32(&prepared.plan, TuneField::Ubatch).unwrap_or(128)]
    } else {
        unique_positive(request.ubatch_sizes)
    };
    let cache_type_k =
        recommended_cache_type(&prepared.plan, TuneField::CacheTypeK).unwrap_or(TuneKvCacheType::Q8_0);
    let cache_type_v =
        recommended_cache_type(&prepared.plan, TuneField::CacheTypeV).unwrap_or(cache_type_k);

    let mut candidates = Vec::new();
    for ctx_size in contexts {
        for batch in &batches {
            for ubatch in &ubatches {
                if *ubatch > *batch {
                    continue;
                }
                candidates.push(TuneBenchmarkCandidate {
                    ctx_size,
                    batch: *batch,
                    ubatch: *ubatch,
                    cache_type_k,
                    cache_type_v,
                });
            }
        }
    }
    candidates
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
    let trial_dir = create_trial_dir(prepared, index)?;
    let config_path = trial_dir.join("config.toml");
    let log_path = trial_dir.join("serve.log");
    std::fs::write(&config_path, trial_config(prepared, candidate)?)?;

    let port = reserve_local_port()?;
    let console = reserve_local_port()?;
    let mut child = TrialChild::spawn(&config_path, &log_path, port, console)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(request.request_timeout_secs))
        .build()?;
    wait_for_trial_ready(
        &client,
        &mut child,
        port,
        request.prompt,
        request.startup_timeout_secs,
    )?;

    let started = std::time::Instant::now();
    let response = send_chat_request(&client, port, request.prompt, request.max_tokens)?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let completion_tokens = response_completion_tokens(&response)
        .ok_or_else(|| anyhow::anyhow!("chat completion response did not include completion_tokens"))?;
    anyhow::ensure!(completion_tokens > 0, "chat completion returned zero completion tokens");
    let decode_tok_s = completion_tokens as f64 / (elapsed_ms / 1000.0);
    child.shutdown();

    Ok(TuneBenchmarkTrial {
        candidate: candidate.clone(),
        status: TuneBenchmarkTrialStatus::Succeeded,
        completion_tokens: Some(completion_tokens),
        elapsed_ms: Some(elapsed_ms),
        decode_tok_s: Some(decode_tok_s),
        log_path: Some(log_path.display().to_string()),
        error: None,
    })
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
) -> anyhow::Result<()> {
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(startup_timeout_secs.max(1));
    let mut last_error = String::new();
    while std::time::Instant::now() < deadline {
        if let Some(status) = child.child.try_wait()? {
            anyhow::bail!("trial server exited before readiness: {status}");
        }
        match send_chat_request(client, port, prompt, 1) {
            Ok(_) => return Ok(()),
            Err(error) => last_error = error.to_string(),
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    anyhow::bail!("trial server did not become ready: {last_error}");
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
        LocalTargetSource, ResolvedTuneTarget, TuneTargetSelection,
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
    }
}
