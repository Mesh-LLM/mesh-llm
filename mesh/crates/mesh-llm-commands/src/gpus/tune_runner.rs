use super::tune::TuneApplyMode;
use super::{tune, tune_apply, tune_hardware, tune_resolver};
use anyhow::{Result, bail};
use mesh_llm_cli::benchmark::{BenchmarkBool, BenchmarkBoolOrAuto, BenchmarkCommand};
use mesh_llm_config::{ConfigStore, load_config};
use mesh_llm_system::hardware;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

pub(crate) fn run_benchmark_tune_command(
    config_path: Option<&Path>,
    command: &BenchmarkCommand,
) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    run_benchmark_tune_command_with_writer(config_path, command, &mut handle)
}

pub(crate) fn run_benchmark_tune_command_with_writer(
    config_path: Option<&Path>,
    command: &BenchmarkCommand,
    writer: &mut impl Write,
) -> Result<()> {
    let args = benchmark_tune_runner_args(command);
    run_tune_request_with_writer(config_path, false, args, writer)
}

fn run_tune_request_with_writer(
    config_path: Option<&Path>,
    json_output: bool,
    args: TuneRunnerArgs<'_>,
    writer: &mut impl Write,
) -> Result<()> {
    let render_json = json_output || args.json;
    let apply_mode = tune_apply_mode(args.launch_args, args.apply, args.replace_existing);
    validate_benchmark_args(args.benchmark.as_ref())?;
    let config = load_config(config_path)?;

    let resolution = if let Some(explicit_model) = args.model {
        tune_resolver::resolve_explicit_tune_targets(&config, &[explicit_model.to_string()])
    } else if !args.models.is_empty() {
        tune_resolver::resolve_explicit_tune_targets(&config, args.models)
    } else {
        tune_resolver::resolve_configured_tune_targets(&config)
    };

    let explicit_inputs = args.model.is_some() || !args.models.is_empty();
    let mut global_safety_errors = Vec::new();
    let mut target_failures = Vec::new();
    for duplicate in &resolution.duplicates {
        let reason = format!(
            "requested target `{}` resolves to duplicate model `{}` (first requested as `{}`)",
            duplicate.input, duplicate.canonical_model_ref, duplicate.first_input
        );
        if explicit_inputs {
            target_failures.push(tune::TuneTargetFailure {
                requested_input: duplicate.input.clone(),
                reason,
            });
        } else {
            global_safety_errors.push(reason);
        }
    }
    target_failures.extend(
        resolution
            .errors
            .iter()
            .map(|error| tune::TuneTargetFailure {
                requested_input: error.input.clone(),
                reason: error.to_string(),
            }),
    );

    let prepared = prepare_tune_plans(&config, &resolution, apply_mode, &mut target_failures);

    let global_context = RunnerOutputContext {
        command: args.command,
        render_json,
        launch_args: args.launch_args,
        config: &config,
        apply_mode,
        prepared: &prepared,
        target_failures: &target_failures,
        global_blockers: &[],
        benchmark_reports: &[],
    };
    bail_on_global_safety_errors(writer, global_context, &global_safety_errors)?;

    let benchmark_reports = maybe_run_benchmark_reports(
        args.benchmark
            .as_ref()
            .map(|benchmark| benchmark_run_request(&config, &prepared, benchmark)),
    )?;
    let output_context = RunnerOutputContext {
        command: args.command,
        render_json,
        launch_args: args.launch_args,
        config: &config,
        apply_mode,
        prepared: &prepared,
        target_failures: &target_failures,
        global_blockers: &[],
        benchmark_reports: &benchmark_reports,
    };

    if resolution.resolved.is_empty() && target_failures.is_empty() {
        bail!(
            "{} found no configured local model targets in the active config",
            command_label(args.command)
        );
    }

    handle_apply_mode(writer, output_context, config_path)?;

    emit_runner_output_for(writer, output_context, &[])?;
    Ok(())
}

fn bail_on_global_safety_errors(
    writer: &mut impl Write,
    context: RunnerOutputContext<'_>,
    errors: &[String],
) -> Result<()> {
    if errors.is_empty() {
        return Ok(());
    }
    emit_runner_output_for(writer, context, errors)?;
    let detail = errors
        .iter()
        .map(|problem| format!("  - {problem}"))
        .collect::<Vec<_>>()
        .join("\n");
    bail!(
        "{} apply aborted before writing config:\n{detail}",
        command_label(context.command)
    )
}

fn handle_apply_mode(
    writer: &mut impl Write,
    context: RunnerOutputContext<'_>,
    config_path: Option<&Path>,
) -> Result<()> {
    if matches!(
        context.apply_mode,
        TuneApplyMode::ApplyMissing | TuneApplyMode::ReplaceExisting
    ) {
        let store = match config_path {
            Some(path) => ConfigStore::open(path),
            None => ConfigStore::default_path()?,
        };
        emit_runner_output_for(writer, context, &[])?;
        let written = tune_apply::apply_prepared_tune_plans(&store, context.prepared)?;
        if written == 0 {
            let mut apply_failures: Vec<String> = context
                .target_failures
                .iter()
                .map(|failure| failure.reason.clone())
                .collect();
            apply_failures.extend(context.prepared.iter().filter_map(apply_failure_reason));
            if apply_failures.is_empty() {
                apply_failures.push(
                    "resolved targets produced no writable tune edits for apply mode".to_string(),
                );
            }
            let detail = apply_failures
                .iter()
                .map(|problem| format!("  - {problem}"))
                .collect::<Vec<_>>()
                .join("\n");
            bail!(
                "{} could not produce any safe config edits:\n{detail}",
                command_label(context.command)
            );
        }
    } else if context.prepared.is_empty() && !context.target_failures.is_empty() {
        emit_runner_output_for(writer, context, &[])?;
        let detail = context
            .target_failures
            .iter()
            .map(|failure| format!("  - {}", failure.reason))
            .collect::<Vec<_>>()
            .join("\n");
        bail!(
            "{} could not prepare any local targets:\n{detail}",
            command_label(context.command)
        );
    }
    Ok(())
}

fn prepare_tune_plans(
    config: &mesh_llm_config::MeshConfig,
    resolution: &tune_resolver::TuneTargetResolution,
    apply_mode: TuneApplyMode,
    target_failures: &mut Vec<tune::TuneTargetFailure>,
) -> Vec<tune_apply::PreparedTunePlan> {
    let survey = hardware::survey();
    let mut prepared = Vec::new();
    for target in &resolution.resolved {
        let metadata = match tune::inspect_tune_target_metadata(
            &target.requested_input,
            &target.resolved_path,
        ) {
            Ok(metadata) => metadata,
            Err(error) => {
                target_failures.push(tune::TuneTargetFailure {
                    requested_input: target.requested_input.clone(),
                    reason: error.to_string(),
                });
                continue;
            }
        };
        let hardware = match tune_hardware::evaluate::evaluate_tune_hardware(
            tune_hardware::types::TuneHardwareEvaluationInput {
                config,
                target,
                survey: &survey,
            },
        ) {
            Ok(hardware) => hardware,
            Err(error) => {
                target_failures.push(tune::TuneTargetFailure {
                    requested_input: target.requested_input.clone(),
                    reason: error.message,
                });
                continue;
            }
        };
        let plan = tune::build_tune_plan(tune::TuneRecommendationInput {
            apply_mode,
            config,
            target,
            metadata: &metadata,
            hardware: &hardware,
            survey: &survey,
        });
        prepared.push(tune_apply::PreparedTunePlan::new(target.clone(), plan));
    }
    prepared
}

fn command_label(command: &'static str) -> &'static str {
    match command {
        "benchmark_tune" => "benchmark tune",
        _ => command,
    }
}

fn apply_failure_reason(prepared: &tune_apply::PreparedTunePlan) -> Option<String> {
    let mut messages = BTreeSet::new();
    for status in &prepared.plan.field_statuses {
        if let tune::TuneFieldStatus::Error { diagnostic, .. } = status {
            messages.insert(diagnostic.message.clone());
        }
    }
    for diagnostic in &prepared.plan.diagnostics {
        if matches!(diagnostic.severity, tune::TuneDiagnosticSeverity::Error) {
            messages.insert(diagnostic.message.clone());
        }
    }
    if !messages.is_empty() {
        return Some(format!(
            "model `{}`: {}",
            prepared.target.requested_input,
            messages.into_iter().collect::<Vec<_>>().join("; "),
        ));
    }
    prepared.plan.config_edits().is_empty().then(|| {
        format!(
            "model `{}`: apply produced no writable tune edits",
            prepared.target.requested_input,
        )
    })
}

fn validate_benchmark_args(args: Option<&BenchmarkTuneArgs<'_>>) -> Result<()> {
    let Some(args) = args else {
        return Ok(());
    };
    if !args.throughput_tolerance_pct.is_finite() || args.throughput_tolerance_pct < 0.0 {
        bail!("--throughput-tolerance-pct must be finite and non-negative");
    }
    if args.max_tokens == 0 {
        bail!("--max-tokens must be greater than zero");
    }
    validate_positive_values("--ctx-sizes", args.ctx_sizes)?;
    validate_positive_values("--batch-sizes", args.batch_sizes)?;
    validate_positive_values("--ubatch-sizes", args.ubatch_sizes)?;
    validate_positive_values("--spec-draft-max-tokens", args.spec_draft_max_tokens)?;
    validate_probability_values(
        "--spec-draft-acceptance-threshold",
        args.spec_draft_acceptance_threshold,
    )?;
    validate_probability_values(
        "--spec-draft-split-probability",
        args.spec_draft_split_probability,
    )?;
    validate_positive_values("--spec-ngram-min", args.spec_ngram_min)?;
    validate_positive_values("--spec-ngram-max", args.spec_ngram_max)?;
    validate_batch_ubatch_pairs(args.batch_sizes, args.ubatch_sizes)?;
    validate_min_max_candidates(
        "--spec-draft-min-tokens",
        args.spec_draft_min_tokens,
        "--spec-draft-max-tokens",
        args.spec_draft_max_tokens,
    )?;
    validate_min_max_candidates(
        "--spec-ngram-min",
        args.spec_ngram_min,
        "--spec-ngram-max",
        args.spec_ngram_max,
    )?;
    Ok(())
}

fn validate_positive_values(name: &str, values: &[u32]) -> Result<()> {
    if !values.is_empty() && !values.iter().any(|value| *value > 0) {
        bail!("{name} must include at least one positive value");
    }
    Ok(())
}

fn validate_probability_values(name: &str, values: &[f64]) -> Result<()> {
    for value in values {
        if !value.is_finite() || *value < 0.0 || *value > 1.0 {
            bail!("{name} values must be finite probabilities in [0.0, 1.0]");
        }
    }
    Ok(())
}

fn validate_batch_ubatch_pairs(batch_sizes: &[u32], ubatch_sizes: &[u32]) -> Result<()> {
    if batch_sizes.is_empty() || ubatch_sizes.is_empty() {
        return Ok(());
    }
    let has_valid_pair = batch_sizes
        .iter()
        .copied()
        .filter(|batch| *batch > 0)
        .any(|batch| {
            ubatch_sizes
                .iter()
                .copied()
                .filter(|ubatch| *ubatch > 0)
                .any(|ubatch| ubatch <= batch)
        });
    if !has_valid_pair {
        bail!("benchmark candidate matrix has no valid batch/ubatch pairs");
    }
    Ok(())
}

fn validate_min_max_candidates(
    min_name: &str,
    mins: &[u32],
    max_name: &str,
    maxes: &[u32],
) -> Result<()> {
    if mins.is_empty() || maxes.is_empty() {
        return Ok(());
    }
    let has_valid_pair = mins.iter().copied().any(|min| {
        maxes
            .iter()
            .copied()
            .filter(|value| *value > 0)
            .any(|max| min <= max)
    });
    if !has_valid_pair {
        bail!("benchmark candidate matrix has no valid {min_name}/{max_name} pairs");
    }
    Ok(())
}

fn benchmark_run_request<'a>(
    config: &'a mesh_llm_config::MeshConfig,
    prepared: &'a [tune_apply::PreparedTunePlan],
    args: &'a BenchmarkTuneArgs<'a>,
) -> tune::TuneBenchmarkRunRequest<'a> {
    tune::TuneBenchmarkRunRequest {
        config,
        prepared,
        ctx_sizes: args.ctx_sizes,
        batch_sizes: args.batch_sizes,
        ubatch_sizes: args.ubatch_sizes,
        mmap_values: args.mmap_values,
        mlock_values: args.mlock_values,
        flash_attention_values: args.flash_attention_values,
        speculative_types: args.speculative_types,
        no_speculative_tune: args.no_speculative_tune,
        spec_draft_models: args.spec_draft_models,
        spec_draft_max_tokens: args.spec_draft_max_tokens,
        spec_draft_min_tokens: args.spec_draft_min_tokens,
        spec_draft_acceptance_threshold: args.spec_draft_acceptance_threshold,
        spec_draft_split_probability: args.spec_draft_split_probability,
        spec_ngram_min: args.spec_ngram_min,
        spec_ngram_max: args.spec_ngram_max,
        throughput_tolerance_pct: args.throughput_tolerance_pct,
        max_tokens: args.max_tokens,
        startup_timeout_secs: args.startup_timeout_secs,
        request_timeout_secs: args.request_timeout_secs,
        debug_telemetry: args.debug_telemetry,
        prompt: args.prompt,
    }
}

struct TuneRunnerArgs<'a> {
    command: &'static str,
    model: Option<&'a str>,
    models: &'a [String],
    json: bool,
    benchmark: Option<BenchmarkTuneArgs<'a>>,
    launch_args: bool,
    apply: bool,
    replace_existing: bool,
}

struct BenchmarkTuneArgs<'a> {
    ctx_sizes: &'a [u32],
    batch_sizes: &'a [u32],
    ubatch_sizes: &'a [u32],
    mmap_values: &'a [BenchmarkBoolOrAuto],
    mlock_values: &'a [BenchmarkBool],
    flash_attention_values: &'a [mesh_llm_cli::benchmark::BenchmarkFlashAttention],
    speculative_types: &'a [mesh_llm_cli::benchmark::BenchmarkSpeculativeType],
    no_speculative_tune: bool,
    spec_draft_models: &'a [std::path::PathBuf],
    spec_draft_max_tokens: &'a [u32],
    spec_draft_min_tokens: &'a [u32],
    spec_draft_acceptance_threshold: &'a [f64],
    spec_draft_split_probability: &'a [f64],
    spec_ngram_min: &'a [u32],
    spec_ngram_max: &'a [u32],
    throughput_tolerance_pct: f64,
    max_tokens: u32,
    startup_timeout_secs: u64,
    request_timeout_secs: u64,
    debug_telemetry: bool,
    prompt: &'a str,
}

fn benchmark_tune_runner_args(command: &BenchmarkCommand) -> TuneRunnerArgs<'_> {
    let BenchmarkCommand::Tune(args) = command else {
        unreachable!("run_benchmark_tune_command called for non-tune benchmark command");
    };
    let args = args.as_ref();
    TuneRunnerArgs {
        command: "benchmark_tune",
        model: args.model.as_deref(),
        models: &args.models,
        json: args.json,
        benchmark: Some(BenchmarkTuneArgs {
            ctx_sizes: &args.ctx_sizes,
            batch_sizes: &args.batch_sizes,
            ubatch_sizes: &args.ubatch_sizes,
            mmap_values: &args.mmap_values,
            mlock_values: &args.mlock_values,
            flash_attention_values: &args.flash_attention,
            speculative_types: &args.speculative_types,
            no_speculative_tune: args.no_speculative_tune,
            spec_draft_models: &args.spec_draft_models,
            spec_draft_max_tokens: &args.spec_draft_max_tokens,
            spec_draft_min_tokens: &args.spec_draft_min_tokens,
            spec_draft_acceptance_threshold: &args.spec_draft_acceptance_threshold,
            spec_draft_split_probability: &args.spec_draft_split_probability,
            spec_ngram_min: &args.spec_ngram_min,
            spec_ngram_max: &args.spec_ngram_max,
            throughput_tolerance_pct: args.throughput_tolerance_pct,
            max_tokens: args.max_tokens,
            startup_timeout_secs: args.startup_timeout_secs,
            request_timeout_secs: args.request_timeout_secs,
            debug_telemetry: args.debug_telemetry,
            prompt: &args.prompt,
        }),
        launch_args: args.launch_args,
        apply: args.apply,
        replace_existing: args.replace_existing,
    }
}

#[derive(Clone, Copy)]
struct RunnerOutputContext<'a> {
    command: &'static str,
    render_json: bool,
    launch_args: bool,
    config: &'a mesh_llm_config::MeshConfig,
    apply_mode: TuneApplyMode,
    prepared: &'a [tune_apply::PreparedTunePlan],
    target_failures: &'a [tune::TuneTargetFailure],
    global_blockers: &'a [String],
    benchmark_reports: &'a [tune::TuneBenchmarkTargetReport],
}

fn emit_runner_output(writer: &mut impl Write, context: RunnerOutputContext<'_>) -> Result<()> {
    tune::emit_tune_output(
        writer,
        tune::TuneOutputRequest {
            command: context.command,
            json_output: context.render_json,
            launch_args: context.launch_args,
            config: context.config,
            apply_mode: context.apply_mode,
            prepared: context.prepared,
            target_failures: context.target_failures,
            global_blockers: context.global_blockers,
            benchmark_reports: context.benchmark_reports,
        },
    )
}

fn emit_runner_output_for(
    writer: &mut impl Write,
    base: RunnerOutputContext<'_>,
    global_blockers: &[String],
) -> Result<()> {
    emit_runner_output(
        writer,
        RunnerOutputContext {
            global_blockers,
            ..base
        },
    )
}

fn maybe_run_benchmark_reports(
    request: Option<tune::TuneBenchmarkRunRequest<'_>>,
) -> Result<Vec<tune::TuneBenchmarkTargetReport>> {
    match request {
        Some(request) => run_benchmark_plans_on_plain_thread(request),
        None => Ok(Vec::new()),
    }
}

fn run_benchmark_plans_on_plain_thread(
    request: tune::TuneBenchmarkRunRequest<'_>,
) -> Result<Vec<tune::TuneBenchmarkTargetReport>> {
    std::thread::scope(|scope| {
        let handle = scope.spawn(move || tune::run_benchmark_plans(request));
        handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

const fn tune_apply_mode(launch_args: bool, apply: bool, replace_existing: bool) -> TuneApplyMode {
    if launch_args {
        TuneApplyMode::LaunchArgs
    } else if apply && replace_existing {
        TuneApplyMode::ReplaceExisting
    } else if apply {
        TuneApplyMode::ApplyMissing
    } else {
        TuneApplyMode::Review
    }
}

#[cfg(test)]
#[path = "tune_runner_tests.rs"]
mod tests;
