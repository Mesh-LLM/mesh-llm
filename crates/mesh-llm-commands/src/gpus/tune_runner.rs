use super::tune::TuneApplyMode;
use super::{tune, tune_apply, tune_hardware, tune_resolver};
use anyhow::{Result, bail};
use mesh_llm_cli::{
    GpuCommand,
    benchmark::{BenchmarkBool, BenchmarkBoolOrAuto, BenchmarkCommand},
};
use mesh_llm_config::{ConfigStore, load_config};
use mesh_llm_system::hardware;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

pub(crate) fn run_tune_command(
    config_path: Option<&Path>,
    json_output: bool,
    command: &GpuCommand,
) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    run_tune_command_with_writer(config_path, json_output, command, &mut handle)
}

fn run_tune_command_with_writer(
    config_path: Option<&Path>,
    json_output: bool,
    command: &GpuCommand,
    writer: &mut impl Write,
) -> Result<()> {
    let args = gpu_tune_runner_args(command);
    run_tune_request_with_writer(config_path, json_output, args, writer)
}

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
    let config = load_config(config_path)?;
    super::tune::recommendation_symbol_anchor();
    tune_resolver::resolver_symbol_anchor();
    tune_hardware::dispatch_symbol_anchor();

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

    let survey = hardware::survey();
    let mut prepared = Vec::new();
    for target in &resolution.resolved {
        let metadata =
            match tune::inspect_local_gguf_metadata(&target.requested_input, &target.resolved_path)
            {
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
                config: &config,
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
            config: &config,
            target,
            metadata: &metadata,
            hardware: &hardware,
            survey: &survey,
        });
        prepared.push(tune_apply::PreparedTunePlan::new(target.clone(), plan));
    }
    let benchmark_reports = maybe_run_benchmark_reports(
        tune::TuneBenchmarkRunRequest {
            prepared: &prepared,
            ctx_sizes: args.ctx_sizes,
            batch_sizes: args.batch_sizes,
            ubatch_sizes: args.ubatch_sizes,
            mmap_values: args.mmap_values,
            mlock_values: args.mlock_values,
            max_tokens: args.max_tokens,
            startup_timeout_secs: args.startup_timeout_secs,
            request_timeout_secs: args.request_timeout_secs,
            prompt: args.prompt,
        },
        args.benchmark,
    );

    if !global_safety_errors.is_empty() {
        emit_runner_output(
            writer,
            RunnerOutputContext {
                command: args.command,
                render_json,
                launch_args: args.launch_args,
                config: &config,
                apply_mode,
                prepared: &prepared,
                target_failures: &target_failures,
                global_blockers: &global_safety_errors,
                benchmark_reports: &benchmark_reports,
            },
        )?;
        let detail = global_safety_errors
            .into_iter()
            .map(|problem| format!("  - {problem}"))
            .collect::<Vec<_>>()
            .join("\n");
        bail!("gpu tune apply aborted before writing config:\n{detail}");
    }

    if resolution.resolved.is_empty() && target_failures.is_empty() {
        bail!("gpu tune found no configured local model targets in the active config");
    }

    if matches!(
        apply_mode,
        TuneApplyMode::ApplyMissing | TuneApplyMode::ReplaceExisting
    ) {
        let store = match config_path {
            Some(path) => ConfigStore::open(path),
            None => ConfigStore::default_path()?,
        };
        let written = tune_apply::apply_prepared_tune_plans(&store, &prepared)?;
        if written == 0 {
            emit_runner_output(
                writer,
                RunnerOutputContext {
                    command: args.command,
                    render_json,
                    launch_args: args.launch_args,
                    config: &config,
                    apply_mode,
                    prepared: &prepared,
                    target_failures: &target_failures,
                    global_blockers: &[],
                    benchmark_reports: &benchmark_reports,
                },
            )?;
            let mut apply_failures = target_failures
                .into_iter()
                .map(|failure| failure.reason)
                .collect::<Vec<_>>();
            apply_failures.extend(prepared.iter().filter_map(apply_failure_reason));
            if apply_failures.is_empty() {
                apply_failures.push(
                    "resolved targets produced no writable tune edits for apply mode".to_string(),
                );
            }
            let detail = apply_failures
                .into_iter()
                .map(|problem| format!("  - {problem}"))
                .collect::<Vec<_>>()
                .join("\n");
            bail!("gpu tune could not produce any safe config edits:\n{detail}");
        }
    } else if prepared.is_empty() && !target_failures.is_empty() {
        emit_runner_output(
            writer,
            RunnerOutputContext {
                command: args.command,
                render_json,
                launch_args: args.launch_args,
                config: &config,
                apply_mode,
                prepared: &prepared,
                target_failures: &target_failures,
                global_blockers: &[],
                benchmark_reports: &benchmark_reports,
            },
        )?;
        let detail = target_failures
            .into_iter()
            .map(|failure| format!("  - {}", failure.reason))
            .collect::<Vec<_>>()
            .join("\n");
        bail!("gpu tune could not prepare any local targets:\n{detail}");
    }

    emit_runner_output(
        writer,
        RunnerOutputContext {
            command: args.command,
            render_json,
            launch_args: args.launch_args,
            config: &config,
            apply_mode,
            prepared: &prepared,
            target_failures: &target_failures,
            global_blockers: &[],
            benchmark_reports: &benchmark_reports,
        },
    )?;
    Ok(())
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

struct TuneRunnerArgs<'a> {
    command: &'static str,
    model: Option<&'a str>,
    models: &'a [String],
    json: bool,
    benchmark: bool,
    ctx_sizes: &'a [u32],
    batch_sizes: &'a [u32],
    ubatch_sizes: &'a [u32],
    mmap_values: &'a [BenchmarkBoolOrAuto],
    mlock_values: &'a [BenchmarkBool],
    max_tokens: u32,
    startup_timeout_secs: u64,
    request_timeout_secs: u64,
    prompt: &'a str,
    launch_args: bool,
    apply: bool,
    replace_existing: bool,
}

fn gpu_tune_runner_args(command: &GpuCommand) -> TuneRunnerArgs<'_> {
    let GpuCommand::Tune {
        model,
        models,
        json,
        launch_args,
        apply,
        replace_existing,
    } = command
    else {
        unreachable!("run_tune_command called for non-tune GPU command");
    };
    TuneRunnerArgs {
        command: "gpu_tune",
        model: model.as_deref(),
        models,
        json: *json,
        benchmark: false,
        ctx_sizes: &[],
        batch_sizes: &[],
        ubatch_sizes: &[],
        mmap_values: &[],
        mlock_values: &[],
        max_tokens: 128,
        startup_timeout_secs: 600,
        request_timeout_secs: 600,
        prompt: "",
        launch_args: *launch_args,
        apply: *apply,
        replace_existing: *replace_existing,
    }
}

fn benchmark_tune_runner_args(command: &BenchmarkCommand) -> TuneRunnerArgs<'_> {
    let BenchmarkCommand::Tune {
        model,
        models,
        json,
        ctx_sizes,
        batch_sizes,
        ubatch_sizes,
        mmap_values,
        mlock_values,
        max_tokens,
        startup_timeout_secs,
        request_timeout_secs,
        prompt,
    } = command
    else {
        unreachable!("run_benchmark_tune_command called for non-tune benchmark command");
    };
    TuneRunnerArgs {
        command: "benchmark_tune",
        model: model.as_deref(),
        models,
        json: *json,
        benchmark: true,
        ctx_sizes,
        batch_sizes,
        ubatch_sizes,
        mmap_values,
        mlock_values,
        max_tokens: *max_tokens,
        startup_timeout_secs: *startup_timeout_secs,
        request_timeout_secs: *request_timeout_secs,
        prompt,
        launch_args: false,
        apply: false,
        replace_existing: false,
    }
}

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

fn maybe_run_benchmark_reports(
    request: tune::TuneBenchmarkRunRequest<'_>,
    benchmark: bool,
) -> Vec<tune::TuneBenchmarkTargetReport> {
    if benchmark {
        run_benchmark_plans_on_plain_thread(request)
    } else {
        Vec::new()
    }
}

fn run_benchmark_plans_on_plain_thread(
    request: tune::TuneBenchmarkRunRequest<'_>,
) -> Vec<tune::TuneBenchmarkTargetReport> {
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
