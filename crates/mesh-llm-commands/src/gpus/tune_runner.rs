use super::tune::TuneApplyMode;
use super::{tune, tune_apply, tune_hardware, tune_resolver};
use anyhow::{Result, bail};
use mesh_llm_cli::GpuCommand;
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

    let render_json = json_output || *json;
    let apply_mode = tune_apply_mode(*launch_args, *apply, *replace_existing);
    let config = load_config(config_path)?;
    super::tune::recommendation_symbol_anchor();
    tune_resolver::resolver_symbol_anchor();
    tune_hardware::dispatch_symbol_anchor();

    let resolution = if let Some(explicit_model) = model.as_deref() {
        tune_resolver::resolve_explicit_tune_targets(&config, &[explicit_model.to_string()])
    } else if !models.is_empty() {
        tune_resolver::resolve_explicit_tune_targets(&config, models)
    } else {
        tune_resolver::resolve_configured_tune_targets(&config)
    };

    let explicit_inputs = model.is_some() || !models.is_empty();
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

    if !global_safety_errors.is_empty() {
        tune::emit_tune_output(
            writer,
            tune::TuneOutputRequest {
                json_output: render_json,
                launch_args: *launch_args,
                config: &config,
                apply_mode,
                prepared: &prepared,
                target_failures: &target_failures,
                global_blockers: &global_safety_errors,
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
            tune::emit_tune_output(
                writer,
                tune::TuneOutputRequest {
                    json_output: render_json,
                    launch_args: *launch_args,
                    config: &config,
                    apply_mode,
                    prepared: &prepared,
                    target_failures: &target_failures,
                    global_blockers: &[],
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
        tune::emit_tune_output(
            writer,
            tune::TuneOutputRequest {
                json_output: render_json,
                launch_args: *launch_args,
                config: &config,
                apply_mode,
                prepared: &prepared,
                target_failures: &target_failures,
                global_blockers: &[],
            },
        )?;
        let detail = target_failures
            .into_iter()
            .map(|failure| format!("  - {}", failure.reason))
            .collect::<Vec<_>>()
            .join("\n");
        bail!("gpu tune could not prepare any local targets:\n{detail}");
    }

    tune::emit_tune_output(
        writer,
        tune::TuneOutputRequest {
            json_output: render_json,
            launch_args: *launch_args,
            config: &config,
            apply_mode,
            prepared: &prepared,
            target_failures: &target_failures,
            global_blockers: &[],
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
