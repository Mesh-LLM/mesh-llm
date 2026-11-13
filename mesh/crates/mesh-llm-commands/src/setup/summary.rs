use super::SetupPlan;
use super::command::{CliSetupActions, SetupPluginOutcome, SetupServiceOutcome};
use super::service::ServiceInstallStatus;
use crate::runtime_native::{
    SetupNativeRuntimeOutcome, SetupNativeRuntimePruneResult, SetupNativeRuntimeStatus,
};
use crate::terminal::{style_muted, style_ok, style_warn};
use skippy_runtime_install::NativeRuntimeInstallStatus;
use std::io::Write;

pub(crate) fn print_runtime_install_result(outcome: &SetupNativeRuntimeOutcome) {
    let mut err = mesh_llm_events::console_err();
    match &outcome.status {
        SetupNativeRuntimeStatus::Skipped => {}
        SetupNativeRuntimeStatus::Installed(installed) => match installed.status {
            NativeRuntimeInstallStatus::Installed => {
                let _ = writeln!(
                    err,
                    "{} Installed native runtime {} for mesh version {}",
                    style_ok("✓"),
                    installed.runtime.native_runtime_id,
                    installed.runtime.release_version
                );
            }
            NativeRuntimeInstallStatus::AlreadyInstalled => {
                let _ = writeln!(
                    err,
                    "{} Native runtime {} is already installed for mesh version {}",
                    style_ok("✓"),
                    installed.runtime.native_runtime_id,
                    installed.runtime.release_version
                );
            }
        },
    }
}

pub(crate) fn print_service_install_result(
    report: &crate::setup::service::ServiceInstallReport,
    verbose: bool,
) {
    if verbose {
        let mut err = mesh_llm_events::console_err();
        for line in &report.messages {
            let _ = writeln!(err, "{line}");
        }
    }
}

pub(crate) fn print_setup_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>, verbose: bool) {
    let mut err = mesh_llm_events::console_err();
    let _ = writeln!(err);
    if verbose {
        let _ = writeln!(err, "Setup summary");
        let _ = writeln!(err, "- Runtime: {}", runtime_summary(plan, actions));
        let _ = writeln!(err, "- Service: {}", service_summary(plan, actions));
        let _ = writeln!(err, "- Plugins: {}", plugin_summary(plan, actions));
        let _ = writeln!(
            err,
            "- GitHub star: {}",
            super::github::github_summary(plan, &actions.github_outcome)
        );
        let _ = writeln!(err, "- Next: {}", next_step(actions));
        return;
    }

    let _ = writeln!(err, "{} Mesh setup complete", style_ok("✓"));
    let _ = writeln!(err, "  Runtime  {}", runtime_brief(plan, actions));
    let _ = writeln!(err, "  Service  {}", service_brief(plan, actions));
    if let Some(plugins) = plugin_brief(plan, actions) {
        let _ = writeln!(err, "  Plugins  {plugins}");
    }
    if let Some(github) = github_brief(actions) {
        let _ = writeln!(err, "  GitHub star  {github}");
    }
    let _ = writeln!(err, "  Next     {}", next_step(actions));
}

/// One line for the optional plugin phase in the concise summary; `None`
/// when nothing happened worth a line.
fn plugin_brief(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> Option<String> {
    let set_name = match &plan.plugins {
        super::SetupPluginPlan::Install(name) => name.as_str(),
        _ => crate::plugin_sets::RECOMMENDED_SET,
    };
    match &plan.plugins {
        super::SetupPluginPlan::Install(_) | super::SetupPluginPlan::Offer => {
            Some(match actions.plugin_outcome {
                SetupPluginOutcome::Installed(count) => format!("installed {count}"),
                SetupPluginOutcome::Incomplete => {
                    format!(
                        "{} incomplete; retry with `mesh-llm plugins sets install {set_name}`",
                        style_warn("installation")
                    )
                }
                SetupPluginOutcome::Skipped | SetupPluginOutcome::NotRequested => {
                    "skipped".to_string()
                }
            })
        }
        super::SetupPluginPlan::Skip(reason) => match reason {
            super::SetupPluginSkipReason::ExplicitSkip => Some("skipped".to_string()),
            super::SetupPluginSkipReason::HiddenPrompt => None,
        },
    }
}

/// The verbose summary line for the optional plugin phase.
fn plugin_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plugin_brief(plan, actions) {
        Some(brief) => format!(
            "{brief} ({})",
            match &plan.plugins {
                super::SetupPluginPlan::Install(name) => format!("--plugin-set {name}"),
                super::SetupPluginPlan::Offer => "recommended offer".to_string(),
                super::SetupPluginPlan::Skip(reason) => match reason {
                    super::SetupPluginSkipReason::ExplicitSkip => {
                        "--skip-plugin-install".to_string()
                    }
                    super::SetupPluginSkipReason::HiddenPrompt => "no prompt available".to_string(),
                },
            }
        ),
        None => "none (no prompt available and no set selected)".to_string(),
    }
}

/// What to run after setup. An installed service runs `serve` on the default
/// ports, now or once started, so another `serve` there would fail: point at
/// starting the service and loading a model into its node instead.
fn next_step(actions: &CliSetupActions<'_>) -> String {
    let SetupServiceOutcome::Installed(report) = &actions.service_outcome else {
        return format!(
            "start a node with `mesh-llm serve --model <model>`. Quickstart: {QUICKSTART_URL}"
        );
    };
    let load = "load a model into it with `mesh-llm load <model>`";
    match report.status {
        // A started service is not yet a ready node: it can still fail to
        // bind, so say where to look if the console never answers.
        ServiceInstallStatus::Started => format!(
            "the service is starting a node: once http://localhost:3131 answers, {load}{}. \
             Quickstart: {QUICKSTART_SERVICE_URL}",
            report_line(report, "Logs: ")
                .map(|logs| format!("; if it never does, see {logs}"))
                .unwrap_or_default()
        ),
        ServiceInstallStatus::StartsAtLogon => format!(
            "the service starts a node at your next logon, or now from a desktop session with \
             `schtasks -Run -TN mesh-llm`; then {load}. Quickstart: {QUICKSTART_SERVICE_URL}"
        ),
        ServiceInstallStatus::NeedsManualStart => format!(
            "start the service{}, then {load}. Quickstart: {QUICKSTART_SERVICE_URL}",
            report_line(report, "Start it with: ")
                .map(|command| format!(" with `{command}`"))
                .unwrap_or_default()
        ),
    }
}

/// The rest of the service report line that starts with `prefix`.
fn report_line<'a>(
    report: &'a crate::setup::service::ServiceInstallReport,
    prefix: &str,
) -> Option<&'a str> {
    report
        .messages
        .iter()
        .find_map(|message| message.strip_prefix(prefix))
}

const QUICKSTART_URL: &str = "https://meshllm.cloud/docs/pages/quickstart/";
/// The quickstart step that covers a node the service already runs.
const QUICKSTART_SERVICE_URL: &str =
    "https://meshllm.cloud/docs/pages/quickstart/#3-start-one-private-node";

fn runtime_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.runtime {
        super::SetupRuntimePlan::Skip => "skipped by --skip-runtime".to_string(),
        super::SetupRuntimePlan::InstallAndPrune => match actions.runtime_outcome.as_ref() {
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Pruned(plan),
            }) => {
                let install_status = match installed.status {
                    NativeRuntimeInstallStatus::Installed => "installed",
                    NativeRuntimeInstallStatus::AlreadyInstalled => "already installed",
                };
                if plan.remove_dirs.is_empty() {
                    format!("{install_status}; cache already clean")
                } else {
                    format!(
                        "{install_status}; pruned {} inactive cache entr{}",
                        plan.remove_dirs.len(),
                        if plan.remove_dirs.len() == 1 {
                            "y"
                        } else {
                            "ies"
                        }
                    )
                }
            }
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Warning(_),
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => {
                    "installed; cache prune warning reported above".to_string()
                }
                NativeRuntimeInstallStatus::AlreadyInstalled => {
                    "already installed; cache prune warning reported above".to_string()
                }
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Skipped,
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => "installed".to_string(),
                NativeRuntimeInstallStatus::AlreadyInstalled => "already installed".to_string(),
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Skipped,
                ..
            }) => "skipped".to_string(),
            None => "not recorded".to_string(),
        },
    }
}

fn runtime_brief(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.runtime {
        super::SetupRuntimePlan::Skip => style_muted("skipped (--skip-runtime)"),
        super::SetupRuntimePlan::InstallAndPrune => match actions.runtime_outcome.as_ref() {
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Warning(_),
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => style_warn("installed; prune warning"),
                NativeRuntimeInstallStatus::AlreadyInstalled => {
                    style_warn("already installed; prune warning")
                }
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                ..
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => style_ok("ready"),
                NativeRuntimeInstallStatus::AlreadyInstalled => style_ok("already ready"),
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Skipped,
                ..
            }) => style_muted("skipped"),
            None => style_muted("not recorded"),
        },
    }
}

fn service_brief(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.service {
        super::SetupServicePlan::Skip => style_muted("not installed"),
        super::SetupServicePlan::Install => match actions.service_outcome {
            SetupServiceOutcome::Installed(ref report) => match report.status {
                ServiceInstallStatus::Started => style_ok("running"),
                ServiceInstallStatus::StartsAtLogon => {
                    style_warn("installed; starts at next logon")
                }
                ServiceInstallStatus::NeedsManualStart => style_warn("installed; start manually"),
            },
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::PrintedGuidance => {
                style_muted("not recorded")
            }
        },
        super::SetupServicePlan::PrintGuidance => match actions.service_outcome {
            SetupServiceOutcome::PrintedGuidance => style_muted("not installed"),
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::Installed(_) => {
                style_muted("not recorded")
            }
        },
    }
}

fn github_brief(actions: &CliSetupActions<'_>) -> Option<String> {
    match actions.github_outcome {
        super::github::SetupGitHubOutcome::Starred => Some(style_ok("starred")),
        super::github::SetupGitHubOutcome::AlreadyStarred => Some(style_ok("already starred")),
        super::github::SetupGitHubOutcome::StarRequestFailed(_)
        | super::github::SetupGitHubOutcome::EligibilityCheckFailed(_) => {
            Some(style_warn("not starred"))
        }
        super::github::SetupGitHubOutcome::CliUnavailable => {
            Some(style_muted("skipped; gh unavailable"))
        }
        super::github::SetupGitHubOutcome::NotAuthenticated => {
            Some(style_muted("skipped; gh not authenticated"))
        }
        super::github::SetupGitHubOutcome::NotEvaluated => Some(style_muted("not recorded")),
        _ => None,
    }
}

pub(crate) fn service_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.service {
        super::SetupServicePlan::Skip => "not requested".to_string(),
        super::SetupServicePlan::Install => match actions.service_outcome {
            SetupServiceOutcome::Installed(ref report) => report.summary.clone(),
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::PrintedGuidance => {
                "not recorded".to_string()
            }
        },
        super::SetupServicePlan::PrintGuidance => match actions.service_outcome {
            SetupServiceOutcome::PrintedGuidance => {
                "not installed; printed follow-up guidance".to_string()
            }
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::Installed(_) => {
                "not recorded".to_string()
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_native::NativeRuntimeConfigSelection;
    use crate::setup::SetupEnvironment;
    use crate::setup::SetupPlatform;
    use crate::setup::github::SetupGitHubOutcome;
    use crate::terminal::strip_ansi_styles;

    fn actions_with_github_outcome(github_outcome: SetupGitHubOutcome) -> CliSetupActions<'static> {
        let mut actions = CliSetupActions::new(
            SetupEnvironment {
                platform: SetupPlatform::Linux,
                interactive: false,
            },
            NativeRuntimeConfigSelection::default(),
            false,
        );
        actions.github_outcome = github_outcome;
        actions
    }

    #[test]
    fn incomplete_plugin_summary_includes_the_selected_set_in_the_retry() {
        let mut actions = actions_with_github_outcome(SetupGitHubOutcome::NotEvaluated);
        actions.plugin_outcome = SetupPluginOutcome::Incomplete;
        for (plugins, name) in [
            (
                crate::setup::SetupPluginPlan::Install("extra".into()),
                "extra",
            ),
            (crate::setup::SetupPluginPlan::Offer, "recommended"),
        ] {
            let plan = SetupPlan::new(
                crate::setup::SetupRuntimePlan::Skip,
                crate::setup::SetupServicePlan::Skip,
                plugins,
                crate::setup::SetupGitHubStarPlan::Skip(
                    crate::setup::SetupGitHubStarSkipReason::HiddenPrompt,
                ),
            );
            let retry = format!("`mesh-llm plugins sets install {name}`");
            assert!(plugin_brief(&plan, &actions).unwrap().contains(&retry));
            assert!(plugin_summary(&plan, &actions).contains(&retry));
        }
    }

    #[test]
    fn github_brief_reports_unavailable_cli() {
        let actions = actions_with_github_outcome(SetupGitHubOutcome::CliUnavailable);

        assert_eq!(
            github_brief(&actions).map(|brief| strip_ansi_styles(&brief)),
            Some("skipped; gh unavailable".to_string())
        );
    }

    fn installed_service(status: ServiceInstallStatus, messages: &[&str]) -> SetupServiceOutcome {
        SetupServiceOutcome::Installed(crate::setup::service::ServiceInstallReport {
            status,
            summary: String::new(),
            messages: messages.iter().map(|message| message.to_string()).collect(),
            service_file: std::path::PathBuf::new(),
            env_file: None,
            runner_file: None,
        })
    }

    fn next_step_for(outcome: SetupServiceOutcome) -> String {
        let mut actions = actions_with_github_outcome(SetupGitHubOutcome::NotAuthenticated);
        actions.service_outcome = outcome;
        next_step(&actions)
    }

    #[test]
    fn an_installed_service_never_gets_a_second_serve() {
        // Started now, at the next logon, or by hand: the service will hold
        // the default ports, so the next step loads into its node.
        for status in [
            ServiceInstallStatus::Started,
            ServiceInstallStatus::StartsAtLogon,
            ServiceInstallStatus::NeedsManualStart,
        ] {
            let next = next_step_for(installed_service(status, &[]));
            assert!(!next.contains("mesh-llm serve"), "{status:?}: {next}");
            assert!(next.contains("mesh-llm load"), "{status:?}: {next}");
        }
    }

    #[test]
    fn without_a_service_the_next_step_starts_a_node() {
        for outcome in [
            SetupServiceOutcome::NotRequested,
            SetupServiceOutcome::PrintedGuidance,
        ] {
            assert!(next_step_for(outcome).contains("mesh-llm serve"));
        }
    }

    #[test]
    fn the_service_report_supplies_the_start_command_and_the_logs() {
        let manual = next_step_for(installed_service(
            ServiceInstallStatus::NeedsManualStart,
            &["Start it with: systemctl --user enable --now mesh-llm"],
        ));
        assert!(manual.contains("systemctl --user enable --now mesh-llm"));
        let started = next_step_for(installed_service(
            ServiceInstallStatus::Started,
            &["Logs: journalctl --user -u mesh-llm -f"],
        ));
        assert!(started.contains("journalctl --user -u mesh-llm -f"));
    }

    #[test]
    fn github_brief_reports_unauthenticated_cli() {
        let actions = actions_with_github_outcome(SetupGitHubOutcome::NotAuthenticated);

        assert_eq!(
            github_brief(&actions).map(|brief| strip_ansi_styles(&brief)),
            Some("skipped; gh not authenticated".to_string())
        );
    }
}
