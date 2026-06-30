use super::SetupPlan;
use super::command::{CliSetupActions, SetupServiceOutcome};
use crate::runtime_native::{
    SetupNativeRuntimeOutcome, SetupNativeRuntimePruneResult, SetupNativeRuntimeStatus,
};
use crate::terminal::{style_muted, style_ok, style_warn};
use mesh_llm_runtime_install::NativeRuntimeInstallStatus;

pub(crate) fn print_runtime_install_result(outcome: &SetupNativeRuntimeOutcome) {
    match &outcome.status {
        SetupNativeRuntimeStatus::Skipped => {}
        SetupNativeRuntimeStatus::Installed(installed) => match installed.status {
            NativeRuntimeInstallStatus::Installed => eprintln!(
                "{} Installed native runtime {} for mesh version {}",
                style_ok("✓"),
                installed.runtime.native_runtime_id,
                installed.runtime.mesh_version
            ),
            NativeRuntimeInstallStatus::AlreadyInstalled => eprintln!(
                "{} Native runtime {} is already installed for mesh version {}",
                style_ok("✓"),
                installed.runtime.native_runtime_id,
                installed.runtime.mesh_version
            ),
        },
    }
}

pub(crate) fn print_service_install_result(
    report: &crate::setup::service::ServiceInstallReport,
    verbose: bool,
) {
    if verbose {
        for line in &report.messages {
            eprintln!("{line}");
        }
    }
}

pub(crate) fn print_setup_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>, verbose: bool) {
    eprintln!();
    if verbose {
        eprintln!("Setup summary");
        eprintln!("- Runtime: {}", runtime_summary(plan, actions));
        eprintln!("- Service: {}", service_summary(plan, actions));
        eprintln!(
            "- GitHub: {}",
            super::github::github_summary(plan, &actions.github_outcome)
        );
        return;
    }

    eprintln!("{} Mesh setup complete", style_ok("✓"));
    eprintln!("  Runtime  {}", runtime_brief(plan, actions));
    eprintln!("  Service  {}", service_brief(plan, actions));
    if let Some(github) = github_brief(actions) {
        eprintln!("  GitHub   {github}");
    }
}

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
            SetupServiceOutcome::Installed(ref report)
                if report.summary == "installed and started" =>
            {
                style_ok("running")
            }
            SetupServiceOutcome::Installed(_) => style_warn("installed; start manually"),
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
