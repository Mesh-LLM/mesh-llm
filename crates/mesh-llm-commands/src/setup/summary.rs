use super::SetupPlan;
use super::command::{CliSetupActions, SetupServiceOutcome};
use crate::runtime_native::{
    SetupNativeRuntimeOutcome, SetupNativeRuntimePruneResult, SetupNativeRuntimeStatus,
};
use mesh_llm_runtime_install::NativeRuntimeInstallStatus;

pub(crate) fn print_runtime_install_result(outcome: &SetupNativeRuntimeOutcome) {
    match &outcome.status {
        SetupNativeRuntimeStatus::Skipped => {}
        SetupNativeRuntimeStatus::Installed(installed) => match installed.status {
            NativeRuntimeInstallStatus::Installed => eprintln!(
                "✅ Installed native runtime {} for mesh version {}",
                installed.runtime.native_runtime_id, installed.runtime.mesh_version
            ),
            NativeRuntimeInstallStatus::AlreadyInstalled => eprintln!(
                "✅ Native runtime {} is already installed for mesh version {}",
                installed.runtime.native_runtime_id, installed.runtime.mesh_version
            ),
        },
    }
}

pub(crate) fn print_service_install_result(report: &crate::setup::service::ServiceInstallReport) {
    for line in &report.messages {
        eprintln!("{line}");
    }
}

pub(crate) fn print_setup_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>) {
    eprintln!();
    eprintln!("Setup summary");
    eprintln!("- Runtime: {}", runtime_summary(plan, actions));
    eprintln!("- Service: {}", service_summary(plan, actions));
    eprintln!(
        "- GitHub: {}",
        super::github::github_summary(plan, &actions.github_outcome)
    );
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
