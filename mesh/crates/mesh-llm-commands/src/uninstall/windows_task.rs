//! The Windows scheduled task that `mesh-llm setup` registers: the uninstall
//! steps that remove it, and carrying out the one that ends and deletes it.

use super::{
    RemovePurpose, SERVICE_NAME, UninstallEnvironment, UninstallOutcome, UninstallStep,
    run_best_effort,
};
use crate::setup::service_host::parse_service_host_record;
use crate::setup::service_paths::{
    WINDOWS_SERVICE_HOST_RECORD, WINDOWS_TASK_DEFINITION_FILE, windows_service_log_dir,
};
use std::{fs, path::Path, process::Command};

/// The step that ends and deletes the task. It runs first: ending the task
/// stops its service host, which would otherwise restart the node that the
/// next step stops.
pub(super) fn remove_task_step(env: &UninstallEnvironment) -> UninstallStep {
    UninstallStep::RemoveScheduledTask {
        service_host_record: windows_service_log_dir(&env.local_data_root)
            .join(WINDOWS_SERVICE_HOST_RECORD),
        binary_name: env
            .binary_path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| SERVICE_NAME.to_string()),
    }
}

/// What setup wrote for the task: its definition and the service logs.
pub(super) fn task_file_steps(env: &UninstallEnvironment) -> [UninstallStep; 2] {
    [
        UninstallStep::RemovePath {
            path: env
                .config_root
                .join("mesh-llm")
                .join(WINDOWS_TASK_DEFINITION_FILE),
            purpose: RemovePurpose::ScheduledTaskDefinition,
        },
        UninstallStep::RemovePath {
            path: windows_service_log_dir(&env.local_data_root),
            purpose: RemovePurpose::WindowsServiceLogs,
        },
    ]
}

/// Carries out [`UninstallStep::RemoveScheduledTask`].
pub(super) fn remove_scheduled_task(
    service_host_record: &Path,
    binary_name: &str,
    outcome: &mut UninstallOutcome,
) {
    // Nothing to report when setup never registered the task, but a
    // failed lookup is not an absent task.
    let registered = match scheduled_task_registered() {
        Ok(registered) => registered,
        Err(error) => {
            outcome.warnings.push(format!(
                "could not check for the {SERVICE_NAME} scheduled task ({error}); if it is still registered, remove it with: schtasks /End /TN {SERVICE_NAME}, then schtasks /Delete /TN {SERVICE_NAME} /F"
            ));
            false
        }
    };
    if registered {
        run_best_effort(
            Command::new("schtasks").args(["/End", "/TN", SERVICE_NAME]),
            outcome,
        );
    }
    // Ending the task ends its service host. Should that have failed,
    // or a host outlived its task, the host would restart the node the
    // next step stops, so stop the host it recorded; its job takes the
    // node down with it.
    stop_recorded_service_host(service_host_record, binary_name, outcome);
    // Best effort, like disabling the systemd unit: a task left behind
    // only fails at the next logon once the binary is gone, so say how
    // to remove it rather than stop the uninstall.
    if registered
        && !Command::new("schtasks")
            .args(["/Delete", "/TN", SERVICE_NAME, "/F"])
            .output()
            .is_ok_and(|output| output.status.success())
    {
        outcome.warnings.push(format!(
            "the {SERVICE_NAME} scheduled task is still registered and will fail at the next logon without the binary; delete it with: schtasks /Delete /TN {SERVICE_NAME} /F"
        ));
    }
}

/// Whether setup's scheduled task is registered. `schtasks /Query /TN` fails
/// the same way for an absent task and for a lookup error, so this lists every
/// task: a failed listing is an error, and a listing without the task means it
/// is absent.
fn scheduled_task_registered() -> Result<bool, String> {
    let output = Command::new("schtasks")
        .args(["/Query", "/FO", "CSV", "/NH"])
        .output()
        .map_err(|error| format!("failed to run schtasks: {error}"))?;
    if !output.status.success() {
        // An unrelated damaged task can fail the listing; a named query that
        // succeeds still proves this one is registered.
        if Command::new("schtasks")
            .args(["/Query", "/TN", SERVICE_NAME])
            .output()
            .is_ok_and(|named| named.status.success())
        {
            return Ok(true);
        }
        let said = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "`schtasks /Query` exited with status {}: {said}",
            output.status
        ));
    }
    Ok(task_listed(
        &String::from_utf8_lossy(&output.stdout),
        SERVICE_NAME,
    ))
}

/// Whether `schtasks /Query /FO CSV /NH` lists the root-folder task `name`.
fn task_listed(listing: &str, name: &str) -> bool {
    let wanted = format!("\"\\{name}\"");
    listing
        .lines()
        .any(|line| line.trim_start().split(',').next() == Some(wanted.as_str()))
}

/// Stops the service host recorded at `record` when a process with that id
/// and start time still runs `binary_name`, so a reused id is never touched.
/// A console process only ends with `/F`.
fn stop_recorded_service_host(record: &Path, binary_name: &str, outcome: &mut UninstallOutcome) {
    let Some((pid, started_at_unix)) = fs::read_to_string(record)
        .ok()
        .and_then(|record| parse_service_host_record(&record))
    else {
        return;
    };
    if !mesh_llm_system::process::validate_pid_matches(pid, binary_name, started_at_unix) {
        return;
    }
    run_best_effort(
        Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]),
        outcome,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uninstall::tests::{env, options};
    use crate::uninstall::{UninstallPlatform, plan_uninstall};

    fn empty_outcome() -> UninstallOutcome {
        UninstallOutcome {
            dry_run: false,
            removed: Vec::new(),
            scheduled_removal: Vec::new(),
            already_absent: Vec::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn windows_plan_ends_the_task_before_stopping_processes_and_removes_its_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plan = plan_uninstall(&options(), &env(temp.path(), UninstallPlatform::Windows));

        assert_eq!(
            &plan.steps[..2],
            &[
                UninstallStep::RemoveScheduledTask {
                    service_host_record: temp
                        .path()
                        .join("local")
                        .join("mesh-llm")
                        .join("logs")
                        .join("service-host.pid"),
                    binary_name: "mesh-llm".to_string(),
                },
                UninstallStep::StopProcesses
            ]
        );
        let removed = |purpose| {
            plan.steps.iter().position(|step| {
                matches!(step, UninstallStep::RemovePath { purpose: p, .. } if *p == purpose)
            })
        };
        let definition = removed(RemovePurpose::ScheduledTaskDefinition)
            .expect("the task definition is removed");
        assert_eq!(
            plan.steps[definition],
            UninstallStep::RemovePath {
                path: temp.path().join("config/mesh-llm/mesh-llm-task.xml"),
                purpose: RemovePurpose::ScheduledTaskDefinition,
            }
        );
        assert!(plan.steps.contains(&UninstallStep::RemovePath {
            path: temp.path().join("local/mesh-llm/logs"),
            purpose: RemovePurpose::WindowsServiceLogs,
        }));
        // The definition sits in the service config directory, which is only
        // removed once empty.
        assert!(
            definition < removed(RemovePurpose::ServiceConfigDir).expect("config directory step")
        );
    }

    #[test]
    fn unix_plans_have_no_scheduled_task_step() {
        let temp = tempfile::tempdir().expect("tempdir");
        for platform in [UninstallPlatform::Linux, UninstallPlatform::MacOs] {
            let plan = plan_uninstall(&options(), &env(temp.path(), platform));
            assert_eq!(plan.steps[0], UninstallStep::StopProcesses);
            assert!(
                !plan
                    .steps
                    .iter()
                    .any(|step| matches!(step, UninstallStep::RemoveScheduledTask { .. }))
            );
        }
    }

    #[test]
    fn the_task_listing_names_only_the_root_mesh_llm_task() {
        let listing = concat!(
            r#""\Microsoft\Windows\UpdateOrchestrator\Schedule Scan","N/A","Ready""#,
            "\r\n",
            r#""\mesh-llm-probe","N/A","Ready""#,
            "\r\n",
            r#""\Vendor\mesh-llm","N/A","Ready""#,
            "\r\n",
        );
        assert!(!task_listed(listing, "mesh-llm"));

        let listing = format!("{listing}{}\r\n", r#""\mesh-llm","N/A","Running""#);
        assert!(task_listed(&listing, "mesh-llm"));
        assert!(!task_listed(
            "INFO: There are no scheduled tasks presently available.\r\n",
            "mesh-llm"
        ));
    }

    #[cfg(windows)]
    #[test]
    fn the_scheduled_task_lookup_runs_on_windows() {
        // Registered or not, listing the tasks must not be an error here.
        assert!(scheduled_task_registered().is_ok());
    }

    #[cfg(windows)]
    fn long_running_child() -> (std::process::Child, i64) {
        let child = Command::new("ping")
            .args(["-n", "30", "127.0.0.1"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn ping");
        let started_at_unix = mesh_llm_system::process::process_started_at_unix(child.id())
            .expect("start time")
            .expect("ping is running");
        (child, started_at_unix)
    }

    #[cfg(windows)]
    #[test]
    fn a_recorded_service_host_that_still_runs_is_stopped() {
        let temp = tempfile::tempdir().expect("tempdir");
        let record = temp.path().join("service-host.pid");
        let (mut child, started_at_unix) = long_running_child();
        fs::write(
            &record,
            crate::setup::service_host::render_service_host_record(child.id(), started_at_unix),
        )
        .expect("write record");
        let mut outcome = empty_outcome();

        stop_recorded_service_host(&record, "PING", &mut outcome);

        let status = child.wait().expect("wait for ping");
        assert!(!status.success(), "ping was killed, not finished: {status}");
        assert!(outcome.warnings.is_empty(), "{:?}", outcome.warnings);
    }

    #[cfg(windows)]
    #[test]
    fn a_record_whose_start_time_differs_stops_nothing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let record = temp.path().join("service-host.pid");
        let (mut child, started_at_unix) = long_running_child();
        // The same id with another start time is a different process.
        fs::write(
            &record,
            crate::setup::service_host::render_service_host_record(
                child.id(),
                started_at_unix - 3600,
            ),
        )
        .expect("write record");
        let mut outcome = empty_outcome();

        stop_recorded_service_host(&record, "PING", &mut outcome);

        assert!(
            child.try_wait().expect("try_wait").is_none(),
            "a reused id must not be killed"
        );
        child.kill().expect("kill ping");
        let _ = child.wait();
    }
}
