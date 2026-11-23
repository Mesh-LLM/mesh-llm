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

/// What `install.ps1` adds beside the binary on Windows: the console launcher,
/// and the MeshLLM Start menu shortcut that runs it. Without this step both
/// would outlive the binary and the shortcut would open a launcher that fails.
pub(super) fn start_menu_step(env: &UninstallEnvironment) -> UninstallStep {
    // The installer records the launcher by absolute path, so a relative
    // `--binary-path` is made absolute before the launcher path is derived.
    let binary = std::path::absolute(&env.binary_path).unwrap_or_else(|_| env.binary_path.clone());
    let install_dir = binary.parent().map(Path::to_path_buf).unwrap_or_default();
    UninstallStep::RemoveStartMenuEntry {
        launcher: install_dir.join(CONSOLE_LAUNCHER),
    }
}

/// The name `install.ps1` gives the launcher, and the line in it that records
/// where the installer put the shortcut.
const CONSOLE_LAUNCHER: &str = "mesh-llm-console.ps1";
const SHORTCUT_RECORD: &str = "# Start menu shortcut: ";

/// Remove this copy's launcher, and the shortcut it records when that shortcut
/// still opens this launcher: a shortcut another installation has since put
/// in the same place opens that copy and is left. Only files are removed.
pub(super) fn remove_start_menu_entry(
    launcher: &Path,
    outcome: &mut UninstallOutcome,
) -> std::io::Result<()> {
    let contents = match fs::symlink_metadata(launcher) {
        Ok(metadata) if metadata.is_file() => fs::read_to_string(launcher)?,
        Ok(_) => {
            outcome.warnings.push(format!(
                "leaving {}: it is not a launcher file",
                launcher.display()
            ));
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            outcome.already_absent.push(launcher.to_path_buf());
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let shortcut = contents
        .lines()
        .find_map(|line| line.trim().strip_prefix(SHORTCUT_RECORD))
        .map(|path| Path::new(path.trim()).to_path_buf());
    if let Some(shortcut) = shortcut {
        match fs::symlink_metadata(&shortcut) {
            Ok(metadata) if metadata.is_file() => {
                if shortcut_opens(&fs::read(&shortcut)?, launcher) {
                    fs::remove_file(&shortcut)?;
                    outcome.removed.push(shortcut);
                } else {
                    outcome.warnings.push(format!(
                        "leaving the Start menu shortcut {}: it opens another installation",
                        shortcut.display()
                    ));
                }
            }
            Ok(_) => outcome.warnings.push(format!(
                "leaving {}: it is not a shortcut file",
                shortcut.display()
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                outcome.already_absent.push(shortcut);
            }
            Err(error) => return Err(error),
        }
    }
    fs::remove_file(launcher)?;
    outcome.removed.push(launcher.to_path_buf());
    Ok(())
}

/// Whether a `.lnk` file runs `launcher`. `install.ps1` writes the arguments as
/// `-File "<launcher>"`, and shell links keep their arguments as UTF-16
/// strings, so that exact quoted argument is looked for in that encoding,
/// ignoring case as Windows paths do. The closing quote keeps a longer path
/// such as `<launcher>.backup.ps1` from matching.
fn shortcut_opens(link: &[u8], launcher: &Path) -> bool {
    let wanted = format!("-file \"{}\"", launcher.to_string_lossy()).to_lowercase();
    // A string can start on an odd byte, so both alignments are read.
    [0, 1].into_iter().any(|start| {
        let (pairs, _) = link.get(start..).unwrap_or_default().as_chunks::<2>();
        let units: Vec<u16> = pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect();
        String::from_utf16_lossy(&units)
            .to_lowercase()
            .contains(&wanted)
    })
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

/// What `mesh-llm stop` did about the Windows background service.
#[derive(Debug, PartialEq, Eq)]
pub enum BackgroundServiceStop {
    /// No service host is recorded, or the recorded one no longer runs.
    NotRunning,
    /// The service host ended, and its node with it.
    Ended,
    /// The service host still runs; what ending it reported.
    StillRunning(Vec<String>),
}

/// For `mesh-llm stop` on Windows. The service host starts `mesh-llm serve`
/// again after any failed exit, and `stop` can only force-end a node that has
/// no console window, so ending the node alone does not stop the service.
/// End the scheduled task instead, then the host itself if it outlived its
/// task, as uninstall does; the host's job takes the node down with it. The
/// task stays registered and starts again at the next logon.
pub fn end_background_service() -> BackgroundServiceStop {
    // Only Windows installs this service; `stop` calls this on every platform.
    if !cfg!(windows) {
        return BackgroundServiceStop::NotRunning;
    }
    let Some(local_data_root) = dirs::data_local_dir() else {
        return BackgroundServiceStop::NotRunning;
    };
    let record = windows_service_log_dir(&local_data_root).join(WINDOWS_SERVICE_HOST_RECORD);
    let binary_name = std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| SERVICE_NAME.to_string());
    end_recorded_service(&record, &binary_name, || {
        let _ = Command::new("schtasks")
            .args(["/End", "/TN", SERVICE_NAME])
            .output();
    })
}

fn end_recorded_service(
    record: &Path,
    binary_name: &str,
    end_task: impl FnOnce(),
) -> BackgroundServiceStop {
    let Some((pid, started_at_unix)) = fs::read_to_string(record)
        .ok()
        .and_then(|record| parse_service_host_record(&record))
    else {
        return BackgroundServiceStop::NotRunning;
    };
    if !mesh_llm_system::process::validate_pid_matches(pid, binary_name, started_at_unix) {
        return BackgroundServiceStop::NotRunning;
    }
    end_task();
    if ends_within_five_seconds(pid) {
        return BackgroundServiceStop::Ended;
    }
    let mut outcome = UninstallOutcome {
        dry_run: false,
        removed: Vec::new(),
        scheduled_removal: Vec::new(),
        already_absent: Vec::new(),
        warnings: Vec::new(),
    };
    stop_recorded_service_host(record, binary_name, &mut outcome);
    if ends_within_five_seconds(pid) {
        BackgroundServiceStop::Ended
    } else {
        BackgroundServiceStop::StillRunning(outcome.warnings)
    }
}

fn ends_within_five_seconds(pid: u32) -> bool {
    for _ in 0..20 {
        if mesh_llm_system::process::process_liveness(pid)
            == mesh_llm_system::process::Liveness::Dead
        {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uninstall::tests::{env, options};
    use crate::uninstall::{UninstallPlatform, plan_uninstall};

    #[cfg(windows)]
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
    fn windows_plan_removes_the_start_menu_shortcut_and_console_launcher() {
        let temp = tempfile::tempdir().expect("tempdir");
        let windows = env(temp.path(), UninstallPlatform::Windows);
        let plan = plan_uninstall(&options(), &windows);
        let install_dir = windows
            .binary_path
            .parent()
            .expect("binary has a directory");
        assert!(plan.steps.contains(&UninstallStep::RemoveStartMenuEntry {
            launcher: install_dir.join("mesh-llm-console.ps1"),
        }));
        for platform in [UninstallPlatform::Linux, UninstallPlatform::MacOs] {
            let plan = plan_uninstall(&options(), &env(temp.path(), platform));
            assert!(
                !plan
                    .steps
                    .iter()
                    .any(|step| matches!(step, UninstallStep::RemoveStartMenuEntry { .. }))
            );
        }
    }

    /// A launcher as `install.ps1` writes it, recording `shortcut`.
    fn write_launcher(dir: &Path, shortcut: &Path) -> std::path::PathBuf {
        let launcher = dir.join("mesh-llm-console.ps1");
        fs::write(
            &launcher,
            format!(
                "# Opens the MeshLLM console.\r\n# Start menu shortcut: {}\r\n$ErrorActionPreference = 'Stop'\r\n",
                shortcut.display()
            ),
        )
        .unwrap();
        launcher
    }

    /// A stand-in `.lnk` whose arguments name `launcher` in UTF-16, after an
    /// odd-length header so the string starts on an odd byte.
    fn write_link(path: &Path, launcher: &Path) {
        let mut bytes = vec![0x4c, 0, 0, 0, 1];
        let arguments = format!("-File \"{}\"", launcher.display());
        for unit in arguments.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    fn outcome() -> UninstallOutcome {
        UninstallOutcome {
            dry_run: false,
            removed: Vec::new(),
            scheduled_removal: Vec::new(),
            already_absent: Vec::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn removes_the_launcher_and_the_shortcut_that_opens_it() {
        let temp = tempfile::tempdir().expect("tempdir");
        let shortcut = temp.path().join("MeshLLM.lnk");
        let launcher = write_launcher(temp.path(), &shortcut);
        // Windows paths ignore case, and so does the check.
        write_link(
            &shortcut,
            Path::new(&launcher.to_string_lossy().to_uppercase()),
        );
        let mut done = outcome();
        remove_start_menu_entry(&launcher, &mut done).unwrap();
        assert!(!shortcut.exists() && !launcher.exists());
        assert_eq!(done.removed, [shortcut, launcher]);
    }

    #[test]
    fn leaves_a_shortcut_that_opens_another_installation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let shortcut = temp.path().join("MeshLLM.lnk");
        let other = temp.path().join("other");
        fs::create_dir(&other).unwrap();
        write_link(&shortcut, &other.join("mesh-llm-console.ps1"));
        let launcher = write_launcher(temp.path(), &shortcut);
        let mut done = outcome();
        remove_start_menu_entry(&launcher, &mut done).unwrap();
        assert!(shortcut.is_file(), "the other copy's shortcut stays");
        assert!(!launcher.exists());
        assert!(done.warnings[0].contains("opens another installation"));

        // A path that only starts with this launcher's is another file.
        let launcher = write_launcher(temp.path(), &shortcut);
        let mut longer = launcher.clone().into_os_string();
        longer.push(".backup.ps1");
        write_link(&shortcut, Path::new(&longer));
        let mut done = outcome();
        remove_start_menu_entry(&launcher, &mut done).unwrap();
        assert!(shortcut.is_file(), "a shortcut to a longer path stays");
    }

    #[test]
    fn a_relative_binary_path_gives_an_absolute_launcher() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut windows = env(temp.path(), UninstallPlatform::Windows);
        windows.binary_path = Path::new("bin").join("mesh-llm.exe");
        let UninstallStep::RemoveStartMenuEntry { launcher } = start_menu_step(&windows) else {
            panic!("expected the Start menu step");
        };
        assert!(launcher.is_absolute(), "{}", launcher.display());
        assert!(launcher.ends_with(Path::new("bin").join("mesh-llm-console.ps1")));
    }

    #[test]
    fn removes_only_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        // A directory where the shortcut is recorded is not removed.
        let shortcut = temp.path().join("MeshLLM.lnk");
        fs::create_dir(&shortcut).unwrap();
        fs::write(shortcut.join("keep.txt"), b"kept").unwrap();
        let launcher = write_launcher(temp.path(), &shortcut);
        let mut done = outcome();
        remove_start_menu_entry(&launcher, &mut done).unwrap();
        assert!(shortcut.join("keep.txt").is_file());
        assert!(!launcher.exists());

        // Nor is a directory at the launcher's path.
        let dir_launcher = temp.path().join("nested").join("mesh-llm-console.ps1");
        fs::create_dir_all(dir_launcher.join("inner")).unwrap();
        let mut done = outcome();
        remove_start_menu_entry(&dir_launcher, &mut done).unwrap();
        assert!(dir_launcher.join("inner").is_dir());
        assert!(done.warnings[0].contains("not a launcher file"));

        // A launcher this copy never wrote is already absent.
        let mut done = outcome();
        remove_start_menu_entry(&temp.path().join("missing.ps1"), &mut done).unwrap();
        assert_eq!(done.already_absent.len(), 1);
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

    #[cfg(windows)]
    #[test]
    fn stop_ends_a_running_service_host_through_its_task_then_directly() {
        let temp = tempfile::tempdir().expect("tempdir");
        let record = temp.path().join("service-host.pid");
        let (mut child, started_at_unix) = long_running_child();
        fs::write(
            &record,
            crate::setup::service_host::render_service_host_record(child.id(), started_at_unix),
        )
        .expect("write record");
        let mut task_ended = 0;

        // The test never touches a real task: ending it does nothing here, so
        // the host outlives it and has to be ended directly.
        let stop = end_recorded_service(&record, "PING", || task_ended += 1);

        assert_eq!(stop, BackgroundServiceStop::Ended);
        assert_eq!(task_ended, 1);
        let status = child.wait().expect("wait for ping");
        assert!(!status.success(), "ping was killed, not finished: {status}");
    }

    #[cfg(windows)]
    #[test]
    fn stop_leaves_the_task_alone_without_a_running_recorded_host() {
        let temp = tempfile::tempdir().expect("tempdir");
        let record = temp.path().join("service-host.pid");
        let mut task_ended = 0;
        assert_eq!(
            end_recorded_service(&record, "PING", || task_ended += 1),
            BackgroundServiceStop::NotRunning
        );

        let (mut child, started_at_unix) = long_running_child();
        fs::write(
            &record,
            crate::setup::service_host::render_service_host_record(
                child.id(),
                started_at_unix - 3600,
            ),
        )
        .expect("write record");
        assert_eq!(
            end_recorded_service(&record, "PING", || task_ended += 1),
            BackgroundServiceStop::NotRunning
        );
        assert_eq!(task_ended, 0);
        assert!(
            child.try_wait().expect("try_wait").is_none(),
            "a reused id must not be killed"
        );
        child.kill().expect("kill ping");
        let _ = child.wait();
    }
}
