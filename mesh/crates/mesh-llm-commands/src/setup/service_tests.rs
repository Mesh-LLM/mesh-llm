use super::SetupPlatform;
use super::service::{ServiceInstallStatus, install_service, windows_task_last_result};
use super::service_paths::{ServiceInstallContext, parse_whoami_user_sid};
use super::service_runner::{ServiceCommand, ServiceCommandRunner};
use super::service_templates::{
    SERVICE_LABEL, render_launchd_plist, render_service_env_file, render_service_runner,
    render_systemd_unit, render_windows_task_xml,
};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default)]
struct FakeRunner {
    commands: Vec<ServiceCommand>,
    failures: HashMap<String, VecDeque<String>>,
    outputs: HashMap<String, String>,
}

impl FakeRunner {
    fn fail_once(&mut self, starts_with: &str, message: &str) {
        self.failures
            .entry(starts_with.to_string())
            .or_default()
            .push_back(message.to_string());
    }

    fn respond(&mut self, starts_with: &str, output: &str) {
        self.outputs
            .insert(starts_with.to_string(), output.to_string());
    }
}

impl ServiceCommandRunner for FakeRunner {
    fn run(&mut self, command: &ServiceCommand) -> Result<()> {
        self.output(command).map(|_| ())
    }

    fn output(&mut self, command: &ServiceCommand) -> Result<String> {
        let rendered = command.display();
        self.commands.push(command.clone());
        for (prefix, failures) in &mut self.failures {
            if rendered.starts_with(prefix)
                && let Some(message) = failures.pop_front()
            {
                return Err(anyhow!(message));
            }
        }
        Ok(self
            .outputs
            .iter()
            .find(|(prefix, _)| rendered.starts_with(prefix.as_str()))
            .map(|(_, output)| output.clone())
            .unwrap_or_default())
    }
}

#[test]
fn rendered_templates_match_existing_unix_service_behavior() {
    let binary_path = PathBuf::from("/Users/example/.local/bin/mesh-llm");
    let env_file = PathBuf::from("/Users/example/.config/mesh-llm/service.env");
    let mesh_config = PathBuf::from("/Users/example/.mesh-llm/config.toml");
    let service_runner = PathBuf::from("/Users/example/.config/mesh-llm/run-service.sh");
    let home_dir = PathBuf::from("/Users/example");
    let stdout_log = PathBuf::from("/Users/example/Library/Logs/mesh-llm/stdout.log");
    let stderr_log = PathBuf::from("/Users/example/Library/Logs/mesh-llm/stderr.log");

    assert_eq!(
        render_service_env_file(),
        "# Optional environment variables for mesh-llm.\n# Use plain KEY=value lines.\n# Example:\n# RUST_LOG=mesh_inference=debug\n\n# Join a private mesh without editing the generated unit.\n# The token file is re-read on every rejoin, so rotating the invite\n# token is just replacing the file's contents.\n# A token at the default location needs no line here at all: an\n# invite.token beside the resolved config file, which is\n# ~/.mesh-llm/invite.token for the default config path.\n# MESH_LLM_JOIN_FILE=/home/you/.mesh-llm/invite.token\n# MESH_LLM_JOIN=<complete invite token>\n"
    );
    assert_eq!(
        render_service_runner(&binary_path, &env_file),
        format!(
            "#!/usr/bin/env bash\n\nset -euo pipefail\n\nBIN=\"{}\"\nENV_FILE=\"{}\"\n\nif [[ ! -x \"$BIN\" ]]; then\n    echo \"mesh-llm binary not found or not executable: $BIN\" >&2\n    exit 1\nfi\n\nif [[ -f \"$ENV_FILE\" ]]; then\n    set -a\n    # shellcheck source=/dev/null\n    . \"$ENV_FILE\"\n    set +a\nfi\n\nexec \"$BIN\" serve\n",
            binary_path.display(),
            env_file.display()
        )
    );
    assert_eq!(
        render_systemd_unit(&binary_path, &env_file, &mesh_config),
        format!(
            "# mesh-llm serve (startup models come from {mesh_config})\n[Unit]\nDescription=Mesh LLM user service\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nEnvironmentFile=-{env_file}\n\nExecStart=\"/Users/example/.local/bin/mesh-llm\" serve\nWorkingDirectory=%h\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
            mesh_config = mesh_config.display(),
            env_file = env_file.display(),
        )
    );
    assert_eq!(
        render_launchd_plist(&service_runner, &home_dir, &stdout_log, &stderr_log),
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"https://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n    <key>Label</key>\n    <string>{SERVICE_LABEL}</string>\n    <key>ProgramArguments</key>\n    <array>\n        <string>{service_runner}</string>\n    </array>\n    <key>WorkingDirectory</key>\n    <string>{home_dir}</string>\n    <key>RunAtLoad</key>\n    <true/>\n    <key>KeepAlive</key>\n    <dict>\n        <key>SuccessfulExit</key>\n        <false/>\n    </dict>\n    <key>ProcessType</key>\n    <string>Background</string>\n    <key>StandardOutPath</key>\n    <string>{stdout_log}</string>\n    <key>StandardErrorPath</key>\n    <string>{stderr_log}</string>\n</dict>\n</plist>\n",
            service_runner = service_runner.display(),
            home_dir = home_dir.display(),
            stdout_log = stdout_log.display(),
            stderr_log = stderr_log.display(),
        )
    );
}

#[test]
fn launchd_runner_escapes_shell_specials_in_paths() {
    let rendered = render_service_runner(
        &PathBuf::from("/Users/example/mesh \"bin\"/$HOME/`mesh`/mesh-llm"),
        &PathBuf::from("/Users/example/config\\dir/service.env"),
    );

    assert!(
        rendered.contains("BIN=\"/Users/example/mesh \\\"bin\\\"/\\$HOME/\\`mesh\\`/mesh-llm\"")
    );
    assert!(rendered.contains("ENV_FILE=\"/Users/example/config\\\\dir/service.env\""));
}

#[test]
fn launchd_plist_escapes_xml_specials_in_paths() {
    let rendered = render_launchd_plist(
        &PathBuf::from("/Users/example/A&B/run-service.sh"),
        &PathBuf::from("/Users/example/<home>"),
        &PathBuf::from("/Users/example/logs/stdout>log"),
        &PathBuf::from("/Users/example/logs/stderr<log"),
    );

    assert!(rendered.contains("<string>/Users/example/A&amp;B/run-service.sh</string>"));
    assert!(rendered.contains("<string>/Users/example/&lt;home&gt;</string>"));
    assert!(rendered.contains("<string>/Users/example/logs/stdout&gt;log</string>"));
    assert!(rendered.contains("<string>/Users/example/logs/stderr&lt;log</string>"));
}

#[test]
fn linux_service_install_writes_systemd_files_and_runs_expected_commands() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let home_dir = temp.path().join("home");
    let config_root = temp.path().join("config");
    let binary_path = temp.path().join("bin/mesh-llm");
    fs::create_dir_all(binary_path.parent().expect("binary parent should exist"))
        .expect("binary dir should exist");
    fs::write(&binary_path, "binary").expect("binary should write");

    let context = ServiceInstallContext {
        platform: SetupPlatform::Linux,
        home_dir,
        config_root,
        local_data_root: temp.path().join("local"),
        binary_path: binary_path.clone(),
        user_id: String::new(),
        start_service: true,
    };
    let mut runner = FakeRunner::default();

    let report = install_service(&context, &mut runner).expect("systemd install should succeed");
    let env_file = report
        .env_file
        .clone()
        .expect("systemd install writes service.env");

    #[cfg(unix)]
    {
        assert_eq!(unix_mode(&env_file), 0o600);
        assert_eq!(
            unix_mode(env_file.parent().expect("env file parent")),
            0o700
        );
    }

    assert_eq!(report.summary, "installed and started");
    assert_eq!(report.status, ServiceInstallStatus::Started);
    assert_eq!(
        fs::read_to_string(&env_file).expect("env file should exist"),
        render_service_env_file()
    );
    assert!(report.runner_file.is_none());
    assert!(
        fs::read_to_string(&report.service_file)
            .expect("unit file should exist")
            .contains("ExecStart=")
    );
    assert_eq!(
        runner.commands,
        vec![
            ServiceCommand::new("systemctl", ["--user", "daemon-reload"]),
            ServiceCommand::new("systemctl", ["--user", "enable", "mesh-llm.service"]),
            ServiceCommand::new("systemctl", ["--user", "restart", "mesh-llm.service"]),
        ]
    );
}

#[test]
fn systemd_unit_escapes_percent_in_environment_file_path() {
    let rendered = render_systemd_unit(
        &PathBuf::from("/Users/example/.local/bin/mesh-llm"),
        &PathBuf::from("/Users/example/.config/mesh-llm/%service.env"),
        &PathBuf::from("/Users/example/.mesh-llm/config.toml"),
    );

    assert!(rendered.contains("EnvironmentFile=-/Users/example/.config/mesh-llm/%%service.env"));
    assert!(
        !rendered
            .lines()
            .any(|line| line == "EnvironmentFile=-/Users/example/.config/mesh-llm/%service.env")
    );
}

#[test]
fn macos_service_install_writes_runner_and_plist_and_preserves_manual_start_guidance() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let home_dir = temp.path().join("home");
    let config_root = temp.path().join("config");
    let binary_path = temp.path().join("bin/mesh-llm");
    fs::create_dir_all(binary_path.parent().expect("binary parent should exist"))
        .expect("binary dir should exist");
    fs::write(&binary_path, "binary").expect("binary should write");

    let context = ServiceInstallContext {
        platform: SetupPlatform::MacOs,
        home_dir: home_dir.clone(),
        config_root,
        local_data_root: temp.path().join("local"),
        binary_path: binary_path.clone(),
        user_id: "501".to_string(),
        start_service: false,
    };
    let mut runner = FakeRunner::default();

    let report = install_service(&context, &mut runner).expect("launchd install should succeed");
    let env_file = report
        .env_file
        .clone()
        .expect("launchd install writes service.env");

    #[cfg(unix)]
    {
        assert_eq!(unix_mode(&env_file), 0o600);
        assert_eq!(
            unix_mode(env_file.parent().expect("env file parent")),
            0o700
        );
    }

    assert_eq!(
        report.summary,
        "installed; automatic start needs manual follow-up"
    );
    let runner_file = report
        .runner_file
        .expect("launchd runner should be recorded");
    assert_eq!(
        fs::read_to_string(&runner_file).expect("runner should exist"),
        render_service_runner(&binary_path, &env_file)
    );
    assert!(
        fs::read_to_string(&report.service_file)
            .expect("plist should exist")
            .contains(SERVICE_LABEL)
    );
    assert!(
        report
            .messages
            .iter()
            .any(|line| line.contains("Start it with: launchctl bootstrap gui/501"))
    );
}

#[test]
fn linux_service_command_failure_is_a_setup_failure_when_starting_service() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let home_dir = temp.path().join("home");
    let config_root = temp.path().join("config");
    let binary_path = temp.path().join("bin/mesh-llm");
    fs::create_dir_all(binary_path.parent().expect("binary parent should exist"))
        .expect("binary dir should exist");
    fs::write(&binary_path, "binary").expect("binary should write");

    let context = ServiceInstallContext {
        platform: SetupPlatform::Linux,
        home_dir,
        config_root,
        local_data_root: temp.path().join("local"),
        binary_path,
        user_id: String::new(),
        start_service: true,
    };
    let mut runner = FakeRunner::default();
    runner.fail_once(
        "systemctl --user enable",
        "systemd user manager unavailable",
    );

    let error =
        install_service(&context, &mut runner).expect_err("systemd enable failure should fail");

    assert!(
        error
            .to_string()
            .contains("enable systemd user service mesh-llm.service"),
        "{error:#}"
    );
}

#[test]
fn macos_service_command_failure_is_a_setup_failure_when_starting_service() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let home_dir = temp.path().join("home");
    let config_root = temp.path().join("config");
    let binary_path = temp.path().join("bin/mesh-llm");
    fs::create_dir_all(binary_path.parent().expect("binary parent should exist"))
        .expect("binary dir should exist");
    fs::write(&binary_path, "binary").expect("binary should write");

    let context = ServiceInstallContext {
        platform: SetupPlatform::MacOs,
        home_dir,
        config_root,
        local_data_root: temp.path().join("local"),
        binary_path,
        user_id: "501".to_string(),
        start_service: true,
    };
    let mut runner = FakeRunner::default();
    runner.fail_once("launchctl bootstrap gui/501", "launchd bootstrap denied");

    let error =
        install_service(&context, &mut runner).expect_err("launchd bootstrap failure should fail");

    assert!(
        error
            .to_string()
            .contains("bootstrap launchd agent com.mesh-llm.mesh-llm"),
        "{error:#}"
    );
}

#[cfg(unix)]
fn unix_mode(path: &std::path::Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).expect("metadata").permissions().mode() & 0o777
}

#[cfg(unix)]
#[test]
fn service_env_file_and_new_directory_are_owner_only() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let env_file = temp.path().join("mesh-llm").join("service.env");
    super::service_files::ensure_service_env_file(&env_file).expect("env file");
    assert_eq!(unix_mode(&env_file), 0o600);
    assert_eq!(unix_mode(&temp.path().join("mesh-llm")), 0o700);
    assert_eq!(
        fs::read_to_string(&env_file).expect("env file contents"),
        render_service_env_file()
    );
}

#[cfg(unix)]
#[test]
fn an_existing_service_env_file_is_tightened_and_kept() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let env_file = temp.path().join("service.env");
    fs::write(&env_file, "MESH_LLM_JOIN=secret\n").expect("seed env file");
    fs::set_permissions(&env_file, fs::Permissions::from_mode(0o644)).expect("chmod");
    super::service_files::ensure_service_env_file(&env_file).expect("env file");
    assert_eq!(unix_mode(&env_file), 0o600);
    assert_eq!(
        fs::read_to_string(&env_file).expect("env file contents"),
        "MESH_LLM_JOIN=secret\n"
    );
}

#[cfg(unix)]
#[test]
fn a_symlinked_service_env_file_is_rejected() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let elsewhere = temp.path().join("elsewhere.env");
    fs::write(&elsewhere, "UNTOUCHED=1\n").expect("seed target");
    let before = unix_mode(&elsewhere);
    let env_file = temp.path().join("service.env");
    std::os::unix::fs::symlink(&elsewhere, &env_file).expect("symlink");
    assert!(super::service_files::ensure_service_env_file(&env_file).is_err());
    assert_eq!(unix_mode(&elsewhere), before);
    assert_eq!(
        fs::read_to_string(&elsewhere).expect("target contents"),
        "UNTOUCHED=1\n"
    );
}

const TEST_SID: &str = "S-1-5-21-1111111111-2222222222-3333333333-1001";

fn windows_context(root: &Path, binary_name: &str, start_service: bool) -> ServiceInstallContext {
    let binary_path = root.join("bin").join(binary_name);
    fs::create_dir_all(binary_path.parent().expect("binary parent should exist"))
        .expect("binary dir should exist");
    fs::write(&binary_path, "binary").expect("binary should write");
    ServiceInstallContext {
        platform: SetupPlatform::Windows,
        home_dir: root.join("home"),
        config_root: root.join("config"),
        local_data_root: root.join("local"),
        binary_path,
        user_id: TEST_SID.to_string(),
        start_service,
    }
}

fn read_utf16_with_bom(path: &Path) -> String {
    let bytes = fs::read(path).expect("task definition should exist");
    assert_eq!(
        &bytes[..2],
        &[0xFF, 0xFE],
        "task definition starts with a BOM"
    );
    let units: Vec<u16> = bytes[2..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    String::from_utf16(&units).expect("task definition is UTF-16")
}

fn schtasks(args: &[&str]) -> ServiceCommand {
    ServiceCommand::new("schtasks", args.iter().copied())
}

/// A `schtasks /Query /FO CSV /NH /V` line, shaped like a real one.
fn task_csv(last_result: &str) -> String {
    format!(
        r#""DESKTOP-EXAMPLE","\mesh-llm","N/A","Ready","Interactive only","11/30/1999 12:00:00 AM","{last_result}","N/A","%SystemRoot%\System32\conhost.exe --headless .\mesh-llm.exe service-host","C:\Users\Example\mesh-llm","Mesh LLM user service, installed by mesh-llm setup","Enabled""#
    ) + "\r\n"
}

const TASK_STATUS_QUERY: &str = "schtasks /Query /TN mesh-llm /FO CSV";

/// A runner whose task is running once started.
fn runner_with_running_task() -> FakeRunner {
    let mut runner = FakeRunner::default();
    runner.respond(TASK_STATUS_QUERY, &task_csv("267009"));
    runner
}

#[test]
fn windows_task_runs_the_service_host_hidden_from_the_binary_directory() {
    let rendered = render_windows_task_xml(
        TEST_SID,
        &PathBuf::from(r"C:\Users\Example User\Programs\mesh & co"),
        "mesh-llm.exe",
    );

    assert_eq!(
        rendered
            .matches(&format!("<UserId>{TEST_SID}</UserId>"))
            .count(),
        2
    );
    for line in [
        "<LogonType>InteractiveToken</LogonType>",
        "<RunLevel>LeastPrivilege</RunLevel>",
        "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>",
        "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>",
        "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>",
        "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>",
        "<Priority>4</Priority>",
        r"<Command>%SystemRoot%\System32\conhost.exe</Command>",
        r"<Arguments>--headless .\mesh-llm.exe service-host</Arguments>",
        r"<WorkingDirectory>C:\Users\Example User\Programs\mesh &amp; co</WorkingDirectory>",
    ] {
        assert!(rendered.contains(line), "missing {line} in:\n{rendered}");
    }
}

#[test]
fn windows_service_install_registers_and_starts_the_task() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let context = windows_context(temp.path(), "mesh-llm.exe", true);
    let mut runner = runner_with_running_task();
    runner.fail_once(
        "schtasks /Query",
        "ERROR: The system cannot find the file specified.",
    );

    let report = install_service(&context, &mut runner).expect("task install should succeed");

    let definition = temp
        .path()
        .join("config")
        .join("mesh-llm")
        .join("mesh-llm-task.xml");
    assert_eq!(report.service_file, definition);
    assert_eq!(report.status, ServiceInstallStatus::Started);
    assert_eq!(report.summary, "installed and started");
    assert!(report.env_file.is_none());
    assert!(report.runner_file.is_none());
    assert_eq!(
        read_utf16_with_bom(&definition),
        render_windows_task_xml(TEST_SID, &temp.path().join("bin"), "mesh-llm.exe")
    );
    let definition_arg = definition.to_string_lossy().to_string();
    assert_eq!(
        runner.commands,
        vec![
            schtasks(&["/Query", "/TN", "mesh-llm"]),
            schtasks(&["/Create", "/TN", "mesh-llm", "/XML", &definition_arg, "/F"]),
            schtasks(&["/Run", "/TN", "mesh-llm"]),
            schtasks(&["/Query", "/TN", "mesh-llm", "/FO", "CSV", "/NH", "/V"]),
        ]
    );
    let log = temp
        .path()
        .join("local")
        .join("mesh-llm")
        .join("logs")
        .join("service.log");
    let invite = temp
        .path()
        .join("home")
        .join(".mesh-llm")
        .join("invite.token");
    for expected in [
        format!("Logs: {}", log.display()),
        format!(
            "Private mesh: write the invite token to {}, then restart the task (schtasks /End /TN mesh-llm, then schtasks /Run /TN mesh-llm): a node that started without a token starts its own mesh, and once it has one the file is re-read on every rejoin",
            invite.display()
        ),
        "Stop it with: schtasks /End /TN mesh-llm".to_string(),
    ] {
        assert!(
            report.messages.contains(&expected),
            "missing {expected:?} in {:#?}",
            report.messages
        );
    }
}

#[test]
fn windows_service_reinstall_ends_the_running_task_first() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let context = windows_context(temp.path(), "mesh-llm.exe", true);
    let mut runner = runner_with_running_task();

    install_service(&context, &mut runner).expect("task reinstall should succeed");

    let rendered: Vec<String> = runner
        .commands
        .iter()
        .map(ServiceCommand::display)
        .collect();
    assert_eq!(rendered[0], "schtasks /Query /TN mesh-llm");
    assert_eq!(rendered[1], "schtasks /End /TN mesh-llm");
    assert!(rendered[2].starts_with("schtasks /Create /TN mesh-llm /XML "));
    assert_eq!(rendered[3], "schtasks /Run /TN mesh-llm");
}

#[test]
fn windows_service_reinstall_warns_when_the_running_task_cannot_be_ended() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let context = windows_context(temp.path(), "mesh-llm.exe", true);
    let mut runner = runner_with_running_task();
    runner.fail_once("schtasks /End", "ERROR: Access is denied.");

    let report = install_service(&context, &mut runner).expect("a failed end only warns");

    assert_eq!(report.status, ServiceInstallStatus::Started);
    assert!(report.messages.iter().any(|line| {
        line.starts_with("warning: could not stop the running scheduled task before reinstalling")
    }));
}

#[test]
fn windows_service_registration_failure_is_a_setup_failure() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let context = windows_context(temp.path(), "mesh-llm.exe", true);
    let mut runner = FakeRunner::default();
    runner.fail_once("schtasks /Query", "not registered");
    runner.fail_once("schtasks /Create", "ERROR: Access is denied.");

    let error = install_service(&context, &mut runner).expect_err("registration must fail");

    assert!(
        error
            .to_string()
            .contains("register scheduled task mesh-llm"),
        "{error:#}"
    );
    assert!(
        !runner
            .commands
            .iter()
            .any(|command| command.display().starts_with("schtasks /Run"))
    );
}

#[test]
fn windows_service_install_without_start_gives_the_manual_commands() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let context = windows_context(temp.path(), "mesh-llm.exe", false);
    let mut runner = FakeRunner::default();

    let report = install_service(&context, &mut runner).expect("task install should succeed");

    assert!(runner.commands.is_empty());
    assert_eq!(report.status, ServiceInstallStatus::NeedsManualStart);
    assert!(report.service_file.exists());
    assert!(report.messages.iter().any(|line| {
        line.starts_with("Start it with: schtasks /Create /TN mesh-llm /XML ")
            && line.ends_with(", then schtasks /Run /TN mesh-llm")
    }));
}

#[test]
fn windows_service_refuses_a_binary_name_conhost_cannot_pass() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let context = windows_context(temp.path(), "mesh llm.exe", true);
    let mut runner = FakeRunner::default();

    let error = install_service(&context, &mut runner).expect_err("a spaced name must fail");

    assert!(
        error.to_string().contains("without spaces or quotes"),
        "{error:#}"
    );
    assert!(runner.commands.is_empty());
}

#[test]
fn windows_service_without_a_desktop_session_starts_at_the_next_logon() {
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let context = windows_context(temp.path(), "mesh-llm.exe", true);
    let mut runner = FakeRunner::default();
    runner.respond(TASK_STATUS_QUERY, &task_csv("267011"));

    let report = install_service(&context, &mut runner).expect("registration still succeeds");

    assert_eq!(report.status, ServiceInstallStatus::StartsAtLogon);
    assert_eq!(report.summary, "installed; starts at the next logon");
    assert!(
        report.messages.contains(
            &"Installed scheduled task: mesh-llm; it starts at your next logon".to_string()
        ),
        "{:#?}",
        report.messages
    );
    assert!(
        runner
            .commands
            .iter()
            .any(|command| command.display() == "schtasks /Run /TN mesh-llm")
    );
}

#[test]
fn schtasks_csv_line_yields_the_last_result() {
    assert_eq!(
        windows_task_last_result(&task_csv("267011")),
        Some("267011")
    );
    assert_eq!(windows_task_last_result(&task_csv("0")), Some("0"));
    assert_eq!(windows_task_last_result(""), None);
}

#[test]
fn whoami_csv_line_yields_the_user_sid() {
    assert_eq!(
        parse_whoami_user_sid(&format!("\"laptop-1\\example\",\"{TEST_SID}\"\r\n")).as_deref(),
        Some(TEST_SID)
    );
    assert_eq!(parse_whoami_user_sid(""), None);
    assert_eq!(
        parse_whoami_user_sid("\"laptop-1\\example\",\"none\""),
        None
    );
}
