use super::SetupPlatform;
use super::service::{ServiceInstallStatus, install_service};
use super::service_paths::ServiceInstallContext;
use super::service_runner::{ServiceCommand, ServiceCommandRunner};
use super::service_templates::{
    SERVICE_LABEL, render_launchd_plist, render_service_env_file, render_service_runner,
    render_systemd_unit,
};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;

#[derive(Default)]
struct FakeRunner {
    commands: Vec<ServiceCommand>,
    failures: HashMap<String, VecDeque<String>>,
}

impl FakeRunner {
    fn fail_once(&mut self, starts_with: &str, message: &str) {
        self.failures
            .entry(starts_with.to_string())
            .or_default()
            .push_back(message.to_string());
    }
}

impl ServiceCommandRunner for FakeRunner {
    fn run(&mut self, command: &ServiceCommand) -> Result<()> {
        let rendered = command.display();
        self.commands.push(command.clone());
        for (prefix, failures) in &mut self.failures {
            if rendered.starts_with(prefix)
                && let Some(message) = failures.pop_front()
            {
                return Err(anyhow!(message));
            }
        }
        Ok(())
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
        "# Optional environment variables for mesh-llm.\n# Use plain KEY=value lines.\n# Example:\n# RUST_LOG=mesh_inference=debug\n"
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
        binary_path: binary_path.clone(),
        user_id: String::new(),
        start_service: true,
    };
    let mut runner = FakeRunner::default();

    let report = install_service(&context, &mut runner).expect("systemd install should succeed");

    assert_eq!(report.summary, "installed and started");
    assert_eq!(report.status, ServiceInstallStatus::Started);
    assert_eq!(
        fs::read_to_string(&report.env_file).expect("env file should exist"),
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
        binary_path: binary_path.clone(),
        user_id: "501".to_string(),
        start_service: false,
    };
    let mut runner = FakeRunner::default();

    let report = install_service(&context, &mut runner).expect("launchd install should succeed");

    assert_eq!(
        report.summary,
        "installed; automatic start needs manual follow-up"
    );
    let runner_file = report
        .runner_file
        .expect("launchd runner should be recorded");
    assert_eq!(
        fs::read_to_string(&runner_file).expect("runner should exist"),
        render_service_runner(&binary_path, &report.env_file)
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
