use std::path::Path;

pub(crate) const SERVICE_NAME: &str = "mesh-llm";
pub(crate) const SERVICE_LABEL: &str = "com.mesh-llm.mesh-llm";

pub(crate) fn render_service_env_file() -> String {
    [
        "# Optional environment variables for mesh-llm.",
        "# Use plain KEY=value lines.",
        "# Example:",
        "# RUST_LOG=mesh_inference=debug",
        "",
        "# Join a private mesh without editing the generated unit.",
        "# The token file is re-read on every rejoin, so rotating the invite",
        "# token is just replacing the file's contents.",
        "# A token at the default location needs no line here at all: an",
        "# invite.token beside the resolved config file, which is",
        "# ~/.mesh-llm/invite.token for the default config path.",
        "# MESH_LLM_JOIN_FILE=/home/you/.mesh-llm/invite.token",
        "# MESH_LLM_JOIN=<complete invite token>",
        "",
    ]
    .join("\n")
}

pub(crate) fn render_service_runner(binary_path: &Path, env_file: &Path) -> String {
    format!(
        "#!/usr/bin/env bash\n\nset -euo pipefail\n\nBIN=\"{}\"\nENV_FILE=\"{}\"\n\nif [[ ! -x \"$BIN\" ]]; then\n    echo \"mesh-llm binary not found or not executable: $BIN\" >&2\n    exit 1\nfi\n\nif [[ -f \"$ENV_FILE\" ]]; then\n    set -a\n    # shellcheck source=/dev/null\n    . \"$ENV_FILE\"\n    set +a\nfi\n\nexec \"$BIN\" serve\n",
        shell_double_quote(&binary_path.to_string_lossy()),
        shell_double_quote(&env_file.to_string_lossy()),
    )
}

pub(crate) fn render_systemd_unit(
    binary_path: &Path,
    service_env_file: &Path,
    mesh_config_file: &Path,
) -> String {
    let exec_line = format!(
        "ExecStart={} serve",
        systemd_quote_token(&binary_path.to_string_lossy())
    );
    let service_env_file = systemd_escape_token(&service_env_file.to_string_lossy());
    format!(
        "# mesh-llm serve (startup models come from {mesh_config_file})\n[Unit]\nDescription=Mesh LLM user service\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nEnvironmentFile=-{service_env_file}\n\n{exec_line}\nWorkingDirectory=%h\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
        mesh_config_file = mesh_config_file.display(),
        service_env_file = service_env_file,
        exec_line = exec_line,
    )
}

pub(crate) fn render_launchd_plist(
    service_runner: &Path,
    home_dir: &Path,
    stdout_log: &Path,
    stderr_log: &Path,
) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"https://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n    <key>Label</key>\n    <string>{service_label}</string>\n    <key>ProgramArguments</key>\n    <array>\n        <string>{service_runner}</string>\n    </array>\n    <key>WorkingDirectory</key>\n    <string>{home_dir}</string>\n    <key>RunAtLoad</key>\n    <true/>\n    <key>KeepAlive</key>\n    <dict>\n        <key>SuccessfulExit</key>\n        <false/>\n    </dict>\n    <key>ProcessType</key>\n    <string>Background</string>\n    <key>StandardOutPath</key>\n    <string>{stdout_log}</string>\n    <key>StandardErrorPath</key>\n    <string>{stderr_log}</string>\n</dict>\n</plist>\n",
        service_label = SERVICE_LABEL,
        service_runner = xml_escape(&service_runner.to_string_lossy()),
        home_dir = xml_escape(&home_dir.to_string_lossy()),
        stdout_log = xml_escape(&stdout_log.to_string_lossy()),
        stderr_log = xml_escape(&stderr_log.to_string_lossy()),
    )
}

/// A per-user scheduled task that starts the service host at the user's
/// logon, with no console window and no elevation.
///
/// `conhost.exe --headless` hosts the console without showing it. It does not
/// keep the quoting of a path with spaces, so the task starts in the binary's
/// directory and names the binary relative to it instead of quoting a path.
/// Task Scheduler expands `%SystemRoot%` in both the command and its
/// arguments. Battery and run-time limits are lifted: the defaults would not
/// start the task on battery and would stop it after three days.
pub(crate) fn render_windows_task_xml(
    user_sid: &str,
    binary_dir: &Path,
    binary_file_name: &str,
) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-16\"?>\n<Task version=\"1.2\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\n  <RegistrationInfo>\n    <Description>Mesh LLM user service, installed by mesh-llm setup</Description>\n  </RegistrationInfo>\n  <Triggers>\n    <LogonTrigger>\n      <Enabled>true</Enabled>\n      <UserId>{user_sid}</UserId>\n    </LogonTrigger>\n  </Triggers>\n  <Principals>\n    <Principal id=\"Author\">\n      <UserId>{user_sid}</UserId>\n      <LogonType>InteractiveToken</LogonType>\n      <RunLevel>LeastPrivilege</RunLevel>\n    </Principal>\n  </Principals>\n  <Settings>\n    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n  </Settings>\n  <Actions Context=\"Author\">\n    <Exec>\n      <Command>%SystemRoot%\\System32\\conhost.exe</Command>\n      <Arguments>--headless .\\{binary_file_name} {WINDOWS_SERVICE_HOST_COMMAND}</Arguments>\n      <WorkingDirectory>{binary_dir}</WorkingDirectory>\n    </Exec>\n  </Actions>\n</Task>\n",
        user_sid = xml_escape(user_sid),
        binary_file_name = xml_escape(binary_file_name),
        binary_dir = xml_escape(&binary_dir.to_string_lossy()),
    )
}

/// The hidden subcommand the scheduled task runs.
pub(crate) const WINDOWS_SERVICE_HOST_COMMAND: &str = "service-host";

fn shell_double_quote(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('$', "\\$")
        .replace('`', "\\`")
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn systemd_quote_token(value: &str) -> String {
    let escaped = systemd_escape_token(value);
    format!("\"{escaped}\"")
}

fn systemd_escape_token(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('$', "$$")
        .replace('%', "%%")
}
