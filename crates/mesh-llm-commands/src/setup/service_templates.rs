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
