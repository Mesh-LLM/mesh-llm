use super::arm_pass::Input;
use crate::command::DynResult;
use std::path::Path;

pub(super) fn execute(
    root: Option<&Path>,
    input: &Input,
    workload: &Path,
    requests: &Path,
    summary: &Path,
    log: &Path,
) -> DynResult<()> {
    if let Some(build) = &input.external {
        let request = super::external_cell::Input {
            build: serde_json::from_value(serde_json::to_value(build)?)?,
            workload: workload.into(),
            requests_output: requests.into(),
            summary_output: summary.into(),
            server_log: log.into(),
            startup_timeout_seconds: input.startup_timeout_seconds,
            timeout_seconds: input.timeout_seconds,
        };
        return super::external_cell::execute(root, &request);
    }
    let mut arguments = Vec::new();
    for (option, path) in [
        ("--binary", input.binary.as_path()),
        ("--native-runtime-root", input.native_runtime_root.as_path()),
        ("--workload", workload),
        ("--requests-output", requests),
        ("--summary-output", summary),
        ("--server-log", log),
    ] {
        arguments.extend([
            option.into(),
            path.to_str().ok_or("non-Unicode replay path")?.into(),
        ]);
    }
    arguments.extend([
        "--model".into(),
        input.model.clone(),
        "--timeout".into(),
        input.timeout_seconds.to_string(),
        "--startup-timeout".into(),
        input.startup_timeout_seconds.to_string(),
    ]);
    if let Some(home) = &input.hf_home {
        arguments.extend([
            "--hf-home".into(),
            home.to_str().ok_or("non-Unicode HF-home")?.into(),
        ]);
    }
    super::server_cell::run(root, &arguments)
}
