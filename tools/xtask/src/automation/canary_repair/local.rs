//! Local entry into the existing trusted build owner, without publication credentials.
use crate::{
    automation::canary_package_closure,
    command::DynResult,
    command_interrupt::Interrupt,
    process::{self, Completion, Limits, ProcessSpec, Readiness, Value},
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation canary-receipts local-upgrade --root ABS --upstream SHA --backend metal|cuda --cache-root ABS [--controller-root ABS] [--state-parent ABS] [--plan]",
    values: &[
        "--root",
        "--controller-root",
        "--upstream",
        "--backend",
        "--cache-root",
        "--state-parent",
    ],
    flags: &["--help", "--plan"],
};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let directory = |key: &str| -> DynResult<PathBuf> {
        let path = Path::new(parsed.last(key).ok_or_else(|| format!("missing {key}"))?);
        if !path.is_absolute() || !path.is_dir() {
            return Err(format!("{key} requires an existing absolute directory").into());
        }
        Ok(path.canonicalize()?)
    };
    let root = directory("--root")?;
    let controller = if parsed.last("--controller-root").is_some() {
        directory("--controller-root")?
    } else {
        root.clone()
    };
    let cache = directory("--cache-root")?;
    let parent = if parsed.last("--state-parent").is_some() {
        directory("--state-parent")?
    } else {
        std::env::temp_dir().canonicalize()?
    };
    if parent.starts_with(&root) || parent.starts_with(&controller) {
        return Err("local upgrade evidence must be outside both checkouts".into());
    }
    let upstream = parsed.last("--upstream").ok_or("missing --upstream")?;
    super::super::canary_build::validate_revision(upstream)?;
    let backend = parsed
        .last("--backend")
        .filter(|s| matches!(*s, "metal" | "cuda"))
        .ok_or("local upgrade requires metal or cuda backend")?;
    if !matches!(
        (std::env::consts::OS, backend),
        ("macos", "metal") | ("linux", "cuda")
    ) {
        return Err("local upgrade backend must match its native macOS or Linux producer".into());
    }
    let selected = canary_package_closure::checkout_revision(&root)?;
    let authority = canary_package_closure::checkout_revision(&controller)?;
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs()
        .to_string();
    let output = parent.join(format!("llama-upgrade-{id}-{}", std::process::id()));
    let request = request(
        &root,
        &controller,
        &selected,
        &authority,
        upstream,
        &id,
        &output,
    );
    if parsed.flag("--plan") {
        return CheckReport::success(format!("{}\n", serde_json::to_string_pretty(&request)?))
            .emit();
    }
    fs::create_dir(&output)?;
    let path = output.join("build.json");
    super::atomic_json(&path, &request)?;
    writeln!(
        crate::cli_output::stdout(),
        "local llama upgrade evidence: {}\nPublication is disabled; successful source is retained in a local candidate bundle.",
        output.display()
    )?;
    launch(&controller, &parent, &output, &path, backend, &cache)?.emit()
}

use std::io::Write;

fn request(
    root: &Path,
    controller: &Path,
    selected: &str,
    authority: &str,
    upstream: &str,
    id: &str,
    output: &Path,
) -> serde_json::Value {
    serde_json::json!({"controller_root":controller,"source_root":root,"controller_revision":authority,"selected_revision":selected,
      "mesh_source":if selected == authority && root == controller { "" } else { selected },"upstream_revision":upstream,"mode":"repair-build","pass_id":"repair-1","run_id":id,"run_attempt":"1",
      "previous":null,"previous_feedback":null,"evidence":output.join("evidence"),"export":output.join("candidate"),"agent_timeout_seconds":41400,"verification_timeout_seconds":43200})
}

fn environment(
    values: impl IntoIterator<Item = (OsString, OsString)>,
) -> BTreeMap<OsString, Value> {
    let mut environment = BTreeMap::new();
    let mut default_rustup = None;
    for (key, value) in values {
        if key == "HOME" {
            default_rustup = Some(
                PathBuf::from(value.clone())
                    .join(".rustup")
                    .into_os_string(),
            );
        }
        if !publication_credential(&key) {
            environment.insert(key, Value::Secret(value));
        }
    }
    if !environment.contains_key(std::ffi::OsStr::new("RUSTUP_HOME"))
        && let Some(path) = default_rustup
    {
        environment.insert("RUSTUP_HOME".into(), Value::Secret(path));
    }
    environment
}

fn publication_credential(key: &std::ffi::OsStr) -> bool {
    let name = key.to_string_lossy();
    if [
        "ZAI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOSE_API_KEY",
    ]
    .contains(&name.as_ref())
    {
        return false;
    }
    [
        "GITHUB_OUTPUT",
        "GITHUB_ENV",
        "GITHUB_STEP_SUMMARY",
        "GITHUB_PATH",
        "GITHUB_STATE",
        "GH_CONFIG_DIR",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "SSH_AUTH_SOCK",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_HTTP_EXTRAHEADER",
        "DOCKER_CONFIG",
        "NETRC",
        "GNUPGHOME",
        "HOME",
        "CARGO_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
    ]
    .contains(&name.as_ref())
        || name.starts_with("GIT_CONFIG_KEY_")
        || name.starts_with("GIT_CONFIG_VALUE_")
        || [
            "_TOKEN",
            "_SECRET",
            "_PASSWORD",
            "_CREDENTIAL",
            "_ACCESS_KEY",
            "_PRIVATE_KEY",
            "_API_KEY",
            "_JWT",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn launch(
    controller: &Path,
    parent: &Path,
    output: &Path,
    input: &Path,
    backend: &str,
    cache: &Path,
) -> DynResult<CheckReport> {
    let executable = std::env::current_exe()?;
    let mut environment = environment(std::env::vars_os());
    let private_home = output.join("home");
    fs::create_dir(&private_home)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&private_home, fs::Permissions::from_mode(0o700))?;
    }
    for (key, value) in [
        ("HOME", private_home.clone()),
        ("XDG_CONFIG_HOME", private_home.join("config")),
        ("XDG_DATA_HOME", private_home.join("data")),
        ("XDG_STATE_HOME", private_home.join("state")),
        ("XDG_CACHE_HOME", private_home.join("cache")),
        ("GH_CONFIG_DIR", private_home.join("gh")),
        ("GIT_CONFIG_GLOBAL", PathBuf::from("/dev/null")),
        ("GIT_CONFIG_NOSYSTEM", PathBuf::from("1")),
        ("GIT_CONFIG_COUNT", PathBuf::from("1")),
        ("GIT_CONFIG_KEY_0", PathBuf::from("credential.helper")),
        ("GIT_CONFIG_VALUE_0", PathBuf::new()),
        ("MESH_LLM_AUTOMATION_BIN", executable.clone()),
        ("RUNNER_TEMP", parent.into()),
        ("LLAMA_STAGE_BUILD_DIR", output.join("native")),
        ("LLAMA_STAGE_BACKEND", backend.into()),
        ("HF_CACHE", cache.into()),
    ] {
        environment.insert(key.into(), Value::Public(value.into()));
    }
    let interrupt = Interrupt::install()?;
    let report = process::supervise_inherited_closed_stdin(
        &ProcessSpec {
            executable,
            cwd: controller.into(),
            environment,
            arguments: ["automation", "canary-receipts", "build", "--input"]
                .map(|s| Value::Public(s.into()))
                .into_iter()
                .chain([Value::Public(input.into())])
                .collect(),
        },
        &Limits {
            execution: Duration::from_secs(85500),
            graceful_shutdown: Duration::from_secs(180),
            forced_shutdown: Duration::from_secs(5),
            retained_bytes_per_stream: 0,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &interrupt.cancellation(),
    )?;
    let cancelled = interrupt.finish().is_err();
    let code = if cancelled
        || !matches!(report.outcome, process::Outcome::Exited)
        || !report.cleanup.complete
        || report.cleanup.forced
        || report.cleanup.graceful_signal_failed
        || report.cleanup.failure.is_some()
        || report.failure.is_some()
    {
        125
    } else {
        report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code)
            .unwrap_or(125)
    };
    Ok(CheckReport {
        stdout: String::new(),
        stderr: format!(
            "local upgrade finished with status {code}; recovery and logs: {}\n",
            output.display()
        ),
        code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_request_freezes_both_revisions_and_exports_no_publication_authority() {
        let value = request(
            Path::new("/source"),
            Path::new("/controller"),
            &"a".repeat(40),
            &"b".repeat(40),
            &"c".repeat(40),
            "123",
            Path::new("/evidence"),
        );
        assert_eq!(value["mesh_source"], "a".repeat(40));
        assert_eq!(value["mode"], "repair-build");
        assert_eq!(value["selected_revision"], "a".repeat(40));
        assert_eq!(value["controller_revision"], "b".repeat(40));
        assert_eq!(
            value["export"],
            serde_json::to_value(Path::new("/evidence").join("candidate")).unwrap()
        );
        let env = environment([
            ("GITHUB_TOKEN".into(), "private".into()),
            ("CANARY_REPAIR_TOKEN".into(), "private".into()),
            ("GH_ENTERPRISE_TOKEN".into(), "private".into()),
            ("AWS_SECRET_ACCESS_KEY".into(), "private".into()),
            ("CLOUDFLARE_API_TOKEN".into(), "private".into()),
            ("ZAI_API_KEY".into(), "provider".into()),
            ("SSH_AUTH_SOCK".into(), "/publisher".into()),
            ("GIT_ASKPASS".into(), "/publisher".into()),
            ("GIT_CONFIG_KEY_0".into(), "credential.helper".into()),
            ("HOME".into(), "/private-home".into()),
            ("CARGO_HOME".into(), "/credential-cache".into()),
            ("GITHUB_OUTPUT".into(), "/publisher".into()),
            ("GITHUB_PATH".into(), "/runner/path".into()),
            ("GITHUB_STATE".into(), "/runner/state".into()),
            ("PATH".into(), "/tools".into()),
        ]);
        assert_eq!(env.len(), 3);
        assert!(env.contains_key(std::ffi::OsStr::new("PATH")));
        assert!(env.contains_key(std::ffi::OsStr::new("ZAI_API_KEY")));
        assert!(
            matches!(env.get(std::ffi::OsStr::new("RUSTUP_HOME")), Some(Value::Secret(path)) if path.as_os_str() == std::ffi::OsStr::new("/private-home/.rustup"))
        );
        let explicit = environment([
            ("HOME".into(), "/private-home".into()),
            ("RUSTUP_HOME".into(), "/installed/toolchains".into()),
        ]);
        assert!(
            matches!(explicit.get(std::ffi::OsStr::new("RUSTUP_HOME")), Some(Value::Secret(path)) if path.as_os_str() == std::ffi::OsStr::new("/installed/toolchains"))
        );
    }
}
