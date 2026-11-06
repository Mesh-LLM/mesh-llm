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
    if !cfg!(any(target_os = "macos", target_os = "linux")) {
        return Err("local upgrades require a native macOS or Linux producer".into());
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
    values
        .into_iter()
        .filter(|(key, _)| {
            ![
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "CANARY_REPAIR_TOKEN",
                "GITHUB_OUTPUT",
                "GITHUB_ENV",
                "GITHUB_STEP_SUMMARY",
            ]
            .iter()
            .any(|name| key == name)
        })
        .map(|(key, value)| (key, Value::Secret(value)))
        .collect()
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
    for (key, value) in [
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
            ("GITHUB_OUTPUT".into(), "/publisher".into()),
            ("PATH".into(), "/tools".into()),
        ]);
        assert_eq!(env.len(), 1);
        assert!(env.contains_key(std::ffi::OsStr::new("PATH")));
    }
}
