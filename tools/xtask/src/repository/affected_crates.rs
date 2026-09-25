//! `repository affected-crates`: the Rust owner of `scripts/affected-crates.sh`.
//! Maps changed paths to owning crates, closes over reverse dependencies, and
//! fails open to the whole workspace exactly where the script does.

mod graph;
mod paths;

use crate::command::DynResult;
use crate::repository::check_report::CheckReport;
use graph::{CrateGraph, FailOpen};
use serde::Serialize;
use std::io::BufRead;
use std::path::Path;
use std::process::Command;

/// The fail-open selection. Order and names are the legacy script's
/// `WORKSPACE_MEMBERS`; tests pin them to that array until it is deleted.
const WORKSPACE_MEMBERS: &[&str] = &[
    "mesh-llm",
    "mesh-llm-build-info",
    "mesh-llm-cli",
    "mesh-llm-commands",
    "mesh-llm-config",
    "mesh-llm-events",
    "mesh-llm-gpu-bench",
    "mesh-llm-host-runtime",
    "mesh-llm-hardware-profile",
    "mesh-llm-identity",
    "mesh-llm-log-store",
    "mesh-llm-native-runtime",
    "mesh-llm-protocol",
    "mesh-llm-release-footer",
    "mesh-llm-routing",
    "mesh-llm-runtime-event-contracts",
    "mesh-llm-runtime-install",
    "mesh-llm-sdk",
    "mesh-llm-guardrails",
    "mesh-llm-system",
    "mesh-llm-types",
    "mesh-llm-console-server",
    "mesh-llm-embedded-runtime",
    "mesh-llm-tui",
    "mesh-llm-ui",
    "mesh-llm-plugin",
    "mesh-llm-skills",
    "mesh-llm-plugin-manager",
    "mesh-llm-client",
    "mesh-mixture-of-agents",
    "mesh-native-serving-plugin-api",
    "mesh-native-serving-plugin-host",
    "mesh-llm-api-client",
    "mesh-llm-api-server",
    "mesh-llm-node",
    "mesh-llm-ffi",
    "mesh-llm-nodejs",
    "mesh-llm-test-harness",
    "model-ref",
    "model-artifact",
    "model-hf",
    "model-resolver",
    "skippy-protocol",
    "skippy-tokenizer",
    "skippy-coordinator",
    "skippy-topology",
    "skippy-cache",
    "skippy-metrics",
    "openai-frontend",
    "skippy-ffi",
    "skippy-model",
    "skippy-package-format",
    "skippy-runtime",
    "skippy-scheduler",
    "skippy-server",
    "metrics-server",
    "skippy-model-package",
    "skippy-quantize",
    "model-package",
    "skippy-correctness",
    "llama-quant-ffi",
    "llama-spec-bench",
    "skippy-bench",
    "skippy-prompt",
    "xtask",
];

/// Signal flags gathered before any Cargo work, so fail-open keeps them.
#[derive(Default)]
struct Signals {
    escalate: bool,
    ui_changed: bool,
    website_changed: bool,
}

impl Signals {
    fn scan(files: &[String]) -> Self {
        files.iter().fold(Self::default(), |signals, file| Self {
            escalate: signals.escalate || paths::escalates(file),
            ui_changed: signals.ui_changed || paths::is_ui_input(file),
            website_changed: signals.website_changed || paths::is_website_input(file),
        })
    }

    /// The heredoc document the script prints for the whole workspace.
    fn all_workspace(&self) -> String {
        let members = WORKSPACE_MEMBERS
            .iter()
            .map(|name| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\n  \"affected\": [{members}],\n  \"test_crates\": [],\n  \"all_rust\": true,\n  \"ui_changed\": {},\n  \"website_changed\": {}\n}}\n",
            self.ui_changed, self.website_changed
        )
    }
}

#[derive(Serialize)]
struct Selection<'a> {
    affected: &'a [String],
    test_crates: &'a [String],
    all_rust: bool,
    ui_changed: bool,
    website_changed: bool,
}

/// Entry point: `--stdin` reads newline-terminated paths, otherwise every
/// argument is a path. `cwd` is where Cargo discovers the workspace.
pub(crate) fn run(cwd: &Path, args: &[String]) -> DynResult<()> {
    let files = match args.first().map(String::as_str) {
        Some("--stdin") => read_terminated_lines(std::io::stdin().lock())?,
        _ => args.to_vec(),
    };
    select(cwd, &files)?.emit()
}

/// Bash `while IFS= read -r line`: an unterminated final line is dropped and
/// empty lines are skipped.
fn read_terminated_lines(mut input: impl BufRead) -> DynResult<Vec<String>> {
    let mut files = Vec::new();
    let mut line = Vec::new();
    while input.read_until(b'\n', &mut line)? > 0 {
        if line.pop() == Some(b'\n') && !line.is_empty() {
            files.push(String::from_utf8_lossy(&line).into_owned());
        }
        line.clear();
    }
    Ok(files)
}

fn select(cwd: &Path, files: &[String]) -> DynResult<CheckReport> {
    let signals = Signals::scan(files);
    if signals.escalate {
        return Ok(CheckReport::success(signals.all_workspace()));
    }
    let graph = match workspace_graph(cwd) {
        Ok(graph) => graph,
        Err(FailOpen(code)) => {
            let mut report = CheckReport::success(signals.all_workspace());
            report.stderr = format!(
                "WARNING: affected-crates.sh encountered an error (exit={code}), falling back to all_rust=true\n"
            );
            return Ok(report);
        }
    };
    let mut test_crates: Vec<String> = Vec::new();
    for owner in files
        .iter()
        .filter(|file| paths::may_own_rust(file))
        .filter_map(|file| graph.owner(file))
    {
        if !test_crates.iter().any(|name| name == owner) {
            test_crates.push(owner.to_owned());
        }
    }
    let affected = graph.closure(&test_crates);
    let selection = Selection {
        affected: &affected,
        test_crates: &test_crates,
        all_rust: false,
        ui_changed: signals.ui_changed,
        website_changed: signals.website_changed,
    };
    Ok(CheckReport::success(
        serde_json::to_string_pretty(&selection)? + "\n",
    ))
}

/// Cargo's stderr is discarded, as in the script; its exit status (or the
/// shell's 127 when Cargo cannot be launched) selects the fallback.
fn workspace_graph(cwd: &Path) -> Result<CrateGraph, FailOpen> {
    let output = Command::new("cargo")
        .current_dir(cwd)
        .args(["metadata", "--format-version=1", "--no-deps"])
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|_| FailOpen(127))?;
    if !output.status.success() {
        return Err(FailOpen(output.status.code().unwrap_or(1)));
    }
    CrateGraph::parse(&output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_repository_affected_reads_only_terminated_lines() {
        let lines = read_terminated_lines(&b"a\n\nb\r\nc"[..]).expect("in-memory read");
        assert_eq!(lines, ["a", "b\r"]);
    }

    #[test]
    fn migration_repository_affected_fallback_list_matches_script() {
        let script = include_str!("../../../../scripts/affected-crates.sh");
        let (_, rest) = script
            .split_once("WORKSPACE_MEMBERS=(\n")
            .expect("member list");
        let (body, _) = rest.split_once("\n)").expect("terminated list");
        let legacy = body
            .lines()
            .map(|line| line.trim().trim_matches('"'))
            .collect::<Vec<_>>();
        assert_eq!(legacy, WORKSPACE_MEMBERS);
    }
}
