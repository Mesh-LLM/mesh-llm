//! Source-owned repair policy and syntax. No cleanup, agent or model is executed.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, OutputFiles, ProcessSpec, Readiness, Value,
};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Duration};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn developer_contract(runbook: &str, wrapper: &str) -> Result<(), String> {
    for required in [
        "Run `scripts/prepare-llama.sh pinned`",
        "Do not run\n   an additional full family battery inside the coding session",
        "Leave the finished changes uncommitted",
        "Stay offline and\ndo not add Actions caching or download logic.",
        "Do not\nchange the target pin, create or switch branches, commit, push, use GitHub\ncredentials",
        "It does not start another\ncandidate or agent session automatically",
        "same named session for another coding turn",
        "full, separately bounded trusted gate pass",
        "both its complete family pass and the fresh independent build/family pass must\nbe green on the same commit",
    ] {
        if !runbook.contains(required) {
            return Err(format!("missing developer boundary: {required}"));
        }
    }
    if runbook.contains("at most three distributed repair attempts") {
        return Err("obsolete distributed repair cycle".into());
    }
    for required in [
        "while remaining_repair_seconds >/dev/null; do",
        "VERIFICATION_DEADLINE_AT=\"$(( $(date +%s) + VERIFICATION_TIMEOUT_SECONDS ))\"",
        "if run_candidate_gates refresh; then",
        "prompt=\"$(agent_feedback_prompt)\"",
        "--name \"$AGENT_SESSION_NAME\"",
        "goose_args+=(--resume)",
    ] {
        if !wrapper.contains(required) {
            return Err(format!("missing bounded session boundary: {required}"));
        }
    }
    for (name, next, boundary) in [
        (
            "agent_prompt",
            "agent_session_step",
            "Do not start an additional full battery in the coding session",
        ),
        (
            "agent_feedback_prompt",
            "snapshot_candidate_tree",
            "do not repeat the full family battery inside the coding session",
        ),
    ] {
        let start = format!("\n{name}() {{\n");
        let end = format!("\n{next}() {{\n");
        let span = wrapper
            .split_once(&start)
            .ok_or_else(|| format!("missing {name}"))?
            .1
            .split_once(&end)
            .ok_or_else(|| format!("missing {next}"))?
            .0;
        if !span.contains(boundary) {
            return Err(format!("missing {name} battery boundary"));
        }
    }
    Ok(())
}

#[test]
fn repair_developer_runbook_preserves_inner_session_and_one_distributed_pass() {
    let root = repository();
    let runbook = fs::read_to_string(root.join("ci/llama-canary/agent-repair-prompt.md")).unwrap();
    let wrapper = fs::read_to_string(root.join("scripts/llama-canary-agent-repair.sh")).unwrap();
    developer_contract(&runbook, &wrapper).unwrap();
    for (before, after) in [
        (
            "same named session for another coding turn",
            "a new session for another coding turn",
        ),
        (
            "full, separately bounded trusted gate pass",
            "focused test pass",
        ),
        (
            "Stay offline and\ndo not add Actions caching or download logic.",
            "Download fresh models during repair.",
        ),
        (
            "It does not start another\ncandidate or agent session automatically",
            "There are at most three distributed repair attempts",
        ),
    ] {
        assert!(runbook.contains(before));
        assert!(developer_contract(&runbook.replace(before, after), &wrapper).is_err());
    }
    for boundary in [
        "Do not start an additional full battery in the coding session",
        "do not repeat the full family battery inside the coding session",
    ] {
        assert!(wrapper.contains(boundary));
        assert!(
            developer_contract(
                &runbook,
                &(wrapper.replace(boundary, "run another full battery")
                    + &format!("\n# {boundary}\n"))
            )
            .is_err()
        );
    }
    let weakened = wrapper.replace(
        "while remaining_repair_seconds >/dev/null; do",
        "while true; do",
    );
    assert!(developer_contract(&runbook, &weakened).is_err());
}

fn scratch_contract(wrapper: &str) -> Result<(), &'static str> {
    for required in [
        "STATE_DIR=\"$ROOT/.deps/llama-canary-state-${RUN_KEY}-${PASS_ID}\"",
        "TARGET_SHA_FILE=\"$ROOT/.deps/llama-canary-target-sha\"",
        "git -C \"$ROOT/.deps/llama.cpp\" worktree prune >/dev/null 2>&1 || true",
        "rm -rf /tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-* 2>/dev/null || true",
    ] {
        if !wrapper.contains(required) {
            return Err("repair scratch naming or existing prune scope changed");
        }
    }
    if wrapper.contains("rm -rf /tmp/*") || wrapper.contains("rm -rf /tmp/llama-*") {
        return Err("repair scratch prune widened");
    }
    Ok(())
}

#[test]
fn repair_scratch_contract_preserves_run_pass_names_and_refuses_widened_prune() {
    let wrapper =
        fs::read_to_string(repository().join("scripts/llama-canary-agent-repair.sh")).unwrap();
    scratch_contract(&wrapper).unwrap();
    for (before, after) in [
        (
            "llama-canary-state-${RUN_KEY}-${PASS_ID}",
            "llama-canary-state-shared",
        ),
        (
            "/tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-*",
            "/tmp/*",
        ),
        (
            "git -C \"$ROOT/.deps/llama.cpp\" worktree prune",
            "git -C \"$ROOT\" worktree prune",
        ),
    ] {
        assert!(wrapper.contains(before));
        assert!(scratch_contract(&wrapper.replace(before, after)).is_err());
    }
    // This guards the existing source scope, not cross-job ownership of the
    // legacy wildcard. Nothing in this test executes rm or worktree prune.
}

#[test]
fn complete_repair_and_publisher_scripts_parse_without_execution() {
    let root = repository();
    let temporary = tempfile::tempdir().unwrap();
    for relative in [
        "scripts/llama-canary-agent-repair.sh",
        "scripts/llama-canary-publish.sh",
    ] {
        let report = process::supervise(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                arguments: vec![
                    Value::Public("-n".into()),
                    Value::Public(root.join(relative).into_os_string()),
                ],
                cwd: root.clone(),
                environment: BTreeMap::from([
                    ("PATH".into(), Value::Public("/usr/bin:/bin".into())),
                    ("HOME".into(), Value::Public(temporary.path().into())),
                ]),
            },
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 8192,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert_eq!(report.outcome, Outcome::Exited);
        assert_eq!(report.status.unwrap().code(), Some(0));
        assert!(report.failure.is_none() && report.cleanup.failure.is_none());
        assert!(
            report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
        );
        for stream in [&report.stdout, &report.stderr] {
            assert_eq!(stream.bytes_seen, 0);
            assert!(stream.bytes_retained.is_empty() && !stream.truncated);
            assert_eq!(stream.suppressed_lines, 0);
        }
    }
    temporary.close().expect("syntax fixture HOME cleanup");
}
