//! Execute retained canary declarations against finite, nonpublishing tool seams.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
const HEAD: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const BASE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const TREE: &str = "cccccccccccccccccccccccccccccccccccccccc";
const BRANCH: &str = "llama-canary/repair-fixture-1-aaaaaaaaaa";
fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn declaration(source: &str, name: &str) -> String {
    let start = source.find(&format!("{name}() {{\n")).unwrap();
    let end = start + source[start..].find("\n}\n").unwrap() + 3;
    source[start..end].into()
}
fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
struct Fixture {
    temporary: tempfile::TempDir,
    root: PathBuf,
    tools: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("owned candidate");
        let tools = temporary.path().join("finite tools");
        for path in [&root, &tools] {
            fs::create_dir_all(path).unwrap();
        }
        fs::write(root.join("bundle"), "inert bundle").unwrap();
        fs::write(root.join("body"), "inert public body").unwrap();
        for name in ["python3", "cargo", "just", "goose"] {
            executable(
                &tools.join(name),
                "#!/bin/bash\nprintf forbidden >> \"$TRACE\"\nexit 97\n",
            );
        }
        Self {
            temporary,
            root,
            tools,
        }
    }
    fn run(&self, script: &str, extra: &[(&str, &str)]) -> (bool, String, String) {
        let path = self.root.join("case.sh");
        executable(&path, script);
        let mut environment: BTreeMap<_, _> = [
            ("PATH", format!("{}:/usr/bin:/bin", self.tools.display())),
            ("HOME", self.root.display().to_string()),
            ("TRACE", self.root.join("trace").display().to_string()),
            ("GIT_MASTER", "1".into()),
            ("GIT_OPTIONAL_LOCKS", "0".into()),
            ("GIT_CONFIG_GLOBAL", "/dev/null".into()),
            ("GIT_CONFIG_NOSYSTEM", "1".into()),
            ("FIXTURE_HEAD", HEAD.into()),
            ("FIXTURE_BASE", BASE.into()),
            ("FIXTURE_TREE", TREE.into()),
            ("FIXTURE_BRANCH", BRANCH.into()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        for (key, value) in extra {
            environment.insert((*key).into(), Value::Public((*value).into()));
        }
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                arguments: vec![Value::Public(path.into_os_string())],
                environment,
            },
            &Limits {
                execution: Duration::from_secs(15),
                graceful_shutdown: Duration::from_secs(2),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(report.process.failure.is_none(), "{report:?}");
        let cleanup = &report.process.cleanup;
        assert!(
            cleanup.complete
                && !cleanup.forced
                && !cleanup.graceful_signal_failed
                && cleanup.failure.is_none(),
            "{report:?}"
        );
        let mut captured = Vec::new();
        for (raw, stream) in [
            (&report.stdout, &report.process.stdout),
            (&report.stderr, &report.process.stderr),
        ] {
            let raw = raw.as_ref().unwrap().as_bytes();
            assert_eq!(raw.len() as u64, stream.bytes_seen, "{report:?}");
            assert!(
                !stream.truncated && stream.line_capture_complete && stream.oversized_lines == 0,
                "{report:?}"
            );
            // Raw bytes are the protocol evidence; sanitizer suppression is separate diagnostics.
            captured.push(String::from_utf8(raw.to_vec()).unwrap());
        }
        (
            report.process.success(),
            captured.remove(0),
            captured.remove(0),
        )
    }
    fn trace(&self) -> String {
        fs::read_to_string(self.root.join("trace")).unwrap_or_default()
    }
    fn close(self) {
        let path = self.temporary.path().to_path_buf();
        self.temporary.close().unwrap();
        assert!(!path.exists());
    }
}
#[test]
fn retained_canary_main_failures_stop_before_snapshot_or_publication() {
    let source =
        fs::read_to_string(repository().join("scripts/llama-canary-agent-repair.sh")).unwrap();
    let tail = &source[source
        .rfind("if [[ \"$HARNESS_MODE\" == repair* ]]; then")
        .unwrap()..];
    for (mode, expected, diagnostic) in [
        ("repair", "37", "agent task failed or timed out"),
        ("verify-build", "41", "final canary verification failed"),
    ] {
        let fixture = Fixture::new();
        let script = format!(
            r#"#!/bin/bash
set -euo pipefail
mark() {{ printf '%s\n' "$1" >> "$TRACE"; }}
write_repair_pin() {{ mark pin; }}
verify_repair_pin() {{ mark verify-pin; }}
repair_candidate_until_green() {{ mark repair; return 37; }}
snapshot_candidate_tree() {{ mark snapshot; }}
write_candidate_bundle() {{ mark bundle; }}
load_candidate_bundle() {{ mark load; }}
materialize_verification_tree() {{ mark materialize; }}
cleanup_verification_worktree() {{ mark cleanup; }}
run_candidate_gates() {{ mark gates; return 41; }}
record_failure_class() {{ mark "failure:$*"; }}
check_split_certification_roster() {{ mark roster; }}
export_family_inputs() {{ mark export; }}
finalize_certified_tree() {{ mark finalize; }}
VERIFICATION_TIMEOUT_SECONDS=60
{tail}
"#
        );
        let output = fixture.run(&script, &[("HARNESS_MODE", mode)]);
        assert!(!output.0 && output.2.contains(diagnostic), "{output:?}");
        assert!(
            output
                .2
                .contains("no canary branch or pull request was published")
        );
        let trace = fixture.trace();
        for forbidden in [
            "snapshot",
            "bundle",
            "finalize",
            "roster",
            "export",
            "forbidden",
        ] {
            assert!(!trace.contains(forbidden), "{trace}");
        }
        if mode == "verify-build" {
            assert!(
                trace.contains("failure:candidate independent-verification")
                    && trace.contains("cleanup")
            );
        }
        // Preserve the actual return status, not merely any nonzero refusal.
        let wrapped =
            format!("#!/bin/bash\n/bin/bash case.sh; status=$?; [[ $status == {expected} ]]\n");
        // The outer case would overwrite case.sh; persist the first script separately.
        fs::write(fixture.root.join("failed-main.sh"), &script).unwrap();
        let wrapped = wrapped.replace("/bin/bash case.sh", "/bin/bash failed-main.sh");
        assert!(fixture.run(&wrapped, &[("HARNESS_MODE", mode)]).0);
        fixture.close();
    }
}
#[test]
fn retained_candidate_import_uses_original_branch_and_restores_commit_tree() {
    let source =
        fs::read_to_string(repository().join("scripts/llama-canary-agent-repair.sh")).unwrap();
    for mode in ["valid", "wrong-head", "bad-branch", "bad-sha"] {
        let fixture = Fixture::new();
        executable(
            &fixture.tools.join("git"),
            r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$TRACE"
case "$1 ${2:-}" in
'check-ref-format '*) [[ "$2" == "refs/heads/$FIXTURE_BRANCH" ]];;
'bundle verify') ;;
'bundle list-heads') [[ "$4" == "refs/heads/$FIXTURE_BRANCH" ]]; printf '%s refs/heads/%s\n' "$ADVERTISED" "$FIXTURE_BRANCH";;
'fetch '*) [[ "$3" == "refs/heads/$FIXTURE_BRANCH" ]];;
'rev-parse '*) if [[ "$2" == "$FIXTURE_HEAD^" ]]; then printf '%s\n' "$FIXTURE_BASE"; elif [[ "$2" == "$FIXTURE_HEAD^{tree}" ]]; then printf '%s\n' "$FIXTURE_TREE"; else exit 96; fi;;
*) exit 95;;
esac
"#,
        );
        let script = format!(
            "#!/bin/bash\nset -euo pipefail\n{}\nBRANCH=llama-canary/repair-a-different-run-9-dddddddddd\nCANARY_INPUT_BUNDLE=bundle\nload_candidate_bundle\nprintf '%s %s %s\\n' \"$CERTIFIED_SHA\" \"$CANDIDATE_BASE_HEAD\" \"$VERIFICATION_TREE\"\n",
            declaration(&source, "load_candidate_bundle")
        );
        let branch = if mode == "bad-branch" {
            "untrusted/branch"
        } else {
            BRANCH
        };
        let head = if mode == "bad-sha" {
            "not-a-commit"
        } else {
            HEAD
        };
        let advertised = if mode == "wrong-head" { BASE } else { HEAD };
        let output = fixture.run(
            &script,
            &[
                ("CANARY_CANDIDATE_BRANCH", branch),
                ("CANARY_CANDIDATE_SHA", head),
                ("ADVERTISED", advertised),
            ],
        );
        assert_eq!(output.0, mode == "valid", "{mode}: {output:?}");
        if mode == "valid" {
            assert_eq!(output.1, format!("{HEAD} {BASE} {TREE}\n"));
        } else {
            assert!(!fixture.trace().contains("fetch"));
        }
        assert!(!fixture.trace().contains("a-different-run"));
        fixture.close();
    }
}
fn snapshot_identity(source: &str) {
    let fixture = Fixture::new();
    executable(
        &fixture.tools.join("git"),
        r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$TRACE"
case "$1 ${2:-}" in
'add -A') ;;
'diff --cached') exit 1;;
'write-tree '*) echo "$FIXTURE_TREE";;
'commit-tree '*) [[ $# == 4 && "$2" == "$FIXTURE_TREE" && "$3" == -p && "$4" == "$FIXTURE_BASE" ]]; /bin/cat >/dev/null; echo "$FIXTURE_HEAD";;
*) exit 95;;
esac
"#,
    );
    let branch = source
        .lines()
        .find(|line| line.starts_with("BRANCH=\"llama-canary/repair-"))
        .unwrap();
    let script = format!(
        r#"#!/bin/bash
set -euo pipefail
assert_agent_control_unchanged() {{ :; }}
verify_repair_pin() {{ :; }}
validate_agent_manifest_changes() {{ :; }}
controller_producer_receipt() {{ echo producer >> "$TRACE"; }}
HARNESS_MODE=repair-build; UPSTREAM_SHA="$FIXTURE_HEAD"; BASE_HEAD="$FIXTURE_BASE"; RUN_KEY=fixture-1
{branch}
{}
snapshot_candidate_tree
printf '%s %s %s\n' "$BRANCH" "$CERTIFIED_SHA" "$VERIFICATION_TREE"
"#,
        declaration(source, "snapshot_candidate_tree")
    );
    let result = fixture.run(&script, &[]);
    assert!(result.0, "{result:?}");
    assert_eq!(result.1, format!("{BRANCH} {HEAD} {TREE}\n"));
    assert!(
        fixture
            .trace()
            .starts_with("producer\nadd -A\ndiff --cached --quiet\nwrite-tree\ncommit-tree ")
    );
    fixture.close();
}
#[test]
fn retained_finalizer_refuses_dirty_or_changed_tree_before_bundle_outputs() {
    let source =
        fs::read_to_string(repository().join("scripts/llama-canary-agent-repair.sh")).unwrap();
    snapshot_identity(&source);
    for mode in [
        "valid",
        "dirty",
        "changed-tree",
        "changed-source",
        "changed-pin",
    ] {
        let fixture = Fixture::new();
        executable(
            &fixture.tools.join("git"),
            r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$TRACE"
case "$1 ${2:-}" in
'status --porcelain') if [[ "$MODE" == dirty ]]; then echo ' M tracked'; fi;;
'rev-parse '*) if [[ "$MODE" == changed-tree ]]; then echo "$FIXTURE_BASE"; else echo "$FIXTURE_TREE"; fi;;
'-c core.hooksPath=/dev/null') [[ "$*" == *"branch -f $FIXTURE_BRANCH $FIXTURE_HEAD" ]];;
'-C '*) [[ "$*" == *'bundle create'* || "$*" == *'bundle verify'* ]];;
*) exit 95;;
esac
"#,
        );
        let script = format!(
            r#"#!/bin/bash
set -euo pipefail
verification_candidate_unchanged() {{ [[ "$MODE" != changed-source ]]; }}
verify_repair_pin() {{ [[ "$MODE" != changed-pin ]]; }}
write_pr_body() {{ printf body >> "$TRACE"; }}
CERTIFIED_SHA="$FIXTURE_HEAD"; VERIFICATION_TREE="$FIXTURE_TREE"; CANDIDATE_BASE_HEAD="$FIXTURE_BASE"
BRANCH="$FIXTURE_BRANCH"; BUNDLE=bundle; PR_BODY=body; TRUSTED_ROOT="$PWD"; GITHUB_OUTPUT=output
{}
finalize_certified_tree
"#,
            declaration(&source, "finalize_certified_tree")
        );
        let output = fixture.run(&script, &[("MODE", mode)]);
        assert_eq!(output.0, mode == "valid", "{mode}: {output:?}");
        if mode == "valid" {
            let fields = fs::read_to_string(fixture.root.join("output")).unwrap();
            assert!(
                fields.contains(&format!("head={HEAD}\n"))
                    && fields.contains(&format!("branch={BRANCH}\n"))
            );
            assert!(
                fields.contains("pr_body=body\n") && fields.contains("candidate_bundle=bundle\n")
            );
            assert!(fixture.trace().contains(&format!("^{BASE}")));
        } else {
            assert!(!fixture.root.join("output").exists());
            assert!(
                !fixture.trace().contains("bundle create") && !fixture.trace().contains("body")
            );
        }
        fixture.close();
    }
}
fn publisher_tools(fixture: &Fixture) {
    executable(
        &fixture.tools.join("git"),
        r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$TRACE"
case "$1 ${2:-}" in
'bundle verify') ;;
'bundle list-heads') if [[ "$MODE" == bundle-mismatch ]]; then head="$FIXTURE_BASE"; else head="$FIXTURE_HEAD"; fi; printf '%s refs/heads/%s\n' "$head" "$FIXTURE_BRANCH";;
'fetch '*) [[ "$3" == "refs/heads/$FIXTURE_BRANCH" ]];;
'checkout --detach') ;;
'rev-parse HEAD') if [[ "$MODE" == checkout-mismatch ]]; then echo "$FIXTURE_BASE"; else echo "$FIXTURE_HEAD"; fi;;
'status --porcelain') if [[ "$MODE" == dirty ]]; then echo ' M tracked'; fi;;
'push '*) [[ "$("$GIT_ASKPASS" Username)" == x-access-token && "$("$GIT_ASKPASS" Password)" == "$CANARY_REPAIR_TOKEN" ]]; [[ $# == 3 && "$2" == https://github.com/fixture/repository.git && "$3" == "HEAD:refs/heads/$FIXTURE_BRANCH" ]]; [[ -x "$GIT_ASKPASS" && "$GIT_TERMINAL_PROMPT" == 0 ]];;
*) exit 95;;
esac
"#,
    );
    executable(
        &fixture.tools.join("gh"),
        r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$TRACE"
[[ "$GH_TOKEN" == "$CANARY_REPAIR_TOKEN" ]]
case "$1 ${2:-}" in
'pr list') ;;
'api --method') [[ "$3" == DELETE && "$MODE" == create-failure ]];;
'api '*) if [[ "$MODE" == remote-changed ]]; then echo "$FIXTURE_BASE"; else echo "$FIXTURE_HEAD"; fi;;
'pr create') [[ "$*" != *--draft* && "$*" == *'--body-file '* ]]; if [[ "$MODE" == create-failure ]]; then exit 23; fi; echo 'https://example.invalid/pull/1';;
*) exit 96;;
esac
"#,
    );
}
#[test]
fn retained_publisher_admits_exact_push_and_refuses_identity_drift_without_unsafe_rollback() {
    for mode in [
        "valid",
        "bundle-mismatch",
        "checkout-mismatch",
        "dirty",
        "remote-changed",
        "create-failure",
    ] {
        let fixture = Fixture::new();
        for path in [
            fixture.root.join("scripts"),
            fixture.root.join("third_party/llama.cpp"),
            fixture.root.join("runner"),
        ] {
            fs::create_dir_all(path).unwrap();
        }
        fs::copy(
            repository().join("scripts/llama-canary-publish.sh"),
            fixture.root.join("scripts/llama-canary-publish.sh"),
        )
        .unwrap();
        fs::write(
            fixture.root.join("third_party/llama.cpp/upstream.txt"),
            HEAD,
        )
        .unwrap();
        publisher_tools(&fixture);
        // Invoke the whole production script: the controller is copied before its redactor is used.
        let script = "#!/bin/bash\nset -euo pipefail\nexport CANARY_PR_BODY=\"$PWD/body\" CANARY_BUNDLE=\"$PWD/bundle\" RUNNER_TEMP=\"$PWD/runner\"\nexec /bin/bash scripts/llama-canary-publish.sh\n";
        let output = fixture.run(
            script,
            &[
                ("MODE", mode),
                ("CANARY_BRANCH", BRANCH),
                ("CANARY_CERTIFIED_SHA", HEAD),
                ("GITHUB_REPOSITORY", "fixture/repository"),
                ("CANARY_REPAIR_TOKEN", "inert-publication-secret"),
                ("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask")),
            ],
        );
        assert_eq!(output.0, mode == "valid", "{mode}: {output:?}");
        assert!(
            !output.1.contains("inert-publication-secret")
                && !output.2.contains("inert-publication-secret")
        );
        let trace = fixture.trace();
        assert!(
            !trace.contains("--force")
                && !trace.contains("--draft")
                && !trace.contains("pr view")
                && !trace.contains("forbidden")
        );
        let pushed = mode == "valid" || mode == "remote-changed" || mode == "create-failure";
        assert_eq!(trace.contains("push "), pushed, "{trace}");
        assert_eq!(
            trace.contains("api --method DELETE"),
            mode == "create-failure",
            "{trace}"
        );
        assert_eq!(
            trace.contains("pr create"),
            mode == "valid" || mode == "create-failure",
            "{trace}"
        );
        if mode == "valid" {
            assert!(trace.find("push ").unwrap() < trace.find("pr create").unwrap());
        }
        if mode == "remote-changed" {
            assert!(output.2.contains("refusing to remove"));
        }
        assert_eq!(
            fs::read_dir(fixture.root.join("runner")).unwrap().count(),
            0
        );
        fixture.close();
    }
}
