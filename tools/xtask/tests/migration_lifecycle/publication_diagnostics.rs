//! Run the publisher against finite Git/gh seams and the real Rust redactor.
#![cfg(unix)]
use std::{
    fs,
    io::Write as _,
    os::unix::fs::PermissionsExt as _,
    path::PathBuf,
    process::{Command, Output, Stdio},
};

const TOKEN: &str = "fixture-token.[x]";
const HEAD: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const URL: &str = "https://example.invalid/pull/1";

fn executable(path: &std::path::Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn redact(token: Option<&str>, input: &[u8], arguments: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .args(["automation", "canary-receipts", "redact-publication-log"])
        .args(arguments)
        .env_remove("CANARY_REPAIR_TOKEN")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(token) = token {
        command.env("CANARY_REPAIR_TOKEN", token);
    }
    let mut child = command.spawn().unwrap();
    if !input.is_empty()
        && let Err(error) = child.stdin.as_mut().unwrap().write_all(input)
    {
        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
    }
    drop(child.stdin.take());
    child.wait_with_output().unwrap()
}

#[test]
fn actual_cli_redacts_unicode_empty_lines_repetitions_and_large_unterminated_input() {
    let input = format!("λ {TOKEN}\r\n\n{} {TOKEN}{TOKEN} tail", "x".repeat(20_000));
    let expected = format!(
        "λ ***redacted***\r\n\n{} ***redacted******redacted*** tail",
        "x".repeat(20_000)
    );
    let output = redact(Some(TOKEN), input.as_bytes(), &[]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, expected.as_bytes());
    assert!(output.stderr.is_empty());
    let empty = redact(Some(TOKEN), b"", &[]);
    assert!(empty.status.success());
    assert!(empty.stdout.is_empty());
}

#[test]
fn actual_cli_refuses_missing_empty_or_malformed_inputs_without_secret_output() {
    for token in [None, Some("")] {
        let output = redact(token, b"", &[]);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
    let output = redact(
        Some(TOKEN),
        format!("{TOKEN}\u{fffd}").as_bytes(),
        &["--unexpected"],
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&output.stderr).contains(TOKEN));
    let invalid = [TOKEN.as_bytes(), b"\xff"].concat();
    let output = redact(Some(TOKEN), &invalid, &[]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&output.stderr).contains(TOKEN));
    let help = redact(None, b"", &["--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("redact-publication-log"));
}

struct Publisher {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    tools: PathBuf,
    runner: PathBuf,
    owner: PathBuf,
}

impl Publisher {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("candidate with spaces");
        let tools = temporary.path().join("finite tools");
        let runner = temporary.path().join("runner temporary");
        for path in [
            root.join("scripts"),
            root.join("skippy/llama_cpp"),
            tools.clone(),
            runner.clone(),
        ] {
            fs::create_dir_all(path).unwrap();
        }
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/llama-canary-publish.sh"),
            root.join("scripts/llama-canary-publish.sh"),
        )
        .unwrap();
        fs::write(root.join("skippy/llama_cpp/upstream.txt"), HEAD).unwrap();
        fs::write(root.join("body.md"), b"public fixture body").unwrap();
        fs::write(root.join("candidate.bundle"), b"inert Git response fixture").unwrap();
        let owner = tools.join("source controller");
        executable(
            &owner,
            r#"#!/bin/bash
if [[ "${FIXTURE_REDACTOR_FAILURE:-0}" == 1 ]]; then /bin/cat >/dev/null; exit 19; fi
exec "$FIXTURE_XTASK" "$@"
"#,
        );
        executable(
            &tools.join("git"),
            r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$1 ${2:-}" >> "$FIXTURE_TRACE"
case "$1 ${2:-}" in
  'bundle verify') ;;
  'bundle list-heads') printf '%s refs/heads/%s\n' "$FIXTURE_HEAD" "$CANARY_BRANCH" ;;
  'fetch '*) ;;
  'checkout --detach')
    printf '#!/bin/bash\necho candidate-owner-must-not-run >&2\nexit 91\n' > "$MESH_LLM_AUTOMATION_BIN"
    chmod 700 "$MESH_LLM_AUTOMATION_BIN"
    ;;
  'rev-parse HEAD') printf '%s\n' "$FIXTURE_HEAD" ;;
  'status --porcelain') ;;
  'push '*) printf 'λ git credential=%s %s\n' "$CANARY_REPAIR_TOKEN" "$CANARY_REPAIR_TOKEN" >&2; exit "$FIXTURE_PUSH_STATUS" ;;
  *) exit 92 ;;
esac
"#,
        );
        executable(
            &tools.join("gh"),
            r#"#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$FIXTURE_TRACE"
case "$1 ${2:-}" in
  'pr list')
    if [[ "$*" == *url,isDraft,headRefOid* && "$FIXTURE_RECONCILE" == 1 ]]; then printf '%s\n' "$FIXTURE_URL"; fi
    ;;
  'api --method') [[ "$3" == DELETE ]] ;;
  'api '*) printf '%s\n' "$FIXTURE_HEAD" ;;
  'pr create')
    printf 'gh credential=%s\n' "$CANARY_REPAIR_TOKEN" >&2
    if [[ "$FIXTURE_CREATE_STATUS" == 0 ]]; then printf '%s\n' "$FIXTURE_URL"; fi
    exit "$FIXTURE_CREATE_STATUS"
    ;;
  *) exit 93 ;;
esac
"#,
        );
        for name in ["python3", "cargo", "just"] {
            executable(
                &tools.join(name),
                "#!/bin/bash\nprintf forbidden > \"$FIXTURE_FORBIDDEN\"\nexit 94\n",
            );
        }
        Self {
            _temporary: temporary,
            root,
            tools,
            runner,
            owner,
        }
    }

    fn invoke(
        &self,
        owner: &str,
        push: u8,
        create: u8,
        reconcile: bool,
        filter_failure: bool,
    ) -> Output {
        Command::new("/bin/bash")
            .arg(self.root.join("scripts/llama-canary-publish.sh"))
            .current_dir(&self.root)
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", self.tools.display()))
            .env("HOME", &self.root)
            .env("RUNNER_TEMP", &self.runner)
            .env("CANARY_BRANCH", "llama-canary/repair-fixture-1-aaaaaaaaaa")
            .env("CANARY_CERTIFIED_SHA", HEAD)
            .env("CANARY_PR_BODY", self.root.join("body.md"))
            .env("CANARY_BUNDLE", self.root.join("candidate.bundle"))
            .env("GITHUB_REPOSITORY", "fixture/repository")
            .env("CANARY_REPAIR_TOKEN", TOKEN)
            .env("MESH_LLM_AUTOMATION_BIN", owner)
            .env("FIXTURE_XTASK", env!("CARGO_BIN_EXE_xtask"))
            .env("FIXTURE_HEAD", HEAD)
            .env("FIXTURE_URL", URL)
            .env("FIXTURE_TRACE", self.root.join("trace"))
            .env("FIXTURE_FORBIDDEN", self.root.join("forbidden"))
            .env("FIXTURE_PUSH_STATUS", push.to_string())
            .env("FIXTURE_CREATE_STATUS", create.to_string())
            .env("FIXTURE_RECONCILE", if reconcile { "1" } else { "0" })
            .env(
                "FIXTURE_REDACTOR_FAILURE",
                if filter_failure { "1" } else { "0" },
            )
            .output()
            .unwrap()
    }
}

#[test]
fn actual_publisher_uses_frozen_redactor_and_preserves_success_failure_and_reconciliation() {
    for (push, create, reconcile, filter_failure, success, pr_called, deleted) in [
        (0, 0, false, false, true, true, false),
        (23, 0, false, false, false, false, false),
        (0, 23, false, false, false, true, true),
        (0, 23, true, false, true, true, false),
        (0, 0, false, true, false, false, true),
    ] {
        let fixture = Publisher::new();
        let output = fixture.invoke(
            fixture.owner.to_str().unwrap(),
            push,
            create,
            reconcile,
            filter_failure,
        );
        assert_eq!(output.status.success(), success, "{output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stdout.contains(TOKEN) && !stderr.contains(TOKEN),
            "credential leaked"
        );
        assert!(!stderr.contains("candidate-owner-must-not-run"));
        if !filter_failure {
            assert!(stderr.contains("***redacted***"), "{stderr}");
        }
        let trace = fs::read_to_string(fixture.root.join("trace")).unwrap();
        assert_eq!(trace.contains("pr create"), pr_called, "{trace}");
        assert_eq!(trace.contains("api --method DELETE"), deleted, "{trace}");
        if pr_called {
            assert!(trace.find("push ").unwrap() < trace.find("pr create").unwrap());
        }
        assert!(!fixture.root.join("forbidden").exists());
        assert_eq!(fs::read_dir(&fixture.runner).unwrap().count(), 0);
        if success {
            assert!(stdout.contains(URL));
        }
    }
}

#[test]
fn actual_publisher_refuses_unprepared_controllers_before_git_or_publication() {
    for owner in ["", "relative-controller", "/fixture/missing-controller"] {
        let fixture = Publisher::new();
        let output = fixture.invoke(owner, 0, 0, false, false);
        assert!(!output.status.success());
        assert!(!fixture.root.join("trace").exists());
        assert!(!fixture.root.join("forbidden").exists());
        assert_eq!(fs::read_dir(&fixture.runner).unwrap().count(), 0);
        assert!(!String::from_utf8_lossy(&output.stderr).contains(TOKEN));
    }
}
