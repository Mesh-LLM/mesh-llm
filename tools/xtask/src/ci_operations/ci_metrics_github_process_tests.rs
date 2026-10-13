use super::*;
use crate::ci_operations::ci_metrics_argv::Args;
use crate::ci_operations::ci_metrics_github::{Gh, fetch_runs};
use crate::ci_operations::ci_metrics_value::Value as Json;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::time::Instant;

struct Fixture {
    root: tempfile::TempDir,
    cancellation: Cancellation,
    environment: Environment,
}
impl Fixture {
    fn new(body: &str) -> Self {
        let root = tempfile::Builder::new()
            .prefix("metrics gh transport ")
            .tempdir()
            .unwrap();
        fs::create_dir(root.path().join("bin")).unwrap();
        let gh = root.path().join("bin/gh");
        fs::write(&gh, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(gh, fs::Permissions::from_mode(0o700)).unwrap();
        let environment = Environment::from([
            (
                "PATH".into(),
                std::env::join_paths([
                    root.path().join("bin"),
                    PathBuf::from("/usr/bin"),
                    PathBuf::from("/bin"),
                ])
                .unwrap(),
            ),
            ("OPERATOR_INPUT".into(), "operator value with spaces".into()),
            ("GH_TOKEN".into(), "fixture-auth-token".into()),
        ]);
        Self {
            root,
            cancellation: Cancellation::default(),
            environment,
        }
    }
    fn run(&self, budget: &Limits, cap: usize) -> io::Result<GhOutput> {
        run_in(
            &["run".to_owned(), "list".to_owned()],
            &self.cancellation,
            self.root.path(),
            &self.environment,
            budget,
            cap,
        )
    }
    fn child_gone(&self) {
        let pid: i32 = fs::read_to_string(self.root.path().join("owned-child"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // SAFETY: signal zero probes only the finite fixture's recorded child.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
    }
}
impl Gh for Fixture {
    fn run(&mut self, arguments: &[String]) -> io::Result<GhOutput> {
        run_in(
            arguments,
            &self.cancellation,
            self.root.path(),
            &self.environment,
            &fast_limits(),
            65536,
        )
    }
}
fn fast_limits() -> Limits {
    let mut value = limits(Duration::from_millis(500));
    value.graceful_shutdown = Duration::from_secs(1);
    value.forced_shutdown = Duration::from_secs(1);
    value
}
fn tree(output: &str) -> String {
    format!(
        "sleep 20 &\nchild=$!\ntrap 'kill \"$child\" 2>/dev/null; wait \"$child\" 2>/dev/null; exit 143' TERM INT\nprintf '%s\\n' \"$child\" > owned-child\n{output}\nwait \"$child\""
    )
}

#[test]
fn metrics_gh_actual_path_stub_collects_runs_jobs_and_preserves_native_arguments() {
    let mut fixture = Fixture::new(
        r#"if read unexpected; then echo inherited-stdin >&2; exit 90; fi
[ "$OPERATOR_INPUT" = 'operator value with spaces' ] || exit 91
[ "$GH_TOKEN" = 'fixture-auth-token' ] || exit 93
{ printf 'cwd=%s\n' "$PWD"; for arg in "$@"; do printf '%s\n' "$arg"; done; printf '<end>\n'; } >> argv
case "$1 $2" in
'run list') printf '[{"databaseId":7}]\r\n' ;;
'api --method') printf '{"total_count":1,"jobs":[{"name":"fixture job"}]}\r\n' ;;
*) exit 92 ;;
esac"#,
    );
    let args = Args {
        workflow: Some("fixture.yml".to_owned()),
        repo: "owner/repo".to_owned(),
        ..Args::default()
    };
    let runs = fetch_runs(&mut fixture, &args).unwrap();
    assert_eq!(runs.len(), 1);
    assert!(matches!(runs[0].get("jobs"), Some(Json::Array(jobs)) if jobs.len() == 1));
    let argv = fs::read_to_string(fixture.root.path().join("argv")).unwrap();
    assert_eq!(argv.matches("<end>").count(), 2);
    assert!(argv.contains("--repo\nowner/repo\n--workflow\nfixture.yml"));
    assert!(argv.contains(
        "repos/owner/repo/actions/runs/7/jobs\n-f\nfilter=latest\n-f\nper_page=100\n-f\npage=1"
    ));
    assert!(argv.contains(&format!(
        "cwd={}",
        fixture.root.path().canonicalize().unwrap().display()
    )));
}

#[test]
fn metrics_gh_actual_failing_status_keeps_collection_failure_intent() {
    let mut fixture = Fixture::new(
        "printf 'fallback output\\r\\n'\nprintf 'permission denied\\r\\n' >&2\nexit 13",
    );
    let args = Args {
        workflow: Some("fixture".to_owned()),
        ..Args::default()
    };
    let error = fetch_runs(&mut fixture, &args).unwrap_err();
    assert!(
        matches!(error, crate::ci_operations::ci_metrics_normalize::Failure::Reported(message) if message.contains("failed: permission denied"))
    );
    let output = fixture.run(&fast_limits(), 65536).unwrap();
    assert!(!output.success);
    assert_eq!(output.stdout, "fallback output\n");
    assert_eq!(output.stderr, "permission denied\n");
}

#[test]
fn metrics_gh_timeout_reaps_owned_descendant_before_return() {
    let fixture = Fixture::new(&tree(""));
    let began = Instant::now();
    let error = fixture.run(&fast_limits(), 65536).err().unwrap();
    assert!(error.to_string().contains("Deadline"));
    assert!(began.elapsed() < Duration::from_secs(4));
    fixture.child_gone();
}

#[test]
fn metrics_gh_json_overflow_refuses_partial_response_and_reaps_descendant() {
    let fixture = Fixture::new(&tree("printf '[%01024d]' 0"));
    let error = fixture.run(&fast_limits(), 64).err().unwrap();
    assert!(error.to_string().contains("RawCaptureOverflow"));
    fixture.child_gone();
}

#[test]
fn metrics_gh_precancellation_refuses_launch_before_path_selection() {
    let fixture = Fixture::new("touch forbidden-launch");
    fixture.cancellation.cancel();
    let error = fixture.run(&fast_limits(), 65536).err().unwrap();
    assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    assert!(!fixture.root.path().join("forbidden-launch").exists());
}

#[test]
fn metrics_gh_live_cancellation_reaps_owned_descendant() {
    let fixture = Fixture::new(&tree(""));
    let cancellation = fixture.cancellation.clone();
    let marker = fixture.root.path().join("owned-child");
    let trigger = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(2);
        while !marker.exists() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(marker.exists(), "fixture child did not start");
        cancellation.cancel();
    });
    let mut budget = fast_limits();
    budget.execution = Duration::from_secs(3);
    let result = fixture.run(&budget, 65536);
    trigger.join().unwrap();
    assert_eq!(result.err().unwrap().kind(), io::ErrorKind::Interrupted);
    fixture.child_gone();
}

#[test]
fn metrics_gh_path_resolution_keeps_relative_selected_spelling_and_skips_nonexecutables() {
    let fixture = Fixture::new("printf '[]'");
    let root = fixture.root.path().canonicalize().unwrap();
    fs::create_dir(root.join("first")).unwrap();
    fs::write(root.join("first/gh"), "not executable").unwrap();
    fs::set_permissions(root.join("first/gh"), fs::Permissions::from_mode(0o600)).unwrap();
    let path = std::env::join_paths([PathBuf::from("first"), PathBuf::from("bin")]).unwrap();
    assert_eq!(executable(&root, Some(path)).unwrap(), root.join("bin/gh"));
    assert_eq!(
        executable(&root, Some(root.join("absent").into_os_string()))
            .err()
            .unwrap()
            .kind(),
        io::ErrorKind::NotFound
    );
}

#[test]
fn metrics_gh_failing_diagnostics_redact_inherited_authentication() {
    let fixture = Fixture::new(
        "printf 'permission denied\\n' >&2\nprintf 'authentication: %s\\n' \"$GH_TOKEN\" >&2\nexit 13",
    );
    let output = fixture.run(&fast_limits(), 65536).unwrap();
    assert!(!output.success);
    assert!(output.stderr.contains("permission denied"));
    assert!(output.stderr.contains("[output line suppressed]"));
    assert!(!output.stderr.contains("fixture-auth-token"));
}
