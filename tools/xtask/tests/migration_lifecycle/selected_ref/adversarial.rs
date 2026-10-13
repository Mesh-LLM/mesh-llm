use super::*;
use std::{
    os::unix::fs::PermissionsExt,
    process::{Child, Command, Stdio},
    thread,
    time::Instant,
};
struct Unrelated(Child);
impl Unrelated {
    fn new() -> Self {
        Self(
            Command::new("/bin/sleep")
                .arg("60")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        )
    }
}
impl Drop for Unrelated {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}
fn interrupted_fetch(cancelled: bool) {
    let fixture = Fixture::new();
    let wrapper = fixture.home.join("git");
    let ready = fixture.home.join("fetch-ready");
    let pid = fixture.home.join("descendant-pid");
    fs::write(&wrapper,format!("#!/bin/sh\nif [ \"$3\" = fetch ]; then\n /bin/sleep 60 &\n printf '%s\\n' \"$!\" > {}\n : > {}\n wait\nelse\n exec {} \"$@\"\nfi\n",quote(&pid),quote(&ready),quote(&fixture.git))).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    let argv = [
        "repository",
        "selected-ref",
        "--repository",
        fixture.checkout.to_str().unwrap(),
        "--ref",
        "candidate",
        "--event",
        "workflow_dispatch",
        "--expected-origin",
        &fixture.url,
        "--github-output",
        fixture.output.to_str().unwrap(),
        "--summary",
        fixture.summary.to_str().unwrap(),
        "--timeout-secs",
        if cancelled { "10" } else { "1" },
    ];
    let path = format!(
        "{}:{}",
        fixture.home.display(),
        std::env::var("PATH").unwrap()
    );
    let spec = ProcessSpec {
        executable: PathBuf::from(env!("CARGO_BIN_EXE_xtask")),
        cwd: fixture.checkout.clone(),
        arguments: argv
            .iter()
            .map(|arg| Value::Public((*arg).into()))
            .collect(),
        environment: BTreeMap::from([
            ("PATH".into(), Value::Public(path.into())),
            (
                "HOME".into(),
                Value::Public(fixture.home.as_os_str().into()),
            ),
            ("GIT_MASTER".into(), Value::Public("1".into())),
        ]),
    };
    let cancellation = Cancellation::default();
    let token = cancellation.clone();
    let ready_path = ready.clone();
    let trigger = thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(6);
        while !ready_path.is_file() && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        let seen = ready_path.is_file();
        if cancelled && seen {
            token.cancel();
        }
        seen
    });
    let limits = Limits {
        execution: Duration::from_secs(15),
        graceful_shutdown: Duration::from_secs(5),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let mut sentinel = Unrelated::new();
    let report = process::supervise(&spec, &limits, &cancellation, OutputFiles::default()).unwrap();
    let seen = trigger.join().unwrap();
    assert_completed(&fixture, &pid, &mut sentinel, &report, seen, cancelled);
}
fn assert_completed(
    fixture: &Fixture,
    pid: &Path,
    sentinel: &mut Unrelated,
    report: &process::ProcessReport,
    seen: bool,
    cancelled: bool,
) {
    assert!(seen, "fetch fixture did not reach blocking operation");
    assert!(
        report.cleanup.complete && !report.cleanup.forced,
        "{report:?}"
    );
    assert!(
        !report.status.is_some_and(|status| status.success()),
        "{report:?}"
    );
    if cancelled {
        assert_eq!(report.outcome, process::Outcome::Cancelled);
    } else {
        assert_eq!(report.outcome, process::Outcome::Exited);
    }
    let descendant = fs::read_to_string(pid)
        .unwrap()
        .trim()
        .parse::<u32>()
        .unwrap();
    crate::support::assert_absent(descendant);
    assert!(sentinel.0.try_wait().unwrap().is_none());
    assert_eq!(
        fs::read_to_string(&fixture.output).unwrap(),
        "prior-output\n"
    );
    assert_eq!(
        fs::read_to_string(&fixture.summary).unwrap(),
        "prior-summary\n"
    );
    assert!(report.stdout.bytes_retained.is_empty(), "{report:?}");
}
#[test]
fn actual_fetch_transaction_deadline_reaps_owned_descendant_and_preserves_unrelated_sentinel() {
    interrupted_fetch(false);
}
#[test]
fn actual_cli_cancellation_reaps_detached_git_group_restores_scope_and_preserves_sentinel() {
    interrupted_fetch(true);
}
