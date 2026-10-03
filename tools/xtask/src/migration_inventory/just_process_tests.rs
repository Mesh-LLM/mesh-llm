//! Transport seam fixtures; the companion suite qualifies actual native Just semantics.
use super::*;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

struct Fixture(tempfile::TempDir);
impl Fixture {
    fn new() -> Self {
        Self(
            tempfile::Builder::new()
                .prefix("native Just transport ")
                .tempdir()
                .unwrap(),
        )
    }
    fn root(&self) -> &Path {
        self.0.path()
    }
    fn command(&self, body: &str) -> ProcessSpec {
        let executable = self.root().join("native parser fixture");
        fs::write(&executable, format!("#!/bin/sh\n{body}\n")).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        ProcessSpec {
            executable,
            arguments: ["--justfile", "Justfile", "--dump", "--dump-format", "json"]
                .into_iter()
                .map(|arg| Value::Public(arg.into()))
                .collect(),
            cwd: self.root().canonicalize().unwrap(),
            environment: BTreeMap::from([
                ("PATH".into(), Value::Public("/usr/bin:/bin".into())),
                (
                    "OPERATOR_INPUT".into(),
                    Value::Public("operator value with spaces".into()),
                ),
            ]),
        }
    }
    fn tree(&self, output: &str) -> ProcessSpec {
        self.command(&format!("sleep 20 &\nchild=$!\ntrap 'kill \"$child\" 2>/dev/null; wait \"$child\" 2>/dev/null; exit 143' TERM INT\nprintf '%s\\n' \"$child\" > owned-child\n{output}\nwait \"$child\""))
    }
    fn assert_child_gone(&self) {
        let pid: i32 = fs::read_to_string(self.root().join("owned-child"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // SAFETY: signal zero probes the fixture-owned, recorded process only.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
use std::fs;

fn fast_limits() -> Limits {
    let mut budget = limits(Duration::from_millis(300));
    budget.graceful_shutdown = Duration::from_secs(1);
    budget.forced_shutdown = Duration::from_secs(1);
    budget
}

#[test]
fn dump_transport_preserves_native_arguments_cwd_environment_and_raw_bytes() {
    let fixture = Fixture::new();
    let spec = fixture.command("printf '%s\\0' \"$PWD\" \"$OPERATOR_INPUT\" \"$@\"");
    let output = invoke(&spec, &fast_limits(), &Cancellation::default(), 4096).unwrap();
    assert_eq!(
        output,
        format!(
            "{}\0operator value with spaces\0--justfile\0Justfile\0--dump\0--dump-format\0json\0",
            fixture.root().canonicalize().unwrap().display()
        )
        .as_bytes()
    );
}

#[test]
fn dump_transport_preserves_nonzero_parse_diagnostic_and_stdout_fallback() {
    let fixture = Fixture::new();
    for (body, expected) in [
        (
            "printf 'stdout detail'; printf 'stderr detail\\n' >&2; exit 19",
            "stderr detail",
        ),
        ("printf 'stdout detail\\n'; exit 19", "stdout detail"),
    ] {
        let error = invoke(
            &fixture.command(body),
            &fast_limits(),
            &Cancellation::default(),
            4096,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("Just recipe: parse failed: {expected}")
        );
    }
}

#[test]
fn dump_transport_deadline_reaps_owned_descendant() {
    let fixture = Fixture::new();
    let error = invoke(
        &fixture.tree(":"),
        &fast_limits(),
        &Cancellation::default(),
        4096,
    )
    .unwrap_err();
    assert!(error.to_string().contains("Deadline"), "{error}");
    fixture.assert_child_gone();
}

#[test]
fn dump_transport_raw_overflow_refuses_partial_json_and_reaps_owned_descendant() {
    let fixture = Fixture::new();
    let spec = fixture.tree("printf 'more than the raw byte bound'");
    let error = invoke(&spec, &fast_limits(), &Cancellation::default(), 8).unwrap_err();
    assert!(error.to_string().contains("RawCaptureOverflow"), "{error}");
    fixture.assert_child_gone();
}

#[test]
fn dump_transport_precancel_refuses_child_launch() {
    let fixture = Fixture::new();
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let error = invoke(
        &fixture.command("touch child-started"),
        &fast_limits(),
        &cancellation,
        4096,
    )
    .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    assert!(!fixture.root().join("child-started").exists());
}

#[test]
fn dump_transport_live_cancellation_reaps_owned_descendant() {
    let fixture = Fixture::new();
    let cancellation = Cancellation::default();
    let flag = cancellation.clone();
    let marker = fixture.root().join("owned-child");
    let observed = Arc::new(AtomicBool::new(false));
    let child_observed = observed.clone();
    let canceller = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(2);
        while !marker.is_file() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        child_observed.store(marker.is_file(), Ordering::SeqCst);
        flag.cancel();
    });
    let mut budget = fast_limits();
    budget.execution = Duration::from_secs(4);
    let result = invoke(&fixture.tree(":"), &budget, &cancellation, 4096);
    canceller.join().unwrap();
    assert!(
        observed.load(Ordering::SeqCst),
        "fixture never admitted owned descendant"
    );
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    fixture.assert_child_gone();
}

#[test]
fn dump_tool_preserves_relative_path_selection_and_skips_nonexecutable_candidates() {
    let fixture = Fixture::new();
    for directory in ["first", "selected"] {
        fs::create_dir(fixture.root().join(directory)).unwrap();
    }
    fs::write(fixture.root().join("first/just"), "not executable").unwrap();
    let target = fixture.root().join("native-just");
    fs::write(&target, "#!/bin/sh\nexit 0\n").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o700)).unwrap();
    std::os::unix::fs::symlink(&target, fixture.root().join("selected/just")).unwrap();
    assert_eq!(
        tool(fixture.root(), Some("first:selected".into())).unwrap(),
        fixture.root().join("selected/just")
    );
}

#[test]
fn inventory_dump_scope_refuses_nested_signal_owner_instead_of_ignoring_cancellation() {
    let parent = Interrupt::install().unwrap();
    let error = operation(|| Ok(())).unwrap_err();
    assert!(error.to_string().contains("signal scope already active"));
    parent.finish().unwrap();
}
