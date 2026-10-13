use super::*;
use std::{fs, path::PathBuf, process::Stdio};

struct Fixture {
    root: PathBuf,
    child: Child,
    _directory: tempfile::TempDir,
}

impl Fixture {
    fn spawn(mode: &str) -> Self {
        let directory = tempfile::Builder::new()
            .prefix("eval-cleanup-")
            .tempdir()
            .unwrap();
        let root = directory.path().canonicalize().unwrap();
        let mut command = Command::new("/bin/bash");
        command.args([
            "-c",
            r#"
case "$FIXTURE_MODE" in
  term-exits) trap 'exit 0' TERM ;;
  term-ignored) trap '' TERM ;;
esac
(trap '' TERM; exec sleep 30) &
printf '%s' "$!" > "$FIXTURE_ROOT/grandchild.pid"
printf 'original stdout\n'
printf 'original stderr\n' >&2
case "$FIXTURE_MODE" in
  exit-zero) exit 0 ;;
  exit-failure) exit 23 ;;
esac
while :; do sleep 0.1; done
"#,
        ]);
        command
            .env("FIXTURE_ROOT", &root)
            .env("FIXTURE_MODE", mode)
            .stdout(Stdio::from(fs::File::create(root.join("stdout")).unwrap()))
            .stderr(Stdio::from(fs::File::create(root.join("stderr")).unwrap()));
        configure_child_group(&mut command);
        let child = command.spawn().unwrap();
        let fixture = Self {
            root,
            child,
            _directory: directory,
        };
        fixture.grandchild();
        fixture
    }

    fn grandchild(&self) -> libc::pid_t {
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(pid) = fs::read_to_string(self.root.join("grandchild.pid"))
                .ok()
                .and_then(|value| value.parse().ok())
            {
                return pid;
            }
            assert!(Instant::now() < until, "fixture grandchild did not start");
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn assert_stopped(&mut self) {
        assert!(
            self.child.try_wait().unwrap().is_some(),
            "direct child not reaped"
        );
        let pid = self.grandchild();
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            let output = Command::new("ps")
                .args(["-o", "stat=", "-p", &pid.to_string()])
                .output()
                .unwrap();
            let state = String::from_utf8(output.stdout).unwrap();
            // Reparented descendants may remain zombies temporarily. They no
            // longer execute or hold live inherited writers; we cannot reap them.
            if state.trim().is_empty() || state.trim().starts_with('Z') {
                break;
            }
            assert!(
                Instant::now() < until,
                "owned grandchild still live: {state}"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: emergency cleanup of the fixture's freshly owned process group.
        if let Ok(pid) = libc::pid_t::try_from(self.child.id()) {
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
        let _ = reap_child(&mut self.child, Duration::from_secs(1));
        // TempDir releases the files only after the owned process cleanup above.
    }
}

#[test]
fn exited_direct_child_preserves_status_and_file_output_but_cleans_descendants() {
    for (mode, code) in [("exit-zero", 0), ("exit-failure", 23)] {
        let mut fixture = Fixture::spawn(mode);
        let result = wait_with_timeout(&mut fixture.child, Some(Duration::from_secs(2))).unwrap();
        assert_eq!(result.exit_status, Some(code));
        assert_eq!(result.success, code == 0);
        assert!(!result.timed_out);
        fixture.assert_stopped();
        assert_eq!(
            fs::read(fixture.root.join("stdout")).unwrap(),
            b"original stdout\n"
        );
        assert_eq!(
            fs::read(fixture.root.join("stderr")).unwrap(),
            b"original stderr\n"
        );
    }
}

#[test]
fn timeout_forces_descendants_even_when_direct_child_exits_during_grace() {
    let mut fixture = Fixture::spawn("term-exits");
    let started = Instant::now();
    let result = wait_polled(
        &mut fixture.child,
        Some(Duration::from_millis(30)),
        Duration::from_millis(300),
        |child| child.try_wait().map_err(Into::into),
    )
    .unwrap();
    assert!(result.timed_out);
    assert!(!result.success);
    assert_eq!(result.exit_status, None);
    assert!(started.elapsed() < Duration::from_secs(2));
    fixture.assert_stopped();
    assert_eq!(fixture.child.try_wait().unwrap().unwrap().code(), Some(0));
}

#[test]
fn term_ignored_direct_child_and_descendant_are_forced_within_cleanup_budget() {
    let mut fixture = Fixture::spawn("term-ignored");
    let started = Instant::now();
    let result = wait_polled(
        &mut fixture.child,
        Some(Duration::from_millis(30)),
        Duration::from_millis(50),
        |child| child.try_wait().map_err(Into::into),
    )
    .unwrap();
    assert!(result.timed_out);
    assert!(started.elapsed() < Duration::from_secs(2));
    fixture.assert_stopped();
}

#[test]
fn polling_error_preserves_original_failure_after_owned_group_cleanup() {
    let mut fixture = Fixture::spawn("term-exits");
    let error = match wait_polled(&mut fixture.child, None, Duration::from_millis(300), |_| {
        Err(io::Error::new(io::ErrorKind::PermissionDenied, "fixture poll refusal").into())
    }) {
        Ok(_) => panic!("polling refusal unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::PermissionDenied
    );
    assert!(format!("{error:#}").contains("fixture poll refusal"));
    fixture.assert_stopped();
}

#[test]
fn capture_observation_error_cleans_direct_child_and_surviving_descendant() {
    let mut fixture = Fixture::spawn("term-exits");
    let capture = fixture.root.join("stdout");
    fs::remove_file(&capture).unwrap();
    let started = Instant::now();
    let error = match wait_with_timeout_observed(&mut fixture.child, None, || {
        fs::metadata(&capture).context("observe owned fixture capture")?;
        Ok(())
    }) {
        Ok(_) => panic!("missing capture unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(
        error.downcast_ref::<io::Error>().unwrap().kind(),
        io::ErrorKind::NotFound
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    fixture.assert_stopped();
}
