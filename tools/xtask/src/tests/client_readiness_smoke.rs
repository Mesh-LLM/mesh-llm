use crate::command::{DynResult, unique_temp_dir};
use std::cell::RefCell;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::time::{Duration, Instant};

const CLIENT: &str = r#"#!/usr/bin/env bash
printf '%s\n' "$$" > "$SMOKE_CHILD_PID"
stop() {
    case "$SMOKE_FIXTURE" in
        nonzero) exit 7 ;;
        hang) return ;;
        *) exit 0 ;;
    esac
}
trap stop TERM
echo '{"event":"passive_mode","status":"ready","role":"client"}'
while :; do sleep 0.1; done
"#;

const PROCESS_ADAPTER: &str = r#"#!/usr/bin/env bash
set -u
if [[ "$1" != */ci-client-readiness-process.py ]]; then
    exec "$SMOKE_REAL_PYTHON" "$@"
fi
shift
operation=$1
shift
case "$operation" in
    run)
        pid_file=$2
        log_file=$4
        shift 5
        "$@" > "$log_file" 2>&1 &
        native_pid=$!
        printf '%s\n' "$native_pid" > "$pid_file"
        wait "$native_pid"
        exit "$?"
        ;;
    ctrl-break)
        if [[ "$SMOKE_FIXTURE" == signal-failed ]]; then
            echo 'fixture: console signal delivery failed' >&2
            exit 2
        fi
        kill -TERM "$2"
        ;;
    is-running)
        # Reproduce the Windows console API's successful, non-liveness result.
        exit 0
        ;;
    force-stop) kill -KILL "$2" ;;
    *) exit 2 ;;
esac
"#;

const LOG_REMOVAL_ADAPTER: &str = r#"#!/usr/bin/env bash
if [[ "$#" == 2 && "$1" == -f && "$2" == */mlc-ready.*.log ]]; then
    case "$SMOKE_FIXTURE" in
        cleanup-transient|cleanup-locked)
            attempts=0
            [[ ! -f "$SMOKE_RM_ATTEMPTS" ]] || attempts=$(<"$SMOKE_RM_ATTEMPTS")
            attempts=$((attempts + 1))
            printf '%s\n' "$attempts" > "$SMOKE_RM_ATTEMPTS"
            printf '%s\n' "$2" > "$SMOKE_LOG_PATH"
            if [[ "$SMOKE_FIXTURE" == cleanup-locked || "$attempts" -lt 3 ]]; then
                echo "rm: cannot remove '$2': Device or resource busy" >&2
                exit 1
            fi
            ;;
    esac
fi
exec "$SMOKE_REAL_RM" "$@"
"#;

struct Fixture {
    root: PathBuf,
    process: RefCell<Option<Child>>,
}

impl Fixture {
    fn new() -> DynResult<Self> {
        let root = unique_temp_dir("client-readiness-windows-adapter");
        fs::create_dir_all(root.join("bin"))?;
        fs::create_dir(root.join("state"))?;
        fs::create_dir(root.join("native-runtimes"))?;
        for (relative, body) in [
            ("client", CLIENT),
            ("bin/python3", PROCESS_ADAPTER),
            ("bin/rm", LOG_REMOVAL_ADAPTER),
            ("bin/uname", "#!/usr/bin/env bash\necho MSYS_NT-10.0\n"),
        ] {
            let path = root.join(relative);
            fs::write(&path, body)?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
        }
        Ok(Self {
            root,
            process: RefCell::new(None),
        })
    }

    fn run(&self, mode: &str) -> DynResult<Output> {
        let python = Command::new("sh")
            .args(["-c", "command -v python3"])
            .output()?;
        assert!(
            python.status.success(),
            "existing Python fixture prerequisite"
        );
        let path = std::env::var("PATH")?;
        let rm = Command::new("sh").args(["-c", "command -v rm"]).output()?;
        assert!(rm.status.success(), "existing rm fixture prerequisite");
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/ci-client-readiness-smoke.sh");
        let stdout_path = self.root.join("stdout.log");
        let stderr_path = self.root.join("stderr.log");
        let child = Command::new("bash")
            .process_group(0)
            .arg(script)
            .arg(self.root.join("client"))
            .arg(self.root.join("native-runtimes"))
            .env(
                "PATH",
                format!("{}:{path}", self.root.join("bin").display()),
            )
            .env(
                "SMOKE_REAL_PYTHON",
                String::from_utf8(python.stdout)?.trim(),
            )
            .env("SMOKE_FIXTURE", mode)
            .env("SMOKE_REAL_RM", String::from_utf8(rm.stdout)?.trim())
            .env("SMOKE_RM_ATTEMPTS", self.root.join("rm-attempts"))
            .env("SMOKE_LOG_PATH", self.root.join("log-path"))
            .env("SMOKE_CHILD_PID", self.root.join("child.pid"))
            .env("MESH_LLM_CLIENT_READY_MAX_WAIT", "3")
            .env("MESH_LLM_CLIENT_SHUTDOWN_MAX_WAIT", "1")
            .env("MESH_LLM_CLIENT_STATE_PARENT", self.root.join("state"))
            .stdout(Stdio::from(fs::File::create(&stdout_path)?))
            .stderr(Stdio::from(fs::File::create(&stderr_path)?))
            .spawn()?;
        *self.process.borrow_mut() = Some(child);
        let status = self.wait_for_smoke()?;
        Ok(Output {
            status,
            stdout: fs::read(stdout_path)?,
            stderr: fs::read(stderr_path)?,
        })
    }

    fn wait_for_smoke(&self) -> DynResult<ExitStatus> {
        let mut process = self.process.borrow_mut();
        let Some(child) = process.as_mut() else {
            return Err("fixture smoke process was not started".into());
        };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err("fixture smoke exceeded independent 15-second deadline".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn assert_child_reaped(&self) -> DynResult<()> {
        let pid = fs::read_to_string(self.root.join("child.pid"))?;
        let result = Command::new("kill").args(["-0", pid.trim()]).output()?;
        assert!(
            !result.status.success(),
            "fixture native child remains alive"
        );
        Ok(())
    }

    fn assert_reaped(&self) -> DynResult<()> {
        self.assert_child_reaped()?;
        assert!(fs::read_dir(self.root.join("state"))?.next().is_none());
        Ok(())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(child) = self.process.get_mut().as_mut() {
            // Contain cleanup to the process group created by this fixture,
            // including on assertions or an independent supervision timeout.
            let _ = Command::new("bash")
                .args(["-c", "kill -KILL -- \"-$1\"", "bash"])
                .arg(child.id().to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn messages(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn windows_adapter_observes_clean_child_exit_despite_false_legacy_liveness() -> DynResult<()> {
    let fixture = Fixture::new()?;
    let output = fixture.run("clean")?;
    assert!(output.status.success(), "{}", messages(&output));
    assert!(messages(&output).contains("CTRL_BREAK_EVENT sent"));
    fixture.assert_reaped()
}

#[test]
fn windows_adapter_preserves_native_nonzero_exit() -> DynResult<()> {
    let fixture = Fixture::new()?;
    let output = fixture.run("nonzero")?;
    assert!(!output.status.success(), "{}", messages(&output));
    assert!(messages(&output).contains("client exited non-cleanly after CTRL_BREAK_EVENT: 7"));
    fixture.assert_reaped()
}

#[test]
fn windows_adapter_rejects_timeout_and_reaps_owned_child() -> DynResult<()> {
    let fixture = Fixture::new()?;
    let output = fixture.run("hang")?;
    assert!(!output.status.success(), "{}", messages(&output));
    assert!(messages(&output).contains("did not stop cleanly after CTRL_BREAK_EVENT within 1s"));
    fixture.assert_reaped()
}

#[test]
fn windows_adapter_reports_signal_failure_and_still_reaps_owned_child() -> DynResult<()> {
    let fixture = Fixture::new()?;
    let output = fixture.run("signal-failed")?;
    assert!(!output.status.success(), "{}", messages(&output));
    assert!(messages(&output).contains("fixture: console signal delivery failed"));
    assert!(messages(&output).contains("failed to send CTRL_BREAK_EVENT"));
    fixture.assert_reaped()
}

#[test]
fn windows_adapter_removes_log_after_transient_windows_lock() -> DynResult<()> {
    let fixture = Fixture::new()?;
    let output = fixture.run("cleanup-transient")?;
    assert!(output.status.success(), "{}", messages(&output));
    assert_eq!(
        fs::read_to_string(fixture.root.join("rm-attempts"))?.trim(),
        "3"
    );
    fixture.assert_reaped()
}

#[test]
fn windows_adapter_fails_bounded_cleanup_and_retains_locked_log() -> DynResult<()> {
    let fixture = Fixture::new()?;
    let output = fixture.run("cleanup-locked")?;
    assert!(!output.status.success(), "{}", messages(&output));
    assert!(messages(&output).contains("client log cleanup failed after 5 attempts"));
    assert!(messages(&output).contains("Device or resource busy"));
    assert_eq!(
        fs::read_to_string(fixture.root.join("rm-attempts"))?.trim(),
        "5"
    );
    let log_path = fs::read_to_string(fixture.root.join("log-path"))?;
    assert!(
        Path::new(log_path.trim()).is_file(),
        "locked log was discarded"
    );
    fixture.assert_child_reaped()
}
