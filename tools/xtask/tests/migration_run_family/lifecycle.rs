use super::support::{Fixture, SHELL_LINE};
use std::{
    fs,
    process::{Child, Command, Output},
    time::{Duration, Instant},
};

struct Sentinel(Child);
impl Drop for Sentinel {
    fn drop(&mut self) {
        let _kill = self.0.kill();
        let _wait = self.0.wait();
    }
}

#[test]
fn timeout_and_cancel_clean_rust_child_reader_tree_and_preserve_independent_sentinel()
-> Result<(), Box<dyn std::error::Error>> {
    for cancel in [false, true] {
        let fixture = Fixture::new()?;
        let mut sentinel = Sentinel(Command::new("/bin/sleep").arg("60").spawn()?);
        let mut command = fixture.command("granite-3.1-2b", Some(if cancel { 30 } else { 3 }));
        command.env("REPLAY_FIXTURE_MODE", "hang");
        command
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let child = command.spawn()?;
        let outer = child.id();
        let started = Instant::now();
        while !fixture.marker.is_file() && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        if !fixture.marker.is_file() {
            let _termination = Command::new("/bin/kill")
                .args(["-TERM", &outer.to_string()])
                .status();
            let output = bounded_output(child)?;
            return Err(format!(
                "reader never started: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        let pids: serde_json::Value = serde_json::from_slice(&fs::read(&fixture.marker)?)?;
        if cancel {
            assert!(
                Command::new("/bin/kill")
                    .args(["-TERM", &outer.to_string()])
                    .status()?
                    .success()
            );
        }
        let output = bounded_output(child)?;
        assert_eq!(output.status.code(), Some(1));
        assert!(fixture.json.is_file());
        assert!(fixture.env.is_file());
        assert!(!String::from_utf8(output.stdout)?.ends_with(SHELL_LINE));
        let stderr = String::from_utf8(output.stderr)?;
        assert!(
            stderr.contains("cleanup=Cleanup { complete: true"),
            "{stderr}"
        );
        if cancel {
            assert!(stderr.contains("run-family cancelled"), "{stderr}");
        } else {
            assert!(stderr.contains("outcome=Deadline"), "{stderr}");
        }
        for name in ["reader", "descendant"] {
            let pid = u32::try_from(pids[name].as_u64().ok_or("fixture PID")?)?;
            let deadline = Instant::now();
            while live(pid)? && deadline.elapsed() < Duration::from_secs(3) {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(!live(pid)?, "owned {name} PID {pid} survived cleanup");
        }
        assert!(
            sentinel.0.try_wait()?.is_none(),
            "independent sentinel was killed"
        );
        assert!(!fixture.root.path().join("build-calls.txt").exists());
    }
    Ok(())
}

fn bounded_output(mut child: Child) -> std::io::Result<Output> {
    let started = Instant::now();
    while child.try_wait()?.is_none() {
        if started.elapsed() > Duration::from_secs(24) {
            child.kill()?;
            let _wait = child.wait();
            return Err(std::io::Error::other(
                "run-family cleanup exceeded fixture bound",
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output()
}

fn live(pid: u32) -> std::io::Result<bool> {
    let output = Command::new("/bin/ps")
        .args(["-p", &pid.to_string(), "-o", "stat="])
        .output()?;
    let state = String::from_utf8_lossy(&output.stdout);
    Ok(!state.trim().is_empty() && !state.trim().starts_with('Z'))
}
