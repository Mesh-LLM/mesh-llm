use super::{fixture, workspace};
use std::io::{BufRead, BufReader};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct OwnedScenario {
    command: Child,
    sentinel: Child,
    listener: Option<UnixListener>,
    lease: Option<UnixStream>,
}

impl Drop for OwnedScenario {
    fn drop(&mut self) {
        self.lease.take();
        self.listener.take();
        if self
            .command
            .try_wait()
            .expect("owned command status")
            .is_none()
        {
            let output = Command::new("/bin/kill")
                .args(["-TERM", "--", &self.command.id().to_string()])
                .output()
                .expect("owned command cancellation");
            assert!(output.status.success(), "{output:?}");
            let deadline = Instant::now() + Duration::from_secs(10);
            while self
                .command
                .try_wait()
                .expect("owned command status")
                .is_none()
            {
                assert!(Instant::now() < deadline, "owned cancellation deadline");
                std::thread::park_timeout(Duration::from_millis(10));
            }
        }
        for child in [&mut self.command, &mut self.sentinel] {
            if child.try_wait().expect("owned child status").is_none() {
                child.kill().expect("owned child cleanup");
            }
            child.wait().expect("owned child reap");
        }
    }
}

#[test]
fn foreground_sigint_cleans_owned_tree_when_unrelated_sentinel_is_running() {
    interrupted("-INT", true);
}

#[test]
fn directed_sigterm_cleans_owned_tree_when_unrelated_sentinel_is_running() {
    interrupted("-TERM", false);
}

fn interrupted(signal: &str, foreground_group: bool) {
    let root = workspace("signal-tree");
    let mut scenario = launch(root.path());
    let (leader, descendant) = ready_tree(&mut scenario);
    assert!(alive(&leader));
    assert!(alive(&descendant));
    assert!(scenario.sentinel.try_wait().unwrap().is_none());
    let target = if foreground_group {
        format!("-{}", scenario.command.id())
    } else {
        scenario.command.id().to_string()
    };

    let delivered = Command::new("/bin/kill")
        .args([signal, "--", &target])
        .output()
        .unwrap();

    assert!(delivered.status.success(), "{delivered:?}");
    let cleanup_deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = scenario.command.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < cleanup_deadline,
            "interruption cleanup deadline"
        );
        std::thread::park_timeout(Duration::from_millis(10));
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    std::io::Read::read_to_end(&mut scenario.command.stdout.take().unwrap(), &mut stdout).unwrap();
    std::io::Read::read_to_end(&mut scenario.command.stderr.take().unwrap(), &mut stderr).unwrap();
    assert_eq!(status.code(), Some(1), "{signal}: {status:?}");
    assert!(
        stdout.is_empty(),
        "interrupted CLI emitted translation JSON"
    );
    assert!(
        String::from_utf8_lossy(&stderr).contains("Cancelled"),
        "{stderr:?}"
    );
    assert!(!alive(&leader), "owned metadata leader survived");
    assert!(!alive(&descendant), "owned metadata descendant survived");
    assert!(
        scenario.sentinel.try_wait().unwrap().is_none(),
        "unrelated sentinel was killed"
    );
}

fn launch(root: &std::path::Path) -> OwnedScenario {
    let socket = UnixListener::bind(root.join("ready.sock")).unwrap();
    socket.set_nonblocking(true).unwrap();
    let sentinel = Command::new(fixture())
        .arg("hold")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    OwnedScenario {
        command: Command::new(env!("CARGO_BIN_EXE_xtask"))
            .arg("--repo-root")
            .arg(root)
            .args([
                "repository",
                "cargo-packages",
                "--generation",
                "legacy",
                "--crates",
                "[\"model-hf\"]",
                "--cargo",
            ])
            .arg(fixture())
            .args(["--timeout", "120"])
            .current_dir(std::env::temp_dir())
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
        sentinel,
        listener: Some(socket),
        lease: None,
    }
}

#[test]
fn owned_tree_stops_when_failure_is_injected_before_pid_handshake() {
    let root = workspace("signal-tree");
    let mut scenario = launch(root.path());
    let deadline = Instant::now() + Duration::from_secs(10);
    while !root.path().join("leader.pid").exists() {
        assert!(Instant::now() < deadline, "fixture launch deadline");
        std::thread::park_timeout(Duration::from_millis(10));
    }
    let leader = std::fs::read_to_string(root.path().join("leader.pid")).unwrap();
    let descendant = std::fs::read_to_string(root.path().join("descendant.pid")).unwrap();
    let sentinel = scenario.sentinel.id().to_string();

    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scenario.command.kill().unwrap();
        scenario.command.wait().unwrap();
        let _owned = scenario;
        panic!("injected failure before accept or PID read");
    }));

    assert!(failure.is_err());
    while alive(&leader) || alive(&descendant) {
        assert!(
            Instant::now() < deadline,
            "pre-handshake tree cleanup deadline"
        );
        std::thread::park_timeout(Duration::from_millis(10));
    }
    assert!(!alive(&sentinel));
}

fn ready_tree(scenario: &mut OwnedScenario) -> (String, String) {
    let ready_deadline = Instant::now() + Duration::from_secs(10);
    let stream = loop {
        match scenario.listener.as_ref().unwrap().accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    Instant::now() < ready_deadline,
                    "fixture readiness deadline"
                );
                assert!(
                    scenario.command.try_wait().unwrap().is_none(),
                    "CLI exited before ready"
                );
                std::thread::park_timeout(Duration::from_millis(10));
            }
            Err(error) => panic!("fixture readiness: {error}"),
        }
    };
    scenario.lease = Some(stream.try_clone().unwrap());
    stream.set_nonblocking(true).unwrap();
    let mut ready = BufReader::new(stream);
    let mut leader = String::new();
    read_pid(&mut ready, &mut leader);
    let leader = leader.trim().parse::<u32>().unwrap().to_string();
    let mut descendant = String::new();
    read_pid(&mut ready, &mut descendant);
    let descendant = descendant.trim().parse::<u32>().unwrap().to_string();
    (leader, descendant)
}

fn read_pid(ready: &mut BufReader<std::os::unix::net::UnixStream>, pid: &mut String) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match ready.read_line(pid) {
            Ok(_) if pid.ends_with('\n') => break,
            Ok(_) => panic!("fixture PID reached EOF"),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(pid.len() <= 11, "fixture PID exceeded bound");
                assert!(Instant::now() < deadline, "fixture PID deadline");
                std::thread::park_timeout(Duration::from_millis(10));
            }
            Err(error) => panic!("fixture PID: {error}"),
        }
    }
}

fn alive(pid: &str) -> bool {
    Command::new("/bin/kill")
        .args(["-0", pid])
        .output()
        .unwrap()
        .status
        .success()
}
