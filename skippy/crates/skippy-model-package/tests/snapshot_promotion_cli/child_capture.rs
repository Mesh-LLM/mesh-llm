//! Drain fixture output while the owned child runs, keeping its existing deadline.

use std::{
    io::{Read, Write},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

const STREAM_LIMIT: usize = 2 * 1024 * 1024;
const FIXTURE_MODE: &str = "SKIPPY_SNAPSHOT_CAPTURE_FIXTURE";

struct CapturedChild {
    output: Output,
    timed_out: bool,
    output_exceeded: bool,
}

pub(super) fn finish(child: Child, timeout: Duration, label: &str) -> Output {
    let captured = capture(child, timeout);
    assert!(
        !captured.timed_out,
        "{label} timed out: {}",
        captured.output.status
    );
    assert!(
        !captured.output_exceeded,
        "{label} exceeded fixture output limit"
    );
    captured.output
}

fn drain(mut stream: impl Read) -> (Vec<u8>, bool) {
    let mut bytes = Vec::new();
    stream
        .by_ref()
        .take((STREAM_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)
        .expect("read fixture output");
    let exceeded = bytes.len() > STREAM_LIMIT;
    bytes.truncate(STREAM_LIMIT);
    // Continue draining excessive output rather than blocking the child after
    // filling the bounded retained buffer. The caller still rejects overflow.
    std::io::copy(&mut stream, &mut std::io::sink()).expect("drain fixture output");
    (bytes, exceeded)
}

fn capture(mut child: Child, timeout: Duration) -> CapturedChild {
    let stdout = child.stdout.take().expect("fixture stdout is piped");
    let stderr = child.stderr.take().expect("fixture stderr is piped");
    thread::scope(|scope| {
        let stdout = scope.spawn(|| drain(stdout));
        let stderr = scope.spawn(|| drain(stderr));
        let deadline = Instant::now() + timeout;
        let (status, timed_out) = loop {
            if let Some(status) = child.try_wait().expect("poll fixture child") {
                break (status, false);
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                break (child.wait().expect("reap timed-out fixture child"), true);
            }
            thread::sleep(Duration::from_millis(5));
        };
        let (stdout, stdout_exceeded) = stdout.join().expect("stdout reader panicked");
        let (stderr, stderr_exceeded) = stderr.join().expect("stderr reader panicked");
        CapturedChild {
            output: Output {
                status,
                stdout,
                stderr,
            },
            timed_out,
            output_exceeded: stdout_exceeded || stderr_exceeded,
        }
    })
}

fn fixture(mode: &str) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .env_clear()
        .env(FIXTURE_MODE, mode)
        .args([
            "--exact",
            "child_capture::fixture_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

#[test]
fn fixture_child() {
    match std::env::var(FIXTURE_MODE).as_deref() {
        Ok("large") => {
            std::io::stdout()
                .write_all(&vec![b'o'; 1024 * 1024])
                .unwrap();
            std::io::stderr()
                .write_all(&vec![b'e'; 1024 * 1024])
                .unwrap();
        }
        Ok("wait") => thread::sleep(Duration::from_secs(60)),
        _ => {}
    }
}

#[test]
fn drains_both_streams_before_waiting_for_exit() {
    let output = finish(fixture("large"), Duration::from_secs(10), "large fixture");
    assert!(output.status.success());
    assert!(
        output
            .stdout
            .windows(1024 * 1024)
            .any(|bytes| bytes.iter().all(|byte| *byte == b'o'))
    );
    assert_eq!(output.stderr, vec![b'e'; 1024 * 1024]);
}

#[test]
fn deadline_still_kills_and_reaps_the_owned_child() {
    let started = Instant::now();
    let captured = capture(fixture("wait"), Duration::from_millis(100));
    assert!(captured.timed_out);
    assert!(!captured.output.status.success());
    assert!(started.elapsed() < Duration::from_secs(5));
}
