use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

const CHILD: &str = "probe::tests::exiting_probe_child";

#[test]
fn exiting_probe_child() {
    if std::env::var_os("SKIPPY_TEST_EXITING_PROBE").is_some() {
        print!("probe-ran");
    }
}

/// #3067: a probe tool that exits by itself must be reaped and return its
/// output. On Windows the job-object wait used to block forever here, which
/// hung every hardware survey.
#[test]
fn probe_that_exits_on_its_own_returns_its_output() {
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = super::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", CHILD, "--nocapture"])
                .env("SKIPPY_TEST_EXITING_PROBE", "1"),
        );
        let _ = done_tx.send(result);
    });
    let output = done_rx
        .recv_timeout(Duration::from_secs(60))
        .expect("a probe that exits on its own must return")
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("probe-ran"),
        "{output:?}"
    );
}
