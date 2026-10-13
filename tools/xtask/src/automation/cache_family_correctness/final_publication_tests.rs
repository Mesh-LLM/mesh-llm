//! Isolated actual signal proof for the production final publication owner.
use super::*;
#[cfg(unix)]
#[test]
fn correctness_final_actual_signal_and_terminal_receipt_custody() {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    for mode in [
        "term",
        "int",
        "success",
        "prior",
        "deadline",
        "late-deadline",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let report = process::supervise(&ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            arguments: ["--ignored", "--exact", "automation::cache_family_correctness::final_publication::tests::correctness_final_signal_worker", "--nocapture"].map(|s| Arg::Public(s.into())).to_vec(),
            cwd: root.clone(), environment: [("CACHE_FINAL_ROOT".into(), Arg::Public(root.clone().into_os_string())), ("CACHE_FINAL_MODE".into(), Arg::Public(mode.into()))].into_iter().collect(),
        }, &Limits { execution: std::time::Duration::from_secs(10), graceful_shutdown: std::time::Duration::from_secs(1), forced_shutdown: std::time::Duration::from_secs(1), retained_bytes_per_stream:65536, readiness:Readiness::None, completion:Completion::Exit }, &Cancellation::default(), OutputFiles::default()).unwrap();
        assert!(
            report.success()
                && report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && report.cleanup.failure.is_none()
                && !report.cleanup.graceful_signal_failed
        );
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            [&report.stdout, &report.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        let receipt: Value =
            serde_json::from_slice(&std::fs::read(root.join("receipt.json")).unwrap()).unwrap();
        assert_eq!(receipt["rows"][0]["observed"], 17);
        assert_eq!(receipt["reason"], "prior diagnostic");
        assert_eq!(receipt["status"] == "completed", mode == "success");
    }
}
#[cfg(unix)]
#[test]
#[ignore = "isolated actual signal after final owned file write"]
fn correctness_final_signal_worker() {
    let root = std::path::PathBuf::from(std::env::var_os("CACHE_FINAL_ROOT").unwrap());
    let mode = std::env::var("CACHE_FINAL_MODE").unwrap();
    let interrupt = crate::automation::command_interrupt::Interrupt::install().unwrap();
    let mut observed = false;
    let deadline = std::time::Instant::now()
        + if mode == "late-deadline" {
            std::time::Duration::from_secs(1)
        } else {
            std::time::Duration::from_secs(5)
        };
    let hook = |seen: &mut bool| {
        *seen = true;
        if mode == "late-deadline" {
            while std::time::Instant::now() < deadline {
                std::thread::yield_now();
            }
        }
        if mode == "term" || mode == "int" {
            let signal = if mode == "term" {
                libc::SIGTERM
            } else {
                libc::SIGINT
            };
            assert_eq!(unsafe { libc::raise(signal) }, 0);
        }
    };
    let mut receipt = serde_json::json!({"status":if mode=="prior" {"failed"} else {"completed"}, "reason":"prior diagnostic", "rows":[{"observed":17}]});
    let result = finish(
        &mut receipt,
        &root.join("receipt.json"),
        if mode == "deadline" {
            std::time::Instant::now()
        } else {
            deadline
        },
        interrupt,
        &mut |_| {
            hook(&mut observed);
            Ok(())
        },
    );
    assert!(observed);
    assert_eq!(result.is_ok(), mode == "success" || mode == "prior");
}
