//! Isolated actual signal proof for the production final publication owner.
use super::*;
#[cfg(unix)]
#[test]
fn preparation_final_actual_signal_and_terminal_receipt_custody() {
    for mode in [
        "term",
        "int",
        "success",
        "prior",
        "deadline",
        "late-deadline",
    ] {
        assert_publication_mode(mode);
    }
}
pub(in crate::automation::cache_family_run::preparation) fn assert_publication_mode(mode: &str) {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let report = process::supervise(&ProcessSpec {
        executable: std::env::current_exe().unwrap(),
        arguments: ["--ignored", "--exact", "automation::cache_family_run::preparation::final_publication::tests::preparation_final_signal_worker", "--nocapture"].map(|s| Arg::Public(s.into())).to_vec(),
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
    assert_eq!(
        root.join("candidate.json").exists(),
        ["success", "prior", "foreign"].contains(&mode)
    );
    if mode == "foreign" {
        assert_eq!(
            std::fs::read(root.join("candidate.json")).unwrap(),
            b"foreign input"
        );
    }
    if mode == "success" || mode == "prior" {
        assert_eq!(
            std::fs::read(root.join("candidate.json")).unwrap(),
            b"observed candidate"
        );
    }
}
#[test]
#[ignore = "isolated actual signal after final owned file write"]
fn preparation_final_signal_worker() {
    let root = std::path::PathBuf::from(std::env::var_os("CACHE_FINAL_ROOT").unwrap());
    let mode = std::env::var("CACHE_FINAL_MODE").unwrap();
    let interrupt = crate::automation::command_interrupt::Interrupt::install().unwrap();
    let cancellation = interrupt.cancellation();
    if mode == "pre-cancel" {
        cancellation.cancel();
    }
    if mode == "foreign" {
        std::fs::write(root.join("candidate.json"), b"foreign input").unwrap();
    }
    let mut observed = false;
    let deadline = std::time::Instant::now()
        + if mode == "late-deadline" {
            std::time::Duration::from_secs(1)
        } else {
            std::time::Duration::from_secs(5)
        };
    let hook = |seen: &mut bool| {
        *seen = true;
        if mode == "cancel" {
            cancellation.cancel();
        }
        if mode == "late-deadline" {
            while std::time::Instant::now() < deadline {
                std::thread::yield_now();
            }
        }
        #[cfg(unix)]
        if mode == "term" || mode == "int" {
            let signal = if mode == "term" {
                libc::SIGTERM
            } else {
                libc::SIGINT
            };
            assert_eq!(unsafe { libc::raise(signal) }, 0);
        }
    };
    let result = finish(
        &root.join("candidate.json"),
        b"observed candidate",
        if mode == "deadline" {
            std::time::Instant::now()
        } else {
            deadline
        },
        interrupt,
        &mut || {
            hook(&mut observed);
            Ok(())
        },
    );
    assert_eq!(result.is_ok(), mode == "success" || mode == "prior");
    assert_eq!(
        observed,
        !["pre-cancel", "deadline", "foreign"].contains(&mode.as_str())
    );
}
