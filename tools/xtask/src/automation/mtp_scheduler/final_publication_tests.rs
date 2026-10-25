//! Isolated helpers exercise the actual retained scheduler publication path.
use super::*;
fn invoke(mode: &str, kind: &str) {
    use crate::process::{
        self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let report = process::supervise(&ProcessSpec {
        executable: std::env::current_exe().unwrap(),
        arguments: ["--ignored", "--exact", "automation::mtp_scheduler::final_publication::tests::mtp_scheduler_final_publication_worker", "--nocapture"].map(|s| Arg::Public(s.into())).to_vec(),
        cwd: root.clone(), environment: [("MTP_FINAL_ROOT".into(), Arg::Public(root.clone().into_os_string())), ("MTP_FINAL_MODE".into(), Arg::Public(mode.into())), ("MTP_FINAL_KIND".into(), Arg::Public(kind.into()))].into_iter().collect(),
    }, &Limits { execution: std::time::Duration::from_secs(10), graceful_shutdown: std::time::Duration::from_secs(1), forced_shutdown: std::time::Duration::from_secs(1), retained_bytes_per_stream:65536, readiness:Readiness::None, completion:Completion::Exit }, &Cancellation::default(), OutputFiles::default()).unwrap();
    assert!(report.success(), "{mode}/{kind}: {report:?}");
    assert!(
        report.cleanup.complete && !report.cleanup.forced && !report.cleanup.graceful_signal_failed
    );
    assert!(report.stdout.line_capture_complete && report.stderr.line_capture_complete);
    let observed: Value =
        serde_json::from_slice(&std::fs::read(root.join(format!("{kind}.json"))).unwrap()).unwrap();
    if mode == "foreign" {
        assert_eq!(observed, serde_json::json!({"foreign":true}));
        return;
    }
    assert_eq!(observed["rows"], serde_json::json!([{"observed":true}]));
    assert_eq!(
        observed["status"],
        if mode == "success" {
            "completed"
        } else {
            "failed"
        }
    );
    assert_eq!(observed["orchestration_complete"], mode == "success");
    if mode == "prior" {
        assert_eq!(observed["failure"], "earlier measurement failure");
    }
}
#[test]
fn mtp_scheduler_final_publication_preserves_rows_and_refuses_late_terminal_states() {
    for kind in ["comparison", "old-result"] {
        for mode in [
            "success",
            "prior",
            "cancel",
            "deadline",
            "late-deadline",
            "foreign",
        ] {
            invoke(mode, kind);
        }
    }
}
#[cfg(unix)]
#[test]
fn mtp_scheduler_final_publication_actual_signals_downgrade_owned_files() {
    for kind in ["comparison", "old-result"] {
        for mode in ["term", "int"] {
            invoke(mode, kind);
        }
    }
}
#[test]
#[ignore = "isolated helper selected by owning production publication tests"]
fn mtp_scheduler_final_publication_worker() {
    let root = PathBuf::from(std::env::var_os("MTP_FINAL_ROOT").unwrap());
    let mode = std::env::var("MTP_FINAL_MODE").unwrap();
    let kind = std::env::var("MTP_FINAL_KIND").unwrap();
    let path = root.join(format!("{kind}.json"));
    if mode == "foreign" {
        std::fs::write(&path, br#"{"foreign":true}"#).unwrap();
        assert!(ReceiptFile::new(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), br#"{"foreign":true}"#);
        return;
    }
    let interrupt = Interrupt::install().unwrap();
    let cancel = interrupt.cancellation();
    let deadline = Instant::now()
        + std::time::Duration::from_secs(if mode == "late-deadline" { 1 } else { 5 });
    let deadline = if mode == "deadline" {
        Instant::now()
    } else {
        deadline
    };
    let mut receipt = serde_json::json!({"status":"measuring","rows":[{"observed":true}]});
    let mut file = ReceiptFile::new(&path).unwrap();
    file.write(&receipt).unwrap();
    let result: DynResult<()> = if mode == "prior" {
        Err("earlier measurement failure".into())
    } else {
        Ok(())
    };
    let mut emitted = false;
    let finalized = file.finish(&mut receipt, result, interrupt, deadline, &mut |value| {
        assert_eq!(value["status"], "completed");
        let durable: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(durable["status"], "completed");
        emitted = true;
        match mode.as_str() {
            "cancel" => cancel.cancel(),
            "late-deadline" => {
                while Instant::now() < deadline {
                    std::thread::yield_now();
                }
            }
            #[cfg(unix)]
            "term" | "int" => {
                let signal = if mode == "term" {
                    libc::SIGTERM
                } else {
                    libc::SIGINT
                };
                assert_eq!(unsafe { libc::raise(signal) }, 0);
                assert!(cancel.is_cancelled());
            }
            _ => {}
        }
        Ok(())
    });
    assert_eq!(finalized.is_ok(), mode == "success");
    assert_eq!(emitted, mode != "prior" && mode != "deadline");
}
