//! Real handler lifetime and retained-file refusal on the production publication path.
use super::*;
#[cfg(unix)]
#[test]
fn quant_delivery_final_actual_signal_deadline_and_foreign_receipt_custody() {
    use crate::process::{
        self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    for mode in [
        "success",
        "prior",
        "term-write",
        "int-emit",
        "deadline",
        "foreign",
        "emit-refusal",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let report = process::supervise(&ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            arguments: ["--ignored", "--exact", "automation::hf_certify::quant_job_delivery::final_publication::tests::quant_delivery_final_signal_worker", "--nocapture"].map(|s| Arg::Public(s.into())).to_vec(),
            cwd: root.clone(), environment: [("QUANT_FINAL_ROOT".into(), Arg::Public(root.clone().into_os_string())), ("QUANT_FINAL_MODE".into(), Arg::Public(mode.into()))].into_iter().collect(),
        }, &Limits { execution: Duration::from_secs(10), graceful_shutdown: Duration::from_secs(1), forced_shutdown: Duration::from_secs(1), retained_bytes_per_stream:65536, readiness:Readiness::None, completion:Completion::Exit }, &Cancellation::default(), OutputFiles::default()).unwrap();
        assert!(
            report.success()
                && report.failure.is_none()
                && report.cleanup.complete
                && report.cleanup.failure.is_none(),
            "mode={mode}; report={report:?}"
        );
        assert!(
            [&report.stdout, &report.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        if mode == "foreign" {
            assert_eq!(
                std::fs::read(root.join("native-job-delivery.json")).unwrap(),
                b"foreign"
            );
        } else {
            let delivery: Value = serde_json::from_slice(
                &std::fs::read(root.join("native-job-delivery.json")).unwrap(),
            )
            .unwrap();
            let native: Value =
                serde_json::from_slice(&std::fs::read(root.join("native-job.json")).unwrap())
                    .unwrap();
            assert_eq!(delivery["native_completed"], mode == "success");
            assert_eq!(delivery["locator"]["delivery_complete"], mode == "success");
            assert_eq!(native["operator"]["windows"][0]["remote_verified"], true);
            if mode != "success" {
                assert_eq!(native["status"], "FAILED");
                assert_eq!(native["operator"]["completed_job"], false);
            }
            if mode == "prior" {
                assert_eq!(native["error"], "earlier window failure");
            }
        }
    }
}
#[cfg(unix)]
#[test]
#[ignore = "isolated real signal during the production quant delivery publication owner"]
fn quant_delivery_final_signal_worker() {
    let root = std::path::PathBuf::from(std::env::var_os("QUANT_FINAL_ROOT").unwrap());
    let mode = std::env::var("QUANT_FINAL_MODE").unwrap();
    let interrupt = Interrupt::install().unwrap();
    let until = Instant::now()
        + if mode == "deadline" {
            Duration::from_millis(100)
        } else {
            Duration::from_secs(5)
        };
    let mut native = json!({"status":"QUANTIZATION_PUBLISHED","error":if mode=="prior"{json!("earlier window failure")}else{Value::Null},"operator":{"completed_job":true,"windows":[{"remote_verified":true}]}});
    let mut native_file = OwnedReceipt::new(root.join("native-job.json"));
    native_file.write(&native).unwrap();
    let mut delivery = json!({"locator":{"delivery_complete":false},"export":{"completed":true}});
    let saved_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
    assert!(saved_stdout >= 0);
    let mut observed = false;
    let result = finish_owned(
        &root,
        Observations {
            native: &mut native,
            native_file: &mut native_file,
            delivery: &mut delivery,
        },
        mode != "prior",
        until,
        interrupt,
        |stage| {
            observed = true;
            if stage == "write" && mode == "emit-refusal" {
                broken_stdout();
            }
            if stage == "write" && mode == "deadline" {
                while Instant::now() < until {
                    std::thread::yield_now();
                }
            }
            if stage == "write" && mode == "foreign" {
                std::fs::remove_file(root.join("native-job-delivery.json"))?;
                std::fs::write(root.join("native-job-delivery.json"), b"foreign")?;
                return Err("foreign path injected".into());
            }
            let signal = match (stage, mode.as_str()) {
                ("write", "term-write") => Some(libc::SIGTERM),
                ("emit", "int-emit") => Some(libc::SIGINT),
                _ => None,
            };
            if let Some(signal) = signal {
                assert_eq!(unsafe { libc::raise(signal) }, 0);
            }
            Ok(())
        },
    );
    assert_eq!(
        unsafe { libc::dup2(saved_stdout, libc::STDOUT_FILENO) },
        libc::STDOUT_FILENO
    );
    assert_eq!(unsafe { libc::close(saved_stdout) }, 0);
    assert!(observed);
    assert_eq!(result.is_ok(), mode == "success");
}

#[cfg(unix)]
fn broken_stdout() {
    let mut current: libc::sigaction = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut current) },
        0
    );
    assert_eq!(
        current.sa_sigaction,
        libc::SIG_IGN,
        "Rust runtime must ignore SIGPIPE for this isolated write-refusal case"
    );
    let mut owned = [-1; 2];
    assert_eq!(unsafe { libc::pipe(owned.as_mut_ptr()) }, 0);
    assert_eq!(unsafe { libc::close(owned[0]) }, 0);
    assert_eq!(
        unsafe { libc::dup2(owned[1], libc::STDOUT_FILENO) },
        libc::STDOUT_FILENO
    );
    assert_eq!(unsafe { libc::close(owned[1]) }, 0);
}
