//! Actual parent CLI with existing inert bootstrap/native tool and private inert HF helper peers.
use crate::process::{self, Cancellation, Outcome};
use serde_json::{Value as Json, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
pub(super) fn hash(b: &[u8]) -> String {
    use sha2::Digest as _;
    hex::encode(sha2::Sha256::digest(b))
}
pub(super) fn pin(p: &Path) -> Json {
    json!({"path":p,"sha256":hash(&std::fs::read(p).unwrap())})
}
pub(super) fn fixture(
    native_mode: &str,
    publisher_mode: &str,
) -> (tempfile::TempDir, PathBuf, PathBuf, Vec<PathBuf>) {
    let (root, input, output, _) = super::hf_job_worker_cli::fixture("ok");
    let base = root.path().canonicalize().unwrap();
    let original: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    let source = base.join("source");
    std::fs::create_dir(&source).unwrap();
    let bodies = std::collections::BTreeMap::from([
        (
            "config.json",
            serde_json::to_vec(&json!({"fixture_mode":native_mode})).unwrap(),
        ),
        (
            "model.safetensors",
            b"finite inert SafeTensors bytes".to_vec(),
        ),
        ("tokenizer.json", b"{}".to_vec()),
        ("tokenizer_config.json", b"{}".to_vec()),
        ("special_tokens_map.json", b"{}".to_vec()),
    ]);
    for (n, b) in &bodies {
        std::fs::write(source.join(n), b).unwrap();
    }
    let profile = base.join("profile.json");
    std::fs::write(&profile,serde_json::to_vec(&json!({"schema_version":1,"config_sha256":hash(&bodies["config.json"]),"tokenizer_sha256":hash(&bodies["tokenizer.json"]),"tokenizer_config_sha256":hash(&bodies["tokenizer_config.json"]),"chat_template_sha256":null,"pre":"qwen2"})).unwrap()).unwrap();
    let staging = super::hf_mtp_default_fixture::wrapper(&base, "staging", "ok", &source);
    let repository = super::hf_mtp_default_fixture::wrapper(&base, "repository", "ok", &source);
    let publisher =
        super::hf_mtp_default_fixture::wrapper(&base, "publisher", publisher_mode, &source);
    let credential = base.join("credential");
    std::fs::write(&credential, b"finite-fixture-not-sent").unwrap();
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(&credential, std::fs::Permissions::from_mode(0o600)).unwrap();
    let parts = (1..=3)
        .map(|i| {
            let p = base.join(format!("Target-{i:05}-of-00003.gguf"));
            std::fs::write(&p, format!("GGUFtarget-{i}")).unwrap();
            p
        })
        .collect::<Vec<_>>();
    let files = bodies
        .iter()
        .map(|(n, b)| ((*n).to_string(), hash(b)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let tokenizer = files
        .iter()
        .filter(|(n, _)| n.starts_with("tokenizer") || n.as_str() == "special_tokens_map.json")
        .map(|(n, h)| (n.clone(), h.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let tool = |p: &Path, name: &str| {
        let mut v = pin(p);
        v["name"] = json!(name);
        v
    };
    let request = json!({"schema_version":1,"bootstrap":original["bootstrap"],"staging_helper":tool(&staging,"stage"),"checkpoint":{"repo":"owner/checkpoint","revision":"a".repeat(40),"files":files},"tokenizer_source":{"repo":"owner/tokenizer","revision":"b".repeat(40),"files":tokenizer},"tokenizer_profile":pin(&profile),"credential_file":credential,"maximum_bytes":1048576,"target_parts":parts.iter().map(|p|pin(p)).collect::<Vec<_>>(),"target_basename":"Target","composite_basename":"Composite","mtp_block":88,"composite_repo":"fixture/composite","sidecars":[],"repository_helper":tool(&repository,"repository"),"repository_helper_source":pin(&repository),"publisher_helper":pin(&publisher),"publisher_source":pin(&publisher),"overall_seconds":60,"publication_reserve_seconds":15,"dry_run":false,"confirm_publication":true});
    std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    (root, input, output, parts)
}
fn invoke(input: &Path, output: &Path, cancel: &Cancellation) -> process::RawProcessReport {
    let spec = process::ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: [
            "automation".into(),
            "hf-certify".into(),
            "compose-default".into(),
            "--input".into(),
            input.as_os_str().into(),
            "--output-directory".into(),
            output.as_os_str().into(),
        ]
        .into_iter()
        .map(process::Value::Public)
        .collect(),
        cwd: input.parent().unwrap().into(),
        environment: Default::default(),
    };
    let raw = process::supervise_raw(
        &spec,
        &process::Limits {
            execution: Duration::from_secs(65),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1048576,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancel,
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(1048576),
            stderr: std::num::NonZeroUsize::new(1048576),
        },
    )
    .unwrap();
    let p = &raw.process;
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
    );
    for (s, r) in [
        (&p.stdout, raw.stdout.as_ref().unwrap()),
        (&p.stderr, raw.stderr.as_ref().unwrap()),
    ] {
        assert!(!s.truncated && s.line_capture_complete);
        assert_eq!(r.as_bytes().len() as u64, s.bytes_seen);
    }
    raw
}
fn receipt(output: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(output.join("report.json")).unwrap()).unwrap()
}
#[test]
#[cfg(target_os = "linux")]
fn actual_default_mtp_compose_cli_chains_staged_custody_native_phases_and_one_ordered_publication()
{
    let (root, input, output, parts) = fixture("ok", "ok");
    let middle = std::fs::read(&parts[1]).unwrap();
    let raw = invoke(&input, &output, &Cancellation::default());
    assert_eq!(raw.process.outcome, Outcome::Exited);
    assert_eq!(raw.process.status.unwrap().code(), Some(0));
    let r = receipt(&output);
    assert_eq!(r["status"], "COMPOSED_PUBLISHED");
    assert_eq!(
        r["staging_receipt"]["checkpoint_files"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
    assert_eq!(r["native_composition"]["source_unchanged"], true);
    assert_eq!(
        r["native_composition"]["native_conversion_verification"]["complete"],
        true
    );
    assert_eq!(
        r["repository_receipt"]["repository"]["observed_parent"],
        "a".repeat(40)
    );
    let p = &r["ordered_publication"]["final_receipt"]["publication"];
    assert_eq!(
        p["ordered_paths"],
        json!([
            "Composite-00001-of-00003.gguf",
            "Composite-00002-of-00003.gguf",
            "Composite-00003-of-00003.gguf",
            "README.md"
        ])
    );
    assert_eq!(p["remote_verified_paths"], p["ordered_paths"]);
    assert_eq!(p["completed"], true);
    assert_eq!(std::fs::read(&parts[1]).unwrap(), middle);
    assert_eq!(r["real_family_qualified"], false);
    root.close().unwrap();
}
#[test]
#[cfg(target_os = "linux")]
fn actual_default_compose_late_native_or_publication_failure_retains_prior_observations() {
    for (native, publisher) in [("bad-verify", "ok"), ("ok", "fail")] {
        let (root, input, output, _) = fixture(native, publisher);
        let raw = invoke(&input, &output, &Cancellation::default());
        assert_eq!(raw.process.outcome, Outcome::Exited);
        assert_eq!(raw.process.status.unwrap().code(), Some(1));
        let r = receipt(&output);
        assert_eq!(r["status"], "FAILED");
        assert!(!r["staging_receipt"].is_null());
        assert!(!r["observed_bootstrap"].is_null());
        if publisher == "fail" {
            assert_eq!(r["native_composition"]["source_unchanged"], true);
            assert_eq!(
                r["ordered_publication"]["final_receipt"]["status"],
                "FAILED"
            );
        } else {
            assert!(!output.join("publisher").exists());
        }
        root.close().unwrap();
    }
}
#[test]
#[cfg(target_os = "linux")]
fn actual_default_compose_cancel_during_publication_keeps_native_rows_and_partial_mutation() {
    let (root, input, output, _) = fixture("ok", "held");
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let child_output = output.clone();
    let thread = std::thread::spawn(move || invoke(&input, &child_output, &child_cancel));
    let marker = output.join("publisher/publication/helper-output/publication-held");
    let until = Instant::now() + Duration::from_secs(30);
    while !marker.exists() && Instant::now() < until {
        std::thread::park_timeout(Duration::from_millis(5));
    }
    let seen = marker.exists();
    cancel.cancel();
    let raw = thread.join().unwrap();
    assert!(seen && cancel.is_cancelled());
    assert_eq!(raw.process.outcome, Outcome::Cancelled);
    let r = receipt(&output);
    assert_eq!(r["status"], "FAILED");
    assert_eq!(r["native_composition"]["source_unchanged"], true);
    assert_eq!(
        r["ordered_publication"]["partial_progress"]["publication"]["commit_attempted"],
        true
    );
    assert!(r["ordered_publication"]["final_receipt"].is_null());
    root.close().unwrap();
}
#[test]
fn actual_default_compose_dry_run_and_writerless_input_require_no_child_or_network() {
    let (root, input, output, _) = fixture("ok", "ok");
    let mut value: Json = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    value["dry_run"] = json!(true);
    value["confirm_publication"] = json!(false);
    std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
    let raw = invoke(&input, &output, &Cancellation::default());
    assert_eq!(raw.process.status.unwrap().code(), Some(0));
    assert_eq!(receipt(&output)["status"], "DRY_RUN_NOT_EXECUTED");
    assert!(!root.path().join("staging-arguments").exists());
    assert!(!output.join("bootstrap").exists());
    let fifo = root.path().join("writerless");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap(); // SAFETY: exclusively owned fixture path.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let fresh = root.path().join("refused");
    let raw = invoke(&fifo, &fresh, &Cancellation::default());
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert!(!fresh.exists());
    value["dry_run"] = json!(false);
    std::fs::write(&input, serde_json::to_vec(&value).unwrap()).unwrap();
    let raw = invoke(&input, &fresh, &Cancellation::default());
    assert_eq!(raw.process.status.unwrap().code(), Some(1));
    assert!(!fresh.exists());
    root.close().unwrap();
}
