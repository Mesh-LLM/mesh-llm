//! Focused owning assertions for frozen transcript, default argv, provenance and typed observation.
use super::{kv_command, kv_identity, kv_manifest, kv_metadata, kv_owner::Owner, kv_report};
use crate::process::{
    self,
    retained::{Coordinator, ExpectedExit, MemberId},
};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
#[test]
fn restart_manifest_is_deterministic_with_canned_assistant_and_user_final_prefix() {
    let a = kv_manifest::build(2, 32, 16).unwrap();
    let b = kv_manifest::build(2, 32, 16).unwrap();
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    let value = serde_json::to_value(&a).unwrap();
    assert_ne!(value["turns"][0]["request"], value["turns"][0]["response"]);
    let first = a.messages(0).unwrap();
    let last = a.messages(1).unwrap();
    assert_eq!(&last[..2], first);
    assert_eq!(last[2]["role"], "assistant");
    assert_eq!(last[2]["content"], value["turns"][0]["response"]);
    assert_eq!(last[3]["role"], "user");
    assert!(a.messages(2).is_err());
    let mut corrupt = value;
    corrupt["turns"][0]["response"] = json!("wrong");
    let bad: kv_manifest::Manifest = serde_json::from_value(corrupt).unwrap();
    assert!(bad.validate().is_err());
    assert!(kv_manifest::build(128, 65536, 65536).is_err());
}
#[test]
fn restart_default_argv_refuses_endpoint_tuning_and_duplicate_identity() {
    let args = kv_command::server_args(Path::new("/model.gguf"), &[]).unwrap();
    assert_eq!(args.len(), 5);
    let cache = kv_command::server_args(
        Path::new("/model.gguf"),
        &[
            "--kv-cache-disk".into(),
            "32GiB".into(),
            "--kv-cache-min-free=16GiB".into(),
        ],
    )
    .unwrap();
    assert_eq!(cache.len(), 8);
    for flag in [
        "--port=9447",
        "--host",
        "--ctx-size=8192",
        "--parallel",
        "--generation-concurrency=2",
        "--generation-queue-capacity",
        "--max-vram=1",
        "--model=other",
        "--log-format=text",
        "--bind-ip=0.0.0.0",
        "--bind-port=9999",
        "--listen-all",
        "--config=other",
        "--kv-cache-disk-dir=other",
        "--gguf=other",
        "--relay-auth=https://relay.invalid=dummy-secret",
        "--join=invite-secret",
        "--join-file=/invite",
        "-j",
        "-jinvite-secret",
    ] {
        assert!(
            kv_command::server_args(Path::new("/model.gguf"), &[flag.into()]).is_err(),
            "{flag}"
        );
    }
}
#[test]
fn restart_singleton_percentile_and_failed_rows_are_honestly_unavailable() {
    let rows = vec![
        json!({"cohort":"restore","ttft_seconds":0.5,"total_seconds":1.0,"prompt_tokens":10,"cached_tokens":0,"error":null}),
        json!({"cohort":"warm","error":"failed"}),
    ];
    let one = kv_report::cohort("restore", &rows);
    assert!(one["ttft_p95_seconds"].is_null());
    assert_eq!(one["ttft_p50_seconds"], 0.5);
    let empty = kv_report::cohort("warm", &rows);
    assert_eq!(empty["failed"], 1);
    assert!(empty["ttft_p50_seconds"].is_null() && empty["cache_pct"].is_null());
}
#[test]
fn restart_linux_memory_requires_unique_real_unit_and_checked_integer() {
    assert_eq!(
        kv_identity::linux_memory(b"MemTotal: 16777216 kB\n"),
        Some(17179869184)
    );
    for bytes in [
        b"MemTotal: 1 MB\n".as_slice(),
        b"MemTotal: 1 kB\nMemTotal: 2 kB\n",
        b"MemTotal: 18446744073709551615 kB\n",
    ] {
        assert_eq!(kv_identity::linux_memory(bytes), None);
    }
}
#[test]
fn restart_readiness_projection_refuses_wrong_member_stream_hash_and_duplicate() {
    let mut owner = Owner {
        server: None,
        worker: None,
        policy: ExpectedExit::new(&[0, 1], Duration::from_secs(2)).unwrap(),
        stopped: false,
        started: Instant::now(),
        request_sha256: "a".repeat(64),
        ready_observed: None,
        ready_ambiguous: false,
    };
    let bytes = serde_json::to_vec(
        &json!({"event":"kv_restart_model_ready","request_sha256":"a".repeat(64)}),
    )
    .unwrap();
    for (member, stream) in [
        (MemberId::Seed, process::Stream::Stdout),
        (MemberId::WorkerOne, process::Stream::Stderr),
    ] {
        owner.captured_line(
            member,
            process::ObservedLine {
                stream,
                bytes: &bytes,
                ending: process::LineEnding::Lf,
            },
        );
    }
    assert!(owner.ready_observed.is_none());
    let wrong = serde_json::to_vec(
        &json!({"event":"kv_restart_model_ready","request_sha256":"b".repeat(64)}),
    )
    .unwrap();
    owner.captured_line(
        MemberId::WorkerOne,
        process::ObservedLine {
            stream: process::Stream::Stdout,
            bytes: &wrong,
            ending: process::LineEnding::Lf,
        },
    );
    assert!(owner.ready_ambiguous && owner.ready_observed.is_none());
    owner.ready_ambiguous = false;
    owner.captured_line(
        MemberId::WorkerOne,
        process::ObservedLine {
            stream: process::Stream::Stdout,
            bytes: &bytes,
            ending: process::LineEnding::Lf,
        },
    );
    assert!(owner.ready_observed.is_some());
    owner.captured_line(
        MemberId::WorkerOne,
        process::ObservedLine {
            stream: process::Stream::Stdout,
            bytes: &bytes,
            ending: process::LineEnding::Lf,
        },
    );
    assert!(owner.ready_ambiguous);
}
#[cfg(unix)]
fn executable(path: &Path, script: &str) {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
#[test]
#[cfg(unix)]
fn restart_sysctl_owned_probe_uses_actual_memsize_and_refuses_invalid_or_suppressed_values() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("sysctl");
    executable(
        &path,
        "#!/bin/sh\n[ \"$1\" = -n ] || exit 64\ncase \"$2\" in machdep.cpu.brand_string) echo 'Apple M2';; hw.model) echo 'Mac14,6';; hw.memsize) echo 17179869184;; *) exit 64;; esac\n",
    );
    let until = Instant::now() + Duration::from_secs(5);
    let value =
        kv_metadata::darwin(&path, root.path(), until, &process::Cancellation::default()).unwrap();
    assert_eq!(value["physical_memory_bytes"], 17179869184_u64);
    assert_eq!(value["machine_model"], "Mac14,6");
    executable(&path, "#!/bin/sh\necho token=secret\n");
    let unavailable = kv_metadata::darwin(
        &path,
        root.path(),
        Instant::now() + Duration::from_secs(5),
        &process::Cancellation::default(),
    )
    .unwrap();
    assert_eq!(unavailable["memory_available"], false);
    assert!(unavailable["chip"].is_null());
    root.close().unwrap();
}
#[test]
#[cfg(unix)]
fn restart_missing_git_keeps_unknown_without_discarding_binary_byte_identity() {
    let root = tempfile::tempdir().unwrap();
    let value = kv_metadata::checkout(
        &root.path().join("missing-git"),
        root.path(),
        Instant::now() + Duration::from_secs(5),
        &process::Cancellation::default(),
    )
    .unwrap();
    assert_eq!(value["source_sha"], "unknown");
    assert_eq!(value["git_describe"], "unknown");
    std::fs::write(root.path().join("binary"), b"binary").unwrap();
    let sha = crate::product::digest::file_sha256(&root.path().join("binary"))
        .unwrap_or_else(|_| panic!("fixture binary digest unavailable"));
    assert_eq!(
        sha,
        "9a3a45d01531a20e89ac6ae10b0b0beb0492acd7216a368aa062d1a5fecaf9cd"
    );
    root.close().unwrap();
}
