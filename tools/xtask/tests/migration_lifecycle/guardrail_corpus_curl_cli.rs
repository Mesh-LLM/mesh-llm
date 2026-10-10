//! Actual guardrail dispatcher with an owned inert curl program; no DNS/TLS/network proof.
use crate::process::{self, Cancellation, Value};
use serde_json::Value as Json;
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}
fn fixture(mode: &str) -> tempfile::TempDir {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let binary = root.path().join("curl");
    let script = format!(
        r#"#!/bin/bash
set -eu
[[ "$1" == --disable ]] || exit 64
if [[ "$2" == --version ]]; then printf 'curl 8.4.0 (owned inert fixture)\nProtocols: http https\n'; exit 0; fi
method='' body='' headers='' payload='' url='' proto='' maximum='' auth=''
while [[ $# -gt 0 ]]; do
 case "$1" in
 --disable|--silent|--show-error|--no-buffer|--http1.1|--suppress-connect-headers) shift;;
 --request) method="$2"; shift 2;; --output) body="$2"; shift 2;; --write-out) [[ "$2" == '%{{http_code}}' ]] || exit 64; shift 2;;
 --data-binary) payload="${{2#@}}"; shift 2;; --url) url="$2"; shift 2;;
 --proto) proto="$2"; shift 2;; --max-filesize) maximum="$2"; shift 2;;
 --proto-redir) [[ "$2" == '=https' ]] || exit 64; shift 2;;
 --header) if [[ "$2" == 'authorization: Bearer mesh-llm-ci' ]]; then auth=1; else [[ "$2" == 'content-type: application/json' ]] || exit 64; fi; shift 2;;
 --max-time|--connect-timeout) shift 2;; *) exit 64;;
 esac
done
[[ "$proto" == '=http,https' && "$maximum" == 1048576 && "$auth" == 1 ]] || exit 64
[[ "$url" == https://fixture.invalid/v1/* || "$url" == http://localhost:1234/v1/* ]] || exit 64
printf '%s %s\n' "$method" "$url" >> {record}
printf '200'
if [[ "$method" == GET ]]; then printf '{{"data":[{{"id":"fixture"}}]}}' > "$body"; exit 0; fi
[[ "$method" == POST && -f "$payload" ]] || exit 64
/bin/cat "$payload" >> {requests}; printf '\n' >> {requests}
if [[ '{mode}' == held ]]; then printf 'owned measured POST admitted' > {marker}; /bin/sleep 30 & pid=$!; trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; exit 0' TERM INT; wait "$pid"; exit 0; fi
if [[ '{mode}' == oversize ]]; then /usr/bin/head -c 1048577 /dev/zero > "$body"; exit 0; fi
if [[ '{mode}' == malformed ]]; then printf 'data: malformed\n\ndata: [DONE]\n\n' > "$body"; exit 0; fi
if /usr/bin/grep -q '"stream":true' "$payload"; then printf 'data: {{"choices":[{{"delta":{{"content":"observed"}},"finish_reason":"stop"}}]}}\n\ndata: [DONE]\n\n' > "$body"; else printf '{{"choices":[{{"message":{{"content":"observed"}}}}]}}' > "$body"; fi
"#,
        record = quote(&root.path().join("commands")),
        requests = quote(&root.path().join("requests")),
        marker = quote(&root.path().join("held"))
    );
    std::fs::write(&binary, script).unwrap();
    std::fs::set_permissions(binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    root
}
fn invoke(
    root: &Path,
    base: &str,
    seconds: u64,
    cancel: &Cancellation,
) -> process::RawProcessReport {
    process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            environment: BTreeMap::from([("PATH".into(), Value::Public(root.as_os_str().into()))]),
            arguments: [
                "automation",
                "guardrail-corpus",
                "--base-url",
                base,
                "--model",
                "fixture",
                "--trials",
                "1",
                "--out",
                root.join("receipt.json").to_str().unwrap(),
                "--timeout-secs",
                &seconds.to_string(),
            ]
            .into_iter()
            .map(|s| Value::Public(s.into()))
            .collect(),
        },
        &process::Limits {
            execution: Duration::from_secs(45),
            graceful_shutdown: Duration::from_secs(10),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancel,
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
fn clean(raw: &process::RawProcessReport) {
    let p = &raw.process;
    assert!(
        p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
            && p.failure.is_none(),
        "{p:?}"
    );
    assert!(p.stdout.line_capture_complete && p.stderr.line_capture_complete);
    assert_eq!(
        raw.stdout.as_ref().unwrap().as_bytes().len() as u64,
        p.stdout.bytes_seen
    );
    assert_eq!(
        raw.stderr.as_ref().unwrap().as_bytes().len() as u64,
        p.stderr.bytes_seen
    );
}
fn report(root: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(root.join("receipt.json")).unwrap()).unwrap()
}
#[test]
fn actual_hostname_https_cli_projects_owned_curl_and_all_five_bodies() {
    for base in ["https://fixture.invalid/v1", "http://localhost:1234/v1"] {
        let root = fixture("success");
        let raw = invoke(root.path(), base, 30, &Cancellation::default());
        clean(&raw);
        assert!(raw.process.success(), "{raw:?}");
        let report = report(root.path());
        assert_eq!(report["backend_mode"], "live");
        assert_eq!(report["total_requests"], 5);
        assert_eq!(report["success_count"], 4);
        assert_eq!(report["failure_count"], 1);
        let commands = std::fs::read_to_string(root.path().join("commands")).unwrap();
        assert_eq!(commands.lines().count(), 6);
        assert!(commands.lines().next().unwrap().starts_with("GET "));
        let bodies = std::fs::read_to_string(root.path().join("requests"))
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str::<Json>(s).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(bodies.len(), 5);
        assert_eq!(bodies[0]["stream"], true);
        assert_eq!(bodies[1]["tools"][0]["function"]["name"], "calculator");
        assert_eq!(bodies[3]["response_format"]["json_schema"]["strict"], true);
        root.close().unwrap();
    }
}
#[test]
fn actual_curl_malformed_and_oversized_measurement_retains_live_partial_failure() {
    for mode in ["malformed", "oversize"] {
        let root = fixture(mode);
        let raw = invoke(
            root.path(),
            "https://fixture.invalid/v1",
            30,
            &Cancellation::default(),
        );
        clean(&raw);
        assert!(!raw.process.status.unwrap().success());
        let report = report(root.path());
        assert_eq!(report["backend_mode"], "live");
        assert_eq!(report["status"], "incomplete");
        assert_eq!(report["total_requests"], 1);
        assert!(report["fallback_reason"].is_null());
        root.close().unwrap();
    }
}
#[test]
fn actual_curl_post_marker_cancellation_and_shared_deadline_own_cleanup_before_publication() {
    for cancelled in [false, true] {
        let root = fixture("held");
        let path = root.path().to_path_buf();
        let cancel = Cancellation::default();
        let scope = cancel.clone();
        let worker = std::thread::spawn(move || {
            invoke(
                &path,
                "https://fixture.invalid/v1",
                if cancelled { 30 } else { 5 },
                &scope,
            )
        });
        let until = Instant::now() + Duration::from_secs(15);
        while !root.path().join("held").exists() && !worker.is_finished() && Instant::now() < until
        {
            std::thread::park_timeout(Duration::from_millis(5));
        }
        let observed = root.path().join("held").exists();
        if cancelled || !observed {
            cancel.cancel();
        }
        let raw = worker.join().unwrap();
        clean(&raw);
        assert!(observed, "owner joined before marker assertion");
        assert_eq!(cancel.is_cancelled(), cancelled);
        let report = report(root.path());
        assert_eq!(report["backend_mode"], "live");
        assert_eq!(report["status"], "incomplete");
        assert_eq!(report["total_requests"], 1);
        assert!(!raw.process.status.unwrap().success());
        root.close().unwrap();
    }
}
#[test]
fn actual_credential_url_refuses_before_curl_probe_or_publication() {
    let root = fixture("success");
    let raw = invoke(
        root.path(),
        "https://user:secret@fixture.invalid/v1",
        30,
        &Cancellation::default(),
    );
    clean(&raw);
    assert!(!raw.process.status.unwrap().success());
    assert!(!root.path().join("commands").exists());
    assert!(!root.path().join("receipt.json").exists());
    root.close().unwrap();
}
