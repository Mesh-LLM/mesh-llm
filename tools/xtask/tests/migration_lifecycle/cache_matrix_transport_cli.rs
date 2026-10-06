//! Actual matrix CLI with an inert supplied curl executable. No DNS/TLS/network occurs.
use super::*;
use std::os::unix::fs::PermissionsExt;
fn quoted(value: &Path) -> String {
    format!("'{}'", value.to_str().unwrap().replace('\'', "'\\''"))
}
fn fake_curl(root: &Path, mode: &str) -> BTreeMap<std::ffi::OsString, Value> {
    let bin = root.join("bin");
    std::fs::create_dir(&bin).unwrap();
    let script = format!(
        r#"#!/bin/sh
if [ "$1" = --disable ] && [ "$2" = --version ]; then
  printf 'curl 8.4.0 (fixture)\nProtocols: http https\n'; exit 0
fi
out= payload= endpoint= auth=none
[ "$1" = --disable ] || exit 31
while [ "$#" -gt 0 ]; do
 case "$1" in
 --output) out="$2"; shift 2;;
 --data-binary) payload="${{2#@}}"; shift 2;;
 --url) endpoint="$2"; shift 2;;
 --header) case "$2" in 'authorization: Bearer inert-private-token') auth=required;; 'authorization:'*) exit 32;; esac; shift 2;;
 --insecure|--location|--netrc) exit 33;;
 *) shift;;
 esac
done
[ -f "$payload" ] && [ -n "$out" ] || exit 34
case "$endpoint" in
 *native*/completion)
  [ "$auth" = none ] || exit 35
  /usr/bin/grep -q '"prompt":' "$payload" || exit 36
  /usr/bin/grep -q '"n_predict":2' "$payload" || exit 37
  native=yes;;
 *skippy*/v1/chat/completions)
  case '{mode}' in no-auth) [ "$auth" = none ] || exit 38;; *) [ "$auth" = required ] || exit 38;; esac
  /usr/bin/grep -q '"model":"inert-model"' "$payload" || exit 39
  /usr/bin/grep -q '"max_tokens":2' "$payload" || exit 40
  /usr/bin/grep -q '"messages":' "$payload" || exit 41
  native=no;;
 *) exit 42;;
esac
printf '%s %s\n' "$endpoint" "$auth" >> {marker}
case "$endpoint" in *warm*) cached=8;; *) cached=0;; esac
if [ "$native" = yes ]; then
 printf '{{"content":"same inert output","stop":true,"tokens_predicted":2,"tokens_evaluated":10,"truncated":false,"stop_type":"limit","timings":{{"prompt_n":2,"cache_n":%s,"predicted_n":2,"prompt_ms":1,"predicted_ms":2}}}}' "$cached" > "$out"
else
 printf '{{"choices":[{{"message":{{"content":"same inert output"}}}}],"usage":{{"prompt_tokens":10,"prompt_tokens_details":{{"cached_tokens":%s}}}}}}' "$cached" > "$out"
fi
case '{mode}' in
 status) case "$endpoint" in *cold-skippy*) printf 503; exit 0;; esac;;
 redirect) case "$endpoint" in *cold-skippy*) printf 302; exit 0;; esac;;
 protocol) case "$endpoint" in *cold-skippy*) printf 'data: [DONE]\n\n' > "$out";; esac;;
 held) case "$endpoint" in *warm-skippy*) while :; do /bin/sleep 1; done;; esac;;
esac
printf 200
"#,
        marker = quoted(&root.join("curl-admission.log"))
    );
    let curl = bin.join("curl");
    std::fs::write(&curl, script).unwrap();
    std::fs::set_permissions(&curl, std::fs::Permissions::from_mode(0o700)).unwrap();
    BTreeMap::from([("PATH".into(), Value::Public(bin.into_os_string()))])
}
fn transport_args(root: &Path) -> Vec<String> {
    let fake = Peer {
        url: "https://matrix.invalid".into(),
        stop: Arc::new(AtomicBool::new(false)),
        requests: Arc::new(Mutex::new(Vec::new())),
        worker: None,
    };
    let mut args = args(root, &fake, "shared-prefix");
    let index = args
        .iter()
        .position(|s| s == "--llama-cold-base-url")
        .unwrap();
    args[index + 1] = "http://matrix.invalid/cold-native".into();
    let index = args.iter().position(|s| s == "--timeout").unwrap();
    args[index + 1] = "8".into();
    args
}
fn call(root: &Path, mode: &str, cancel: &Cancellation) -> process::RawProcessReport {
    cli_with_environment(
        root,
        transport_args(root),
        cancel,
        fake_curl(root, mode),
        Duration::from_secs(30),
        Duration::from_secs(8),
    )
}
#[test]
fn hostname_https_actual_matrix_cli_preserves_auth_four_serial_arms_and_excluded_warmups() {
    let root = tempfile::tempdir().unwrap();
    let result = call(root.path(), "success", &Cancellation::default());
    assert!(result.process.success(), "{result:?}");
    let report = receipt(root.path());
    assert_eq!(report["status"], "completed");
    let rows = report["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 4);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row["runs"].as_array().unwrap().len(), 2);
        assert_eq!(row["warmup"].is_null(), index < 2);
    }
    let markers = std::fs::read_to_string(root.path().join("curl-admission.log")).unwrap();
    let lines = markers.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 10);
    for (index, line) in lines.iter().enumerate() {
        let expected = if index < 2 {
            "cold-native"
        } else if index < 4 {
            "cold-skippy"
        } else if index < 7 {
            "warm-native"
        } else {
            "warm-skippy"
        };
        assert!(line.contains(expected));
        assert!(line.ends_with(if expected.contains("skippy") {
            "required"
        } else {
            "none"
        }));
    }
    for name in ["cache-matrix.json", "cache-matrix.md"] {
        assert!(
            !std::fs::read_to_string(root.path().join("new-parent/results").join(name))
                .unwrap()
                .contains("inert-private-token")
        );
    }
    root.close().unwrap();
}
#[test]
fn hostname_https_actual_cli_refuses_redirect_and_nonstream_protocol_preserving_prior_row() {
    for mode in ["redirect", "status", "protocol"] {
        let root = tempfile::tempdir().unwrap();
        let result = call(root.path(), mode, &Cancellation::default());
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let report = receipt(root.path());
        assert_eq!(report["status"], "failed");
        assert_eq!(report["rows"].as_array().unwrap().len(), 1);
        assert!(report["incomplete_row"].is_object());
        assert_eq!(
            std::fs::read_to_string(root.path().join("curl-admission.log"))
                .unwrap()
                .lines()
                .count(),
            3
        );
        root.close().unwrap();
    }
}
#[test]
fn hostname_https_actual_cli_causal_cancellation_joins_owned_curl_and_keeps_three_rows() {
    let root = tempfile::tempdir().unwrap();
    let cancel = Cancellation::default();
    let environment = fake_curl(root.path(), "held");
    let result = std::thread::scope(|scope| {
        let token = cancel.clone();
        let marker = root.path().join("curl-admission.log");
        let waiter = scope.spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(20);
            let mut observed = false;
            while Instant::now() < deadline {
                if std::fs::read_to_string(&marker).is_ok_and(|s| s.lines().count() == 8) {
                    observed = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            token.cancel();
            observed
        });
        let result = cli_with_environment(
            root.path(),
            transport_args(root.path()),
            &cancel,
            environment,
            Duration::from_secs(30),
            Duration::from_secs(8),
        );
        let observed = waiter.join().unwrap();
        assert!(
            observed,
            "owned curl POST must causally precede cancellation"
        );
        result
    });
    assert!(cancel.is_cancelled());
    assert_eq!(result.process.outcome, process::Outcome::Cancelled);
    let report = receipt(root.path());
    assert_eq!(report["status"], "cancelled");
    assert_eq!(report["rows"].as_array().unwrap().len(), 3);
    assert_eq!(
        std::fs::read_to_string(root.path().join("curl-admission.log"))
            .unwrap()
            .lines()
            .count(),
        8
    );
    root.close().unwrap();
}

#[test]
fn hostname_https_actual_cli_request_deadline_keeps_prior_rows_and_owned_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let result = call(root.path(), "held", &Cancellation::default());
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    let report = receipt(root.path());
    assert_eq!(report["status"], "failed");
    assert_eq!(report["rows"].as_array().unwrap().len(), 3);
    assert_eq!(
        std::fs::read_to_string(root.path().join("curl-admission.log"))
            .unwrap()
            .lines()
            .count(),
        8
    );
    root.close().unwrap();
}

#[test]
fn hostname_https_actual_cli_refuses_auth_controls_before_output_and_empty_means_no_auth() {
    for control in ['\r', '\n'] {
        let root = tempfile::tempdir().unwrap();
        let mut arguments = transport_args(root.path());
        let key = arguments.iter().position(|s| s == "--api-key").unwrap();
        arguments[key + 1] = format!("owned{control}value");
        let result = cli_with_environment(
            root.path(),
            arguments,
            &Cancellation::default(),
            fake_curl(root.path(), "success"),
            Duration::from_secs(10),
            Duration::from_secs(3),
        );
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        assert!(!root.path().join("new-parent").exists());
        assert!(!root.path().join("curl-admission.log").exists());
        root.close().unwrap();
    }
    let root = tempfile::tempdir().unwrap();
    let mut arguments = transport_args(root.path());
    let key = arguments.iter().position(|s| s == "--api-key").unwrap();
    arguments[key + 1] = String::new();
    let result = cli_with_environment(
        root.path(),
        arguments,
        &Cancellation::default(),
        fake_curl(root.path(), "no-auth"),
        Duration::from_secs(30),
        Duration::from_secs(8),
    );
    assert!(result.process.success());
    assert_eq!(receipt(root.path())["status"], "completed");
    let markers = std::fs::read_to_string(root.path().join("curl-admission.log")).unwrap();
    assert_eq!(markers.lines().count(), 10);
    assert!(markers.lines().all(|line| line.ends_with("none")));
    root.close().unwrap();
}
