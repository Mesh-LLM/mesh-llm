use super::{contract::Input, guard};
use crate::process::Cancellation;
use serde_json::json;
use std::time::{Duration, Instant};
fn input() -> Input {
    serde_json::from_value(json!({"schema_version":1,"source_repo":"fixture/source","target_repo":"fixture/result","mesh_revision":"b".repeat(40),"output_basename":"model-BF16","upload_only":true,"timeout_seconds":60})).unwrap()
}
#[cfg(unix)]
#[test]
fn generic_conversion_defaults_and_exact_native_split_spool_status_argv_preserve_original() {
    let input = input();
    input.validate().unwrap();
    assert_eq!(input.source.to_str(), Some("/mnt/checkpoint"));
    assert_eq!(input.work_directory.to_str(), Some("/data/skippy-convert"));
    assert_eq!(input.shards(input.expected_splits), ["model-BF16.gguf"]);
    let args = input.convert_args();
    for (k, v) in [
        ("--output-type", "bf16"),
        ("--expected-splits", "1"),
        ("--window-size", "1"),
        ("--split-max-size", "50G"),
        ("--max-memory", "24G"),
        ("--stream-buffer-bytes", "8388608"),
        ("--json-event-interval-seconds", "60"),
        ("--json-event-window", "8"),
        ("--watchdog-seconds", "300"),
        ("--spool-dir", "/data/skippy-convert/spool"),
        ("--record-dir", "/data/skippy-convert/records"),
        ("--json-event-file", "/data/skippy-convert/status.json"),
    ] {
        let at = args.iter().position(|s| s == k).unwrap();
        assert_eq!(args[at + 1], v);
    }
    assert!(args.contains(&"--mtp".to_string()));
    assert!(!args.contains(&"--remote".to_string()));
    let card = String::from_utf8(input.card()).unwrap();
    for marker in [
        "license: apache-2.0",
        "base_model: fixture/source",
        "Inkling MTP sidecar (beta)",
        "not a standalone chat model",
        "not a promoted mesh-llm catalog entry",
    ] {
        assert!(card.contains(marker));
    }
    assert!(card.contains(&"b".repeat(40)));
}
#[cfg(unix)]
#[test]
fn generic_conversion_closed_requests_preserve_upload_only_and_dry_run_refusals() {
    let mut input = input();
    input.expected_splits = 3;
    assert_eq!(
        input.shards(input.expected_splits)[2],
        "model-BF16-00003-of-00003.gguf"
    );
    input.dry_run = true;
    assert!(input.validate().is_err());
    input.dry_run = false;
    input.publish_confirmed = true;
    assert!(input.validate().is_err());
    input.publish_confirmed = false;
    input.target_prefix = "../BF16".into();
    assert!(input.validate().is_err());
    input.target_prefix = "BF16".into();
    input.upload_only = false;
    assert!(input.validate().is_err());
}
#[test]
fn generic_conversion_terminal_budget_and_cancellation_refuse_after_observations() {
    let cancel = Cancellation::default();
    assert!(guard(Instant::now() + Duration::from_secs(1), &cancel).is_ok());
    assert!(guard(Instant::now(), &cancel).is_err());
    cancel.cancel();
    assert!(guard(Instant::now() + Duration::from_secs(1), &cancel).is_err());
}

#[cfg(unix)]
#[test]
fn generic_conversion_native_size_units_zero_and_literal_argv_are_preserved() {
    for literal in ["0", "24GiB", "24gib", " 24 GiB ", "1b", "2KB", "3m", "4TiB"] {
        let mut request = input();
        request.split_max_size = literal.into();
        request.max_memory = literal.into();
        request.validate().unwrap();
        let args = request.convert_args();
        for flag in ["--split-max-size", "--max-memory"] {
            let at = args.iter().position(|arg| arg == flag).unwrap();
            assert_eq!(args[at + 1], literal);
        }
    }
    for literal in [
        "",
        " ",
        "-1G",
        "1.5G",
        "1GBextra",
        "18446744073709551615TiB",
        "18446744073709551616",
        "1G;echo secret",
    ] {
        let mut request = input();
        request.split_max_size = literal.into();
        assert!(
            request.validate().is_err(),
            "accepted invalid native size {literal:?}"
        );
        request.split_max_size = "0".into();
        request.max_memory = literal.into();
        assert!(
            request.validate().is_err(),
            "accepted invalid memory size {literal:?}"
        );
    }
}
