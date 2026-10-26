//! `validate_pair` and producer/source binding unit tests.
use super::{validate_pair, verify_runtime_source};
use crate::ci_plan::document::Json;

const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const TARGET: &str = "x86_64-unknown-linux-gnu";

fn cli_contract() -> Json {
    Json::Object(
        [
            ("source_sha", SOURCE_SHA),
            ("os", "linux"),
            ("architecture", "x86_64"),
            ("runtime_release", "1.0.0"),
            ("skippy_abi", "7"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), Json::String(value.to_owned())))
        .collect(),
    )
}

fn runtime_manifest(backend: &str) -> Json {
    let build = if backend == "rocm" { "hip" } else { backend };
    crate::qualification::contract::parse_strict(
        format!(
            r#"{{"schema_version":2,"runtime":{{"id":"runtime-1","release_version":"1.0.0","skippy_abi":"7","platform":{{"target":"{TARGET}","os":"linux","arch":"x86_64"}},"backend":{{"kind":"{backend}"}}}},"build":{{"backend":"{build}"}}}}"#
        )
        .as_bytes(),
    )
    .unwrap()
}

#[test]
fn runtime_target_backend_release_and_abi_must_match_cli() {
    assert!(
        validate_pair(
            &cli_contract(),
            &runtime_manifest("cpu"),
            SOURCE_SHA,
            TARGET,
            "cpu"
        )
        .is_ok()
    );
    assert!(
        validate_pair(
            &cli_contract(),
            &runtime_manifest("cuda"),
            SOURCE_SHA,
            TARGET,
            "cpu"
        )
        .is_err()
    );
    assert!(
        validate_pair(
            &cli_contract(),
            &runtime_manifest("cpu"),
            &"f".repeat(40),
            TARGET,
            "cpu"
        )
        .is_err()
    );
    let foreign_os = crate::qualification::contract::parse_strict(
        br#"{"schema_version":2,"runtime":{"id":"r","release_version":"1.0.0","skippy_abi":"7","platform":{"target":"x86_64-unknown-linux-gnu","os":"windows","arch":"x86_64"},"backend":{"kind":"cpu"}}}"#,
    )
    .unwrap();
    assert!(validate_pair(&cli_contract(), &foreign_os, SOURCE_SHA, TARGET, "cpu").is_err());
}

#[test]
fn rocm_accepts_the_hip_build_alias() {
    assert!(
        validate_pair(
            &cli_contract(),
            &runtime_manifest("rocm"),
            SOURCE_SHA,
            TARGET,
            "rocm"
        )
        .is_ok()
    );
}

#[test]
fn unsupported_targets_and_backends_fail() {
    assert!(
        validate_pair(
            &cli_contract(),
            &runtime_manifest("cpu"),
            SOURCE_SHA,
            "plan9",
            "cpu"
        )
        .is_err()
    );
    assert!(
        validate_pair(
            &cli_contract(),
            &runtime_manifest("cpu"),
            SOURCE_SHA,
            TARGET,
            "tpu"
        )
        .is_err()
    );
}

#[test]
fn runtime_producer_source_is_required() {
    let temp = tempfile::tempdir().unwrap();
    assert!(verify_runtime_source(temp.path(), SOURCE_SHA).is_err());
    std::fs::write(
        temp.path().join("ci-source.json"),
        format!("{{\"source_sha\":\"{SOURCE_SHA}\"}}"),
    )
    .unwrap();
    assert!(verify_runtime_source(temp.path(), SOURCE_SHA).is_ok());
    std::fs::write(
        temp.path().join("ci-source.json"),
        format!("{{\"source_sha\":\"{}\",}}", "f".repeat(40)).replace(",}", "}"),
    )
    .unwrap();
    assert!(verify_runtime_source(temp.path(), SOURCE_SHA).is_err());
    std::fs::write(
        temp.path().join("ci-source.json"),
        format!("{{\"source_sha\":\"{}\",\"extra\":true}}", SOURCE_SHA),
    )
    .unwrap();
    assert!(verify_runtime_source(temp.path(), SOURCE_SHA).is_err());
}
