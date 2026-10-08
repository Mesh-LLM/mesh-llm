//! Actual runtime-preparation adapter, with finite product/build boundaries.
use super::fixture::{Fixture, capture};
use crate::process::{ProcessSpec, RawProcessReport, Value};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, os::unix::fs::PermissionsExt, path::Path};
fn executable(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/bash\nset -euo pipefail\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn row(supported: bool, backend: &str) -> Json {
    json!({"id":"runtime","backend":backend,"supported":supported})
}
fn fixture(report: &Json, backend: &str) -> Fixture {
    let fixture = Fixture::new(report);
    let host = fixture.root.join("host-input/mesh-llm");
    let source = fs::read_to_string(&host).unwrap().replace(
        "[[ \"$MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK\" == 0 ]] || exit 97",
        "[[ \"$MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK\" == 0 || \"$MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK\" == 1 ]] || exit 97",
    );
    executable(&host, &source);
    let runtime = fixture.root.join("host-input/native-runtimes/runtime");
    fs::create_dir_all(runtime.join("lib")).unwrap();
    fs::copy(
        fixture.root.join("runtime-input/runtime/lib/runtime.bin"),
        runtime.join("lib/runtime.bin"),
    )
    .unwrap();
    let mut manifest: Json = serde_json::from_slice(
        &fs::read(fixture.root.join("runtime-input/runtime/manifest.json")).unwrap(),
    )
    .unwrap();
    manifest["runtime"]["backend"]["kind"] = backend.into();
    manifest["build"]["backend"] = backend.into();
    fs::write(
        runtime.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    fixture
}
fn run(fixture: &Fixture, fallback: &str) -> RawProcessReport {
    let root = &fixture.root;
    let environment = [
        (
            "PATH",
            format!("{}:/usr/bin:/bin", root.join("bin").display()),
        ),
        ("HOME", root.join("ambient-home").display().to_string()),
        ("RUNNER_TEMP", root.join("tmp").display().to_string()),
        ("GITHUB_WORKSPACE", root.display().to_string()),
        (
            "MESH_LLM_AUTOMATION_BIN",
            env!("CARGO_BIN_EXE_xtask").into(),
        ),
        ("MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK", fallback.into()),
        ("MESH_LLM_CONFIG", "ambient-config".into()),
        (
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
            "ambient-bundle".into(),
        ),
        ("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", "ambient-cache".into()),
        ("CI", "true".into()),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), Value::Public(value.into())))
    .collect::<BTreeMap<_, _>>();
    capture(ProcessSpec {
        executable: "/bin/bash".into(),
        cwd: root.clone(),
        environment,
        arguments: [
            root.join("scripts/ci-prepare-native-runtime.sh")
                .into_os_string(),
            root.join("fallback").into_os_string(),
            "cpu".into(),
            "--reuse-from-binary".into(),
            root.join("host-input/mesh-llm").into_os_string(),
        ]
        .into_iter()
        .map(Value::Public)
        .collect(),
    })
}
fn diagnostic(result: &RawProcessReport) -> String {
    String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes()).into_owned()
}
fn no_build(fixture: &Fixture) {
    assert!(!fixture.root.join("fallback-ran").exists());
    assert!(!fixture.root.join("forbidden").exists());
}
#[test]
fn actual_runtime_preparation_reuses_flat_wrapped_and_sole_accelerated_reports() {
    for (report, backend) in [
        (json!([row(true, "cpu")]), "cpu"),
        (json!({"catalogs":{},"runtimes":[row(true,"cpu")]}), "cpu"),
        (json!([row(true, "vulkan")]), "vulkan"),
    ] {
        let fixture = fixture(&report, backend);
        let result = run(&fixture, "0");
        assert!(result.process.success(), "{}", diagnostic(&result));
        assert_eq!(
            result.stdout.as_ref().unwrap().as_bytes(),
            format!(
                "{}\n",
                fixture
                    .root
                    .join("host-input/native-runtimes/runtime")
                    .display()
            )
            .as_bytes()
        );
        assert!(diagnostic(&result).contains("verified native runtime artifact"));
        assert_eq!(
            fs::read_to_string(fixture.root.join("events")).unwrap(),
            "sdk\n"
        );
        assert!(!fixture.root.join("fallback").exists());
        no_build(&fixture);
    }
}
#[test]
fn actual_runtime_preparation_rejects_malformed_ambiguous_and_incompatible_reports_without_building()
 {
    let cases = [
        Json::Null,
        json!({}),
        json!({"runtimes":null}),
        json!({"runtimes":{}}),
        json!([null]),
        json!({"runtimes":[null]}),
        json!([row(false, "cpu")]),
        json!({"runtimes":[row(false,"cpu")]}),
        json!({"runtimes":[row(true,"cpu"),{"id":"another","backend":"cpu","supported":true}]}),
    ];
    for report in cases {
        let fixture = fixture(&report, "cpu");
        let result = run(&fixture, "1");
        assert!(!result.process.success());
        assert!(
            diagnostic(&result).contains("native runtime compatibility")
                || diagnostic(&result).contains("expected exactly one compatible")
        );
        assert!(result.stdout.as_ref().unwrap().as_bytes().is_empty());
        assert!(!fixture.root.join("fallback").exists());
        no_build(&fixture);
    }
}
#[test]
fn actual_runtime_preparation_wrong_abi_cannot_enable_fallback() {
    for report in [
        json!([row(true, "cpu")]),
        json!({"runtimes":[row(true,"cpu")]}),
    ] {
        let fixture = fixture(&report, "cpu");
        let path = fixture
            .root
            .join("host-input/native-runtimes/runtime/manifest.json");
        let mut manifest: Json = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest["runtime"]["skippy_abi"] = "99.99.99".into();
        fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let result = run(&fixture, "1");
        assert!(!result.process.success());
        assert!(diagnostic(&result).contains("has Skippy ABI 99.99.99"));
        assert!(!fixture.root.join("fallback").exists());
        no_build(&fixture);
    }
}
#[test]
fn actual_runtime_preparation_requires_bundle_in_ci_and_preserves_existing_output() {
    let fixture = fixture(&json!([row(true, "cpu")]), "cpu");
    fs::remove_dir_all(fixture.root.join("host-input/native-runtimes")).unwrap();
    fs::create_dir(fixture.root.join("fallback")).unwrap();
    fs::write(
        fixture.root.join("fallback/preserve"),
        b"owned prior result",
    )
    .unwrap();
    let result = run(&fixture, "0");
    assert!(!result.process.success());
    assert!(diagnostic(&result).contains("Adjacent native runtime bundle is required in CI"));
    assert_eq!(
        fs::read(fixture.root.join("fallback/preserve")).unwrap(),
        b"owned prior result"
    );
    assert!(!fixture.root.join("events").exists());
    no_build(&fixture);
}
#[test]
fn actual_runtime_preparation_explicit_fallback_preserves_native_primitive_arguments() {
    let fixture = fixture(&json!([row(true, "cpu")]), "cpu");
    fs::remove_dir_all(fixture.root.join("host-input/native-runtimes")).unwrap();
    executable(
        &fixture.root.join("scripts/package-native-runtime.sh"),
        r#"[[ "$#" == 5 && "$1" == --build && "$2" == --backend && "$3" == cpu && "$4" == --out && "$5" == "$GITHUB_WORKSPACE/fallback" ]]
[[ "$LLAMA_STAGE_LINK_MODE" == dynamic && "$LLAMA_STAGE_BACKEND" == cpu ]]
printf 'package\n' >> "$GITHUB_WORKSPACE/fallback-events"
mkdir -p "$5/meshllm-native-runtime-fallback"
printf '{}' > "$5/meshllm-native-runtime-fallback/manifest.json"
printf 'finite archive' > "$5/meshllm-native-runtime-fallback.tar.gz"
"#,
    );
    executable(
        &fixture
            .root
            .join("scripts/verify-native-runtime-package.sh"),
        r#"[[ "$#" == 1 && "$1" == "$GITHUB_WORKSPACE/fallback/meshllm-native-runtime-fallback.tar.gz" && -f "$1" ]]
printf 'verify\n' >> "$GITHUB_WORKSPACE/fallback-events"
"#,
    );
    let result = run(&fixture, "1");
    assert!(result.process.success(), "{}", diagnostic(&result));
    assert_eq!(
        result.stdout.as_ref().unwrap().as_bytes(),
        format!(
            "{}\n",
            fixture
                .root
                .join("fallback/meshllm-native-runtime-fallback")
                .display()
        )
        .as_bytes()
    );
    assert_eq!(
        fs::read_to_string(fixture.root.join("fallback-events")).unwrap(),
        "package\nverify\n"
    );
    assert!(!fixture.root.join("events").exists());
    assert!(!fixture.root.join("forbidden").exists());
}
#[test]
fn actual_sdk_smoke_callers_bind_reuse_to_the_selected_binary() {
    for path in [
        "scripts/ci-rust-sdk-smoke.sh",
        "scripts/ci-kotlin-sdk-smoke.sh",
        "scripts/ci-swift-sdk-smoke.sh",
    ] {
        let source = fs::read_to_string(super::fixture::repository().join(path)).unwrap();
        let lines = source
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
            .replace("\\\n", " ");
        assert!(
            lines
                .lines()
                .any(|line| line.contains("ci-prepare-native-runtime.sh")
                    && line.contains("--reuse-from-binary")
                    && line.contains("\"$1\"")),
            "{path}"
        );
    }
}
