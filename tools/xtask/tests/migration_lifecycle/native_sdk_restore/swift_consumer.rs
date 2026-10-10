//! Actual Swift adapter handoff and archive extraction with finite component observers.
//! This does not qualify Mach-O, privacy metadata, Swift compilation, or a model.
use super::kotlin_consumer::{consumer_fixture, write_executable};
use super::*;
#[path = "../../migration_archives/zip_writers.rs"]
mod zip_writers;

fn swift_fixture() -> Fixture {
    let fixture = consumer_fixture();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::copy(
        repository.join("scripts/ci-swift-sdk-smoke.sh"),
        fixture.root.join("scripts/ci-swift-sdk-smoke.sh"),
    )
    .unwrap();
    fs::copy(
        repository.join("Package.swift"),
        fixture.root.join("Package.swift"),
    )
    .unwrap();
    fs::create_dir_all(
        fixture
            .root
            .join("mesh/sdk/swift/Sources/MeshLLM/Generated"),
    )
    .unwrap();
    fs::write(
        fixture.root.join("input-binding.swift"),
        b"inert binding bytes",
    )
    .unwrap();
    fs::write(
        fixture.root.join("mesh/sdk/swift/PrivacyInfo.xcprivacy"),
        b"inert privacy observer input",
    )
    .unwrap();
    let bytes = zip_writers::zip(&[
        zip_writers::ZipEntry::raw(
            "MeshLLMFFI.xcframework/",
            (zip_writers::S_IFDIR | 0o755) << 16,
            b"",
        ),
        zip_writers::ZipEntry::file(
            "MeshLLMFFI.xcframework/fixture",
            0o644,
            b"producer archive bytes",
        )
        .deflated(),
    ]);
    fs::write(fixture.root.join("swift.zip"), bytes).unwrap();
    write_component_observers(&fixture.root);
    write_client_observers(&fixture.root);
    fixture
}

fn write_component_observers(root: &Path) {
    write_executable(
        root,
        "scripts/verify-swift-release-artifact.sh",
        r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 2 && "$1" == "$GITHUB_WORKSPACE/swift.zip" && "$2" == host-only ]] || exit 80
[[ "$(cat mesh/sdk/swift/Sources/MeshLLM/Generated/mesh_ffi.swift)" == 'inert binding bytes' ]] || exit 81
[[ ! -f fail-artifact ]] || exit 21
printf 'artifact-component-observed\n' > artifact-observer
"#,
    );
    write_executable(
        root,
        "scripts/verify-swift-privacy-manifest.sh",
        r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 2 && "$1" == mesh/sdk/swift/PrivacyInfo.xcprivacy && "$2" == mesh/sdk/swift/Generated/MeshLLMFFI.xcframework ]] || exit 80
[[ "$(cat "$2/fixture")" == 'producer archive bytes' ]] || exit 81
printf 'privacy-component-observed\n' > privacy-observer
"#,
    );
}

fn write_client_observers(root: &Path) {
    write_executable(
        root,
        "scripts/ci-prepare-native-runtime.sh",
        r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 4 && "$1" == "$GITHUB_WORKSPACE/target/swift-native-runtime" ]] || exit 80
[[ "$2" == cpu && "$3" == --reuse-from-binary && "$4" == "$GITHUB_WORKSPACE/selected-host" ]] || exit 81
printf '%s\n' "$GITHUB_WORKSPACE/observed-runtime"
"#,
    );
    write_executable(
        root,
        "scripts/ci-sdk-fixture.sh",
        r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 7 && "$1" == "$GITHUB_WORKSPACE/selected-host" ]] || exit 80
[[ "$2" == "$GITHUB_WORKSPACE/bin" && "$3" == "$GITHUB_WORKSPACE/selected-model" ]] || exit 81
[[ "$4" == -- && "$5" == bash && "$6" == -lc ]] || exit 82
[[ "$MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR" == "$GITHUB_WORKSPACE/observed-runtime" ]] || exit 83
export MESH_SDK_INVITE_TOKEN=inert-invite
/bin/bash -c "$7"
"#,
    );
    write_executable(
        root,
        "bin/swift",
        r#"#!/bin/bash
set -euo pipefail
[[ "$PWD" == "$GITHUB_WORKSPACE" ]] || exit 80
[[ "$#" == 5 && "$1" == run && "$2" == --package-path && "$3" == mesh/sdk/swift/example/MeshExampleApp && "$4" == MeshExampleApp && "$5" == inert-invite ]] || exit 81
[[ "$(cat mesh/sdk/swift/Generated/MeshLLMFFI.xcframework/fixture)" == 'producer archive bytes' ]] || exit 82
printf 'literal-root-client-body\n' > client-body-observer
[[ ! -f fail-client-body ]] || exit 23
"#,
    );
}

fn invoke(fixture: &Fixture) -> process::RawProcessReport {
    let arguments = [
        "scripts/ci-swift-sdk-smoke.sh",
        "selected-host",
        "bin",
        "selected-model",
        "swift.zip",
    ]
    .into_iter()
    .map(|p| fixture.root.join(p).display().to_string())
    .chain([
        "host-only".to_owned(),
        fixture
            .root
            .join("input-binding.swift")
            .display()
            .to_string(),
        "--skip-build".to_owned(),
    ])
    .map(|p| Value::Public(p.into()))
    .collect();
    fixture.command_with_env(
        arguments,
        &[
            ("GITHUB_WORKSPACE", fixture.root.display().to_string()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                fixture
                    .root
                    .join("bin/sdk-source-observer")
                    .display()
                    .to_string(),
            ),
            (
                "MESH_FIXTURE_NATIVE_AUTOMATION",
                env!("CARGO_BIN_EXE_xtask").to_owned(),
            ),
        ],
    )
}

#[test]
fn swift_actual_handoff_keeps_literal_root_and_restores_producer_archive_bytes() {
    let mut fixture = swift_fixture();
    let renamed = fixture
        .root
        .parent()
        .unwrap()
        .join("Swift with 'quotes' and $literal");
    fs::rename(&fixture.root, &renamed).unwrap();
    fixture.root = renamed;
    let before = fs::read(fixture.root.join("swift.zip")).unwrap();
    let report = invoke(&fixture);
    assert!(
        report.process.status.unwrap().success(),
        "{}",
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
    );
    assert!(fixture.root.join("client-body-observer").is_file());
    let native_argv = fs::read_to_string(fixture.root.join("native-owner-argv")).unwrap();
    for owner in ["sdk-console-manifest", "sdk-console-verify"] {
        assert!(
            native_argv.lines().any(|argument| argument == owner),
            "native owner not delegated: {owner}"
        );
    }
    assert_eq!(fs::read(fixture.root.join("swift.zip")).unwrap(), before);
    assert!(
        fixture
            .root
            .join("mesh/sdk/swift/Sources/MeshLLM/Resources/Console/manifest.txt")
            .is_file()
    );
}

#[test]
fn swift_actual_handoff_propagates_client_failure_after_verified_restore() {
    let fixture = swift_fixture();
    fs::write(fixture.root.join("fail-client-body"), b"finite failure").unwrap();
    assert_eq!(invoke(&fixture).process.status.unwrap().code(), Some(23));
    assert!(fixture.root.join("client-body-observer").is_file());
}

#[test]
fn swift_actual_handoff_stops_on_artifact_component_failure() {
    let fixture = swift_fixture();
    fs::write(fixture.root.join("fail-artifact"), b"finite failure").unwrap();
    assert_eq!(invoke(&fixture).process.status.unwrap().code(), Some(21));
    assert!(!fixture.root.join("privacy-observer").exists());
    assert!(!fixture.root.join("client-body-observer").exists());
}

#[test]
fn swift_actual_handoff_refuses_archive_escape_before_client_and_preserves_sentinel() {
    let fixture = swift_fixture();
    let archive = zip_writers::zip(&[zip_writers::ZipEntry::symlink(
        "MeshLLMFFI.xcframework",
        "../outside-sentinel",
    )]);
    fs::write(fixture.root.join("swift.zip"), &archive).unwrap();
    assert!(!invoke(&fixture).process.status.unwrap().success());
    assert!(!fixture.root.join("client-body-observer").exists());
    assert_eq!(fs::read(fixture.root.join("swift.zip")).unwrap(), archive);
}
