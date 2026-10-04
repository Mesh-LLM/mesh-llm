//! Actual Kotlin adapter with real archive/console owners and finite process observers.
//! This does not execute Java, Gradle, a model, or the nested login shell.
use super::*;

fn write_executable(root: &Path, name: &str, body: &str) {
    let path = root.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn consumer_fixture() -> Fixture {
    let fixture = Fixture::new(false);
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in [
        "scripts/ci-kotlin-sdk-smoke.sh",
        "scripts/package-sdk-console-assets.sh",
        "scripts/verify-sdk-console-assets.sh",
        "scripts/check-sdk-contract.sh",
        "docs/SDK.md",
        "sdk/swift/Sources/MeshLLM/Node.swift",
        "sdk/swift/Sources/MeshLLM/EventStream.swift",
        "sdk/kotlin/src/main/kotlin/ai/meshllm/Node.kt",
        "sdk/kotlin/build.gradle.kts",
        "sdk/node/index.js",
        "sdk/node/inference.js",
        "sdk/node/index.d.ts",
        "crates/mesh-llm-nodejs/src/lib.rs",
        // Unchanged compatibility sources inspected by the existing shell contract.
        // Neither file is executed as an oracle or automation tool.
        "sdk/python/src/meshllm/client.py",
        "sdk/python/src/meshllm/types.py",
    ] {
        let destination = fixture.root.join(name);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(repository.join(name), destination).unwrap();
    }
    let dist = fixture.root.join("crates/mesh-llm-ui/dist");
    fs::create_dir_all(dist.join("assets")).unwrap();
    fs::write(
        dist.join("index.html"),
        "<script src=\"/assets/app.js\"></script>",
    )
    .unwrap();
    fs::write(dist.join("assets/app.js"), b"inert console fixture").unwrap();
    fs::create_dir_all(fixture.root.join("observed-runtime")).unwrap();
    fs::write(fixture.root.join("selected-host"), b"inert selected host").unwrap();
    fs::write(fixture.root.join("selected-model"), b"inert selected model").unwrap();
    for name in [
        "bin/cargo",
        "bin/just",
        "bin/rustc",
        "bin/cmake",
        "scripts/build-ui.sh",
    ] {
        write_executable(
            &fixture.root,
            name,
            "#!/bin/bash\nprintf 'forbidden build\\n' >&2\nexit 91\n",
        );
    }
    write_executable(
        &fixture.root,
        "scripts/ci-prepare-native-runtime.sh",
        r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 4 ]] || exit 80
[[ "$1" == "$GITHUB_WORKSPACE/target/kotlin-native-runtime" ]] || exit 81
[[ "$2" == cpu && "$3" == --reuse-from-binary ]] || exit 82
[[ "$4" == "$GITHUB_WORKSPACE/selected-host" ]] || exit 83
printf 'selected-host-reuse\n' > "$GITHUB_WORKSPACE/runtime-observer"
[[ ! -f "$GITHUB_WORKSPACE/fail-runtime" ]] || exit 17
printf '%s\n' "$GITHUB_WORKSPACE/observed-runtime"
"#,
    );
    write_executable(
        &fixture.root,
        "scripts/ci-sdk-fixture.sh",
        r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 7 ]] || exit 80
[[ "$1" == "$GITHUB_WORKSPACE/selected-host" ]] || exit 81
[[ "$2" == "$GITHUB_WORKSPACE/bin" && "$3" == "$GITHUB_WORKSPACE/selected-model" ]] || exit 82
[[ "$4" == -- && "$5" == bash && "$6" == -lc ]] || exit 83
[[ "$MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR" == "$GITHUB_WORKSPACE/observed-runtime" ]] || exit 84
[[ "$MESHLLM_KOTLIN_JNA_LIBRARY_PATH" == /*/extracted/meshllm-native-linux-x86_64-cpu/lib ]] || exit 85
[[ "$(cat "$MESHLLM_KOTLIN_JNA_LIBRARY_PATH/libmesh_llm_uniffi.so")" == 'inert native SDK bytes' ]] || exit 86
printf '%s' "$7" > "$GITHUB_WORKSPACE/nested-client-body"
printf 'verified-library-client-handoff\n' > "$GITHUB_WORKSPACE/client-observer"
printf '%s' "$MESHLLM_KOTLIN_JNA_LIBRARY_PATH" > "$GITHUB_WORKSPACE/consumed-library-path"
[[ ! -f "$GITHUB_WORKSPACE/fail-client" ]] || exit 18
"#,
    );
    fixture
}

fn run_consumer(
    fixture: &Fixture,
    target: &str,
    profile: &str,
    flag: Option<&str>,
) -> process::RawProcessReport {
    let mut arguments = vec![
        fixture
            .root
            .join("scripts/ci-kotlin-sdk-smoke.sh")
            .display()
            .to_string(),
        fixture.root.join("selected-host").display().to_string(),
        fixture.root.join("bin").display().to_string(),
        fixture.root.join("selected-model").display().to_string(),
        fixture.root.join("download").display().to_string(),
        target.into(),
        "cpu".into(),
        profile.into(),
    ];
    if let Some(flag) = flag {
        arguments.push(flag.into());
    }
    fixture.command_with_env(
        arguments
            .into_iter()
            .map(|word| Value::Public(word.into()))
            .collect(),
        &[("GITHUB_WORKSPACE", fixture.root.display().to_string())],
    )
}

#[test]
fn kotlin_actual_consumer_uses_verified_sdk_library_and_selected_host_reuse() {
    let fixture = consumer_fixture();
    let original = fs::read(fixture.root.join("download/sdk.tar.gz")).unwrap();
    let report = run_consumer(
        &fixture,
        "x86_64-unknown-linux-gnu",
        "debug",
        Some("--skip-build"),
    );
    assert!(
        report.process.status.unwrap().success(),
        "{}",
        String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
    );
    assert_eq!(
        fs::read(fixture.root.join("download/sdk.tar.gz")).unwrap(),
        original
    );
    assert!(fixture.root.join("runtime-observer").is_file());
    assert!(fixture.root.join("client-observer").is_file());
    let body = fs::read_to_string(fixture.root.join("nested-client-body")).unwrap();
    for required in [
        "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR:?",
        "MESHLLM_KOTLIN_JNA_LIBRARY_PATH",
        "./gradlew --no-daemon run",
        "MESH_SDK_INVITE_TOKEN",
    ] {
        assert!(
            body.contains(required),
            "missing client contract: {required}"
        );
    }
    let console = fixture
        .root
        .join("sdk/kotlin/src/main/resources/mesh-llm/console");
    assert_eq!(
        fs::read(console.join("assets/app.js")).unwrap(),
        b"inert console fixture"
    );
    assert!(console.join("manifest.txt").is_file());
    let consumed = fs::read_to_string(fixture.root.join("consumed-library-path")).unwrap();
    assert!(
        !Path::new(&consumed).exists(),
        "owned extraction must be cleaned"
    );
    assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
}

#[test]
fn kotlin_actual_consumer_refuses_corrupt_or_extra_sdk_upload_before_client() {
    for corrupt in [true, false] {
        let fixture = consumer_fixture();
        if corrupt {
            fs::write(fixture.root.join("download/sdk.tar.gz"), b"corrupt archive").unwrap();
        } else {
            fs::write(fixture.root.join("download/extra"), b"unexpected payload").unwrap();
        }
        let report = run_consumer(
            &fixture,
            "x86_64-unknown-linux-gnu",
            "debug",
            Some("--skip-build"),
        );
        assert!(!report.process.status.unwrap().success());
        assert!(!fixture.root.join("runtime-observer").exists());
        assert!(!fixture.root.join("client-observer").exists());
        assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
    }
}

#[test]
fn kotlin_actual_consumer_propagates_runtime_and_client_failures_and_cleans_extraction() {
    for (marker, code, client_reached) in [("fail-runtime", 17, false), ("fail-client", 18, true)] {
        let fixture = consumer_fixture();
        fs::write(fixture.root.join(marker), b"bounded failure").unwrap();
        let original = fs::read(fixture.root.join("download/sdk.tar.gz")).unwrap();
        let report = run_consumer(
            &fixture,
            "x86_64-unknown-linux-gnu",
            "debug",
            Some("--skip-build"),
        );
        assert_eq!(report.process.status.unwrap().code(), Some(code));
        assert!(fixture.root.join("runtime-observer").exists());
        assert_eq!(
            fixture.root.join("client-observer").exists(),
            client_reached
        );
        if client_reached {
            let consumed = fs::read_to_string(fixture.root.join("consumed-library-path")).unwrap();
            assert!(
                !Path::new(&consumed).exists(),
                "owned extraction must be cleaned"
            );
        }
        assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
        assert_eq!(
            fs::read(fixture.root.join("download/sdk.tar.gz")).unwrap(),
            original
        );
    }
}

#[test]
fn kotlin_actual_consumer_refuses_invalid_sdk_identity_before_runtime_or_client() {
    for (target, profile) in [
        ("aarch64-unknown-linux-gnu", "debug"),
        ("x86_64-unknown-linux-gnu", "release"),
    ] {
        let fixture = consumer_fixture();
        let report = run_consumer(&fixture, target, profile, Some("--skip-build"));
        assert!(!report.process.status.unwrap().success());
        assert!(!fixture.root.join("runtime-observer").exists());
        assert!(!fixture.root.join("client-observer").exists());
    }
}

#[test]
fn kotlin_actual_consumer_stops_before_client_on_console_build_or_argument_failure() {
    for flag in [None, Some("--invalid")] {
        let fixture = consumer_fixture();
        let report = run_consumer(&fixture, "x86_64-unknown-linux-gnu", "debug", flag);
        assert!(!report.process.status.unwrap().success());
        assert!(!fixture.root.join("runtime-observer").exists());
        assert!(!fixture.root.join("client-observer").exists());
    }
    let fixture = consumer_fixture();
    fs::remove_file(fixture.root.join("crates/mesh-llm-ui/dist/index.html")).unwrap();
    let report = run_consumer(
        &fixture,
        "x86_64-unknown-linux-gnu",
        "debug",
        Some("--skip-build"),
    );
    assert!(!report.process.status.unwrap().success());
    assert!(!fixture.root.join("client-observer").exists());
}
