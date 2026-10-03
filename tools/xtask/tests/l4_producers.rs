use serde_json::Value;
use std::fs;
use std::process::Command;

#[path = "l4_producers/cargo_projection.rs"]
mod cargo_projection;

fn runtime_args(root: &std::path::Path) -> Vec<String> {
    let output = root.join("manifest.json").to_string_lossy().into_owned();
    [
        "native",
        "runtime-manifest-write",
        &output,
        "runtime-fixture",
        "0.80.0",
        "1.2.3",
        "macos",
        "aarch64",
        "aarch64-apple-darwin",
        "darwin-aarch64",
        "metal",
        "",
        "lib/libllama.dylib",
        "upstream",
        "patched",
        "queue",
        "lib/libllama.dylib",
        "--",
        "tools/package",
        "--",
        "licenses/license.txt",
        "--",
    ]
    .map(str::to_owned)
    .to_vec()
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    for directory in ["lib", "tools", "licenses"] {
        fs::create_dir(root.path().join(directory)).unwrap();
    }
    fs::write(root.path().join("lib/libllama.dylib"), b"abc").unwrap();
    fs::write(root.path().join("tools/package"), b"abc").unwrap();
    fs::write(root.path().join("licenses/license.txt"), b"abc").unwrap();
    root
}

fn command(args: &[String]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .env_remove("MESH_LLM_NATIVE_RUNTIME_RANK")
        .output()
        .unwrap()
}

#[test]
fn l4_runtime_manifest_records_consumed_shape_when_packaged_files_exist() {
    let root = fixture();
    let args = runtime_args(root.path());
    let output = command(&args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = fs::read(root.path().join("manifest.json")).unwrap();
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(manifest["runtime"]["files"]["lib/libllama.dylib"], digest);
    assert_eq!(manifest["runtime"]["files"]["licenses/license.txt"], digest);
    assert_eq!(manifest["runtime"]["tools"]["tools/package"], digest);
    assert_eq!(manifest["build"]["library_sha256"], digest);
    assert_eq!(
        manifest["runtime"]["platform"],
        serde_json::json!({"os":"macos","arch":"aarch64","target":"aarch64-apple-darwin","min_glibc":null})
    );
    assert_eq!(
        manifest["runtime"]["backend"],
        serde_json::json!({"kind":"metal"})
    );
    assert_eq!(manifest["runtime"]["skippy_abi"], "1.2.3");
    assert_eq!(manifest["runtime"]["mesh_version"], "0.80.0");
    assert_eq!(manifest["build"]["llama_patch_digest"], "queue");
    assert_eq!(bytes.last(), Some(&b'\n'));
}

#[test]
fn l4_runtime_manifest_preserves_output_when_packaged_file_is_missing() {
    let root = fixture();
    fs::remove_file(root.path().join("tools/package")).unwrap();
    fs::write(root.path().join("manifest.json"), b"prior manifest").unwrap();
    let output = command(&runtime_args(root.path()));
    assert!(!output.status.success());
    assert_eq!(
        fs::read(root.path().join("manifest.json")).unwrap(),
        b"prior manifest"
    );
}

#[test]
fn l4_runtime_manifest_rejects_traversal_when_library_path_escapes() {
    let root = fixture();
    let mut args = runtime_args(root.path());
    args[12] = "../outside".into();
    args[16] = "../outside".into();
    let output = command(&args);
    assert!(!output.status.success());
    assert!(!root.path().join("manifest.json").exists());
}

#[test]
fn l4_runtime_manifest_normalizes_backend_when_cuda_blackwell_is_selected() {
    let root = fixture();
    let mut args = runtime_args(root.path());
    args[6] = "windows".into();
    args[7] = "x86_64".into();
    args[8] = "x86_64-pc-windows-msvc".into();
    args[10] = "cuda-blackwell".into();
    args[11] = "13".into();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&args)
        .env("LLAMA_STAGE_CUDA_ARCHITECTURES", "sm_86; sm_120,sm_90")
        .env("MESH_LLM_NATIVE_RUNTIME_RANK", "7")
        .env("MESH_LLM_CUDA_MIN_DRIVER", "570")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.path().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["runtime"]["backend"],
        serde_json::json!({"kind":"cuda","cuda":{"toolkit_major":13,"gpu_arches":["sm_86","sm_120","sm_90"],"min_driver":"570"}})
    );
    assert_eq!(manifest["runtime"]["rank"], 7);
    assert_eq!(manifest["build"]["backend"], "cuda-blackwell");
}

#[cfg(unix)]
#[test]
fn l4_runtime_manifest_rejects_symlink_when_it_escapes_artifact() {
    let root = fixture();
    let outside = tempfile::NamedTempFile::new().unwrap();
    fs::remove_file(root.path().join("tools/package")).unwrap();
    std::os::unix::fs::symlink(outside.path(), root.path().join("tools/package")).unwrap();
    let output = command(&runtime_args(root.path()));
    assert!(!output.status.success());
    assert!(!root.path().join("manifest.json").exists());
}

fn sdk_args(root: &std::path::Path) -> Vec<String> {
    let output = root.join("manifest.json").to_string_lossy().into_owned();
    [
        "prepared-input",
        "native-sdk-manifest-write",
        &output,
        "meshllm-native-darwin-aarch64-cpu",
        "0.80.0",
        "aarch64-apple-darwin",
        "darwin-aarch64",
        "macos",
        "aarch64",
        "cpu",
        "cpu",
        "release",
        "lib/libmeshllm_ffi.dylib",
        "lib/libuniffi_mesh_ffi.dylib",
        "",
        "",
        "",
    ]
    .map(str::to_owned)
    .to_vec()
}

#[test]
fn l4_sdk_manifest_passes_existing_consumer_when_library_alias_matches() {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("meshllm-native-darwin-aarch64-cpu");
    fs::create_dir_all(artifact.join("lib")).unwrap();
    for library in ["libmeshllm_ffi.dylib", "libuniffi_mesh_ffi.dylib"] {
        fs::write(artifact.join("lib").join(library), b"abc").unwrap();
    }
    let runner_build = root.path().join("runner private llama build");
    fs::create_dir(&runner_build).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(sdk_args(&artifact))
        .env_remove("MESH_LLM_NATIVE_RUNTIME_RANK")
        .env("LLAMA_STAGE_BUILD_DIR", &runner_build)
        .env("LLAMA_BUILD_DIR", &runner_build)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let consumer = command(&[
        "prepared-input".into(),
        "native-sdk-manifest".into(),
        artifact.display().to_string(),
        artifact.join("manifest.json").display().to_string(),
    ]);
    assert!(
        consumer.status.success(),
        "{}",
        String::from_utf8_lossy(&consumer.stderr)
    );
    let bytes = fs::read(artifact.join("manifest.json")).unwrap();
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(manifest.get("llama_build_dir").is_none());
    let local_root = root.path().to_string_lossy();
    for emitted in [
        bytes.as_slice(),
        output.stdout.as_slice(),
        output.stderr.as_slice(),
    ] {
        let text = String::from_utf8_lossy(emitted);
        assert!(
            !text.contains(local_root.as_ref()),
            "writer leaked a runner-local path: {text}"
        );
        assert!(!text.contains("runner private llama build"));
    }
    assert_eq!(
        manifest["library_sha256"],
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(manifest["cargo_profile"], "release");
    assert_eq!(manifest["requirements"], serde_json::json!([]));
    assert!(manifest["llama_patch_digest"].is_null());
}

#[test]
fn l4_sdk_manifest_rejects_mismatched_alias_before_writing() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("lib")).unwrap();
    fs::write(root.path().join("lib/libmeshllm_ffi.dylib"), b"abc").unwrap();
    fs::write(root.path().join("lib/libuniffi_mesh_ffi.dylib"), b"other").unwrap();
    let output = command(&sdk_args(root.path()));
    assert!(!output.status.success());
    assert!(!root.path().join("manifest.json").exists());
}
