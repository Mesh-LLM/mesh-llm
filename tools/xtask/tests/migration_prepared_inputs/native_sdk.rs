use crate::support::{Case, Edit, Legacy, Scratch, TestResult, assert_output, sha256, snapshot};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

const VERIFIER: Legacy = Legacy::Heredoc("scripts/verify-native-sdk-package.sh", 1);
const IDENTITY: Legacy = Legacy::Heredoc(".github/actions/prepare-native-sdk-input/action.yml", 1);
const RESTORE_IDENTITY: Legacy = Legacy::Heredoc("scripts/restore-native-sdk-input.sh", 1);
const KOTLIN: Legacy = Legacy::Heredoc("scripts/ci-kotlin-sdk-smoke.sh", 1);
const ARTIFACT: &str = "meshllm-native-linux-x86_64-cpu";
const LIBRARY: &[u8] = b"native-lib";

fn manifest() -> Value {
    json!({
        "schema_version": 1,
        "artifact_id": ARTIFACT,
        "native_runtime_id": ARTIFACT,
        "sdk_version": "1.2.3",
        "mesh_version": "1.2.3",
        "target_triple": "x86_64-unknown-linux-gnu",
        "platform": "linux-x86_64",
        "os": "linux",
        "arch": "x86_64",
        "backend": "cpu",
        "flavor": "cpu",
        "cargo_profile": "release",
        "library": "lib/libmesh_llm_ffi.so",
        "uniffi_library": "lib/libmesh_llm_ffi.so",
        "library_paths": ["lib/libmesh_llm_ffi.so"],
        "library_sha256": sha256(LIBRARY),
        "requirements": [],
        "features": ["mesh-inference", "model-management", "local-serving", "chat", "responses"],
    })
}

/// Writes one artifact directory with `manifest` and returns its path.
fn artifact(scratch: &Scratch, manifest: &Value) -> Result<PathBuf, Box<dyn std::error::Error>> {
    scratch.write(&format!("{ARTIFACT}/lib/libmesh_llm_ffi.so"), LIBRARY)?;
    scratch.write(
        &format!("{ARTIFACT}/manifest.json"),
        &serde_json::to_vec(manifest)?,
    )?;
    Ok(scratch.join(ARTIFACT))
}

fn verify(
    scratch: &Scratch,
    dir: &Path,
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let dir = dir.to_str().ok_or("path")?;
    let manifest = format!("{dir}/manifest.json");
    Case::same(&["native-sdk-manifest"], &[dir, &manifest], VERIFIER).run(scratch.path())
}

fn verify_edit(
    edit: impl FnOnce(&mut Value),
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let scratch = Scratch::new("sdk-manifest")?;
    let mut value = manifest();
    edit(&mut value);
    let dir = artifact(&scratch, &value)?;
    let before = snapshot(scratch.path())?;
    let output = verify(&scratch, &dir)?;
    assert_eq!(
        snapshot(scratch.path())?,
        before,
        "verification must not modify the input"
    );
    Ok(output)
}

#[test]
fn migration_prepared_inputs_sdk_manifest_accepts_matching_artifact() -> TestResult {
    assert_output(&verify_edit(|_| {})?, 0, "", "");
    assert_output(
        &verify_edit(|value| value["schema_version"] = json!(1.0))?,
        0,
        "",
        "",
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_manifest_rejects_field_contract_violations() -> TestResult {
    let sha = sha256(LIBRARY);
    let cases: Vec<(Edit, String)> = vec![
        (Box::new(|v| { v.as_object_mut().map(|o| (o.remove("os"), o.remove("features"))); }),
            "missing manifest field(s): os, features".into()),
        (Box::new(|v| v["schema_version"] = json!("1")), "unsupported schema_version: '1'".into()),
        (Box::new(|v| v["sdk_version"] = json!(3)), "sdk_version must be a non-empty string".into()),
        (Box::new(|v| v["flavor"] = json!("metal")),
            "artifact_id does not match platform/flavor: meshllm-native-linux-x86_64-cpu != meshllm-native-linux-x86_64-metal".into()),
        (Box::new(|v| v["mesh_version"] = json!("1.2.4")), "mesh_version must match sdk_version: 1.2.4 != 1.2.3".into()),
        (Box::new(|v| v["target_triple"] = json!("riscv64gc-unknown-linux-gnu")),
            "unsupported target_triple: riscv64gc-unknown-linux-gnu".into()),
        (Box::new(|v| v["arch"] = json!("aarch64")), "arch does not match target_triple: aarch64 != x86_64".into()),
        (Box::new(|v| v["backend"] = json!("tpu")), "unsupported native SDK backend: tpu".into()),
        (Box::new(|v| v["library_paths"] = json!(["lib/other.so"])), "library_paths must include the primary library".into()),
        (Box::new(|v| v["requirements"] = json!({})), "requirements must be a list".into()),
        (Box::new(|v| v["library_sha256"] = json!("0")),
            format!("library_sha256 mismatch for lib/libmesh_llm_ffi.so: {sha} != 0")),
        (Box::new(|v| v["features"] = json!(["chat", ""])), "features must be a list of non-empty strings".into()),
        (Box::new(|v| v["features"] = json!(["chat"])), "missing feature marker: mesh-inference".into()),
    ];
    for (edit, expected) in cases {
        assert_output(&verify_edit(edit)?, 1, "", &format!("{expected}\n"));
    }
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_manifest_rejects_escaping_library_paths() -> TestResult {
    for (raw, expected) in [
        (
            json!(""),
            "library_paths entry path must be a non-empty string",
        ),
        (
            json!("lib\\x.so"),
            "library_paths entry path must use forward slashes inside the artifact: lib\\x.so",
        ),
        (
            json!("../x.so"),
            "library_paths entry must be a relative path inside the artifact: ../x.so",
        ),
        (
            json!("/x.so"),
            "library_paths entry must be a relative path inside the artifact: /x.so",
        ),
        (
            json!("C:x.so"),
            "library_paths entry must be a relative path inside the artifact: C:x.so",
        ),
    ] {
        let output = verify_edit(|v| v["library_paths"] = json!(["lib/libmesh_llm_ffi.so", raw]))?;
        assert_output(&output, 1, "", &format!("{expected}\n"));
    }
    let scratch = Scratch::new("sdk-manifest-missing")?;
    let mut value = manifest();
    value["library_paths"] = json!(["lib/libmesh_llm_ffi.so", "lib/./absent.so"]);
    let dir = artifact(&scratch, &value)?;
    let expected = format!(
        "missing library_paths entry: {}/lib/absent.so\n",
        dir.display()
    );
    assert_output(&verify(&scratch, &dir)?, 1, "", &expected);
    Ok(())
}

#[cfg(unix)]
#[test]
fn migration_prepared_inputs_sdk_manifest_rejects_symlink_escape() -> TestResult {
    let scratch = Scratch::new("sdk-manifest-link")?;
    let dir = artifact(&scratch, &manifest())?;
    let outside = scratch.write("outside.so", LIBRARY)?;
    fs::remove_file(dir.join("lib/libmesh_llm_ffi.so"))?;
    crate::support::symlink(&outside, &dir.join("lib/libmesh_llm_ffi.so"))?;
    let expected =
        "library_paths entry path resolves outside the artifact: lib/libmesh_llm_ffi.so\n";
    assert_output(&verify(&scratch, &dir)?, 1, "", expected);
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_manifest_checks_uniffi_and_library_kind() -> TestResult {
    let scratch = Scratch::new("sdk-uniffi")?;
    let mut value = manifest();
    value["uniffi_library"] = json!("lib/uniffi.so");
    scratch.write(&format!("{ARTIFACT}/lib/uniffi.so"), b"other")?;
    let dir = artifact(&scratch, &value)?;
    let expected = format!(
        "uniffi_library checksum mismatch: {} != {}\n",
        sha256(b"other"),
        sha256(LIBRARY)
    );
    assert_output(&verify(&scratch, &dir)?, 1, "", &expected);
    let scratch = Scratch::new("sdk-kind")?;
    let mut value = manifest();
    value["library"] = json!("lib/libmesh.dylib");
    value["library_paths"] = json!(["lib/libmesh.dylib"]);
    scratch.write(&format!("{ARTIFACT}/lib/libmesh.dylib"), LIBRARY)?;
    let dir = artifact(&scratch, &value)?;
    let expected = "linux-x86_64 artifact must contain a .so: libmesh.dylib\n";
    assert_output(&verify(&scratch, &dir)?, 1, "", expected);
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_manifest_rejects_renamed_directory_and_malformed_json()
-> TestResult {
    let scratch = Scratch::new("sdk-rename")?;
    let dir = artifact(&scratch, &manifest())?;
    let renamed = scratch.join("meshllm-native-other");
    fs::rename(&dir, &renamed)?;
    let expected = "artifact directory name does not match artifact_id: meshllm-native-other != meshllm-native-linux-x86_64-cpu\n";
    assert_output(&verify(&scratch, &renamed)?, 1, "", expected);
    fs::write(renamed.join("manifest.json"), "{\"schema_version\": 1,")?;
    let dir = renamed.to_str().ok_or("path")?;
    let manifest = format!("{dir}/manifest.json");
    let output = Case::same(&["native-sdk-manifest"], &[dir, &manifest], VERIFIER)
        .status_only()
        .run(scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    Ok(())
}

fn identity(legacy: Legacy, manifest: &Path, expected: [&str; 3]) -> Case {
    let manifest = manifest.to_str().expect("UTF-8 path");
    let args = [manifest, expected[0], expected[1], expected[2]];
    Case::same(&["native-sdk-identity"], &args, legacy)
}

#[test]
fn migration_prepared_inputs_sdk_identity_rejects_mismatched_prepared_input() -> TestResult {
    let scratch = Scratch::new("sdk-identity")?;
    let dir = artifact(&scratch, &manifest())?;
    let path = dir.join("manifest.json");
    for legacy in [IDENTITY, RESTORE_IDENTITY] {
        let ok = identity(
            legacy,
            &path,
            ["x86_64-unknown-linux-gnu", "cpu", "release"],
        );
        assert_output(&ok.run(scratch.path())?, 0, "", "");
        let wrong = identity(
            legacy,
            &path,
            ["x86_64-unknown-linux-gnu", "cuda", "release"],
        );
        let expected = "native SDK manifest backend mismatch: expected 'cuda', got 'cpu'\n";
        assert_output(&wrong.run(scratch.path())?, 1, "", expected);
    }
    let mut value = manifest();
    value
        .as_object_mut()
        .ok_or("object")?
        .remove("cargo_profile");
    fs::write(&path, serde_json::to_vec(&value)?)?;
    let missing = identity(
        IDENTITY,
        &path,
        ["x86_64-unknown-linux-gnu", "cpu", "debug"],
    );
    let expected = "native SDK manifest cargo_profile mismatch: expected 'debug', got None\n";
    assert_output(&missing.run(scratch.path())?, 1, "", expected);
    Ok(())
}

#[test]
fn migration_prepared_inputs_sdk_library_dir_prefers_uniffi_library() -> TestResult {
    let scratch = Scratch::new("sdk-library-dir")?;
    for (uniffi, library, expected) in [
        (json!("jna/libuniffi.so"), "lib/libmesh.so", "jna\n"),
        (json!(""), "lib/deep//libmesh.so", "lib/deep\n"),
        (Value::Null, "libmesh.so", "\n"),
    ] {
        let path = scratch.write(
            "manifest.json",
            &serde_json::to_vec(&json!({"uniffi_library": uniffi, "library": library}))?,
        )?;
        let case = Case::same(
            &["native-sdk-library-dir"],
            &[path.to_str().ok_or("path")?],
            KOTLIN,
        );
        assert_output(&case.run(scratch.path())?, 0, expected, "");
    }
    Ok(())
}
