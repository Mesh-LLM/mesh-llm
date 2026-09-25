use crate::support::{Case, Edit, Legacy, Scratch, TestResult, assert_output, sha256, snapshot};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

const DESCRIBE: Legacy = Legacy::Heredoc(".github/actions/prepare-static-abi-input/action.yml", 2);
const VERIFY: Legacy = Legacy::Heredoc("scripts/restore-static-abi-input.sh", 1);
const TARGET: &str = "x86_64-unknown-linux-gnu";
const BUILD_DIR: &str = "build-stage-abi-static";
const STAMP: &str = "stamp-version=3\npatched-sha=abc\nbackend=cpu\nlink-mode=static\n\
                     toolchain-epoch=e1\ncmake-arg=-DGGML_NATIVE=OFF\n";

struct Input {
    scratch: Scratch,
    manifest: PathBuf,
    stamp: PathBuf,
}

fn input() -> Result<Input, Box<dyn std::error::Error>> {
    let scratch = Scratch::new("abi-input")?;
    let stamp = scratch.write("stage/.mesh-llm-build-stamp", STAMP.as_bytes())?;
    let manifest = scratch.join("stage/.mesh-llm-static-abi-input.json");
    Ok(Input {
        scratch,
        manifest,
        stamp,
    })
}

fn path(path: &Path) -> &str {
    path.to_str().expect("UTF-8 scratch path")
}

fn describe(
    input: &Input,
    patched_sha: &str,
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let sha_file = input.scratch.write("patched-sha", patched_sha.as_bytes())?;
    let args = [
        path(&input.manifest),
        TARGET,
        "cpu",
        BUILD_DIR,
        "e1",
        path(&sha_file),
        path(&input.stamp),
    ];
    Case::new(
        &[&["static-abi-manifest", "describe"][..], &args].concat(),
        DESCRIBE,
        &args,
    )
    .writing(&input.manifest)
    .run(input.scratch.path())
}

fn verify(input: &Input, target: &str, epoch: &str) -> Case {
    let args = [
        path(&input.manifest),
        path(&input.stamp),
        TARGET,
        "cpu",
        BUILD_DIR,
        epoch,
    ];
    let mut case = Case::new(
        &[&["static-abi-manifest", "verify"][..], &args].concat(),
        VERIFY,
        &args,
    );
    case.args[4] = target.to_owned();
    case.legacy_args[2] = target.to_owned();
    case
}

fn golden() -> String {
    format!(
        "{{\n  \"backend\": \"cpu\",\n  \"build_directory\": \"{BUILD_DIR}\",\n  \
         \"build_stamp_sha256\": \"{}\",\n  \"contract\": \"mesh-llm-static-abi-v3\",\n  \
         \"schema_version\": 3,\n  \"target_triple\": \"{TARGET}\",\n  \"toolchain_epoch\": \"e1\"\n}}\n",
        sha256(STAMP.as_bytes())
    )
}

fn described() -> Result<Input, Box<dyn std::error::Error>> {
    let input = input()?;
    assert_output(&describe(&input, "abc\n")?, 0, "", "");
    Ok(input)
}

fn rewrite(input: &Input, edit: impl FnOnce(&mut Value)) -> TestResult {
    let mut value: Value = serde_json::from_slice(&fs::read(&input.manifest)?)?;
    edit(&mut value);
    fs::write(&input.manifest, serde_json::to_vec(&value)?)?;
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_manifest_describe_writes_exact_bytes() -> TestResult {
    let input = described()?;
    assert_eq!(fs::read_to_string(&input.manifest)?, golden());
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_manifest_describe_requires_patched_sha() -> TestResult {
    let input = input()?;
    let output = describe(&input, " \n")?;
    assert_output(&output, 1, "", "prepared llama.cpp patched SHA is empty\n");
    assert!(!input.manifest.exists());
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_manifest_verify_accepts_matching_input_unchanged() -> TestResult {
    let input = described()?;
    let before = snapshot(input.scratch.path())?;
    assert_output(
        &verify(&input, TARGET, "e1").run(input.scratch.path())?,
        0,
        "",
        "",
    );
    assert_eq!(
        snapshot(input.scratch.path())?,
        before,
        "consumer must not rebuild its input"
    );
    rewrite(&input, |value| value["schema_version"] = json!(3.0))?;
    assert_output(
        &verify(&input, TARGET, "e1").run(input.scratch.path())?,
        0,
        "",
        "",
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_manifest_verify_rejects_changed_identity() -> TestResult {
    let input = described()?;
    let target = "aarch64-unknown-linux-gnu";
    let expected = format!(
        "static ABI manifest target_triple mismatch: expected '{target}', got '{TARGET}'\n"
    );
    assert_output(
        &verify(&input, target, "e1").run(input.scratch.path())?,
        1,
        "",
        &expected,
    );
    let expected = "static ABI manifest toolchain_epoch mismatch: expected 'e2', got 'e1'\n";
    assert_output(
        &verify(&input, TARGET, "e2").run(input.scratch.path())?,
        1,
        "",
        expected,
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_manifest_verify_reports_python_values() -> TestResult {
    let input = described()?;
    let cases: [(Edit, &str); 3] = [
        (
            Box::new(|v| v["schema_version"] = json!("3")),
            "schema_version mismatch: expected 3, got '3'",
        ),
        (
            Box::new(|v| {
                v.as_object_mut().map(|o| o.remove("contract"));
            }),
            "contract mismatch: expected 'mesh-llm-static-abi-v3', got None",
        ),
        (
            Box::new(|v| v["backend"] = json!(["cpu", true, null, 1.5])),
            "backend mismatch: expected 'cpu', got ['cpu', True, None, 1.5]",
        ),
    ];
    for (edit, message) in cases {
        let fresh = described()?;
        rewrite(&fresh, edit)?;
        let expected = format!("static ABI manifest {message}\n");
        assert_output(
            &verify(&fresh, TARGET, "e1").run(fresh.scratch.path())?,
            1,
            "",
            &expected,
        );
    }
    drop(input);
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_manifest_verify_rejects_stamp_checksum_drift() -> TestResult {
    let input = described()?;
    fs::write(
        &input.stamp,
        STAMP.replace("e1", "e1\ncmake-arg=-DEXTRA=ON"),
    )?;
    let expected = "static ABI build stamp checksum mismatch\n";
    assert_output(
        &verify(&input, TARGET, "e1").run(input.scratch.path())?,
        1,
        "",
        expected,
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_manifest_verify_rejects_malformed_json() -> TestResult {
    let input = described()?;
    fs::write(&input.manifest, "{\"schema_version\": 3,")?;
    let output = verify(&input, TARGET, "e1")
        .status_only()
        .run(input.scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    Ok(())
}
