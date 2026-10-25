use crate::support::{Case, Legacy, Scratch, TestResult, assert_output};
use std::fs;

const ACTION: &str = ".github/actions/prepare-static-abi-input/action.yml";

#[test]
fn migration_prepared_inputs_abi_cache_cli_retains_exact_bytes_when_mixed() -> TestResult {
    let scratch = Scratch::new("abi-cache-producer")?;
    scratch.write("source", b"CMAKE_HOME_DIRECTORY:PATH=/producer\r\nGGML_OPENMP_ENABLED:BOOL=ON\rOpenMP_C_LIB_NAMES:STRING=gomp")?;
    let destination = scratch.join("filtered");
    let case = Case::same(
        &["static-abi-cache-filter"],
        &["source", "filtered"],
        Legacy::Heredoc(ACTION, 1),
    )
    .writing(&destination);

    let output = case.run(scratch.path())?;

    assert_output(&output, 0, "", "");
    assert_eq!(fs::read(destination)?, b"# Portable MeshLLM static ABI link metadata\nGGML_OPENMP_ENABLED:BOOL=ON\nOpenMP_C_LIB_NAMES:STRING=gomp");
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_cache_cli_rejects_when_source_has_invalid_utf8() -> TestResult {
    let scratch = Scratch::new("abi-cache-invalid")?;
    scratch.write("source", b"\xff")?;
    let case = Case::same(
        &["static-abi-cache-filter"],
        &["source", "filtered"],
        Legacy::Heredoc(ACTION, 1),
    )
    .status_only();

    let output = case.run(scratch.path())?;

    assert_output(
        &output,
        1,
        "",
        "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte\n",
    );
    assert!(!scratch.join("filtered").exists());
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_scan_cli_rejects_when_binary_contains_forbidden_path() -> TestResult
{
    let scratch = Scratch::new("abi-scan-producer")?;
    scratch.write("stage/a/file.a", b"\xff/second\x00/first\xfe")?;
    scratch.write("stage/a.a", b"/second")?;
    let case = Case::same(
        &["static-abi-path-scan"],
        &["stage", "", "/first", "/first", "/second"],
        Legacy::Heredoc(ACTION, 3),
    );

    let output = case.run(scratch.path())?;

    assert_output(
        &output,
        1,
        "",
        "portable static ABI retained producer-local path '/first' in a/file.a\n",
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_scan_cli_refuses_when_stage_is_missing() -> TestResult {
    let scratch = Scratch::new("abi-scan-missing")?;
    let case = Case::same(
        &["static-abi-path-scan"],
        &["missing", "/producer"],
        Legacy::Heredoc(ACTION, 3),
    );

    let output = case.run(scratch.path())?;

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("could not read static ABI stage directory")
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_abi_scan_cli_refuses_stage_file_without_mutation() -> TestResult {
    let scratch = Scratch::new("abi-scan-file")?;
    scratch.write("stage", b"/producer")?;
    let case = Case::same(
        &["static-abi-path-scan"],
        &["stage", "/producer"],
        Legacy::Heredoc(ACTION, 3),
    );
    let output = case.run(scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("could not read static ABI stage directory")
    );
    assert_eq!(fs::read(scratch.join("stage"))?, b"/producer");
    Ok(())
}
