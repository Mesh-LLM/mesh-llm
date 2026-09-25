use super::script_source_calls::check_script_source_calls;
use super::shard_rows::ScriptSourceCall;
use crate::command::DynResult;
use sha2::Digest;
use std::fs;

#[test]
fn migration_inventory_rejects_unowned_continued_interpreter_probe() -> DynResult<()> {
    // Given a version probe on a tokenless continuation line.
    let root = crate::command::unique_temp_dir("migration-script-continuation");
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(
        root.join("scripts/package-native-runtime.sh"),
        "python_bin() {\n  for candidate in python3 python; do\n    if command -v \"$candidate\" &&\n      \"$candidate\" -c 'import sys'; then\n      :\n    fi\n  done\n}\n",
    )?;
    // When no independent source-call record owns it.
    let error = check_script_source_calls(&root, &[]).unwrap_err();
    // Then the continuation is rejected even though the token scanner misses it.
    assert!(error.to_string().contains("unowned"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn migration_inventory_accepts_source_bound_independent_call() -> DynResult<()> {
    // Given an independently reviewed CMake discovery call.
    let root = crate::command::unique_temp_dir("migration-cmake-source-call");
    fs::create_dir_all(root.join("tools/skippy-stage-rewriter"))?;
    let line = "find_package(Python3 REQUIRED COMPONENTS Interpreter)";
    fs::write(
        root.join("tools/skippy-stage-rewriter/CMakeLists.txt"),
        line,
    )?;
    let digest = hex::encode(sha2::Sha256::digest(line.as_bytes()));
    let record = ScriptSourceCall {
        id: format!(
            "tools/skippy-stage-rewriter/CMakeLists.txt:1:{}",
            &digest[..16]
        ),
        context: "CMake configure-time Python3 Interpreter discovery".to_owned(),
        target: "Python3 Interpreter".to_owned(),
        boundary: "BUILD_TESTING".to_owned(),
        owner: "native test".to_owned(),
        replacement: "native CTest".to_owned(),
        deletion_condition: "native parity".to_owned(),
    };
    // When the exact source call is reconciled.
    check_script_source_calls(&root, &[record])?;
    // Then the source-bound record is accepted without consulting scanner output.
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn migration_inventory_rejects_changed_cmake_continued_test_target() -> DynResult<()> {
    // Given a CTest interpreter call whose target no longer names the reviewed fixture.
    let root = crate::command::unique_temp_dir("migration-cmake-target");
    fs::create_dir_all(root.join("tools/skippy-stage-rewriter"))?;
    fs::write(
        root.join("tools/skippy-stage-rewriter/CMakeLists.txt"),
        "add_test(\n COMMAND\n ${Python3_EXECUTABLE}\n ${CMAKE_CURRENT_SOURCE_DIR}/tests/other.py)\n",
    )?;
    // When its continued argument is validated.
    let error = check_script_source_calls(&root, &[]).unwrap_err();
    // Then the wrong CTest target is rejected rather than treated as data.
    assert!(error.to_string().contains("CTest"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn migration_inventory_rejects_duplicate_independent_call() -> DynResult<()> {
    // Given two records claiming one CMake source call.
    let root = crate::command::unique_temp_dir("migration-duplicate-source-call");
    fs::create_dir_all(root.join("tools/skippy-stage-rewriter"))?;
    let line = "find_package(Python3 REQUIRED COMPONENTS Interpreter)";
    fs::write(
        root.join("tools/skippy-stage-rewriter/CMakeLists.txt"),
        line,
    )?;
    let digest = hex::encode(sha2::Sha256::digest(line.as_bytes()));
    let record = ScriptSourceCall {
        id: format!(
            "tools/skippy-stage-rewriter/CMakeLists.txt:1:{}",
            &digest[..16]
        ),
        context: "CMake configure-time Python3 Interpreter discovery".to_owned(),
        target: "Python3 Interpreter".to_owned(),
        boundary: "BUILD_TESTING".to_owned(),
        owner: "native test".to_owned(),
        replacement: "native CTest".to_owned(),
        deletion_condition: "native parity".to_owned(),
    };
    let duplicate = ScriptSourceCall {
        id: record.id.clone(),
        context: record.context.clone(),
        target: record.target.clone(),
        boundary: record.boundary.clone(),
        owner: record.owner.clone(),
        replacement: record.replacement.clone(),
        deletion_condition: record.deletion_condition.clone(),
    };
    // When the records are checked against the single call.
    let error = check_script_source_calls(&root, &[record, duplicate]).unwrap_err();
    // Then one call cannot be counted twice.
    assert!(error.to_string().contains("duplicate"), "{error}");
    fs::remove_dir_all(root)?;
    Ok(())
}
