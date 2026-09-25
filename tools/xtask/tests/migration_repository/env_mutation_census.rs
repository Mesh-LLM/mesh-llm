//! `repository env-mutation-census` parity with
//! `scripts/check-env-mutation-contract.py`.

use crate::support::{
    Invocation, Legacy, LegacyKind, Scratch, TestResult, assert_output, repository_root,
};
use std::path::Path;
use std::process::Output;

const SCRIPT: &str = "scripts/check-env-mutation-contract.py";
const AUDITED_FILE: &str = "crates/model-hf/src/store/local.rs";
const TODO: &str =
    "// TODO: Audit that the environment access only happens in single-threaded code.";
const BOOTSTRAP_FILE: &str =
    "crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs";
const HEADER: &str = "environment mutation contract violations:\n";

/// Writes a fixture source, expanding `{SET}` / `{REMOVE}` into the mutation
/// calls. Spelling the calls out here would make this test file itself a
/// mutation site that the real-checkout census rejects as unregistered.
fn write_source(scratch: &Scratch, relative: &str, source: &str) -> TestResult {
    let set = ["std", "env", "set_var"].join("::");
    let remove = ["env", "remove_var"].join("::");
    scratch.write(
        relative,
        &source.replace("{SET}", &set).replace("{REMOVE}", &remove),
    )?;
    Ok(())
}

fn census(root: &Path, files: &[&str]) -> Result<Output, Box<dyn std::error::Error>> {
    let root_arg = root.to_str().ok_or("non-UTF8 root")?;
    let mut args = vec!["--root", root_arg];
    for file in files {
        args.extend_from_slice(&["--file", file]);
    }
    let mut ported = vec!["repository", "env-mutation-census"];
    ported.extend_from_slice(&args);
    Invocation {
        cwd: root,
        args: &ported,
        stdin: None,
        env: &[],
    }
    .run_with_legacy(Legacy {
        kind: LegacyKind::Python,
        script: SCRIPT,
        args: &args,
    })
}

fn violations(lines: &[&str]) -> String {
    lines.iter().fold(HEADER.to_owned(), |report, line| {
        report + "- " + line + "\n"
    })
}

#[test]
fn migration_repository_census_matches_repository_baseline() -> TestResult {
    // Given/When: the real checkout is censused.
    let output = census(&repository_root(), &[])?;
    // Then: the frozen census summary is reported on stdout.
    assert_output(
        &output,
        0,
        "environment mutation contract: discovered 38 Rust files and 237 mutation sites; 22 contract-audited files; unresolved runtime sites remain explicit\n",
        "",
    );
    Ok(())
}

#[test]
fn migration_repository_census_rejects_stale_census() -> TestResult {
    // Given: an unregistered file, a changed frozen count, a bare build
    // script, ignored target/.git copies and no bootstrap caller.
    let scratch = Scratch::new("census-stale")?;
    write_source(
        &scratch,
        "crates/new-crate/src/lib.rs",
        "unsafe { {SET}(\"X\", \"1\") };\n",
    )?;
    write_source(
        &scratch,
        "crates/model-hf/src/cache_paths.rs",
        "fn a() {\n    unsafe { {SET}(\"A\", \"1\") };\n}\n",
    )?;
    write_source(
        &scratch,
        "crates/skippy-protocol/build.rs",
        "fn main() {\n    // SAFETY: fine\n    unsafe { {SET}(\"A\", \"1\") };\n}\n",
    )?;
    write_source(&scratch, "target/x/a.rs", "{SET}(\n")?;
    write_source(&scratch, ".git/a.rs", "{SET}(\n")?;
    // When: the whole root is censused.
    let output = census(scratch.path(), &[])?;
    // Then: each violation, in discovery order, and exit 1.
    assert_output(
        &output,
        1,
        "",
        &violations(&[
            "crates/new-crate/src/lib.rs: unregistered process-environment mutation file (1 sites)",
            "crates/model-hf/src/cache_paths.rs: unaudited mutation census changed from 2 to 1 sites",
            "crates/skippy-protocol/build.rs:3 (main): build-script environment mutation needs a build-script SAFETY comment",
            "crates/mesh-llm/src/main.rs: bootstrap caller is missing",
        ]),
    );
    Ok(())
}

#[test]
fn migration_repository_census_checks_bootstrap_order() -> TestResult {
    // Given: a main.rs that starts the application thread before the mutation.
    let scratch = Scratch::new("census-order")?;
    scratch.write(
        "crates/mesh-llm/src/main.rs",
        "fn main() {\n    run_on_application_thread(|| {});\n    configure_metal_pipeline_cache();\n    tokio::runtime::Builder::new_multi_thread()\n}\n",
    )?;
    let output = census(scratch.path(), &[])?;
    assert_output(
        &output,
        1,
        "",
        &violations(&[
            "crates/mesh-llm/src/main.rs: Metal cache environment mutation must run before application-thread and Tokio runtime construction",
        ]),
    );
    // When: the order is correct, the empty census passes.
    scratch.write(
        "crates/mesh-llm/src/main.rs",
        "fn main() {\n    configure_metal_pipeline_cache();\n    run_on_application_thread(|| {\n        tokio::runtime::Builder::new_multi_thread()\n    });\n}\n",
    )?;
    let output = census(scratch.path(), &[])?;
    assert_output(
        &output,
        0,
        "environment mutation contract: discovered 0 Rust files and 0 mutation sites; 0 contract-audited files; unresolved runtime sites remain explicit\n",
        "",
    );
    Ok(())
}

fn strict(relative: &str, source: &str) -> Result<Output, Box<dyn std::error::Error>> {
    let scratch = Scratch::new("census-strict")?;
    write_source(&scratch, relative, source)?;
    census(scratch.path(), &[relative])
}

#[test]
fn migration_repository_census_strict_test_contracts() -> TestResult {
    let unserialized = strict(
        AUDITED_FILE,
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn mutates() {\n        // SAFETY: this comment cannot replace the required test lock.\n        unsafe { {SET}(\"X\", \"1\") };\n    }\n}\n",
    )?;
    assert_output(
        &unserialized,
        1,
        "",
        &violations(&[
            "crates/model-hf/src/store/local.rs:6 (mutates): test environment mutation is not covered by #[serial]",
        ]),
    );
    let distant = strict(
        AUDITED_FILE,
        "#[cfg(test)]\nmod tests {\n    #[test]\n    #[serial]\n    fn distant() {\n        // SAFETY: this applies only to the first mutation.\n        unsafe { {SET}(\"FIRST\", \"1\") };\n        let _x = true;\n        unsafe { {SET}(\"SECOND\", \"2\") };\n    }\n}\n",
    )?;
    assert_output(
        &distant,
        1,
        "",
        &violations(&[
            "crates/model-hf/src/store/local.rs:9 (distant): test environment mutation needs a SAFETY comment",
        ]),
    );
    let helper = strict(
        AUDITED_FILE,
        "#[cfg(test)]\nmod tests {}\nfn restore_env() {\n    /* SAFETY: callers are\n     * #[serial] tests. */\n    unsafe { {REMOVE}(\"X\") };\n}\n",
    )?;
    assert_output(
        &helper,
        0,
        "environment mutation contract: discovered 1 Rust files and 1 mutation sites; 1 contract-audited files; unresolved runtime sites remain explicit\n",
        "",
    );
    Ok(())
}

#[test]
fn migration_repository_census_strict_runtime_contracts() -> TestResult {
    let deferred = strict(
        "crates/skippy-runtime/src/logging.rs",
        &format!(
            "fn configure_runtime() {{\n    {TODO}\n    unsafe {{ {{SET}}(\"R\", \"1\") }};\n}}\n"
        ),
    )?;
    assert_output(
        &deferred,
        1,
        "",
        &violations(&[
            "crates/skippy-runtime/src/logging.rs:3 (configure_runtime): deferred runtime mutation needs adjacent SAFETY and audit TODO comments",
        ]),
    );
    let stale_todo = strict(
        BOOTSTRAP_FILE,
        &format!(
            "fn configure_metal_pipeline_cache() {{\n    // SAFETY: single-threaded bootstrap before the Tokio runtime.\n    {TODO}\n    unsafe {{ {{SET}}(\"M\", \"1\") }};\n}}\n"
        ),
    )?;
    assert_output(
        &stale_todo,
        1,
        "",
        &violations(&[&format!(
            "{BOOTSTRAP_FILE}:3: stale environment audit TODO remains"
        )]),
    );
    let wrong_function = strict(
        BOOTSTRAP_FILE,
        "fn configure_late() {\n    // SAFETY: generic claim.\n    unsafe { {SET}(\"M\", \"1\") };\n}\n",
    )?;
    assert_output(
        &wrong_function,
        1,
        "",
        &violations(&[
            &format!(
                "{BOOTSTRAP_FILE}:3 (configure_late): bootstrap mutation must remain in configure_metal_pipeline_cache"
            ),
            &format!(
                "{BOOTSTRAP_FILE}:3 (configure_late): bootstrap mutation needs an adjacent SAFETY comment with the single-threaded pre-Tokio ordering guarantee"
            ),
        ]),
    );
    let outside = strict(
        "crates/mesh-llm-system/src/autoupdate.rs",
        "fn update<T>() {\n    // SAFETY: none\n    unsafe { {SET}(\"U\", \"1\") };\n}\n",
    )?;
    assert_output(
        &outside,
        1,
        "",
        &violations(&[
            "crates/mesh-llm-system/src/autoupdate.rs:3 (update): audited mutation is outside a recognized test module",
        ]),
    );
    Ok(())
}

#[test]
fn migration_repository_census_strict_missing_file() -> TestResult {
    let scratch = Scratch::new("census-missing")?;
    let output = census(scratch.path(), &[AUDITED_FILE, "crates/nope.rs"])?;
    assert_output(
        &output,
        1,
        "",
        &violations(&[
            "crates/model-hf/src/store/local.rs: audited source file is missing",
            "crates/nope.rs: audited source file is missing",
        ]),
    );
    Ok(())
}
