use std::{fs, path::Path, process::Command};

use super::TestResult;

fn workflow() -> Result<String, Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("repository root")?;
    Ok(fs::read_to_string(
        root.join(".github/workflows/ci-quality-slice.yml"),
    )?)
}

fn admission_script() -> Result<String, Box<dyn std::error::Error>> {
    let workflow = workflow()?;
    let step = workflow
        .split("      - name: Admit standalone API source\n")
        .nth(1)
        .ok_or("source-admission step")?
        .split("      - name: Verify C++ API toolchain\n")
        .next()
        .ok_or("step boundary")?;
    let script = step.split("        run: |\n").nth(1).ok_or("script")?;
    script
        .lines()
        .map(|line| {
            line.strip_prefix("          ")
                .map(str::to_owned)
                .ok_or_else(|| "invalid shell indentation".into())
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()
        .map(|lines| lines.join("\n"))
}

fn admit(root: &Path) -> Result<(bool, String), Box<dyn std::error::Error>> {
    let output_path = root.join("output");
    fs::write(&output_path, [])?;
    let result = Command::new("bash")
        .args(["-c", &admission_script()?])
        .current_dir(root)
        .env("GITHUB_OUTPUT", &output_path)
        .env("GITHUB_STEP_SUMMARY", root.join("summary"))
        .output()?;
    Ok((
        result.status.success(),
        fs::read_to_string(output_path).unwrap_or_default(),
    ))
}

fn install_source(root: &Path) -> TestResult {
    for marker in [
        "skippy/skippy_api/CMakeLists.txt",
        "skippy/skippy-api-rs/Cargo.toml",
        "skippy/skippy-api-rs/skippy-sys/Cargo.toml",
        "skippy/skippy-api-rs/skippy/Cargo.toml",
    ] {
        let path = root.join(marker);
        fs::create_dir_all(path.parent().ok_or("marker parent")?)?;
        fs::write(path, "fixture\n")?;
    }
    Ok(())
}

#[test]
fn api_source_admission_distinguishes_absent_complete_and_partial_sources() -> TestResult {
    let root = tempfile::tempdir()?;
    assert_eq!(admit(root.path())?, (true, "present=false\n".into()));
    fs::create_dir_all(root.path().join("skippy/skippy_api"))?;
    assert!(!admit(root.path())?.0, "a partial source must fail");
    install_source(root.path())?;
    assert_eq!(admit(root.path())?, (true, "present=true\n".into()));
    fs::remove_file(root.path().join("skippy/skippy-api-rs/skippy/Cargo.toml"))?;
    assert!(!admit(root.path())?.0, "a missing nested crate must fail");
    Ok(())
}

#[test]
fn api_source_admission_rejects_symlinked_sources_and_manifests() -> TestResult {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir()?;
    install_source(root.path())?;
    let manifest = root.path().join("skippy/skippy-api-rs/Cargo.toml");
    fs::remove_file(&manifest)?;
    symlink("missing-target", &manifest)?;
    assert!(!admit(root.path())?.0);
    fs::remove_file(manifest)?;
    fs::remove_dir_all(root.path().join("skippy/skippy-api-rs"))?;
    symlink("missing-target", root.path().join("skippy/skippy-api-rs"))?;
    assert!(!admit(root.path())?.0);
    Ok(())
}

#[test]
fn api_validation_requires_every_suite_and_all_three_model_fixtures() -> TestResult {
    let workflow = workflow()?;
    let job = workflow
        .split("  skippy_cpp_api:\n")
        .nth(1)
        .ok_or("API job")?
        .split("  rust_fmt:\n")
        .next()
        .ok_or("job boundary")?;
    for command in [
        "just llama-prepare",
        "just skippy-llama-api-format-check",
        "just skippy-api-all-targets-build",
        "target/skippy-api-all/skippy_api_stage_pipeline_probe \"$CAUSAL_MODEL\" 3 30 'the quick brown fox jumps over the lazy dog' 16",
        "just skippy-llama-api-test",
        "just skippy-api-asan",
        "just skippy-api-rs-lint",
        "just skippy-api-downstream-smoke",
        "just skippy-tidy-plugin-build",
        "just skippy-llama-api-tidy '*'",
        "just skippy-api-rs-e2e \"$EMBED_MODEL\" \"$RERANK_MODEL\" \"$CAUSAL_MODEL\"",
    ] {
        assert!(
            job.contains(command),
            "missing required validation: {command}"
        );
    }
    for fixture in [
        "minilm-q8-cpp-api",
        "bge-reranker-q4-cpp-api",
        "smollm2-q8-inference",
    ] {
        assert!(job.contains(&format!("model_artifact_id: {fixture}")));
    }
    assert!(job.contains("SKIPPY_TEST_REQUIRE_MODELS: '1'"));
    let witness = job
        .split("      - name: Verify a real three-stage causal pipeline\n")
        .nth(1)
        .ok_or("three-stage witness")?
        .split("      - name: Run the complete C++ API suite\n")
        .next()
        .ok_or("witness boundary")?;
    assert!(witness.contains("if: steps.api_source.outputs.present == 'true'"));
    assert!(witness.contains("CAUSAL_MODEL: ${{ steps.api_causal.outputs.model_path }}"));
    assert!(witness.contains("test -s \"$CAUSAL_MODEL\""));
    assert!(
        job.find("just skippy-api-all-targets-build")
            < job.find("target/skippy-api-all/skippy_api_stage_pipeline_probe")
    );
    assert!(job.contains("needs.runner_policy.outputs.runner_8"));
    assert!(job.contains("needs: [runner_policy, quality_contracts]"));
    assert!(job.contains("fromJson(inputs.authority_linux_x64).binary_sha256"));
    assert!(!job.contains("continue-on-error"));
    assert!(!job.contains("save_model_cache: 'true'"));
    Ok(())
}
