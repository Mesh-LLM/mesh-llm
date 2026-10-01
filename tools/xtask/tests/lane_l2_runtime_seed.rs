use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, process::Command};

type TestResult = Result<(), Box<dyn Error>>;
const IMAGE: &str = "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:f499b79bc52dc7492d57397fdbec9f890c6f6bb1d8c1fcde9c1c97d45c0541a7";
const EPOCH: &str =
    "mesh-llm-cuda-runner-sha256-f499b79bc52dc7492d57397fdbec9f890c6f6bb1d8c1fcde9c1c97d45c0541a7";

fn samples(root: &Path, benefit: bool) -> TestResult {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("root")?;
    for pair in 1..=3 {
        for arm in ["cold", "warm"] {
            let source = repository.join(format!(
                "ci/runtime-seed-evidence/34272984200-1/runtime-seed-evidence-{pair}-{arm}-1"
            ));
            let target = root.join(format!("{pair}-{arm}"));
            fs::create_dir(&target)?;
            let mut result: Value = serde_json::from_slice(&fs::read(source.join("result.json"))?)?;
            let mut raw: Value = serde_json::from_slice(&fs::read(source.join("raw-stats.json"))?)?;
            result["image"] = IMAGE.into();
            result["epoch"] = EPOCH.into();
            result["host_cpu"] = json!({"Model name:":"fixture"});
            result["kernel"] = "Linux fixture x86_64".into();
            result["runner_image_version"] = Value::Null;
            result["total_seconds"] = if arm == "cold" { 20 } else { 10 }.into();
            if benefit && arm == "warm" {
                raw["stats"]["cache_hits"]["counts"] = json!({"C/C++":603});
                raw["stats"]["cache_misses"]["counts"] = json!({"Assembler":139,"Rust":304});
                result["language_hits"] = json!({"C/C++":603});
                result["language_misses"] = json!({"Assembler":139,"Rust":304});
                result["native_hits"] = 603.into();
                result["hit_rate"] = json!(603.0 / 1046.0);
                result["warm_floor_passed"] = true.into();
                result["classification"] = "measured".into();
            }
            fs::write(target.join("result.json"), serde_json::to_vec(&result)?)?;
            fs::write(target.join("raw-stats.json"), serde_json::to_vec(&raw)?)?;
        }
    }
    Ok(())
}

fn summarize(root: &Path) -> TestResult {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci-ops", "runtime-seed", "summarize"])
        .arg(root)
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
    }
    Ok(())
}

#[test]
fn complete_negative_measurements_preserve_floor_and_eligibility() -> TestResult {
    let directory = tempfile::tempdir()?;
    samples(directory.path(), false)?;
    summarize(directory.path())?;
    let summary: Value = serde_json::from_slice(&fs::read(directory.path().join("summary.json"))?)?;
    assert_eq!(summary["classification"], "not-qualified");
    assert_eq!(summary["paired_seconds_saved"], json!([10.0, 10.0, 10.0]));
    assert_eq!(
        summary["reasons"],
        json!(["warm-floor-failure", "no-incremental-c-cpp-coverage"])
    );
    assert_eq!(summary["eligibility_changed"], false);
    Ok(())
}

#[test]
fn comparable_native_hits_and_total_time_report_observed_benefit() -> TestResult {
    let directory = tempfile::tempdir()?;
    samples(directory.path(), true)?;
    summarize(directory.path())?;
    let summary: Value = serde_json::from_slice(&fs::read(directory.path().join("summary.json"))?)?;
    assert_eq!(summary["classification"], "observed-benefit");
    assert_eq!(summary["native_coverage_observed"], true);
    assert_eq!(summary["eligibility_changed"], false);
    Ok(())
}

#[test]
fn corrupted_counter_identity_host_and_missing_samples_fail_closed() -> TestResult {
    for mutation in [
        "missing",
        "duplicate",
        "host",
        "attempt",
        "floor",
        "boolean",
        "counter",
        "image",
        "timing",
    ] {
        let directory = tempfile::tempdir()?;
        samples(directory.path(), true)?;
        let path = directory.path().join("1-warm/result.json");
        let mut value: Value = serde_json::from_slice(&fs::read(&path)?)?;
        match mutation {
            "missing" => fs::remove_file(&path)?,
            "duplicate" => {
                value["pair"] = 2.into();
                fs::write(&path, serde_json::to_vec(&value)?)?;
            }
            "boolean" | "counter" => {
                let raw_path = path.with_file_name("raw-stats.json");
                let mut raw: Value = serde_json::from_slice(&fs::read(&raw_path)?)?;
                if mutation == "boolean" {
                    raw["stats"]["compile_requests"] = true.into();
                } else {
                    raw["stats"]
                        .as_object_mut()
                        .ok_or("stats")?
                        .remove("requests_not_cacheable");
                }
                fs::write(raw_path, serde_json::to_vec(&raw)?)?;
            }
            _ => {
                match mutation {
                    "host" => value["host_cpu"] = json!({"Model name:":"different"}),
                    "attempt" => value["run_attempt"] = "2".into(),
                    "floor" => value["warm_floor_passed"] = false.into(),
                    "image" => value["image"] = "different".into(),
                    "timing" => value["total_seconds"] = (-1).into(),
                    _ => unreachable!(),
                }
                fs::write(&path, serde_json::to_vec(&value)?)?;
            }
        }
        assert!(summarize(directory.path()).is_err(), "{mutation}");
        let summary: Value =
            serde_json::from_slice(&fs::read(directory.path().join("summary.json"))?)?;
        assert_eq!(summary["classification"], "inconclusive", "{mutation}");
        assert_eq!(summary["eligibility_changed"], false);
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn preflight_rejects_populated_measured_cache_before_any_child() -> TestResult {
    let directory = tempfile::tempdir()?;
    let cache = directory.path().join("mesh-llm-sccache");
    fs::create_dir(&cache)?;
    fs::write(cache.join("existing"), "cache")?;
    let evidence = directory.path().join("evidence");
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(directory.path())
        .args(["ci-ops", "runtime-seed", "preflight"])
        .arg(&evidence)
        .env_clear()
        .env("GITHUB_EVENT_NAME", "workflow_dispatch")
        .env("RUNNER_ENVIRONMENT", "github-hosted")
        .env("RUNNER_ARCH", "X64")
        .env(
            "LLAMA_STAGE_BUILD_DIR",
            ".deps/llama.cpp/build-stage-abi-dynamic-cpu",
        )
        .env("RUSTC_WRAPPER", "sccache")
        .env("MESH_LLM_REQUIRE_SCCACHE", "1")
        .env("CARGO_INCREMENTAL", "0")
        .env("CACHE_NAMESPACE", "mesh-llm")
        .env("SCCACHE_GHA_ENABLED", "false")
        .env("SCCACHE_MULTILEVEL_CHAIN", "disk")
        .env("SCCACHE_CACHE_SIZE", "2G")
        .env("LLAMA_STAGE_BACKEND", "cpu")
        .env("RUNNER_TEMP", directory.path())
        .output()?;
    assert!(!output.status.success());
    let failure: Value =
        serde_json::from_slice(&fs::read(evidence.join("preflight-failure.json"))?)?;
    assert_eq!(failure["reason"], "initial compiler cache is not empty");
    assert_eq!(failure["eligibility_changed"], false);
    Ok(())
}

#[test]
fn existing_preflight_directory_is_rejected_without_mutating_evidence() -> TestResult {
    let directory = tempfile::tempdir()?;
    fs::write(directory.path().join("context.json"), "existing evidence")?;
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci-ops", "runtime-seed", "preflight"])
        .arg(directory.path())
        .output()?;
    assert!(!output.status.success());
    assert_eq!(
        fs::read_to_string(directory.path().join("context.json"))?,
        "existing evidence"
    );
    assert!(!directory.path().join("preflight-failure.json").exists());
    Ok(())
}
