//! `models generate` parity with `scripts/generate-test-model-manifests.py`:
//! write-mode projection bytes, stale detection and every registry rejection.

use crate::support::{
    GENERATOR, Stage, TestResult, assert_same, assert_streams, code, field, fixture, fixture_dir,
    legacy, run_legacy, xtask,
};
use serde_json::Value;

const REGISTRY: &str = "ci/model-artifacts/registry.json";

/// serde_json's parser wording replaces Python's `JSONDecodeError` text.
const EXPLAINED: &[&str] = &["registry-invalid-json"];

fn small_registry() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Ok(std::fs::read(fixture_dir().join("registry-small.json"))?)
}

/// Applies recorded `{set,value}` / `{delete}` path operations.
fn apply(mut registry: Value, ops: &[Value]) -> Result<Value, Box<dyn std::error::Error>> {
    for op in ops {
        let (path, deleting) = match (op.get("set"), op.get("delete")) {
            (Some(path), _) => (path, false),
            (None, Some(path)) => (path, true),
            (None, None) => return Err("unknown registry op".into()),
        };
        let steps = path.as_array().ok_or("op path must be an array")?;
        let (last, parents) = match steps.split_last() {
            Some(split) => split,
            None => {
                registry = op["value"].clone();
                continue;
            }
        };
        let mut target = &mut registry;
        for step in parents {
            target = match step {
                Value::String(key) => target.get_mut(key.as_str()),
                Value::Number(index) => {
                    target.get_mut(usize::try_from(index.as_u64().ok_or("index")?)?)
                }
                _ => None,
            }
            .ok_or("op path does not exist")?;
        }
        match (last, deleting) {
            (Value::String(key), true) => {
                target.as_object_mut().ok_or("not an object")?.remove(key);
            }
            (Value::String(key), false) => {
                target
                    .as_object_mut()
                    .ok_or("not an object")?
                    .insert(key.clone(), op["value"].clone());
            }
            (Value::Number(index), false) => {
                let index = usize::try_from(index.as_u64().ok_or("index")?)?;
                *target.get_mut(index).ok_or("index out of range")? = op["value"].clone();
            }
            _ => return Err("unsupported op".into()),
        }
    }
    Ok(registry)
}

fn stage_registry(stage: &Stage, case: &Value) -> TestResult {
    let _fresh = std::fs::remove_dir_all(stage.path().join("ci"));
    std::fs::create_dir_all(stage.path().join("ci/llama-canary"))?;
    let registry = &case["registry"];
    if let Some(text) = registry["text"].as_str() {
        return stage.write(REGISTRY, text.as_bytes());
    }
    if registry["absent"].as_bool() == Some(true) {
        return Ok(());
    }
    let ops = registry["ops"].as_array().ok_or("ops must be an array")?;
    let base = serde_json::from_slice(&small_registry()?)?;
    stage.write(REGISTRY, &serde_json::to_vec_pretty(&apply(base, ops)?)?)
}

fn case_args(case: &Value) -> Result<Vec<&str>, Box<dyn std::error::Error>> {
    let args = case["args"].as_array().ok_or("args must be an array")?;
    args.iter()
        .map(|arg| arg.as_str().ok_or_else(|| "arg must be a string".into()))
        .collect()
}

#[test]
fn migration_models_generator_rejections_and_staleness_match_legacy() -> TestResult {
    // Given: every captured registry case in an otherwise empty checkout.
    let stage = Stage::new("generator")?;
    let cases = fixture("generator-cases.json")?;
    let cases = cases.as_array().ok_or("cases must be an array")?;
    let python = legacy("MIGRATION_MODELS_LEGACY_PYTHON");
    let script = python
        .as_ref()
        .map(|_| stage.legacy_script(GENERATOR))
        .transpose()?;
    for case in cases {
        let name = field(case, "name")?;
        stage_registry(&stage, case)?;
        let args = case_args(case)?;
        let mut ported_args = vec!["models", "generate"];
        ported_args.extend_from_slice(&args);
        // When: the ported generator runs with the same argv.
        let ported = xtask(stage.path(), &ported_args)?;
        // Then: status and both streams equal the legacy capture.
        let stderr = stage.expand(field(case, "stderr")?)?;
        if EXPLAINED.contains(&name) {
            assert_eq!(ported.status.code(), Some(code(case)?), "{name}: status");
            let actual = String::from_utf8_lossy(&ported.stderr);
            assert!(
                actual.starts_with("test-model registry error: "),
                "{name}: {actual}"
            );
        } else {
            assert_streams(name, &ported, code(case)?, field(case, "stdout")?, &stderr);
        }
        if let (Some(python), Some(script)) = (&python, &script) {
            let original = run_legacy(python, script, stage.path(), &args)?;
            if !EXPLAINED.contains(&name) {
                assert_same(name, &original, &ported);
            }
        }
    }
    assert_eq!(cases.len(), 85);
    Ok(())
}

#[test]
fn migration_models_generator_writes_legacy_projection_bytes() -> TestResult {
    // Given: a registry covering every suite, family evidence, mmproj
    // artifacts, multi-file rows, quantizations, floats and non-ASCII text.
    let stage = Stage::new("generate")?;
    stage.write(REGISTRY, &small_registry()?)?;
    // When: the generator writes its outputs.
    let written = xtask(stage.path(), &["models", "generate"])?;
    // Then: every projected file equals the legacy bytes.
    assert_streams("write", &written, 0, "", "");
    let expected = fixture("generated-small.json")?;
    let expected = expected
        .as_array()
        .ok_or("generated files must be an array")?;
    assert_eq!(expected.len(), 12);
    for file in expected {
        let path = field(file, "path")?;
        assert_eq!(stage.read(path)?, field(file, "text")?, "{path}");
    }
    // When/Then: a fresh projection passes the check silently.
    assert_streams(
        "fresh",
        &xtask(stage.path(), &["models", "generate", "--check"])?,
        0,
        "",
        "",
    );
    Ok(())
}

#[test]
fn migration_models_generator_check_names_only_the_stale_output() -> TestResult {
    // Given: a current projection with one hand-edited suite manifest.
    let stage = Stage::new("stale")?;
    stage.write(REGISTRY, &small_registry()?)?;
    assert_eq!(
        xtask(stage.path(), &["models", "generate"])?.status.code(),
        Some(0)
    );
    let edited = "ci/model-artifacts/manifests/sdk-smoke.json";
    let text = stage.read(edited)?.replace("\"release\"", "\"nightly\"");
    stage.write(edited, text.as_bytes())?;
    // When: the check runs.
    let output = xtask(stage.path(), &["models", "generate", "--check"])?;
    // Then: exactly the edited output is reported stale and nothing is rewritten.
    let message = format!("generated test-model manifests are stale:\n  {edited}\n");
    assert_streams("stale", &output, 1, "", &message);
    assert_eq!(stage.read(edited)?, text);
    Ok(())
}
