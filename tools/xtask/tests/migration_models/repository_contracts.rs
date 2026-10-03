//! Independent current registry/schema/consumer contracts; no model execution.
use crate::support::{TestResult, repository_root};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

fn document(path: &str) -> Value {
    serde_json::from_slice(
        &fs::read(repository_root().join(path)).expect("registry contract source"),
    )
    .expect("valid authored JSON")
}
fn array(value: &Value) -> &[Value] {
    value.as_array().expect("declared array")
}
fn text(value: &Value) -> &str {
    value.as_str().expect("declared string")
}
fn string_set(value: &Value) -> BTreeSet<String> {
    array(value)
        .iter()
        .map(|value| text(value).to_owned())
        .collect()
}
fn manifest(suite: &str) -> Value {
    document(&format!("ci/model-artifacts/manifests/{suite}.json"))
}
fn members(suite: &str) -> BTreeMap<String, Value> {
    array(&manifest(suite)["artifacts"])
        .iter()
        .map(|row| (text(&row["id"]).to_owned(), row.clone()))
        .collect()
}

#[test]
fn model_registry_published_schema_tracks_current_classes_profiles_and_evidence() -> TestResult {
    let roster = document("ci/llama-canary/family-certified.json");
    let schema = document("ci/llama-canary/family-certified.schema.json");
    let model_schema = &schema["$defs"]["model"];
    let required = string_set(&model_schema["required"]);
    assert!(required.contains("class") && required.contains("architecture"));
    let architecture =
        regex::Regex::new(text(&model_schema["properties"]["architecture"]["pattern"]))?;
    for row in array(&roster["models"]) {
        assert!(architecture.is_match(text(&row["architecture"])));
    }
    let classes: BTreeSet<_> = array(&roster["models"])
        .iter()
        .map(|row| text(&row["class"]).to_owned())
        .collect();
    assert_eq!(
        classes,
        string_set(&model_schema["properties"]["class"]["enum"])
    );
    let profiles: BTreeSet<_> = roster["policy"]["profiles"]
        .as_object()
        .expect("profile policy")
        .keys()
        .cloned()
        .collect();
    assert_eq!(
        profiles,
        string_set(&model_schema["properties"]["profile"]["enum"])
    );
    assert_eq!(
        profiles,
        string_set(&schema["properties"]["policy"]["properties"]["profiles"]["required"])
    );
    assert_eq!(
        string_set(&model_schema["properties"]["evidence"]["required"]),
        BTreeSet::from(["fixture".to_owned(), "comparison".to_owned()])
    );
    Ok(())
}

#[test]
fn model_registry_every_current_suite_has_ordered_registry_membership_and_identity() -> TestResult {
    let bytes = fs::read(repository_root().join("ci/model-artifacts/registry.json"))?;
    let registry: Value = serde_json::from_slice(&bytes)?;
    let digest = hex::encode(Sha256::digest(bytes));
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(repository_root().join("ci/model-artifacts/manifests"))? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let generated: Value = serde_json::from_slice(&fs::read(path)?)?;
        let suite = text(&generated["suite"]);
        assert!(
            observed.insert(suite.to_owned()),
            "duplicate generated suite {suite}"
        );
        assert_eq!(generated["registry_sha256"], digest);
        let expected: Vec<_> = array(&registry["artifacts"])
            .iter()
            .filter(|row| array(&row["suites"]).iter().any(|value| value == suite))
            .collect();
        let rows = array(&generated["artifacts"]);
        assert_eq!(rows.len(), expected.len(), "{suite} membership count");
        for (row, source) in rows.iter().zip(expected) {
            assert_eq!(row["id"], source["id"], "{suite} ordered identity");
            for key in ["repo", "revision"] {
                assert_eq!(row[key], source["artifact"][key], "{suite}/{key}");
            }
            let files: Vec<_> = array(&source["artifact"]["files"])
                .iter()
                .map(|file| file["path"].clone())
                .collect();
            assert_eq!(array(&row["files"]), files);
        }
    }
    let mut expected = string_set(&registry["suites"]);
    expected.remove("llama-family-certification");
    assert_eq!(observed, expected);
    Ok(())
}

#[test]
fn model_registry_every_artifact_admits_its_executable_suite_cadences() {
    for (suite, cadences) in [
        ("product-smoke", &["pull-request", "main", "release"][..]),
        (
            "scripted-binary-smoke",
            &["pull-request", "main", "release"],
        ),
        ("sdk-smoke", &["pull-request", "main", "release"]),
        ("hf-download-smoke", &["pull-request", "main", "manual"]),
        ("openai-smoke", &["manual"]),
        ("skippy-correctness", &["pull-request", "main", "manual"]),
        ("safetensors-runtime-smoke", &["pull-request"]),
        ("skippy-ci-smoke", &["manual"]),
        ("skippy-parity", &["manual"]),
        ("competitive-benchmark", &["manual"]),
        ("radix-cache", &["manual"]),
    ] {
        let generated = manifest(suite);
        assert!(!array(&generated["artifacts"]).is_empty());
        for row in array(&generated["artifacts"]) {
            let actual = string_set(&row["cadences"]);
            for cadence in cadences {
                assert!(
                    actual.contains(*cadence),
                    "{suite}/{} lacks {cadence}",
                    row["id"]
                );
            }
        }
    }
}

#[test]
fn model_registry_product_smoke_is_the_exact_pinned_dense_recurrent_laya_set() {
    let rows = members("product-smoke");
    let expected = [
        (
            "smollm2-q8-inference",
            "unsloth/SmolLM2-135M-Instruct-GGUF:Q8_0",
        ),
        (
            "family-granite-hybrid",
            "ibm-granite/granite-4.0-h-350m-GGUF:Q4_K_M",
        ),
        (
            "family-laya-multilingual",
            "meshllm/laya-multilingual-F16-GGUF:F16",
        ),
    ];
    let expected_ids: BTreeSet<_> = expected.iter().map(|row| row.0).collect();
    assert_eq!(
        rows.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        expected_ids
    );
    for (id, model_ref) in expected {
        let row = &rows[id];
        assert_eq!(row["model_ref"], model_ref);
        assert_eq!(array(&row["files"]).len(), 1);
        assert_eq!(
            row["sha256"],
            row["file_integrity"][text(&row["file"])]["blob_id"]
        );
    }
    for suite in ["product-smoke", "scripted-binary-smoke"] {
        assert_eq!(
            manifest(suite)["default_artifact_id"],
            "smollm2-q8-inference"
        );
    }
}

#[test]
fn model_registry_current_family_order_policy_class_and_architecture_match_registry() {
    let registry = document("ci/model-artifacts/registry.json");
    let roster = document("ci/llama-canary/family-certified.json");
    let expected: Vec<_> = array(&registry["artifacts"])
        .iter()
        .filter(|row| {
            array(&row["suites"])
                .iter()
                .any(|suite| suite == "llama-family-certification")
        })
        .collect();
    assert_eq!(roster["policy"], registry["family_policy"]);
    let models = array(&roster["models"]);
    assert_eq!(models.len(), expected.len());
    for (model, source) in models.iter().zip(expected) {
        assert_eq!(model["family"], source["family"]);
        for key in ["class", "architecture"] {
            assert_eq!(model[key], source["certification"][key]);
        }
    }
}

#[test]
fn model_registry_optional_suite_configs_reference_registered_variants() -> TestResult {
    let competitive = members("competitive-benchmark");
    for row in array(&document("skippy/evals/skippy-competitive-benchmark.json")["models"]) {
        let source = &competitive[text(&row["artifact_id"])];
        for (left, right) in [
            ("repo", "repo"),
            ("revision", "revision"),
            ("filename", "file"),
            ("sha256", "sha256"),
        ] {
            assert_eq!(row[left], source[right]);
        }
    }
    let radix = members("radix-cache");
    for row in array(&document("skippy/evals/skippy-radix-cache-models.json")["cases"]) {
        let Some(id) = row.get("artifact_id").filter(|value| !value.is_null()) else {
            continue;
        };
        let source = &radix[text(id)];
        for (left, right) in [
            ("repo", "repo"),
            ("revision", "revision"),
            ("filename", "file"),
        ] {
            assert_eq!(row["source"][left], source[right]);
        }
    }
    let parity = members("skippy-parity");
    for row in array(&document("skippy/docs/llama-parity-candidates.json")["candidates"]) {
        let Some(id) = row.get("artifact_id").filter(|value| !value.is_null()) else {
            continue;
        };
        let source = &parity[text(id)];
        assert_eq!(row["repo"], source["repo"]);
        let patterns = match row.get("include") {
            None => vec!["*.gguf"],
            Some(Value::String(value)) => vec![value.as_str()],
            Some(value) => array(value).iter().map(text).collect(),
        };
        let patterns = patterns
            .into_iter()
            .map(glob::Pattern::new)
            .collect::<Result<Vec<_>, _>>()?;
        for file in array(&source["files"]) {
            assert!(
                patterns.iter().any(|pattern| pattern.matches(text(file))),
                "unselected parity file {file}"
            );
        }
    }
    Ok(())
}

#[test]
fn model_registry_actual_manifest_consumers_declare_authorized_cadence() -> TestResult {
    let invocation = regex::Regex::new(r"models (?:resolve|restore-inputs)")?;
    let cadence = regex::Regex::new(r"--(?:model-)?cadence")?;
    for path in [
        ".github/actions/restore-test-model/action.yml",
        ".github/workflows/ci-rust-tests-slice.yml",
        "scripts/ci-hf-download-smoke.sh",
        "skippy/scripts/materialize-competitive-inputs.sh",
        "skippy/scripts/skippy-ci-smoke.sh",
        "skippy/scripts/skippy-openai-smoke.sh",
    ] {
        let content = fs::read_to_string(repository_root().join(path))?;
        let calls: Vec<_> = invocation.find_iter(&content).collect();
        assert!(!calls.is_empty(), "model consumer missing: {path}");
        for call in calls {
            let args: String = content[call.start()..].chars().take(500).collect();
            assert!(
                cadence.is_match(&args),
                "model consumer omitted cadence: {path}"
            );
        }
    }
    let parity = fs::read_to_string(
        repository_root().join("skippy/scripts/download-skippy-parity-candidates.sh"),
    )?;
    assert!(parity.contains("\"manual\" not in artifact.get(\"cadences\", [])"));
    Ok(())
}

#[test]
fn model_registry_smoke_identity_overrides_use_nonempty_values() -> TestResult {
    let skippy = fs::read_to_string(repository_root().join("skippy/scripts/skippy-ci-smoke.sh"))?;
    let openai =
        fs::read_to_string(repository_root().join("skippy/scripts/skippy-openai-smoke.sh"))?;
    for legacy in ["DENSE_MODEL_REPO+x", "RECURRENT_MODEL_REPO+x"] {
        assert!(!skippy.contains(legacy));
    }
    assert!(!openai.contains("MODEL_REPO+x"));
    for prefix in ["DENSE_MODEL", "RECURRENT_MODEL"] {
        for suffix in ["REPO", "FILE", "SELECTOR", "REVISION", "PATH"] {
            assert!(
                skippy.contains(&format!("${{{prefix}_{suffix}:-}}")),
                "{prefix}_{suffix} must be nonempty"
            );
        }
    }
    for suffix in ["REPO", "FILE", "SELECTOR", "REVISION", "PATH"] {
        assert!(openai.contains(&format!("${{MODEL_{suffix}:-}}")));
    }
    Ok(())
}

#[test]
fn model_registry_hf_smoke_passes_selected_manifest_to_runtime_fixture() -> TestResult {
    let smoke = fs::read_to_string(repository_root().join("scripts/ci-hf-download-smoke.sh"))?;
    let fixture = fs::read_to_string(
        repository_root().join("skippy/crates/skippy-model-hf/tests/hf_download.rs"),
    )?;
    assert!(smoke.contains("export MESH_HF_DOWNLOAD_TEST_MANIFEST=\"$MODEL_MANIFEST\""));
    assert!(fixture.contains("std::env::var_os(\"MESH_HF_DOWNLOAD_TEST_MANIFEST\")"));
    assert!(!fixture.contains("include_str!"));
    Ok(())
}
