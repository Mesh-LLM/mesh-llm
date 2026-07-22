use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};

use crate::generation_manifest::{
    GLM_DSA_COMPACT_FLASH_MIN_KV, GLM_DSA_DENSE_MASK_MAX_BYTES,
    GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K, GLM_DSA_POLICY_DECODE, GLM_DSA_POLICY_INDEXSHARE,
    GLM_DSA_POLICY_LONG_PREFILL, GLM_DSA_POLICY_PROFILE, GLM_DSA_POLICY_SELECTED_ROW_FLASH,
    GLM_DSA_POLICY_SHORT_PREFILL, GLM_DSA_POLICY_VERIFY, GLM_DSA_SHORT_PREFILL_MAX_TOKENS,
};
use crate::glm_dsa_contract::{self, GlmDsaContractOptions, GlmDsaContractReport};

#[derive(Debug, Serialize)]
struct GlmDsaGenerationPolicyRepairOutput {
    repaired: bool,
    package: String,
    manifest_path: String,
    preserved_speculative_decoding: bool,
    validation: GlmDsaContractReport,
}

pub(crate) fn repair_package(package: &Path, in_place: bool) -> Result<()> {
    ensure!(
        in_place,
        "repair-glm-dsa-generation-policy requires --in-place"
    );
    ensure!(
        package.is_dir(),
        "package must be a directory: {}",
        package.display()
    );

    let base_validation = glm_dsa_contract::validate_path(package)?;
    ensure!(
        base_validation.metadata_errors.is_empty() && base_validation.tensor_errors.is_empty(),
        "cannot repair GLM-DSA generation policy while tensor/metadata errors are present"
    );

    let manifest_path = package.join("model-package.json");
    let original = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let mut manifest: Value = serde_json::from_str(&original)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    let preserved_speculative_decoding = manifest
        .pointer("/generation/speculative_decoding")
        .is_some();
    let repaired = ensure_manifest_generation_policy(&mut manifest)?;
    if repaired {
        write_json_file(&manifest_path, &manifest)?;
    }

    let validation = glm_dsa_contract::validate_path_with_options(
        package,
        GlmDsaContractOptions {
            require_generation_policy: true,
        },
    )?;
    ensure!(
        validation.valid,
        "repaired GLM-DSA package still fails strict generation contract"
    );

    let output = GlmDsaGenerationPolicyRepairOutput {
        repaired,
        package: package.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        preserved_speculative_decoding,
        validation,
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn ensure_manifest_generation_policy(manifest: &mut Value) -> Result<bool> {
    let object = manifest
        .as_object_mut()
        .context("model-package.json root must be a JSON object")?;
    let generation = object
        .entry("generation")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .context("model-package.json generation must be a JSON object")?;

    let expected_policy = expected_policy();
    let expected_thresholds = expected_thresholds();
    let repaired = generation.get("policy") != Some(&expected_policy)
        || generation.get("thresholds") != Some(&expected_thresholds);
    generation.insert("policy".to_string(), expected_policy);
    generation.insert("thresholds".to_string(), expected_thresholds);
    Ok(repaired)
}

fn expected_policy() -> Value {
    json!({
        "profile": GLM_DSA_POLICY_PROFILE,
        "decode": GLM_DSA_POLICY_DECODE,
        "short_prefill": GLM_DSA_POLICY_SHORT_PREFILL,
        "long_prefill": GLM_DSA_POLICY_LONG_PREFILL,
        "verify": GLM_DSA_POLICY_VERIFY,
        "indexshare": GLM_DSA_POLICY_INDEXSHARE,
        "experimental": {
            "selected_row_flash": GLM_DSA_POLICY_SELECTED_ROW_FLASH
        }
    })
}

fn expected_thresholds() -> Value {
    json!({
        "short_prefill_max_tokens": GLM_DSA_SHORT_PREFILL_MAX_TOKENS,
        "direct_sparse_decode_max_top_k": GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K,
        "compact_flash_min_kv": GLM_DSA_COMPACT_FLASH_MIN_KV,
        "dense_mask_max_bytes": GLM_DSA_DENSE_MASK_MAX_BYTES
    })
}

fn write_json_file(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let tmp_path = path.with_file_name(format!(
        ".{}.repair-{}",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("model-package.json"),
        std::process::id()
    ));
    fs::write(&tmp_path, bytes).with_context(|| format!("write {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("replace {} with {}", path.display(), tmp_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserts_generation_policy_and_preserves_speculation() {
        let mut manifest = json!({
            "schema_version": 1,
            "generation": {
                "speculative_decoding": {
                    "default": "native-mtp-n1",
                    "strategies": {
                        "native-mtp-n1": {
                            "type": "native-mtp",
                            "prediction_depth": 1
                        }
                    }
                }
            }
        });

        let repaired = ensure_manifest_generation_policy(&mut manifest).unwrap();

        assert!(repaired);
        assert_eq!(
            manifest.pointer("/generation/policy/profile"),
            Some(&json!("glm-dsa-v1"))
        );
        assert_eq!(
            manifest.pointer("/generation/thresholds/compact_flash_min_kv"),
            Some(&json!(1))
        );
        assert_eq!(
            manifest.pointer("/generation/thresholds/direct_sparse_decode_max_top_k"),
            Some(&json!(256))
        );
        assert_eq!(
            manifest.pointer("/generation/speculative_decoding/default"),
            Some(&json!("native-mtp-n1"))
        );
    }

    #[test]
    fn generation_policy_repair_is_idempotent() {
        let mut manifest = json!({});

        assert!(ensure_manifest_generation_policy(&mut manifest).unwrap());
        assert!(!ensure_manifest_generation_policy(&mut manifest).unwrap());
    }

    #[test]
    fn rejects_non_object_generation() {
        let mut manifest = json!({
            "generation": []
        });

        let error = ensure_manifest_generation_policy(&mut manifest)
            .expect_err("generation array should be rejected");

        assert!(
            error
                .to_string()
                .contains("generation must be a JSON object")
        );
    }
}
