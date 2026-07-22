use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use model_ref::split_gguf_shard_info;
use serde::{Deserialize, Serialize};
use skippy_runtime::ModelInfo;

use crate::generation_manifest::{
    GLM_DSA_COMPACT_FLASH_MIN_KV, GLM_DSA_DENSE_MASK_MAX_BYTES,
    GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K, GLM_DSA_POLICY_DECODE, GLM_DSA_POLICY_INDEXSHARE,
    GLM_DSA_POLICY_LONG_PREFILL, GLM_DSA_POLICY_PROFILE, GLM_DSA_POLICY_SELECTED_ROW_FLASH,
    GLM_DSA_POLICY_SHORT_PREFILL, GLM_DSA_POLICY_VERIFY, GLM_DSA_SHORT_PREFILL_MAX_TOKENS,
    PackageGeneration, PackageGenerationThresholds,
};
#[cfg(test)]
use crate::generation_manifest::{PackageGenerationExperimentalPolicy, PackageGenerationPolicy};

const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_ARRAY_DEPTH: usize = 64;
const MAX_GGUF_HEADER_KV_COUNT: u64 = 1_000_000;
const MAX_GGUF_TENSOR_COUNT: u64 = 1_000_000;

const REQUIRED_U32_METADATA: &[&str] = &[
    "glm-dsa.context_length",
    "glm-dsa.embedding_length",
    "glm-dsa.block_count",
    "glm-dsa.feed_forward_length",
    "glm-dsa.attention.head_count",
    "glm-dsa.attention.head_count_kv",
    "glm-dsa.attention.key_length",
    "glm-dsa.attention.value_length",
    "glm-dsa.attention.q_lora_rank",
    "glm-dsa.attention.kv_lora_rank",
    "glm-dsa.rope.dimension_count",
    "glm-dsa.expert_count",
    "glm-dsa.expert_used_count",
    "glm-dsa.expert_shared_count",
    "glm-dsa.expert_feed_forward_length",
    "glm-dsa.leading_dense_block_count",
    "glm-dsa.attention.indexer.head_count",
    "glm-dsa.attention.indexer.key_length",
    "glm-dsa.attention.indexer.top_k",
];

const REQUIRED_F32_METADATA: &[&str] = &[
    "glm-dsa.attention.layer_norm_rms_epsilon",
    "glm-dsa.expert_weights_scale",
];

const REQUIRED_BOOL_METADATA: &[&str] = &["glm-dsa.expert_weights_norm"];

const INDEXER_TENSORS: &[&str] = &[
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.proj.weight",
    "indexer.attn_k.weight",
    "indexer.attn_q_b.weight",
];

const BASE_LAYER_TENSORS: &[&str] = &[
    "attn_norm.weight",
    "attn_q_a_norm.weight",
    "attn_kv_a_norm.weight",
    "attn_q_a.weight",
    "attn_q_b.weight",
    "attn_kv_a_mqa.weight",
    "attn_k_b.weight",
    "attn_v_b.weight",
    "attn_output.weight",
    "ffn_norm.weight",
];

const DENSE_FFN_TENSORS: &[&str] = &["ffn_gate.weight", "ffn_down.weight", "ffn_up.weight"];

const MOE_LAYER_TENSORS: &[&str] = &[
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_down_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_shexp.weight",
    "ffn_down_shexp.weight",
    "ffn_up_shexp.weight",
];

const NEXTN_TENSORS: &[&str] = &[
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
];

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GlmDsaContractOptions {
    pub(crate) require_generation_policy: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct GlmDsaContractReport {
    pub(crate) valid: bool,
    pub(crate) path: String,
    pub(crate) artifact_kind: String,
    pub(crate) gguf_files: Vec<String>,
    pub(crate) architecture: Option<String>,
    pub(crate) tensor_count: usize,
    pub(crate) layer_count: Option<u32>,
    pub(crate) effective_decoder_layers: Option<u32>,
    pub(crate) nextn_predict_layers: u32,
    pub(crate) role_source: Option<String>,
    pub(crate) full_layers: Vec<u32>,
    pub(crate) shared_layers: Vec<u32>,
    pub(crate) generation_policy_required: bool,
    pub(crate) generation_policy: Option<GlmDsaGenerationPolicyReport>,
    pub(crate) generation_thresholds: Option<GlmDsaGenerationThresholdReport>,
    pub(crate) metadata_errors: Vec<String>,
    pub(crate) tensor_errors: Vec<String>,
    pub(crate) generation_policy_errors: Vec<String>,
    pub(crate) generation_threshold_errors: Vec<String>,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GlmDsaGenerationPolicyReport {
    pub(crate) profile: String,
    pub(crate) decode: String,
    pub(crate) short_prefill: String,
    pub(crate) long_prefill: String,
    pub(crate) verify: String,
    pub(crate) indexshare: Option<String>,
    pub(crate) selected_row_flash: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GlmDsaGenerationThresholdReport {
    pub(crate) short_prefill_max_tokens: Option<u32>,
    pub(crate) direct_sparse_decode_max_top_k: Option<u32>,
    pub(crate) compact_flash_min_kv: Option<u32>,
    pub(crate) dense_mask_max_bytes: Option<u64>,
}

#[derive(Debug, Default)]
struct GgufMetadata {
    strings: BTreeMap<String, String>,
    u32s: BTreeMap<String, u32>,
    f32s: BTreeMap<String, f32>,
    bools: BTreeMap<String, bool>,
    array_strings: BTreeMap<String, Vec<String>>,
}

#[derive(Debug)]
struct ContractInput {
    path: String,
    artifact_kind: String,
    gguf_files: Vec<String>,
    metadata: GgufMetadata,
    tensors: BTreeSet<String>,
    generation: Option<PackageGeneration>,
}

#[derive(Debug, Deserialize)]
struct PackageManifest {
    #[serde(default)]
    generation: Option<PackageGeneration>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndexShareRole {
    Full,
    Shared,
}

pub(crate) fn validate_path(path: &Path) -> Result<GlmDsaContractReport> {
    validate_path_with_options(path, GlmDsaContractOptions::default())
}

pub(crate) fn validate_path_with_options(
    path: &Path,
    options: GlmDsaContractOptions,
) -> Result<GlmDsaContractReport> {
    let files = collect_gguf_files(path)?;
    ensure!(
        !files.is_empty(),
        "no GGUF files found under {}",
        path.display()
    );

    let mut metadata = GgufMetadata::default();
    let mut tensors = BTreeSet::new();
    for file in &files {
        metadata.merge(read_gguf_metadata(file)?);
        let info = ModelInfo::open(file).with_context(|| format!("open {}", file.display()))?;
        for tensor in info
            .tensors()
            .with_context(|| format!("read tensors from {}", file.display()))?
        {
            tensors.insert(tensor.name);
        }
    }

    let input = ContractInput {
        path: path.display().to_string(),
        artifact_kind: artifact_kind(path).to_string(),
        gguf_files: files
            .iter()
            .map(|file| file.display().to_string())
            .collect(),
        metadata,
        tensors,
        generation: read_package_generation(path)?,
    };
    Ok(validate_contract(input, options))
}

fn artifact_kind(path: &Path) -> &'static str {
    if path.is_dir() {
        if path.join("model-package.json").is_file() {
            "model_package"
        } else {
            "gguf_directory"
        }
    } else {
        "gguf_file"
    }
}

fn validate_contract(input: ContractInput, options: GlmDsaContractOptions) -> GlmDsaContractReport {
    let mut report = GlmDsaContractReport {
        valid: false,
        path: input.path,
        artifact_kind: input.artifact_kind,
        gguf_files: input.gguf_files,
        architecture: input.metadata.strings.get("general.architecture").cloned(),
        tensor_count: input.tensors.len(),
        layer_count: input.metadata.u32s.get("glm-dsa.block_count").copied(),
        effective_decoder_layers: None,
        nextn_predict_layers: input
            .metadata
            .u32s
            .get("glm-dsa.nextn_predict_layers")
            .copied()
            .unwrap_or(0),
        role_source: None,
        full_layers: Vec::new(),
        shared_layers: Vec::new(),
        generation_policy_required: options.require_generation_policy,
        generation_policy: None,
        generation_thresholds: None,
        metadata_errors: Vec::new(),
        tensor_errors: Vec::new(),
        generation_policy_errors: Vec::new(),
        generation_threshold_errors: Vec::new(),
        warnings: Vec::new(),
    };

    validate_metadata(&input.metadata, &mut report);
    validate_tensors(&input.metadata, &input.tensors, &mut report);
    validate_generation_policy(input.generation.as_ref(), options, &mut report);
    report.valid = report.metadata_errors.is_empty()
        && report.tensor_errors.is_empty()
        && report.generation_policy_errors.is_empty()
        && report.generation_threshold_errors.is_empty();
    report
}

fn read_package_generation(path: &Path) -> Result<Option<PackageGeneration>> {
    let manifest_path = path.join("model-package.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: PackageManifest = serde_json::from_str(&manifest)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    Ok(manifest.generation)
}

fn validate_generation_policy(
    generation: Option<&PackageGeneration>,
    options: GlmDsaContractOptions,
    report: &mut GlmDsaContractReport,
) {
    let Some(generation) = generation else {
        if options.require_generation_policy {
            report
                .generation_policy_errors
                .push("model-package.json missing generation block".to_string());
        }
        return;
    };

    if let Some(policy) = generation.policy.as_ref() {
        let selected_row_flash = policy
            .experimental
            .as_ref()
            .and_then(|experimental| experimental.selected_row_flash.clone());
        report.generation_policy = Some(GlmDsaGenerationPolicyReport {
            profile: policy.profile.clone(),
            decode: policy.decode.clone(),
            short_prefill: policy.short_prefill.clone(),
            long_prefill: policy.long_prefill.clone(),
            verify: policy.verify.clone(),
            indexshare: policy.indexshare.clone(),
            selected_row_flash: selected_row_flash.clone(),
        });

        expect_generation_value(
            &mut report.generation_policy_errors,
            "generation.policy.profile",
            &policy.profile,
            GLM_DSA_POLICY_PROFILE,
        );
        expect_generation_value(
            &mut report.generation_policy_errors,
            "generation.policy.decode",
            &policy.decode,
            GLM_DSA_POLICY_DECODE,
        );
        expect_generation_value(
            &mut report.generation_policy_errors,
            "generation.policy.short_prefill",
            &policy.short_prefill,
            GLM_DSA_POLICY_SHORT_PREFILL,
        );
        expect_generation_value(
            &mut report.generation_policy_errors,
            "generation.policy.long_prefill",
            &policy.long_prefill,
            GLM_DSA_POLICY_LONG_PREFILL,
        );
        expect_generation_value(
            &mut report.generation_policy_errors,
            "generation.policy.verify",
            &policy.verify,
            GLM_DSA_POLICY_VERIFY,
        );
        expect_generation_optional_value(
            &mut report.generation_policy_errors,
            "generation.policy.indexshare",
            policy.indexshare.as_deref(),
            GLM_DSA_POLICY_INDEXSHARE,
        );
        expect_generation_optional_value(
            &mut report.generation_policy_errors,
            "generation.policy.experimental.selected_row_flash",
            selected_row_flash.as_deref(),
            GLM_DSA_POLICY_SELECTED_ROW_FLASH,
        );
    } else if options.require_generation_policy {
        report
            .generation_policy_errors
            .push("generation.policy is required for GLM-DSA packages".to_string());
    }

    validate_generation_thresholds(generation.thresholds.as_ref(), options, report);
}

fn validate_generation_thresholds(
    thresholds: Option<&PackageGenerationThresholds>,
    options: GlmDsaContractOptions,
    report: &mut GlmDsaContractReport,
) {
    let Some(thresholds) = thresholds else {
        if options.require_generation_policy {
            report
                .generation_threshold_errors
                .push("generation.thresholds is required for GLM-DSA packages".to_string());
        }
        return;
    };

    report.generation_thresholds = Some(GlmDsaGenerationThresholdReport {
        short_prefill_max_tokens: thresholds.short_prefill_max_tokens,
        direct_sparse_decode_max_top_k: thresholds.direct_sparse_decode_max_top_k,
        compact_flash_min_kv: thresholds.compact_flash_min_kv,
        dense_mask_max_bytes: thresholds.dense_mask_max_bytes,
    });
    expect_generation_optional_u32(
        &mut report.generation_threshold_errors,
        "generation.thresholds.short_prefill_max_tokens",
        thresholds.short_prefill_max_tokens,
        GLM_DSA_SHORT_PREFILL_MAX_TOKENS,
    );
    expect_generation_optional_u32(
        &mut report.generation_threshold_errors,
        "generation.thresholds.direct_sparse_decode_max_top_k",
        thresholds.direct_sparse_decode_max_top_k,
        GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K,
    );
    expect_generation_optional_u32(
        &mut report.generation_threshold_errors,
        "generation.thresholds.compact_flash_min_kv",
        thresholds.compact_flash_min_kv,
        GLM_DSA_COMPACT_FLASH_MIN_KV,
    );
    expect_generation_optional_u64(
        &mut report.generation_threshold_errors,
        "generation.thresholds.dense_mask_max_bytes",
        thresholds.dense_mask_max_bytes,
        GLM_DSA_DENSE_MASK_MAX_BYTES,
    );
}

fn expect_generation_value(errors: &mut Vec<String>, field: &str, actual: &str, expected: &str) {
    if actual != expected {
        errors.push(format!("{field} must be {expected}, got {actual}"));
    }
}

fn expect_generation_optional_value(
    errors: &mut Vec<String>,
    field: &str,
    actual: Option<&str>,
    expected: &str,
) {
    match actual {
        Some(actual) if actual == expected => {}
        Some(actual) => errors.push(format!("{field} must be {expected}, got {actual}")),
        None => errors.push(format!("missing {field}")),
    }
}

fn expect_generation_optional_u32(
    errors: &mut Vec<String>,
    field: &str,
    actual: Option<u32>,
    expected: u32,
) {
    match actual {
        Some(actual) if actual == expected => {}
        Some(actual) => errors.push(format!("{field} must be {expected}, got {actual}")),
        None => errors.push(format!("missing {field}")),
    }
}

fn expect_generation_optional_u64(
    errors: &mut Vec<String>,
    field: &str,
    actual: Option<u64>,
    expected: u64,
) {
    match actual {
        Some(actual) if actual == expected => {}
        Some(actual) => errors.push(format!("{field} must be {expected}, got {actual}")),
        None => errors.push(format!("missing {field}")),
    }
}

fn validate_metadata(metadata: &GgufMetadata, report: &mut GlmDsaContractReport) {
    if report.architecture.as_deref() != Some("glm-dsa") {
        report
            .metadata_errors
            .push("general.architecture must be glm-dsa".to_string());
    }

    for key in REQUIRED_U32_METADATA {
        match metadata.u32s.get(*key) {
            Some(value) if *value > 0 => {}
            Some(_) => report.metadata_errors.push(format!("{key} must be > 0")),
            None => report.metadata_errors.push(format!("missing {key}")),
        }
    }
    for key in REQUIRED_F32_METADATA {
        if !metadata.f32s.contains_key(*key) {
            report.metadata_errors.push(format!("missing {key}"));
        }
    }
    for key in REQUIRED_BOOL_METADATA {
        if !metadata.bools.contains_key(*key) {
            report.metadata_errors.push(format!("missing {key}"));
        }
    }
    validate_cross_field_metadata(metadata, report);

    let Some(block_count) = metadata.u32s.get("glm-dsa.block_count").copied() else {
        return;
    };
    if report.nextn_predict_layers >= block_count {
        report.metadata_errors.push(format!(
            "glm-dsa.nextn_predict_layers {} must be less than block_count {block_count}",
            report.nextn_predict_layers
        ));
        return;
    }
    let effective_layers = block_count - report.nextn_predict_layers;
    report.effective_decoder_layers = Some(effective_layers);

    if let Some(freq) = metadata
        .u32s
        .get("glm-dsa.attention.indexer.top_k_frequency")
        && *freq == 0
    {
        report.metadata_errors.push(
            "glm-dsa.attention.indexer.top_k_frequency must be positive when present".to_string(),
        );
    }
    if metadata
        .u32s
        .contains_key("glm-dsa.attention.indexer.top_k_frequency")
        && !metadata
            .u32s
            .contains_key("glm-dsa.attention.indexer.skip_top_k_offset")
    {
        report.metadata_errors.push(
            "glm-dsa.attention.indexer.skip_top_k_offset is required when top_k_frequency is present"
                .to_string(),
        );
    }

    if let Some(types) = metadata
        .array_strings
        .get("glm-dsa.attention.indexer.types")
    {
        if types.len() != effective_layers as usize {
            report.metadata_errors.push(format!(
                "glm-dsa.attention.indexer.types length {} must match effective decoder layers {effective_layers}",
                types.len()
            ));
        }
        for (index, role) in types.iter().enumerate() {
            if role != "full" && role != "shared" {
                report.metadata_errors.push(format!(
                    "glm-dsa.attention.indexer.types[{index}] must be full or shared"
                ));
            }
        }
        validate_indexshare_role_metadata_consistency(metadata, types, effective_layers, report);
    } else if !metadata
        .u32s
        .contains_key("glm-dsa.attention.indexer.top_k_frequency")
    {
        report.metadata_errors.push(
            "missing IndexShare role metadata; expected indexer.types or top_k_frequency/skip_top_k_offset"
                .to_string(),
        );
    }
}

fn validate_cross_field_metadata(metadata: &GgufMetadata, report: &mut GlmDsaContractReport) {
    if let (Some(indexer_key_length), Some(rope_dimension_count)) = (
        metadata.u32s.get("glm-dsa.attention.indexer.key_length"),
        metadata.u32s.get("glm-dsa.rope.dimension_count"),
    ) && indexer_key_length <= rope_dimension_count
    {
        report.metadata_errors.push(format!(
            "glm-dsa.attention.indexer.key_length {indexer_key_length} must be greater than glm-dsa.rope.dimension_count {rope_dimension_count}"
        ));
    }
    if let (Some(expert_used_count), Some(expert_count)) = (
        metadata.u32s.get("glm-dsa.expert_used_count"),
        metadata.u32s.get("glm-dsa.expert_count"),
    ) && expert_used_count > expert_count
    {
        report.metadata_errors.push(format!(
            "glm-dsa.expert_used_count {expert_used_count} must not exceed glm-dsa.expert_count {expert_count}"
        ));
    }
}

fn validate_indexshare_role_metadata_consistency(
    metadata: &GgufMetadata,
    types: &[String],
    effective_layers: u32,
    report: &mut GlmDsaContractReport,
) {
    if types.len() != effective_layers as usize {
        return;
    }
    let Some(freq) = metadata
        .u32s
        .get("glm-dsa.attention.indexer.top_k_frequency")
        .copied()
        .filter(|freq| *freq > 0)
    else {
        return;
    };
    let Some(offset) = metadata
        .u32s
        .get("glm-dsa.attention.indexer.skip_top_k_offset")
        .copied()
    else {
        return;
    };

    for (layer, role) in types.iter().enumerate() {
        if role != "full" && role != "shared" {
            continue;
        }
        let frequency_role = if frequency_indexshare_role_is_full(layer as u32, offset, freq) {
            "full"
        } else {
            "shared"
        };
        if role != frequency_role {
            report.metadata_errors.push(format!(
                "glm-dsa.attention.indexer.types conflicts with top_k_frequency at layer {layer}: types={role}, frequency={frequency_role}"
            ));
        }
    }
}

fn validate_tensors(
    metadata: &GgufMetadata,
    tensors: &BTreeSet<String>,
    report: &mut GlmDsaContractReport,
) {
    let Some(effective_layers) = report.effective_decoder_layers else {
        return;
    };
    require_tensor(tensors, "token_embd.weight", &mut report.tensor_errors);
    require_tensor(tensors, "output_norm.weight", &mut report.tensor_errors);

    let dense_lead = metadata
        .u32s
        .get("glm-dsa.leading_dense_block_count")
        .copied()
        .unwrap_or(0);
    let roles = indexshare_roles(metadata, tensors, effective_layers, report);
    for layer in 0..effective_layers {
        validate_decoder_layer(layer, dense_lead, roles[layer as usize], tensors, report);
    }
    validate_mtp_layers(
        effective_layers,
        report.nextn_predict_layers,
        tensors,
        report,
    );
}

fn validate_decoder_layer(
    layer: u32,
    dense_lead: u32,
    role: IndexShareRole,
    tensors: &BTreeSet<String>,
    report: &mut GlmDsaContractReport,
) {
    for suffix in BASE_LAYER_TENSORS {
        require_layer_tensor(tensors, layer, suffix, &mut report.tensor_errors);
    }
    reject_unsplit_kv_b(tensors, layer, &mut report.tensor_errors);

    if layer < dense_lead {
        for suffix in DENSE_FFN_TENSORS {
            require_layer_tensor(tensors, layer, suffix, &mut report.tensor_errors);
        }
    } else {
        for suffix in MOE_LAYER_TENSORS {
            require_layer_tensor(tensors, layer, suffix, &mut report.tensor_errors);
        }
    }

    let indexer_count = indexer_tensor_count(tensors, layer);
    if indexer_count != 0 && indexer_count != INDEXER_TENSORS.len() {
        report.tensor_errors.push(format!(
            "blk.{layer} has partial GLM-DSA indexer tensor group ({indexer_count}/{})",
            INDEXER_TENSORS.len()
        ));
    }

    match role {
        IndexShareRole::Full => {
            report.full_layers.push(layer);
            if indexer_count != INDEXER_TENSORS.len() {
                report.tensor_errors.push(format!(
                    "blk.{layer} is a Full IndexShare layer but lacks complete indexer tensors"
                ));
            }
        }
        IndexShareRole::Shared => {
            report.shared_layers.push(layer);
            if indexer_count == INDEXER_TENSORS.len() {
                report.tensor_errors.push(format!(
                    "blk.{layer} is declared Shared but contains complete indexer tensors"
                ));
            }
        }
    }
}

fn validate_mtp_layers(
    effective_layers: u32,
    nextn_predict_layers: u32,
    tensors: &BTreeSet<String>,
    report: &mut GlmDsaContractReport,
) {
    for layer in effective_layers..effective_layers + nextn_predict_layers {
        for suffix in BASE_LAYER_TENSORS {
            require_layer_tensor(tensors, layer, suffix, &mut report.tensor_errors);
        }
        reject_unsplit_kv_b(tensors, layer, &mut report.tensor_errors);
        for suffix in MOE_LAYER_TENSORS {
            require_layer_tensor(tensors, layer, suffix, &mut report.tensor_errors);
        }
        for suffix in NEXTN_TENSORS {
            require_layer_tensor(tensors, layer, suffix, &mut report.tensor_errors);
        }

        let indexer_count = indexer_tensor_count(tensors, layer);
        if indexer_count != INDEXER_TENSORS.len() {
            report.tensor_errors.push(format!(
                "blk.{layer} is an MTP/NextN GLM-DSA layer but lacks complete indexer tensors ({indexer_count}/{})",
                INDEXER_TENSORS.len()
            ));
        }
    }
}

fn reject_unsplit_kv_b(tensors: &BTreeSet<String>, layer: u32, errors: &mut Vec<String>) {
    if tensors.contains(&format!("blk.{layer}.attn_kv_b.weight")) {
        errors.push(format!(
            "blk.{layer} has stale unsplit attn_kv_b.weight; GLM-DSA sparse attention requires split attn_k_b/attn_v_b only"
        ));
    }
}

fn indexshare_roles(
    metadata: &GgufMetadata,
    tensors: &BTreeSet<String>,
    effective_layers: u32,
    report: &mut GlmDsaContractReport,
) -> Vec<IndexShareRole> {
    if let Some(types) = metadata
        .array_strings
        .get("glm-dsa.attention.indexer.types")
        && types.len() == effective_layers as usize
    {
        report.role_source = Some("metadata_types".to_string());
        return types
            .iter()
            .map(|role| {
                if role == "full" {
                    IndexShareRole::Full
                } else {
                    IndexShareRole::Shared
                }
            })
            .collect();
    }

    if let Some(freq) = metadata
        .u32s
        .get("glm-dsa.attention.indexer.top_k_frequency")
        .copied()
        && freq > 0
    {
        report.role_source = Some("metadata_frequency".to_string());
        let offset = metadata
            .u32s
            .get("glm-dsa.attention.indexer.skip_top_k_offset")
            .copied()
            .unwrap_or(0);
        return (0..effective_layers)
            .map(|layer| {
                if frequency_indexshare_role_is_full(layer, offset, freq) {
                    IndexShareRole::Full
                } else {
                    IndexShareRole::Shared
                }
            })
            .collect();
    }

    report.role_source = Some("tensor_presence".to_string());
    report
        .warnings
        .push("using indexer tensor presence as IndexShare role fallback".to_string());
    (0..effective_layers)
        .map(|layer| {
            if indexer_tensor_count(tensors, layer) == INDEXER_TENSORS.len() {
                IndexShareRole::Full
            } else {
                IndexShareRole::Shared
            }
        })
        .collect()
}

fn frequency_indexshare_role_is_full(layer: u32, offset: u32, freq: u32) -> bool {
    layer < offset || (layer - offset + 1).is_multiple_of(freq)
}

fn require_layer_tensor(
    tensors: &BTreeSet<String>,
    layer: u32,
    suffix: &str,
    errors: &mut Vec<String>,
) {
    require_tensor(tensors, &format!("blk.{layer}.{suffix}"), errors);
}

fn require_tensor(tensors: &BTreeSet<String>, name: &str, errors: &mut Vec<String>) {
    if !tensors.contains(name) {
        errors.push(format!("missing tensor {name}"));
    }
}

fn indexer_tensor_count(tensors: &BTreeSet<String>, layer: u32) -> usize {
    INDEXER_TENSORS
        .iter()
        .filter(|suffix| tensors.contains(&format!("blk.{layer}.{suffix}")))
        .count()
}

fn collect_gguf_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_dir() {
        let manifest = path.join("model-package.json");
        if manifest.is_file() {
            return collect_package_gguf_files(path, &manifest);
        }
        let mut files = Vec::new();
        collect_gguf_files_recursive(path, &mut files)?;
        files.sort();
        return Ok(files);
    }
    resolve_gguf_shard_paths(path)
}

fn collect_package_gguf_files(package: &Path, manifest: &Path) -> Result<Vec<PathBuf>> {
    let value: serde_json::Value = serde_json::from_slice(
        &fs::read(manifest).with_context(|| format!("read {}", manifest.display()))?,
    )
    .with_context(|| format!("parse {}", manifest.display()))?;
    let mut files = Vec::new();
    for field in ["metadata", "embeddings", "output"] {
        if let Some(path) = value
            .pointer(&format!("/shared/{field}/path"))
            .and_then(serde_json::Value::as_str)
        {
            files.push(package.join(path));
        }
    }
    if let Some(layers) = value.get("layers").and_then(serde_json::Value::as_array) {
        for layer in layers {
            if let Some(path) = layer.get("path").and_then(serde_json::Value::as_str) {
                files.push(package.join(path));
            }
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn collect_gguf_files_recursive(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path).with_context(|| format!("read directory {}", path.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files_recursive(&path, files)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("gguf") {
            files.push(path);
        }
    }
    Ok(())
}

fn resolve_gguf_shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some(shard) = split_gguf_shard_info(file_name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let total = shard
        .total
        .parse::<usize>()
        .with_context(|| format!("parse GGUF shard total from {file_name}"))?;
    if total <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }

    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let mut paths = Vec::with_capacity(total);
    for part in 1..=total {
        let shard_name = format!("{}-{part:05}-of-{}.gguf", shard.prefix, shard.total);
        let shard_path = parent.join(shard_name);
        if !shard_path.exists() {
            bail!(
                "split GGUF shard {} is missing sibling {}",
                path.display(),
                shard_path.display()
            );
        }
        paths.push(shard_path);
    }
    Ok(paths)
}

fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .with_context(|| format!("read GGUF magic from {}", path.display()))?;
    ensure!(&magic == b"GGUF", "not a GGUF file: {}", path.display());

    let version = read_u32(&mut file)?;
    ensure!(
        version >= 2,
        "unsupported GGUF version {version} in {}",
        path.display()
    );
    let _tensor_count = read_header_count(&mut file, MAX_GGUF_TENSOR_COUNT, "tensor")?;
    let kv_count = read_header_count(&mut file, MAX_GGUF_HEADER_KV_COUNT, "metadata")?;

    let mut metadata = GgufMetadata::default();
    for _ in 0..kv_count {
        let key = read_string(&mut file)?;
        let value_type = GgufValueType::from_u32(read_u32(&mut file)?)?;
        read_metadata_value(&mut file, value_type, key, &mut metadata)?;
    }
    Ok(metadata)
}

fn read_metadata_value(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
    key: String,
    metadata: &mut GgufMetadata,
) -> Result<()> {
    match value_type {
        GgufValueType::Uint32 => {
            metadata.u32s.insert(key, read_u32(reader)?);
        }
        GgufValueType::Int32 => {
            let value = read_i32(reader)?;
            if let Ok(value) = u32::try_from(value) {
                metadata.u32s.insert(key, value);
            }
        }
        GgufValueType::Uint16 => {
            metadata.u32s.insert(key, u32::from(read_u16(reader)?));
        }
        GgufValueType::Uint8 => {
            metadata.u32s.insert(key, u32::from(read_u8(reader)?));
        }
        GgufValueType::Float32 => {
            metadata.f32s.insert(key, read_f32(reader)?);
        }
        GgufValueType::Bool => {
            metadata.bools.insert(key, read_bool(reader)?);
        }
        GgufValueType::String => {
            metadata.strings.insert(key, read_string(reader)?);
        }
        GgufValueType::Array => {
            read_metadata_array(reader, key, metadata)?;
        }
        other => {
            skip_value(reader, other)?;
        }
    }
    Ok(())
}

fn read_metadata_array(
    reader: &mut (impl Read + Seek),
    key: String,
    metadata: &mut GgufMetadata,
) -> Result<()> {
    let item_type = GgufValueType::from_u32(read_u32(reader)?)?;
    let len = read_u64(reader)?;
    ensure!(
        len <= MAX_GGUF_ARRAY_ELEMENTS,
        "GGUF array length {len} exceeds safety limit {MAX_GGUF_ARRAY_ELEMENTS}"
    );
    if item_type == GgufValueType::String {
        let mut values = Vec::with_capacity(usize::try_from(len).unwrap_or_default());
        for _ in 0..len {
            values.push(read_string(reader)?);
        }
        metadata.array_strings.insert(key, values);
    } else {
        skip_array_items(reader, item_type, len, 0)?;
    }
    Ok(())
}

impl GgufMetadata {
    fn merge(&mut self, other: Self) {
        self.strings.extend(other.strings);
        self.u32s.extend(other.u32s);
        self.f32s.extend(other.f32s);
        self.bools.extend(other.bools);
        self.array_strings.extend(other.array_strings);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl GgufValueType {
    fn from_u32(value: u32) -> Result<Self> {
        Ok(match value {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            other => bail!("unsupported GGUF metadata value type {other}"),
        })
    }

    fn fixed_width(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> Result<i32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF i32")?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read GGUF u64")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(reader: &mut impl Read) -> Result<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read GGUF i64")?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes).context("read GGUF u16")?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes).context("read GGUF u8")?;
    Ok(bytes[0])
}

fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF f32")?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_bool(reader: &mut impl Read) -> Result<bool> {
    Ok(read_u8(reader)? != 0)
}

fn read_header_count(reader: &mut impl Read, max: u64, label: &str) -> Result<u64> {
    let count = read_i64(reader)?;
    ensure!(count >= 0, "GGUF {label} count is negative: {count}");
    let count = u64::try_from(count).context("GGUF header count does not fit u64")?;
    ensure!(
        count <= max,
        "GGUF {label} count {count} exceeds safety limit {max}"
    );
    Ok(count)
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let len = read_u64(reader)?;
    ensure!(
        len <= MAX_GGUF_STRING_BYTES,
        "GGUF string length {len} exceeds safety limit {MAX_GGUF_STRING_BYTES}"
    );
    let len = usize::try_from(len).context("GGUF string length does not fit usize")?;
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .context("read GGUF string bytes")?;
    String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
}

fn skip_value(reader: &mut (impl Read + Seek), value_type: GgufValueType) -> Result<()> {
    skip_value_with_depth(reader, value_type, 0)
}

fn skip_value_with_depth(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
    depth: usize,
) -> Result<()> {
    ensure!(
        depth <= MAX_GGUF_ARRAY_DEPTH,
        "GGUF array nesting exceeds safety limit {MAX_GGUF_ARRAY_DEPTH}"
    );
    if let Some(width) = value_type.fixed_width() {
        skip_bytes(reader, width)
    } else if value_type == GgufValueType::String {
        let len = read_u64(reader)?;
        ensure!(
            len <= MAX_GGUF_STRING_BYTES,
            "GGUF string length {len} exceeds safety limit {MAX_GGUF_STRING_BYTES}"
        );
        skip_bytes(reader, len)
    } else {
        let item_type = GgufValueType::from_u32(read_u32(reader)?)?;
        let len = read_u64(reader)?;
        ensure!(
            len <= MAX_GGUF_ARRAY_ELEMENTS,
            "GGUF array length {len} exceeds safety limit {MAX_GGUF_ARRAY_ELEMENTS}"
        );
        skip_array_items(reader, item_type, len, depth)
    }
}

fn skip_array_items(
    reader: &mut (impl Read + Seek),
    item_type: GgufValueType,
    len: u64,
    depth: usize,
) -> Result<()> {
    if let Some(width) = item_type.fixed_width() {
        let bytes = width
            .checked_mul(len)
            .context("GGUF array byte size overflows u64")?;
        skip_bytes(reader, bytes)
    } else {
        for _ in 0..len {
            skip_value_with_depth(reader, item_type, depth + 1)?;
        }
        Ok(())
    }
}

fn skip_bytes(reader: &mut impl Seek, len: u64) -> Result<()> {
    let offset = i64::try_from(len).context("GGUF value is too large to seek over")?;
    reader
        .seek(SeekFrom::Current(offset))
        .context("skip GGUF metadata value")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_minimal_glm_dsa_contract() {
        let input = mock_input(false);
        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(report.valid, "{report:#?}");
        assert_eq!(report.effective_decoder_layers, Some(3));
        assert_eq!(report.nextn_predict_layers, 1);
        assert_eq!(report.full_layers, vec![0]);
        assert_eq!(report.shared_layers, vec![1, 2]);
    }

    #[test]
    fn strict_contract_rejects_missing_generation_policy() {
        let input = mock_input(false);
        let report = validate_contract(
            input,
            GlmDsaContractOptions {
                require_generation_policy: true,
            },
        );

        assert!(!report.valid);
        assert!(report.generation_policy_required);
        assert!(report.generation_policy_errors.iter().any(|error| {
            error.contains("model-package.json missing generation block")
                || error.contains("generation.policy is required")
        }));
    }

    #[test]
    fn strict_contract_accepts_glm_dsa_generation_policy() {
        let mut input = mock_input(false);
        input.generation = Some(mock_generation());
        let report = validate_contract(
            input,
            GlmDsaContractOptions {
                require_generation_policy: true,
            },
        );

        assert!(report.valid, "{report:#?}");
        let policy = report.generation_policy.expect("policy should be reported");
        assert_eq!(policy.profile, GLM_DSA_POLICY_PROFILE);
        assert_eq!(policy.decode, GLM_DSA_POLICY_DECODE);
        assert_eq!(
            policy.indexshare.as_deref(),
            Some(GLM_DSA_POLICY_INDEXSHARE)
        );
        let thresholds = report
            .generation_thresholds
            .expect("thresholds should be reported");
        assert_eq!(
            thresholds.short_prefill_max_tokens,
            Some(GLM_DSA_SHORT_PREFILL_MAX_TOKENS)
        );
        assert_eq!(
            thresholds.direct_sparse_decode_max_top_k,
            Some(GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K)
        );
        assert_eq!(
            thresholds.dense_mask_max_bytes,
            Some(GLM_DSA_DENSE_MASK_MAX_BYTES)
        );
    }

    #[test]
    fn strict_contract_rejects_wrong_glm_dsa_generation_threshold() {
        let mut input = mock_input(false);
        let mut generation = mock_generation();
        generation
            .thresholds
            .as_mut()
            .expect("thresholds")
            .compact_flash_min_kv = Some(256);
        input.generation = Some(generation);

        let report = validate_contract(
            input,
            GlmDsaContractOptions {
                require_generation_policy: true,
            },
        );

        assert!(!report.valid);
        assert!(report.generation_threshold_errors.iter().any(|error| {
            error.contains("generation.thresholds.compact_flash_min_kv must be 1")
        }));
    }

    #[test]
    fn rejects_indexer_key_length_not_greater_than_rope_dimension() {
        let mut input = mock_input(false);
        input
            .metadata
            .u32s
            .insert("glm-dsa.attention.indexer.key_length".to_string(), 128);
        input
            .metadata
            .u32s
            .insert("glm-dsa.rope.dimension_count".to_string(), 128);

        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(report.metadata_errors.iter().any(|error| {
            error.contains("indexer.key_length 128 must be greater than")
                && error.contains("rope.dimension_count 128")
        }));
    }

    #[test]
    fn rejects_expert_used_count_above_expert_count() {
        let mut input = mock_input(false);
        input
            .metadata
            .u32s
            .insert("glm-dsa.expert_count".to_string(), 8);
        input
            .metadata
            .u32s
            .insert("glm-dsa.expert_used_count".to_string(), 9);

        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(report.metadata_errors.iter().any(|error| {
            error.contains("expert_used_count 9 must not exceed")
                && error.contains("expert_count 8")
        }));
    }

    #[test]
    fn rejects_stale_unsplit_kv_b_tensor() {
        let mut input = mock_input(false);
        input.tensors.insert("blk.1.attn_kv_b.weight".to_string());

        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(
            report
                .tensor_errors
                .iter()
                .any(|error| { error.contains("blk.1 has stale unsplit attn_kv_b.weight") })
        );
    }

    #[test]
    fn rejects_partial_indexer_group() {
        let input = mock_input(true);
        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(
            report
                .tensor_errors
                .iter()
                .any(|error| error.contains("partial GLM-DSA indexer"))
        );
    }

    #[test]
    fn rejects_full_layer_without_indexer_group() {
        let mut input = mock_input(false);
        remove_indexer_tensors(&mut input.tensors, 0);

        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(
            report
                .tensor_errors
                .iter()
                .any(|error| error.contains("Full IndexShare layer")
                    && error.contains("lacks complete indexer tensors")),
            "{report:#?}"
        );
    }

    #[test]
    fn rejects_shared_layer_with_complete_indexer_group() {
        let mut input = mock_input(false);
        add_indexer_tensors(&mut input.tensors, 1);

        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(
            report
                .tensor_errors
                .iter()
                .any(|error| error.contains("declared Shared")
                    && error.contains("complete indexer tensors")),
            "{report:#?}"
        );
    }

    #[test]
    fn rejects_mtp_layer_without_complete_indexer_group() {
        let mut input = mock_input(false);
        remove_indexer_tensors(&mut input.tensors, 3);

        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(
            report.tensor_errors.iter().any(|error| {
                error.contains("MTP/NextN GLM-DSA layer")
                    && error.contains("lacks complete indexer tensors")
            }),
            "{report:#?}"
        );
    }

    #[test]
    fn derives_glm52_frequency_cadence() {
        let input = mock_frequency_input(true);
        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(report.valid, "{report:#?}");
        assert_eq!(report.role_source.as_deref(), Some("metadata_frequency"));
        assert_eq!(report.full_layers, vec![0, 1, 2, 6]);
        assert_eq!(report.shared_layers, vec![3, 4, 5, 7]);
    }

    #[test]
    fn rejects_frequency_without_skip_offset() {
        let input = mock_frequency_input(false);
        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(report.metadata_errors.iter().any(|error| {
            error.contains("skip_top_k_offset is required when top_k_frequency is present")
        }));
    }

    #[test]
    fn rejects_indexer_types_frequency_conflict() {
        let mut input = mock_input(false);
        input
            .metadata
            .u32s
            .insert("glm-dsa.attention.indexer.top_k_frequency".to_string(), 2);
        input
            .metadata
            .u32s
            .insert("glm-dsa.attention.indexer.skip_top_k_offset".to_string(), 2);

        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(
            report.metadata_errors.iter().any(|error| error.contains(
                "glm-dsa.attention.indexer.types conflicts with top_k_frequency at layer 1"
            )),
            "{report:#?}"
        );
    }

    #[test]
    fn classifies_contract_artifact_kind() {
        let root = std::env::temp_dir().join(format!(
            "glm-dsa-contract-artifact-kind-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let package = root.join("package");
        let raw_dir = root.join("raw");
        let file = root.join("model.gguf");
        fs::create_dir(&package).unwrap();
        fs::create_dir(&raw_dir).unwrap();
        fs::write(package.join("model-package.json"), "{}").unwrap();
        fs::write(&file, b"").unwrap();

        assert_eq!(artifact_kind(&package), "model_package");
        assert_eq!(artifact_kind(&raw_dir), "gguf_directory");
        assert_eq!(artifact_kind(&file), "gguf_file");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn rejects_tensor_presence_role_fallback() {
        let mut input = mock_input(false);
        input
            .metadata
            .array_strings
            .remove("glm-dsa.attention.indexer.types");
        let report = validate_contract(input, GlmDsaContractOptions::default());

        assert!(!report.valid);
        assert!(
            report
                .metadata_errors
                .iter()
                .any(|error| { error.contains("missing IndexShare role metadata") })
        );
        assert_eq!(report.role_source.as_deref(), Some("tensor_presence"));
    }

    fn mock_frequency_input(include_skip_offset: bool) -> ContractInput {
        let mut metadata = mock_metadata();
        metadata.u32s.insert("glm-dsa.block_count".to_string(), 9);
        metadata
            .u32s
            .insert("glm-dsa.nextn_predict_layers".to_string(), 1);
        metadata
            .u32s
            .insert("glm-dsa.leading_dense_block_count".to_string(), 3);
        metadata
            .u32s
            .insert("glm-dsa.attention.indexer.top_k_frequency".to_string(), 4);
        if include_skip_offset {
            metadata
                .u32s
                .insert("glm-dsa.attention.indexer.skip_top_k_offset".to_string(), 3);
        }

        let mut tensors = BTreeSet::from([
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
        ]);
        for layer in 0..8 {
            add_base_layer_tensors(&mut tensors, layer);
            if layer < 3 {
                for suffix in ["ffn_gate.weight", "ffn_down.weight", "ffn_up.weight"] {
                    tensors.insert(format!("blk.{layer}.{suffix}"));
                }
            } else {
                add_moe_layer_tensors(&mut tensors, layer);
            }
        }
        for layer in [0, 1, 2, 6] {
            add_indexer_tensors(&mut tensors, layer);
        }
        add_nextn_tensors(&mut tensors, 8);

        ContractInput {
            path: "mock-frequency".to_string(),
            artifact_kind: "test".to_string(),
            gguf_files: Vec::new(),
            metadata,
            tensors,
            generation: None,
        }
    }

    fn mock_input(partial_shared_indexer: bool) -> ContractInput {
        let mut metadata = mock_metadata();
        metadata.u32s.insert("glm-dsa.block_count".to_string(), 4);
        metadata
            .u32s
            .insert("glm-dsa.nextn_predict_layers".to_string(), 1);
        metadata
            .u32s
            .insert("glm-dsa.leading_dense_block_count".to_string(), 1);
        metadata
            .f32s
            .insert("glm-dsa.attention.layer_norm_rms_epsilon".to_string(), 1e-5);
        metadata
            .f32s
            .insert("glm-dsa.expert_weights_scale".to_string(), 2.5);
        metadata
            .bools
            .insert("glm-dsa.expert_weights_norm".to_string(), true);
        metadata.array_strings.insert(
            "glm-dsa.attention.indexer.types".to_string(),
            vec![
                "full".to_string(),
                "shared".to_string(),
                "shared".to_string(),
            ],
        );

        let mut tensors = BTreeSet::from([
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
        ]);
        for layer in 0..3 {
            add_base_layer_tensors(&mut tensors, layer);
            if layer == 0 {
                for suffix in ["ffn_gate.weight", "ffn_down.weight", "ffn_up.weight"] {
                    tensors.insert(format!("blk.{layer}.{suffix}"));
                }
                add_indexer_tensors(&mut tensors, layer);
            } else {
                for suffix in [
                    "ffn_gate_inp.weight",
                    "ffn_gate_exps.weight",
                    "ffn_down_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_gate_shexp.weight",
                    "ffn_down_shexp.weight",
                    "ffn_up_shexp.weight",
                ] {
                    tensors.insert(format!("blk.{layer}.{suffix}"));
                }
            }
        }
        if partial_shared_indexer {
            tensors.insert("blk.1.indexer.k_norm.weight".to_string());
        }
        add_nextn_tensors(&mut tensors, 3);

        ContractInput {
            path: "mock".to_string(),
            artifact_kind: "test".to_string(),
            gguf_files: Vec::new(),
            metadata,
            tensors,
            generation: None,
        }
    }

    fn mock_metadata() -> GgufMetadata {
        let mut metadata = GgufMetadata::default();
        metadata
            .strings
            .insert("general.architecture".to_string(), "glm-dsa".to_string());
        for key in REQUIRED_U32_METADATA {
            metadata.u32s.insert((*key).to_string(), 1);
        }
        metadata
            .u32s
            .insert("glm-dsa.attention.indexer.key_length".to_string(), 2);
        metadata
            .f32s
            .insert("glm-dsa.attention.layer_norm_rms_epsilon".to_string(), 1e-5);
        metadata
            .f32s
            .insert("glm-dsa.expert_weights_scale".to_string(), 2.5);
        metadata
            .bools
            .insert("glm-dsa.expert_weights_norm".to_string(), true);
        metadata
    }

    fn mock_generation() -> PackageGeneration {
        PackageGeneration {
            policy: Some(PackageGenerationPolicy {
                profile: GLM_DSA_POLICY_PROFILE.to_string(),
                decode: GLM_DSA_POLICY_DECODE.to_string(),
                short_prefill: GLM_DSA_POLICY_SHORT_PREFILL.to_string(),
                long_prefill: GLM_DSA_POLICY_LONG_PREFILL.to_string(),
                verify: GLM_DSA_POLICY_VERIFY.to_string(),
                indexshare: Some(GLM_DSA_POLICY_INDEXSHARE.to_string()),
                experimental: Some(PackageGenerationExperimentalPolicy {
                    selected_row_flash: Some(GLM_DSA_POLICY_SELECTED_ROW_FLASH.to_string()),
                }),
            }),
            thresholds: Some(PackageGenerationThresholds {
                short_prefill_max_tokens: Some(GLM_DSA_SHORT_PREFILL_MAX_TOKENS),
                direct_sparse_decode_max_top_k: Some(GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K),
                compact_flash_min_kv: Some(GLM_DSA_COMPACT_FLASH_MIN_KV),
                dense_mask_max_bytes: Some(GLM_DSA_DENSE_MASK_MAX_BYTES),
            }),
            speculative_decoding: None,
        }
    }

    fn add_base_layer_tensors(tensors: &mut BTreeSet<String>, layer: u32) {
        for suffix in BASE_LAYER_TENSORS {
            tensors.insert(format!("blk.{layer}.{suffix}"));
        }
    }

    fn add_moe_layer_tensors(tensors: &mut BTreeSet<String>, layer: u32) {
        for suffix in MOE_LAYER_TENSORS {
            tensors.insert(format!("blk.{layer}.{suffix}"));
        }
    }

    fn add_indexer_tensors(tensors: &mut BTreeSet<String>, layer: u32) {
        for suffix in INDEXER_TENSORS {
            tensors.insert(format!("blk.{layer}.{suffix}"));
        }
    }

    fn remove_indexer_tensors(tensors: &mut BTreeSet<String>, layer: u32) {
        for suffix in INDEXER_TENSORS {
            tensors.remove(&format!("blk.{layer}.{suffix}"));
        }
    }

    fn add_nextn_tensors(tensors: &mut BTreeSet<String>, layer: u32) {
        add_base_layer_tensors(tensors, layer);
        add_moe_layer_tensors(tensors, layer);
        add_indexer_tensors(tensors, layer);
        for suffix in NEXTN_TENSORS {
            tensors.insert(format!("blk.{layer}.{suffix}"));
        }
    }
}
