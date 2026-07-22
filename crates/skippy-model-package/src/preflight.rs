use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

mod artifact_io;

use crate::generation_manifest::{
    PackageGeneration, PackageGenerationExperimentalPolicy, PackageGenerationPolicy,
    PackageGenerationThresholds,
};
use artifact_io::{file_sha256, safe_relative_path, sha256_bytes};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PackagePreflightOptions {
    pub stages: Option<usize>,
    pub verify_sha256: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackagePreflightReport {
    pub schema_version: u32,
    pub package_path: String,
    pub valid: bool,
    pub model_id: Option<String>,
    pub layer_count: Option<u32>,
    pub activation_width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<PreflightGeneration>,
    pub manifest_sha256: Option<String>,
    pub checked_artifact_count: usize,
    pub missing_artifact_count: usize,
    pub issue_count: usize,
    pub issues: Vec<PreflightIssue>,
    pub artifacts: Vec<PreflightArtifact>,
    pub stages: Vec<PreflightStage>,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PreflightSeverity {
    Error,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub(crate) struct PreflightIssue {
    pub severity: PreflightSeverity,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub remediation: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightArtifact {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<u32>,
    pub path: String,
    pub present: bool,
    pub declared_artifact_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_artifact_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_matches_manifest: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256_matches_manifest: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightStage {
    pub stage_index: usize,
    pub layer_start: u32,
    pub layer_end: u32,
    pub includes_embeddings: bool,
    pub includes_output: bool,
    pub part_count: usize,
    pub artifact_bytes: u64,
    pub parts: Vec<String>,
    pub missing_parts: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightGeneration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<PreflightGenerationPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thresholds: Option<PreflightGenerationThresholds>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_decoding: Option<PreflightSpeculativeDecoding>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightGenerationPolicy {
    pub profile: String,
    pub decode: String,
    pub short_prefill: String,
    pub long_prefill: String,
    pub verify: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexshare: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<PreflightGenerationExperimentalPolicy>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightGenerationExperimentalPolicy {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_row_flash: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightGenerationThresholds {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_prefill_max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_sparse_decode_max_top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_flash_min_kv: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_mask_max_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightSpeculativeDecoding {
    pub default: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub proposers: Vec<PreflightSpeculativeProposer>,
    pub strategies: Vec<PreflightSpeculativeStrategy>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightSpeculativeProposer {
    pub name: String,
    #[serde(rename = "type")]
    pub proposer_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction_depth: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub layer_indices: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ngram_min: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ngram_max: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_proposal_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history_scope: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightSpeculativeStrategy {
    pub name: String,
    #[serde(rename = "type")]
    pub strategy_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction_depth: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub layer_indices: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_policy: Option<PreflightWindowPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extender: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extension_policy: Option<PreflightExtensionPolicy>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightExtensionPolicy {
    pub initial_tokens: u32,
    pub max_tokens: u32,
    pub tail_backoff_proposals: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightWindowPolicy {
    pub default: String,
    pub initial_window: u32,
    pub min_window: u32,
    pub max_window: u32,
}

#[derive(Debug, Deserialize)]
struct PackageManifest {
    schema_version: u32,
    model_id: String,
    source_model: PackageSourceModel,
    format: String,
    layer_count: u32,
    #[serde(default)]
    activation_width: Option<u32>,
    #[serde(default)]
    generation: Option<PackageGeneration<PackageSpeculativeDecoding>>,
    shared: PackageShared,
    #[serde(default)]
    projectors: Vec<PackageProjector>,
    layers: Vec<PackageLayer>,
    skippy_abi_version: String,
}

#[derive(Debug, Deserialize)]
struct PackageSourceModel {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PackageShared {
    metadata: PackageArtifact,
    embeddings: PackageArtifact,
    output: PackageArtifact,
}

#[derive(Debug, Deserialize)]
struct PackageSpeculativeDecoding {
    default: String,
    #[serde(default)]
    proposers: BTreeMap<String, PackageSpeculativeProposer>,
    #[serde(default)]
    strategies: BTreeMap<String, PackageSpeculativeStrategy>,
}

#[derive(Debug, Deserialize)]
struct PackageSpeculativeProposer {
    #[serde(rename = "type")]
    proposer_type: String,
    #[serde(default)]
    prediction_depth: Option<u32>,
    #[serde(default)]
    layer_indices: Vec<u32>,
    #[serde(default)]
    ngram_min: Option<u32>,
    #[serde(default)]
    ngram_max: Option<u32>,
    #[serde(default)]
    max_proposal_tokens: Option<u32>,
    #[serde(default)]
    history_scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PackageSpeculativeStrategy {
    #[serde(rename = "type")]
    strategy_type: String,
    #[serde(default)]
    prediction_depth: Option<u32>,
    #[serde(default)]
    layer_indices: Vec<u32>,
    #[serde(default)]
    window_policy: Option<PackageWindowPolicy>,
    #[serde(default)]
    proposer: Option<String>,
    #[serde(default)]
    primary: Option<String>,
    #[serde(default)]
    extender: Option<String>,
    #[serde(default)]
    extension_policy: Option<PackageExtensionPolicy>,
}

#[derive(Debug, Deserialize)]
struct PackageExtensionPolicy {
    initial_tokens: u32,
    max_tokens: u32,
    tail_backoff_proposals: u32,
}

#[derive(Debug, Deserialize)]
struct PackageWindowPolicy {
    default: String,
    initial_window: u32,
    min_window: u32,
    max_window: u32,
}

#[derive(Debug, Deserialize)]
struct PackageArtifact {
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PackageProjector {
    kind: String,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PackageLayer {
    layer_index: u32,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone)]
struct ArtifactSpec {
    role: &'static str,
    layer_index: Option<u32>,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

pub(crate) fn preflight_package(
    package: &Path,
    options: &PackagePreflightOptions,
) -> PackagePreflightReport {
    let mut report = PackagePreflightReport::new(package);
    let manifest_path = package.join("model-package.json");
    let manifest_contents = match fs::read(&manifest_path) {
        Ok(contents) => contents,
        Err(error) => {
            report.error(
                "missing_manifest",
                format!("cannot read package manifest: {error}"),
                Some("model-package.json".to_string()),
                "ensure the package directory contains model-package.json",
            );
            return report.finalize();
        }
    };
    report.manifest_sha256 = Some(sha256_bytes(&manifest_contents));
    let manifest = match serde_json::from_slice::<PackageManifest>(&manifest_contents) {
        Ok(manifest) => manifest,
        Err(error) => {
            report.error(
                "invalid_manifest_json",
                format!("cannot parse package manifest: {error}"),
                Some("model-package.json".to_string()),
                "rebuild the layer package manifest with skippy-model-package write-package",
            );
            return report.finalize();
        }
    };

    report.model_id = Some(manifest.model_id.clone());
    report.layer_count = Some(manifest.layer_count);
    report.activation_width = manifest.activation_width;
    report.generation = manifest.generation.as_ref().map(preflight_generation);
    validate_manifest_header(&manifest, &mut report);
    validate_generation(
        manifest.generation.as_ref(),
        manifest.layer_count,
        &mut report,
    );
    let artifacts = collect_artifacts(&manifest);
    validate_layer_coverage(&manifest, &mut report);
    validate_artifacts(package, &artifacts, options.verify_sha256, &mut report);
    build_stage_reports(&manifest, options.stages, &mut report);
    report.finalize()
}

impl PackagePreflightReport {
    fn new(package: &Path) -> Self {
        Self {
            schema_version: 1,
            package_path: package.display().to_string(),
            valid: true,
            model_id: None,
            layer_count: None,
            activation_width: None,
            generation: None,
            manifest_sha256: None,
            checked_artifact_count: 0,
            missing_artifact_count: 0,
            issue_count: 0,
            issues: Vec::new(),
            artifacts: Vec::new(),
            stages: Vec::new(),
        }
    }

    fn error(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        path: Option<String>,
        remediation: impl Into<String>,
    ) {
        self.issues.push(PreflightIssue {
            severity: PreflightSeverity::Error,
            code: code.into(),
            message: message.into(),
            path,
            remediation: remediation.into(),
        });
    }

    fn finalize(mut self) -> Self {
        self.issue_count = self.issues.len();
        self.checked_artifact_count = self.artifacts.len();
        self.missing_artifact_count = self
            .artifacts
            .iter()
            .filter(|artifact| !artifact.present)
            .count();
        self.valid = !self
            .issues
            .iter()
            .any(|issue| issue.severity == PreflightSeverity::Error);
        self
    }
}

fn validate_manifest_header(manifest: &PackageManifest, report: &mut PackagePreflightReport) {
    if manifest.schema_version != 1 {
        report.error(
            "unsupported_schema_version",
            format!(
                "unsupported package manifest schema_version {}",
                manifest.schema_version
            ),
            Some("model-package.json".to_string()),
            "rebuild the package with a compatible skippy-model-package binary",
        );
    }
    if manifest.format != "layer-package" {
        report.error(
            "invalid_format",
            "package manifest format must be layer-package",
            Some("model-package.json".to_string()),
            "rebuild the package with skippy-model-package write-package",
        );
    }
    if manifest.model_id.trim().is_empty() {
        report.error(
            "empty_model_id",
            "package manifest model_id must not be empty",
            Some("model-package.json".to_string()),
            "rebuild the package with a real model coordinate",
        );
    }
    match manifest.activation_width {
        Some(0) => report.error(
            "invalid_activation_width",
            "package manifest activation_width must be greater than zero",
            Some("model-package.json".to_string()),
            "rebuild the package manifest from the source GGUF metadata",
        ),
        Some(_) => {}
        None => report.error(
            "missing_activation_width",
            "package manifest is missing activation_width",
            Some("model-package.json".to_string()),
            "rebuild the package manifest with a current skippy-model-package write-package",
        ),
    }
    if manifest.source_model.path.trim().is_empty() {
        report.error(
            "empty_source_model_path",
            "package manifest source_model.path must not be empty",
            Some("model-package.json".to_string()),
            "rebuild the package manifest with source model provenance",
        );
    }
    if !is_sha256(&manifest.source_model.sha256) {
        report.error(
            "invalid_source_model_sha256",
            "package manifest source_model.sha256 must be a 64-character hex digest",
            Some("model-package.json".to_string()),
            "rebuild the package manifest from the source GGUF",
        );
    }
    if manifest.skippy_abi_version.trim().is_empty() {
        report.error(
            "missing_skippy_abi_version",
            "package manifest skippy_abi_version is empty",
            Some("model-package.json".to_string()),
            "rebuild the package so runtime compatibility can be checked before serving",
        );
    }
    for projector in &manifest.projectors {
        if projector.kind.trim().is_empty() {
            report.error(
                "empty_projector_kind",
                "package projector kind must not be empty",
                Some(projector.path.clone()),
                "rebuild the package manifest so projector sidecars have a supported kind",
            );
        } else if projector.kind != "mmproj" {
            report.error(
                "unsupported_projector_kind",
                format!("unsupported package projector kind {}", projector.kind),
                Some(projector.path.clone()),
                "rebuild the package with supported mmproj projector sidecars only",
            );
        }
    }
}

fn validate_generation(
    generation: Option<&PackageGeneration<PackageSpeculativeDecoding>>,
    layer_count: u32,
    report: &mut PackagePreflightReport,
) {
    if let Some(generation) = generation {
        if let Some(policy) = generation.policy.as_ref() {
            validate_generation_policy(policy, report);
        }
        if let Some(thresholds) = generation.thresholds.as_ref() {
            validate_generation_thresholds(thresholds, report);
        }
    }
    let Some(speculative) =
        generation.and_then(|generation| generation.speculative_decoding.as_ref())
    else {
        return;
    };
    if speculative.default.trim().is_empty() {
        report.error(
            "empty_speculative_default",
            "generation.speculative_decoding.default must not be empty",
            Some("model-package.json".to_string()),
            "set the default speculative decoding strategy name or remove the generation block",
        );
    } else if !speculative.strategies.contains_key(&speculative.default) {
        report.error(
            "missing_speculative_default_strategy",
            format!(
                "generation.speculative_decoding.default {} is not present in strategies",
                speculative.default
            ),
            Some("model-package.json".to_string()),
            "add the default strategy entry or point default at an existing strategy",
        );
    }
    for (name, proposer) in &speculative.proposers {
        validate_speculative_proposer(name, proposer, layer_count, report);
    }
    for (name, strategy) in &speculative.strategies {
        validate_speculative_strategy(name, strategy, &speculative.proposers, layer_count, report);
    }
}

fn validate_speculative_proposer(
    name: &str,
    proposer: &PackageSpeculativeProposer,
    layer_count: u32,
    report: &mut PackagePreflightReport,
) {
    if name.trim().is_empty() {
        report.error(
            "empty_speculative_proposer_name",
            "generation.speculative_decoding proposer names must not be empty",
            Some("model-package.json".to_string()),
            "use a stable non-empty proposer id such as mtp or ngram-cache",
        );
    }
    match proposer.proposer_type.as_str() {
        "native-mtp" => validate_native_mtp_parts(
            name,
            proposer.prediction_depth,
            &proposer.layer_indices,
            layer_count,
            report,
        ),
        "ngram-simple" | "ngram-cache" => validate_ngram_proposer(name, proposer, report),
        _ => report.error(
            "unsupported_speculative_proposer_type",
            format!(
                "speculative proposer {name} has unsupported type {}",
                proposer.proposer_type
            ),
            Some("model-package.json".to_string()),
            "use native-mtp, ngram-simple, or ngram-cache",
        ),
    }
}

fn validate_generation_policy(
    policy: &PackageGenerationPolicy,
    report: &mut PackagePreflightReport,
) {
    for (field, value) in [
        ("profile", &policy.profile),
        ("decode", &policy.decode),
        ("short_prefill", &policy.short_prefill),
        ("long_prefill", &policy.long_prefill),
        ("verify", &policy.verify),
    ] {
        if value.trim().is_empty() {
            report.error(
                "empty_generation_policy_field",
                format!("generation.policy.{field} must not be empty"),
                Some("model-package.json".to_string()),
                "set a stable package execution policy value or remove generation.policy",
            );
        }
    }
    if let Some(indexshare) = &policy.indexshare
        && indexshare.trim().is_empty()
    {
        report.error(
            "empty_generation_policy_field",
            "generation.policy.indexshare must not be empty when present",
            Some("model-package.json".to_string()),
            "set indexshare to a stable value such as required or remove the field",
        );
    }
    if let Some(selected_row_flash) = policy
        .experimental
        .as_ref()
        .and_then(|experimental| experimental.selected_row_flash.as_ref())
        && selected_row_flash.trim().is_empty()
    {
        report.error(
            "empty_generation_policy_field",
            "generation.policy.experimental.selected_row_flash must not be empty when present",
            Some("model-package.json".to_string()),
            "set selected_row_flash to a stable value such as evidence-gated or remove the field",
        );
    }
}

fn validate_generation_thresholds(
    thresholds: &PackageGenerationThresholds,
    report: &mut PackagePreflightReport,
) {
    if thresholds.short_prefill_max_tokens == Some(0) {
        report.error(
            "invalid_generation_threshold_zero",
            "generation.thresholds.short_prefill_max_tokens must be greater than zero",
            Some("model-package.json".to_string()),
            "set a positive token threshold or remove the field",
        );
    }
    if thresholds.direct_sparse_decode_max_top_k == Some(0) {
        report.error(
            "invalid_generation_threshold_zero",
            "generation.thresholds.direct_sparse_decode_max_top_k must be greater than zero",
            Some("model-package.json".to_string()),
            "set a positive top-k threshold or remove the field",
        );
    }
    if thresholds.compact_flash_min_kv == Some(0) {
        report.error(
            "invalid_generation_threshold_zero",
            "generation.thresholds.compact_flash_min_kv must be greater than zero",
            Some("model-package.json".to_string()),
            "set a positive KV threshold or remove the field",
        );
    }
    if thresholds.dense_mask_max_bytes == Some(0) {
        report.error(
            "invalid_generation_threshold_zero",
            "generation.thresholds.dense_mask_max_bytes must be greater than zero",
            Some("model-package.json".to_string()),
            "set a positive byte threshold or remove the field",
        );
    }
}

fn validate_speculative_strategy(
    name: &str,
    strategy: &PackageSpeculativeStrategy,
    proposers: &BTreeMap<String, PackageSpeculativeProposer>,
    layer_count: u32,
    report: &mut PackagePreflightReport,
) {
    if name.trim().is_empty() {
        report.error(
            "empty_speculative_strategy_name",
            "generation.speculative_decoding strategy names must not be empty",
            Some("model-package.json".to_string()),
            "use a stable non-empty strategy id such as mtp",
        );
    }
    if strategy.strategy_type.trim().is_empty() {
        report.error(
            "empty_speculative_strategy_type",
            format!("speculative strategy {name} type must not be empty"),
            Some("model-package.json".to_string()),
            "set a supported strategy type such as native-mtp",
        );
    }
    if let Some(proposer) = &strategy.proposer {
        validate_proposer_reference(name, "proposer", proposer, proposers, report);
    }
    if strategy.strategy_type == "native-mtp" {
        validate_native_mtp_strategy_proposer_or_inline(
            name,
            strategy,
            proposers,
            layer_count,
            report,
        );
    }
    if matches!(
        strategy.strategy_type.as_str(),
        "ngram-simple" | "ngram-cache"
    ) {
        validate_ngram_strategy_proposer_type(name, strategy, proposers, report);
    }
    if strategy.strategy_type == "composite" {
        validate_composite_strategy(name, strategy, proposers, report);
    }
    if let Some(policy) = &strategy.extension_policy {
        validate_extension_policy(name, policy, report);
    }
    if let Some(window) = &strategy.window_policy {
        validate_window_policy(name, window, report);
    }
}

fn validate_native_mtp_strategy_proposer_or_inline(
    strategy_name: &str,
    strategy: &PackageSpeculativeStrategy,
    proposers: &BTreeMap<String, PackageSpeculativeProposer>,
    layer_count: u32,
    report: &mut PackagePreflightReport,
) {
    let Some(proposer_name) = strategy.proposer.as_deref() else {
        validate_native_mtp_strategy(strategy_name, strategy, layer_count, report);
        return;
    };
    let Some(proposer) = proposers.get(proposer_name) else {
        return;
    };
    if proposer.proposer_type != "native-mtp" {
        report.error(
            "native_mtp_strategy_proposer_type_mismatch",
            format!(
                "native MTP speculative strategy {strategy_name} references proposer {proposer_name} with type {}",
                proposer.proposer_type
            ),
            Some("model-package.json".to_string()),
            "set proposer to a declared native-mtp proposer",
        );
    }
}

fn validate_ngram_strategy_proposer_type(
    strategy_name: &str,
    strategy: &PackageSpeculativeStrategy,
    proposers: &BTreeMap<String, PackageSpeculativeProposer>,
    report: &mut PackagePreflightReport,
) {
    let Some(proposer_name) = strategy.proposer.as_deref() else {
        report.error(
            "missing_ngram_strategy_proposer",
            format!("N-gram speculative strategy {strategy_name} must declare a proposer"),
            Some("model-package.json".to_string()),
            "set proposer to a declared ngram-simple or ngram-cache proposer",
        );
        return;
    };
    let Some(proposer) = proposers.get(proposer_name) else {
        return;
    };
    if proposer.proposer_type != strategy.strategy_type {
        report.error(
            "ngram_strategy_proposer_type_mismatch",
            format!(
                "N-gram speculative strategy {strategy_name} type {} does not match proposer {proposer_name} type {}",
                strategy.strategy_type, proposer.proposer_type
            ),
            Some("model-package.json".to_string()),
            "make the strategy type match its referenced N-gram proposer",
        );
    }
}

fn validate_proposer_reference(
    strategy_name: &str,
    field: &str,
    proposer_name: &str,
    proposers: &BTreeMap<String, PackageSpeculativeProposer>,
    report: &mut PackagePreflightReport,
) {
    if !proposers.contains_key(proposer_name) {
        report.error(
            "missing_speculative_proposer",
            format!("speculative strategy {strategy_name} references missing {field} proposer {proposer_name}"),
            Some("model-package.json".to_string()),
            "declare the referenced proposer under generation.speculative_decoding.proposers",
        );
    }
}

fn validate_composite_strategy(
    name: &str,
    strategy: &PackageSpeculativeStrategy,
    proposers: &BTreeMap<String, PackageSpeculativeProposer>,
    report: &mut PackagePreflightReport,
) {
    let Some(primary) = strategy.primary.as_deref() else {
        report.error(
            "missing_composite_primary",
            format!("composite speculative strategy {name} must declare primary"),
            Some("model-package.json".to_string()),
            "set primary to a declared native-mtp proposer",
        );
        return;
    };
    let Some(extender) = strategy.extender.as_deref() else {
        report.error(
            "missing_composite_extender",
            format!("composite speculative strategy {name} must declare extender"),
            Some("model-package.json".to_string()),
            "set extender to a declared ngram-simple or ngram-cache proposer",
        );
        return;
    };
    validate_proposer_reference(name, "primary", primary, proposers, report);
    validate_proposer_reference(name, "extender", extender, proposers, report);
    if proposers
        .get(primary)
        .is_some_and(|proposer| proposer.proposer_type != "native-mtp")
    {
        report.error(
            "invalid_composite_primary_type",
            format!("composite speculative strategy {name} primary {primary} must be native-mtp"),
            Some("model-package.json".to_string()),
            "set primary to a native-mtp proposer",
        );
    }
    if proposers.get(extender).is_some_and(|proposer| {
        !matches!(
            proposer.proposer_type.as_str(),
            "ngram-simple" | "ngram-cache"
        )
    }) {
        report.error(
            "invalid_composite_extender_type",
            format!("composite speculative strategy {name} extender {extender} must be an N-gram proposer"),
            Some("model-package.json".to_string()),
            "set extender to an ngram-simple or ngram-cache proposer",
        );
    }
}

fn validate_native_mtp_strategy(
    name: &str,
    strategy: &PackageSpeculativeStrategy,
    layer_count: u32,
    report: &mut PackagePreflightReport,
) {
    validate_native_mtp_parts(
        name,
        strategy.prediction_depth,
        &strategy.layer_indices,
        layer_count,
        report,
    );
}

fn validate_native_mtp_parts(
    name: &str,
    prediction_depth: Option<u32>,
    layer_indices: &[u32],
    layer_count: u32,
    report: &mut PackagePreflightReport,
) {
    if prediction_depth != Some(1) {
        report.error(
            "unsupported_native_mtp_prediction_depth",
            format!("native MTP strategy {name} must use prediction_depth 1"),
            Some("model-package.json".to_string()),
            "rebuild the package with the mtp policy supported by this runtime",
        );
    }
    if layer_indices.is_empty() {
        report.error(
            "missing_native_mtp_layers",
            format!("native MTP strategy {name} must declare MTP layer_indices"),
            Some("model-package.json".to_string()),
            "rebuild the package from a GGUF with native MTP tensors",
        );
    }
    for layer_index in layer_indices {
        if *layer_index >= layer_count {
            report.error(
                "native_mtp_layer_out_of_range",
                format!(
                    "native MTP strategy {name} references layer {layer_index}, but layer_count is {layer_count}"
                ),
                Some("model-package.json".to_string()),
                "rebuild the package manifest so MTP layer indices are within the package layer range",
            );
        }
    }
}

fn validate_ngram_proposer(
    name: &str,
    proposer: &PackageSpeculativeProposer,
    report: &mut PackagePreflightReport,
) {
    let min = proposer.ngram_min.unwrap_or_default();
    let max = proposer.ngram_max.unwrap_or_default();
    if min == 0 || max == 0 || min > max {
        report.error(
            "invalid_ngram_proposer_window",
            format!("N-gram proposer {name} must set ngram_min and ngram_max with 1 <= min <= max"),
            Some("model-package.json".to_string()),
            "set positive ngram_min and ngram_max values with min less than or equal to max",
        );
    }
    if proposer.max_proposal_tokens.unwrap_or_default() == 0 {
        report.error(
            "invalid_ngram_proposer_max_tokens",
            format!("N-gram proposer {name} must set max_proposal_tokens greater than zero"),
            Some("model-package.json".to_string()),
            "set max_proposal_tokens to a positive value",
        );
    }
    if proposer.proposer_type == "ngram-cache"
        && max as usize > skippy_runtime::NGRAM_CACHE_MAX_NGRAM
    {
        report.error(
            "unsupported_ngram_cache_max_window",
            format!(
                "N-gram cache proposer {name} ngram_max {max} exceeds llama.cpp limit {}",
                skippy_runtime::NGRAM_CACHE_MAX_NGRAM
            ),
            Some("model-package.json".to_string()),
            format!(
                "set ngram_max to at most {} while keeping max_proposal_tokens independent",
                skippy_runtime::NGRAM_CACHE_MAX_NGRAM
            ),
        );
    }
    if proposer.proposer_type == "ngram-cache"
        && proposer.history_scope.as_deref() != Some("request")
    {
        report.error(
            "invalid_ngram_cache_history_scope",
            format!("N-gram cache proposer {name} must set history_scope to request"),
            Some("model-package.json".to_string()),
            "set history_scope to request; shared cache history is not supported",
        );
    }
}

fn validate_extension_policy(
    name: &str,
    policy: &PackageExtensionPolicy,
    report: &mut PackagePreflightReport,
) {
    if policy.initial_tokens == 0
        || policy.max_tokens == 0
        || policy.initial_tokens > policy.max_tokens
    {
        report.error(
            "invalid_extension_policy_tokens",
            format!("speculative strategy {name} extension_policy must satisfy 1 <= initial_tokens <= max_tokens"),
            Some("model-package.json".to_string()),
            "set positive initial_tokens and max_tokens with initial_tokens no larger than max_tokens",
        );
    }
}

fn validate_window_policy(
    name: &str,
    window: &PackageWindowPolicy,
    report: &mut PackagePreflightReport,
) {
    if window.default.trim().is_empty() {
        report.error(
            "empty_window_policy_default",
            format!("speculative strategy {name} window_policy.default must not be empty"),
            Some("model-package.json".to_string()),
            "set the window policy default to fixed or adaptive",
        );
    }
    if window.min_window == 0 || window.max_window == 0 || window.initial_window == 0 {
        report.error(
            "invalid_window_policy_zero",
            format!("speculative strategy {name} window_policy values must be greater than zero"),
            Some("model-package.json".to_string()),
            "use positive window sizes",
        );
    }
    if window.min_window > window.max_window {
        report.error(
            "invalid_window_policy_bounds",
            format!(
                "speculative strategy {name} window_policy min_window {} exceeds max_window {}",
                window.min_window, window.max_window
            ),
            Some("model-package.json".to_string()),
            "set min_window less than or equal to max_window",
        );
    }
    if window.initial_window < window.min_window || window.initial_window > window.max_window {
        report.error(
            "invalid_window_policy_initial",
            format!(
                "speculative strategy {name} window_policy initial_window {} is outside {}..{}",
                window.initial_window, window.min_window, window.max_window
            ),
            Some("model-package.json".to_string()),
            "set initial_window inside the declared min/max range",
        );
    }
}

fn preflight_generation(
    generation: &PackageGeneration<PackageSpeculativeDecoding>,
) -> PreflightGeneration {
    PreflightGeneration {
        policy: generation.policy.as_ref().map(preflight_generation_policy),
        thresholds: generation
            .thresholds
            .as_ref()
            .map(preflight_generation_thresholds),
        speculative_decoding: generation
            .speculative_decoding
            .as_ref()
            .map(preflight_speculative_decoding),
    }
}

fn preflight_generation_policy(policy: &PackageGenerationPolicy) -> PreflightGenerationPolicy {
    PreflightGenerationPolicy {
        profile: policy.profile.clone(),
        decode: policy.decode.clone(),
        short_prefill: policy.short_prefill.clone(),
        long_prefill: policy.long_prefill.clone(),
        verify: policy.verify.clone(),
        indexshare: policy.indexshare.clone(),
        experimental: policy
            .experimental
            .as_ref()
            .map(preflight_generation_experimental_policy),
    }
}

fn preflight_generation_experimental_policy(
    policy: &PackageGenerationExperimentalPolicy,
) -> PreflightGenerationExperimentalPolicy {
    PreflightGenerationExperimentalPolicy {
        selected_row_flash: policy.selected_row_flash.clone(),
    }
}

fn preflight_generation_thresholds(
    thresholds: &PackageGenerationThresholds,
) -> PreflightGenerationThresholds {
    PreflightGenerationThresholds {
        short_prefill_max_tokens: thresholds.short_prefill_max_tokens,
        direct_sparse_decode_max_top_k: thresholds.direct_sparse_decode_max_top_k,
        compact_flash_min_kv: thresholds.compact_flash_min_kv,
        dense_mask_max_bytes: thresholds.dense_mask_max_bytes,
    }
}

fn preflight_speculative_decoding(
    speculative: &PackageSpeculativeDecoding,
) -> PreflightSpeculativeDecoding {
    PreflightSpeculativeDecoding {
        default: speculative.default.clone(),
        proposers: speculative
            .proposers
            .iter()
            .map(|(name, proposer)| PreflightSpeculativeProposer {
                name: name.clone(),
                proposer_type: proposer.proposer_type.clone(),
                prediction_depth: proposer.prediction_depth,
                layer_indices: proposer.layer_indices.clone(),
                ngram_min: proposer.ngram_min,
                ngram_max: proposer.ngram_max,
                max_proposal_tokens: proposer.max_proposal_tokens,
                history_scope: proposer.history_scope.clone(),
            })
            .collect(),
        strategies: speculative
            .strategies
            .iter()
            .map(|(name, strategy)| PreflightSpeculativeStrategy {
                name: name.clone(),
                strategy_type: strategy.strategy_type.clone(),
                prediction_depth: strategy.prediction_depth,
                layer_indices: strategy.layer_indices.clone(),
                window_policy: strategy.window_policy.as_ref().map(preflight_window_policy),
                proposer: strategy.proposer.clone(),
                primary: strategy.primary.clone(),
                extender: strategy.extender.clone(),
                extension_policy: strategy.extension_policy.as_ref().map(|policy| {
                    PreflightExtensionPolicy {
                        initial_tokens: policy.initial_tokens,
                        max_tokens: policy.max_tokens,
                        tail_backoff_proposals: policy.tail_backoff_proposals,
                    }
                }),
            })
            .collect(),
    }
}

fn preflight_window_policy(window: &PackageWindowPolicy) -> PreflightWindowPolicy {
    PreflightWindowPolicy {
        default: window.default.clone(),
        initial_window: window.initial_window,
        min_window: window.min_window,
        max_window: window.max_window,
    }
}

fn collect_artifacts(manifest: &PackageManifest) -> Vec<ArtifactSpec> {
    let mut artifacts = vec![
        artifact_spec("metadata", None, &manifest.shared.metadata),
        artifact_spec("embeddings", None, &manifest.shared.embeddings),
        artifact_spec("output", None, &manifest.shared.output),
    ];
    artifacts.extend(
        manifest
            .layers
            .iter()
            .map(|layer| layer_artifact_spec(layer.layer_index, layer)),
    );
    artifacts.extend(manifest.projectors.iter().map(projector_artifact_spec));
    artifacts
}

fn artifact_spec(
    role: &'static str,
    layer_index: Option<u32>,
    artifact: &PackageArtifact,
) -> ArtifactSpec {
    ArtifactSpec {
        role,
        layer_index,
        path: artifact.path.clone(),
        tensor_count: artifact.tensor_count,
        tensor_bytes: artifact.tensor_bytes,
        artifact_bytes: artifact.artifact_bytes,
        sha256: artifact.sha256.clone(),
    }
}

fn layer_artifact_spec(layer_index: u32, layer: &PackageLayer) -> ArtifactSpec {
    ArtifactSpec {
        role: "layer",
        layer_index: Some(layer_index),
        path: layer.path.clone(),
        tensor_count: layer.tensor_count,
        tensor_bytes: layer.tensor_bytes,
        artifact_bytes: layer.artifact_bytes,
        sha256: layer.sha256.clone(),
    }
}

fn projector_artifact_spec(projector: &PackageProjector) -> ArtifactSpec {
    ArtifactSpec {
        role: "projector",
        layer_index: None,
        path: projector.path.clone(),
        tensor_count: projector.tensor_count,
        tensor_bytes: projector.tensor_bytes,
        artifact_bytes: projector.artifact_bytes,
        sha256: projector.sha256.clone(),
    }
}

fn validate_layer_coverage(manifest: &PackageManifest, report: &mut PackagePreflightReport) {
    let mut counts = BTreeMap::<u32, usize>::new();
    for layer in &manifest.layers {
        *counts.entry(layer.layer_index).or_default() += 1;
        if layer.layer_index >= manifest.layer_count {
            report.error(
                "layer_index_out_of_range",
                format!(
                    "package layer index {} exceeds layer_count {}",
                    layer.layer_index, manifest.layer_count
                ),
                Some(layer.path.clone()),
                "rebuild the package so layer indexes are contiguous and in range",
            );
        }
    }
    for layer_index in 0..manifest.layer_count {
        if !counts.contains_key(&layer_index) {
            report.error(
                "missing_layer",
                format!("package manifest is missing layer {layer_index}"),
                Some("model-package.json".to_string()),
                "rebuild the package so every transformer layer has one artifact",
            );
        }
    }
    for (layer_index, count) in counts {
        if count > 1 {
            report.error(
                "duplicate_layer",
                format!("package manifest contains layer {layer_index} {count} times"),
                Some("model-package.json".to_string()),
                "rebuild the package so each layer appears exactly once",
            );
        }
    }
}

fn validate_artifacts(
    package: &Path,
    artifacts: &[ArtifactSpec],
    verify_sha256: bool,
    report: &mut PackagePreflightReport,
) {
    for artifact in artifacts {
        report.artifacts.push(preflight_artifact(
            package,
            artifact,
            verify_sha256,
            &mut report.issues,
        ));
    }
}

fn preflight_artifact(
    package: &Path,
    artifact: &ArtifactSpec,
    verify_sha256: bool,
    issues: &mut Vec<PreflightIssue>,
) -> PreflightArtifact {
    let path = match safe_relative_path(&artifact.path) {
        Ok(path) => path,
        Err(message) => {
            push_error(
                issues,
                "unsafe_artifact_path",
                format!(
                    "package {} artifact path is unsafe: {message}",
                    artifact.role
                ),
                Some(artifact.path.clone()),
                "rebuild the package so artifact paths stay inside the package directory",
            );
            return artifact_output(artifact, false, None, None, None);
        }
    };
    validate_artifact_manifest(artifact, issues);
    let absolute = package.join(&path);
    let metadata = match fs::metadata(&absolute) {
        Ok(metadata) if metadata.is_file() => metadata,
        Ok(_) => {
            push_error(
                issues,
                "artifact_not_file",
                format!("package artifact {} is not a file", artifact.path),
                Some(artifact.path.clone()),
                "replace the artifact path with a regular GGUF file",
            );
            return artifact_output(artifact, false, None, None, None);
        }
        Err(error) => {
            push_error(
                issues,
                "missing_artifact",
                format!("package artifact {} is missing: {error}", artifact.path),
                Some(artifact.path.clone()),
                "download or rebuild the package artifact before starting split serving",
            );
            return artifact_output(artifact, false, None, None, None);
        }
    };
    let actual_len = metadata.len();
    let size_matches = actual_len == artifact.artifact_bytes;
    if !size_matches {
        push_error(
            issues,
            "artifact_size_mismatch",
            format!(
                "package artifact {} has {} bytes, manifest expects {}",
                artifact.path, actual_len, artifact.artifact_bytes
            ),
            Some(artifact.path.clone()),
            "redownload or rebuild the package artifact so manifest sizes match",
        );
    }
    let sha_matches = if verify_sha256 {
        Some(validate_artifact_sha(&absolute, artifact, issues))
    } else {
        None
    };
    artifact_output(
        artifact,
        true,
        Some(actual_len),
        Some(size_matches),
        sha_matches,
    )
}

fn validate_artifact_manifest(artifact: &ArtifactSpec, issues: &mut Vec<PreflightIssue>) {
    if artifact.artifact_bytes == 0 {
        push_error(
            issues,
            "empty_artifact",
            format!("package {} artifact declares zero bytes", artifact.role),
            Some(artifact.path.clone()),
            "rebuild the package; split artifacts must be non-empty files",
        );
    }
    if artifact.tensor_count == 0 && artifact.tensor_bytes > 0 {
        push_error(
            issues,
            "invalid_tensor_bytes",
            format!(
                "package {} artifact declares tensor_bytes without tensors",
                artifact.role
            ),
            Some(artifact.path.clone()),
            "rebuild the package manifest so tensor counts and bytes agree",
        );
    }
    if artifact.tensor_count > 0 && artifact.tensor_bytes == 0 {
        push_error(
            issues,
            "invalid_tensor_bytes",
            format!(
                "package {} artifact declares tensors but zero tensor_bytes",
                artifact.role
            ),
            Some(artifact.path.clone()),
            "rebuild the package manifest so tensor counts and bytes agree",
        );
    }
    if !is_sha256(&artifact.sha256) {
        push_error(
            issues,
            "invalid_artifact_sha256",
            format!(
                "package {} artifact sha256 is not a hex digest",
                artifact.role
            ),
            Some(artifact.path.clone()),
            "rebuild the package manifest so artifact checksums are valid",
        );
    }
}

fn validate_artifact_sha(
    path: &Path,
    artifact: &ArtifactSpec,
    issues: &mut Vec<PreflightIssue>,
) -> bool {
    match file_sha256(path) {
        Ok(actual) if actual == artifact.sha256.to_ascii_lowercase() => true,
        Ok(actual) => {
            push_error(
                issues,
                "artifact_sha256_mismatch",
                format!(
                    "package artifact {} checksum mismatch: expected {}, got {}",
                    artifact.path, artifact.sha256, actual
                ),
                Some(artifact.path.clone()),
                "redownload or rebuild the package artifact so checksums match",
            );
            false
        }
        Err(error) => {
            push_error(
                issues,
                "artifact_sha256_unreadable",
                format!("cannot hash package artifact {}: {error}", artifact.path),
                Some(artifact.path.clone()),
                "ensure the artifact is readable before enabling checksum verification",
            );
            false
        }
    }
}

fn artifact_output(
    artifact: &ArtifactSpec,
    present: bool,
    actual_artifact_bytes: Option<u64>,
    size_matches_manifest: Option<bool>,
    sha256_matches_manifest: Option<bool>,
) -> PreflightArtifact {
    PreflightArtifact {
        role: artifact.role.to_string(),
        layer_index: artifact.layer_index,
        path: artifact.path.clone(),
        present,
        declared_artifact_bytes: artifact.artifact_bytes,
        actual_artifact_bytes,
        size_matches_manifest,
        sha256_matches_manifest,
    }
}

fn build_stage_reports(
    manifest: &PackageManifest,
    stages: Option<usize>,
    report: &mut PackagePreflightReport,
) {
    let Some(stage_count) = stages else {
        return;
    };
    if stage_count == 0 {
        report.error(
            "invalid_stage_count",
            "--stages must be greater than zero",
            Some("model-package.json".to_string()),
            "choose a positive stage count for split preflight",
        );
        return;
    }
    if stage_count as u32 > manifest.layer_count {
        report.error(
            "stage_count_exceeds_layer_count",
            format!(
                "--stages {stage_count} exceeds package layer_count {}",
                manifest.layer_count
            ),
            Some("model-package.json".to_string()),
            "use at most one split stage per transformer layer",
        );
        return;
    }
    let artifact_map = stage_artifacts(report);
    for (stage_index, (layer_start, layer_end)) in
        partition_layers(manifest.layer_count, stage_count)
            .into_iter()
            .enumerate()
    {
        report.stages.push(stage_report(
            stage_index,
            layer_start,
            layer_end,
            stage_count,
            &artifact_map,
        ));
    }
}

fn stage_report(
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    stage_count: usize,
    artifact_map: &BTreeMap<String, StageArtifact>,
) -> PreflightStage {
    let includes_embeddings = stage_index == 0;
    let includes_output = stage_index + 1 == stage_count;
    let mut parts = vec!["metadata".to_string()];
    if includes_embeddings {
        parts.push("embeddings".to_string());
    }
    for layer_index in layer_start..layer_end {
        parts.push(format!("layer:{layer_index}"));
    }
    if includes_output {
        parts.push("output".to_string());
    }
    let artifact_bytes = parts
        .iter()
        .filter_map(|part| artifact_map.get(part))
        .filter(|artifact| artifact.present)
        .map(|artifact| artifact.bytes)
        .sum();
    let missing_parts = parts
        .iter()
        .filter(|part| {
            !artifact_map
                .get(*part)
                .is_some_and(|artifact| artifact.present)
        })
        .cloned()
        .collect::<Vec<_>>();
    PreflightStage {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        part_count: parts.len(),
        artifact_bytes,
        parts,
        missing_parts,
    }
}

#[derive(Clone, Copy)]
struct StageArtifact {
    present: bool,
    bytes: u64,
}

fn stage_artifacts(report: &PackagePreflightReport) -> BTreeMap<String, StageArtifact> {
    report
        .artifacts
        .iter()
        .map(|artifact| {
            (
                stage_part_key(artifact),
                StageArtifact {
                    present: artifact.present,
                    bytes: artifact
                        .actual_artifact_bytes
                        .unwrap_or(artifact.declared_artifact_bytes),
                },
            )
        })
        .collect()
}

fn stage_part_key(artifact: &PreflightArtifact) -> String {
    match (artifact.role.as_str(), artifact.layer_index) {
        ("layer", Some(layer)) => format!("layer:{layer}"),
        (role, _) => role.to_string(),
    }
}

fn partition_layers(layer_count: u32, stages: usize) -> Vec<(u32, u32)> {
    let base = layer_count / stages as u32;
    let extra = layer_count % stages as u32;
    let mut start = 0;
    (0..stages)
        .map(|stage_index| {
            let width = base + u32::from((stage_index as u32) < extra);
            let end = start + width;
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

fn push_error(
    issues: &mut Vec<PreflightIssue>,
    code: impl Into<String>,
    message: impl Into<String>,
    path: Option<String>,
    remediation: impl Into<String>,
) {
    issues.push(PreflightIssue {
        severity: PreflightSeverity::Error,
        code: code.into(),
        message: message.into(),
        path,
        remediation: remediation.into(),
    });
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn preflight_accepts_complete_package_and_reports_stage_parts() {
        let dir = unique_test_dir("valid");
        let package = write_package_fixture(&dir, true);

        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: Some(2),
                verify_sha256: true,
            },
        );

        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.activation_width, Some(4096));
        assert_eq!(report.checked_artifact_count, 5);
        assert_eq!(report.stages.len(), 2);
        assert_eq!(
            report.stages[0].parts,
            ["metadata", "embeddings", "layer:0"]
        );
        assert_eq!(report.stages[1].parts, ["metadata", "layer:1", "output"]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_rejects_missing_activation_width() {
        let dir = unique_test_dir("missing-width");
        let package = write_package_fixture(&dir, false);

        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: None,
                verify_sha256: false,
            },
        );

        assert!(!report.valid);
        assert_issue(&report, "missing_activation_width");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_reports_missing_shared_embedding_before_split_startup() {
        let dir = unique_test_dir("missing-embeddings");
        let package = write_package_fixture(&dir, true);
        fs::remove_file(package.join("shared/embeddings.gguf")).unwrap();

        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: Some(2),
                verify_sha256: false,
            },
        );

        assert!(!report.valid);
        assert_issue(&report, "missing_artifact");
        assert!(
            report.stages[0]
                .missing_parts
                .contains(&"embeddings".to_string())
        );
        assert_eq!(
            report.stages[0].artifact_bytes,
            (b"metadata".len() + b"layer0".len()) as u64
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_detects_artifact_sha_mismatch_when_requested() {
        let dir = unique_test_dir("sha-mismatch");
        let package = write_package_fixture(&dir, true);
        fs::write(package.join("layers/layer-001.gguf"), b"corrupt1").unwrap();

        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: None,
                verify_sha256: true,
            },
        );

        assert!(!report.valid);
        assert_issue(&report, "artifact_sha256_mismatch");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn artifact_sha_verification_uses_resolved_safe_path() {
        let dir = unique_test_dir("resolved-sha-path");
        fs::create_dir_all(&dir).unwrap();
        let resolved = dir.join("resolved.gguf");
        fs::write(&resolved, b"resolved").unwrap();
        fs::write(dir.join("manifest-path.gguf"), b"other").unwrap();
        let artifact = ArtifactSpec {
            role: "metadata",
            layer_index: None,
            path: "manifest-path.gguf".to_string(),
            tensor_count: 1,
            tensor_bytes: b"resolved".len() as u64,
            artifact_bytes: b"resolved".len() as u64,
            sha256: file_sha256(&resolved).unwrap(),
        };
        let mut issues = Vec::new();

        assert!(validate_artifact_sha(&resolved, &artifact, &mut issues));
        assert!(issues.is_empty(), "{issues:?}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_rejects_stage_count_above_layer_count() {
        let dir = unique_test_dir("too-many-stages");
        let package = write_package_fixture(&dir, true);

        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: Some(3),
                verify_sha256: false,
            },
        );

        assert!(!report.valid);
        assert_issue(&report, "stage_count_exceeds_layer_count");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_reports_native_mtp_generation_defaults() {
        let dir = unique_test_dir("native-mtp-generation");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "mtp",
                    "strategies": {
                        "mtp": {
                            "type": "native-mtp",
                            "prediction_depth": 1,
                            "layer_indices": [1],
                            "window_policy": {
                                "default": "fixed",
                                "initial_window": 1,
                                "min_window": 1,
                                "max_window": 1
                            }
                        }
                    }
                }
            }),
        );

        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: None,
                verify_sha256: false,
            },
        );

        assert!(report.valid, "{:?}", report.issues);
        let generation = report.generation.expect("generation should be reported");
        let speculative = generation
            .speculative_decoding
            .expect("speculative decoding should be reported");
        assert_eq!(speculative.default, "mtp");
        assert_eq!(speculative.strategies.len(), 1);
        assert_eq!(speculative.strategies[0].name, "mtp");
        assert_eq!(speculative.strategies[0].strategy_type, "native-mtp");
        assert_eq!(speculative.strategies[0].prediction_depth, Some(1));
        assert_eq!(speculative.strategies[0].layer_indices, [1]);
        let window_policy = speculative.strategies[0]
            .window_policy
            .as_ref()
            .expect("window policy should be reported");
        assert_eq!(window_policy.default, "fixed");
        assert_eq!(window_policy.initial_window, 1);
        assert_eq!(window_policy.min_window, 1);
        assert_eq!(window_policy.max_window, 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_accepts_request_local_ngram_cache_composite_strategy() {
        let dir = unique_test_dir("ngram-cache-composite");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "mtp-cache",
                    "proposers": {
                        "mtp": {
                            "type": "native-mtp",
                            "prediction_depth": 1,
                            "layer_indices": [1]
                        },
                        "cache": {
                            "type": "ngram-cache",
                            "ngram_min": 2,
                            "ngram_max": 4,
                            "max_proposal_tokens": 4,
                            "history_scope": "request"
                        }
                    },
                    "strategies": {
                        "mtp-cache": {
                            "type": "composite",
                            "primary": "mtp",
                            "extender": "cache",
                            "extension_policy": {
                                "initial_tokens": 2,
                                "max_tokens": 4,
                                "tail_backoff_proposals": 6
                            }
                        }
                    }
                }
            }),
        );

        let report = preflight_package(&package, &PackagePreflightOptions::default());

        assert!(report.valid, "{:?}", report.issues);
        let speculative = report
            .generation
            .and_then(|generation| generation.speculative_decoding)
            .expect("generation strategy should be reported");
        assert_eq!(speculative.proposers.len(), 2);
        assert_eq!(speculative.strategies[0].primary.as_deref(), Some("mtp"));
        assert_eq!(speculative.strategies[0].extender.as_deref(), Some("cache"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_accepts_complete_native_and_ngram_strategy_matrix() {
        let dir = unique_test_dir("complete-speculative-matrix");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "mtp",
                    "proposers": {
                        "mtp": {
                            "type": "native-mtp",
                            "prediction_depth": 1,
                            "layer_indices": [1]
                        },
                        "simple": {
                            "type": "ngram-simple",
                            "ngram_min": 2,
                            "ngram_max": 6,
                            "max_proposal_tokens": 6
                        },
                        "cache": {
                            "type": "ngram-cache",
                            "ngram_min": 2,
                            "ngram_max": 4,
                            "max_proposal_tokens": 6,
                            "history_scope": "request"
                        }
                    },
                    "strategies": {
                        "mtp": {
                            "type": "native-mtp",
                            "proposer": "mtp"
                        },
                        "ngram-simple": {
                            "type": "ngram-simple",
                            "proposer": "simple"
                        },
                        "ngram-cache": {
                            "type": "ngram-cache",
                            "proposer": "cache"
                        },
                        "mtp-simple": {
                            "type": "composite",
                            "primary": "mtp",
                            "extender": "simple",
                            "extension_policy": {
                                "initial_tokens": 2,
                                "max_tokens": 6,
                                "tail_backoff_proposals": 2
                            }
                        },
                        "mtp-cache": {
                            "type": "composite",
                            "primary": "mtp",
                            "extender": "cache",
                            "extension_policy": {
                                "initial_tokens": 2,
                                "max_tokens": 6,
                                "tail_backoff_proposals": 2
                            }
                        }
                    }
                }
            }),
        );

        let report = preflight_package(&package, &PackagePreflightOptions::default());

        assert!(report.valid, "{:?}", report.issues);
        let strategies = report
            .generation
            .and_then(|generation| generation.speculative_decoding)
            .expect("generation strategies should be reported")
            .strategies;
        assert_eq!(strategies.len(), 5);
        assert!(
            strategies
                .iter()
                .any(|strategy| strategy.name == "mtp-simple")
        );
        assert!(
            strategies
                .iter()
                .any(|strategy| strategy.name == "mtp-cache")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_rejects_ngram_strategy_with_mismatched_proposer_type() {
        let dir = unique_test_dir("ngram-strategy-type-mismatch");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "simple",
                    "proposers": {
                        "cache": {
                            "type": "ngram-cache",
                            "ngram_min": 2,
                            "ngram_max": 4,
                            "max_proposal_tokens": 4,
                            "history_scope": "request"
                        }
                    },
                    "strategies": {
                        "simple": { "type": "ngram-simple", "proposer": "cache" }
                    }
                }
            }),
        );

        let report = preflight_package(&package, &PackagePreflightOptions::default());

        assert!(!report.valid);
        assert_issue(&report, "ngram_strategy_proposer_type_mismatch");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_rejects_native_mtp_strategy_with_ngram_proposer() {
        let dir = unique_test_dir("native-mtp-strategy-type-mismatch");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "mtp",
                    "proposers": {
                        "simple": {
                            "type": "ngram-simple",
                            "ngram_min": 2,
                            "ngram_max": 4,
                            "max_proposal_tokens": 4
                        }
                    },
                    "strategies": {
                        "mtp": { "type": "native-mtp", "proposer": "simple" }
                    }
                }
            }),
        );

        let report = preflight_package(&package, &PackagePreflightOptions::default());

        assert!(!report.valid);
        assert_issue(&report, "native_mtp_strategy_proposer_type_mismatch");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_rejects_shared_ngram_cache_history() {
        let dir = unique_test_dir("ngram-cache-shared-history");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "cache",
                    "proposers": {
                        "cache": {
                            "type": "ngram-cache",
                            "ngram_min": 2,
                            "ngram_max": 4,
                            "max_proposal_tokens": 4,
                            "history_scope": "shared"
                        }
                    },
                    "strategies": {
                        "cache": { "type": "ngram-cache", "proposer": "cache" }
                    }
                }
            }),
        );

        let report = preflight_package(&package, &PackagePreflightOptions::default());

        assert!(!report.valid);
        assert_issue(&report, "invalid_ngram_cache_history_scope");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_rejects_ngram_cache_window_above_llama_limit() {
        let dir = unique_test_dir("ngram-cache-max-window");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "cache",
                    "proposers": {
                        "cache": {
                            "type": "ngram-cache",
                            "ngram_min": 2,
                            "ngram_max": 5,
                            "max_proposal_tokens": 6,
                            "history_scope": "request"
                        }
                    },
                    "strategies": {
                        "cache": { "type": "ngram-cache", "proposer": "cache" }
                    }
                }
            }),
        );

        let report = preflight_package(&package, &PackagePreflightOptions::default());

        assert!(!report.valid);
        assert_issue(&report, "unsupported_ngram_cache_max_window");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preflight_rejects_native_mtp_layer_out_of_range() {
        let dir = unique_test_dir("native-mtp-out-of-range");
        let package = write_package_fixture(&dir, true);
        write_generation_to_manifest(
            &package,
            serde_json::json!({
                "speculative_decoding": {
                    "default": "mtp",
                    "strategies": {
                        "mtp": {
                            "type": "native-mtp",
                            "prediction_depth": 1,
                            "layer_indices": [2],
                            "window_policy": {
                                "default": "fixed",
                                "initial_window": 1,
                                "min_window": 1,
                                "max_window": 1
                            }
                        }
                    }
                }
            }),
        );

        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: None,
                verify_sha256: false,
            },
        );

        assert!(!report.valid);
        assert_issue(&report, "native_mtp_layer_out_of_range");
        fs::remove_dir_all(dir).unwrap();
    }

    fn assert_issue(report: &PackagePreflightReport, code: &str) {
        assert!(
            report.issues.iter().any(|issue| issue.code == code),
            "missing issue {code}; issues: {:?}",
            report.issues
        );
    }

    fn write_package_fixture(root: &Path, include_activation_width: bool) -> PathBuf {
        let package = root.join("package");
        fs::create_dir_all(package.join("shared")).unwrap();
        fs::create_dir_all(package.join("layers")).unwrap();
        write_artifact(&package, "shared/metadata.gguf", b"metadata");
        write_artifact(&package, "shared/embeddings.gguf", b"embeddings");
        write_artifact(&package, "shared/output.gguf", b"output");
        write_artifact(&package, "layers/layer-000.gguf", b"layer0");
        write_artifact(&package, "layers/layer-001.gguf", b"layer1");
        let mut manifest = serde_json::json!({
            "schema_version": 1,
            "model_id": "meshllm/test-model-layers",
            "source_model": {
                "path": "test-model.gguf",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "format": "layer-package",
            "layer_count": 2,
            "shared": {
                "metadata": artifact_json(&package, "shared/metadata.gguf", b"metadata"),
                "embeddings": artifact_json(&package, "shared/embeddings.gguf", b"embeddings"),
                "output": artifact_json(&package, "shared/output.gguf", b"output")
            },
            "layers": [
                layer_json(&package, 0, "layers/layer-000.gguf", b"layer0"),
                layer_json(&package, 1, "layers/layer-001.gguf", b"layer1")
            ],
            "skippy_abi_version": "1.0.0"
        });
        if include_activation_width {
            manifest["activation_width"] = serde_json::json!(4096);
        }
        fs::write(
            package.join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        package
    }

    fn write_generation_to_manifest(package: &Path, generation: serde_json::Value) {
        let manifest_path = package.join("model-package.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["generation"] = generation;
        fs::write(manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }

    fn write_artifact(package: &Path, relative_path: &str, bytes: &[u8]) {
        let path = package.join(relative_path);
        fs::write(path, bytes).unwrap();
    }

    fn artifact_json(package: &Path, path: &str, bytes: &[u8]) -> serde_json::Value {
        serde_json::json!({
            "path": path,
            "tensor_count": 1,
            "tensor_bytes": bytes.len(),
            "artifact_bytes": bytes.len(),
            "sha256": file_sha256(&package.join(path)).unwrap()
        })
    }

    fn layer_json(package: &Path, layer_index: u32, path: &str, bytes: &[u8]) -> serde_json::Value {
        let mut value = artifact_json(package, path, bytes);
        value["layer_index"] = serde_json::json!(layer_index);
        value
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skippy-model-package-preflight-{name}-{}-{nanos}",
            std::process::id()
        ))
    }
}
