//! DFlash drafts pinned on Hugging Face: the pairing index that names the
//! default draft for a plain GGUF target, and the validation a package's
//! `dflash` proposers and strategies share with it.

use serde::{Deserialize, Serialize};

use super::{ValidationCode, ValidationErrors, ValidationIssue, push_issue, validate_sha256};

/// A DFlash draft GGUF pinned to a Hugging Face commit and digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DFlashDraft {
    /// Hugging Face model repo, `namespace/name`.
    pub repo: String,
    /// The 40-character commit the draft is pinned to.
    pub revision: String,
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
}

impl DFlashDraft {
    /// Whether the draft names one file in a Hugging Face repo at an immutable
    /// commit, with its size and SHA-256: what a pairing index or package
    /// requires of it.
    pub fn is_pinned(&self) -> bool {
        let mut issues = Vec::new();
        validate_dflash_draft("draft", self, &mut issues);
        issues.is_empty()
    }
}

/// Where a DFlash proposer applies, so a package can choose a different draft
/// per backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DFlashBackends {
    #[default]
    All,
    Metal,
    NotMetal,
}

impl DFlashBackends {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Metal => "metal",
            Self::NotMetal => "not-metal",
        }
    }

    /// Whether this policy covers a Metal (`true`) or other backend.
    pub fn applies_on(self, metal: bool) -> bool {
        match self {
            Self::All => true,
            Self::Metal => metal,
            Self::NotMetal => !metal,
        }
    }
}

/// The schema version of a DFlash pairing index.
pub const DFLASH_PAIRING_INDEX_SCHEMA_VERSION: u32 = 1;

/// The DFlash drafts chosen by default for plain GGUF targets, published as
/// `index.json` in a Hugging Face repo pinned by commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DFlashPairingIndex {
    pub schema_version: u32,
    /// In order of preference: the first pairing that matches a target wins.
    pub pairings: Vec<DFlashPairing>,
}

/// A DFlash draft and the target it is the default for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DFlashPairing {
    pub target: DFlashTarget,
    pub draft: DFlashDraft,
    /// Most draft tokens per block the draft can propose (its block size - 1).
    pub max_draft_tokens: u32,
    /// The backends this pairing is the default on.
    #[serde(default)]
    pub backends: DFlashBackends,
}

/// GGUF metadata that identifies a target model, never its file name or
/// quant, so every quant of a model gets the same draft.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DFlashTarget {
    /// `general.architecture`.
    pub architecture: String,
    /// Executable decoder layers, excluding NextN/MTP blocks.
    pub layer_count: u32,
    pub embedding_length: u32,
    /// Routed experts; `0` for a dense model.
    pub expert_count: u32,
    /// Model family named in `general.basename` (else `general.name`), for
    /// targets that share a shape. With a family, the name must carry it.
    /// Without one, the pairing yields to a same-shape pairing whose family
    /// the name carries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
}

impl DFlashTarget {
    /// Whether `other` has the same architecture, depth, width and experts.
    pub fn same_shape(&self, other: &DFlashTarget) -> bool {
        self.architecture == other.architecture
            && self.layer_count == other.layer_count
            && self.embedding_length == other.embedding_length
            && self.expert_count == other.expert_count
    }
}

impl DFlashPairingIndex {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let index: Self = serde_json::from_str(json).map_err(|error| error.to_string())?;
        index.validate().map_err(|error| error.to_string())?;
        Ok(index)
    }

    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        if self.schema_version != DFLASH_PAIRING_INDEX_SCHEMA_VERSION {
            push_issue(
                &mut issues,
                ValidationCode::UnsupportedSchemaVersion,
                "schema_version",
                format!(
                    "expected schema version {DFLASH_PAIRING_INDEX_SCHEMA_VERSION}, got {}",
                    self.schema_version
                ),
            );
        }
        for (index, pairing) in self.pairings.iter().enumerate() {
            let base = format!("pairings[{index}]");
            let target = &pairing.target;
            if target.architecture.trim().is_empty() {
                push_issue(
                    &mut issues,
                    ValidationCode::MissingValue,
                    format!("{base}.target.architecture"),
                    "architecture must not be empty",
                );
            }
            if target.layer_count == 0 || target.embedding_length == 0 {
                push_issue(
                    &mut issues,
                    ValidationCode::InvalidDimension,
                    format!("{base}.target"),
                    "layer_count and embedding_length must be greater than zero",
                );
            }
            if target
                .family
                .as_ref()
                .is_some_and(|family| family.trim().is_empty())
            {
                push_issue(
                    &mut issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("{base}.target.family"),
                    "family must not be empty when set",
                );
            }
            if !(1..=DFLASH_MAX_DRAFT_TOKENS).contains(&pairing.max_draft_tokens) {
                push_issue(
                    &mut issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("{base}.max_draft_tokens"),
                    format!("max_draft_tokens must be between 1 and {DFLASH_MAX_DRAFT_TOKENS}"),
                );
            }
            validate_dflash_draft(&format!("{base}.draft"), &pairing.draft, &mut issues);
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(ValidationErrors { issues })
        }
    }
}

/// The most draft tokens a DFlash block can carry: the split wire encodes a
/// requested draft depth in four bits.
pub const DFLASH_MAX_DRAFT_TOKENS: u32 = 15;

pub(crate) fn validate_dflash_proposer(
    base: &str,
    draft: &DFlashDraft,
    target_layers: &[u32],
    (block_size, max_draft_tokens): (u32, u32),
    layer_count: u32,
    issues: &mut Vec<ValidationIssue>,
) {
    validate_dflash_draft(&format!("{base}.draft"), draft, issues);
    let mut invalid = |field: &str, message: String| {
        push_issue(
            issues,
            ValidationCode::InvalidGenerationValue,
            format!("{base}.{field}"),
            message,
        );
    };
    if target_layers.is_empty() {
        invalid(
            "target_layers",
            "target_layers must not be empty".to_string(),
        );
    }
    if target_layers.windows(2).any(|pair| pair[0] >= pair[1]) {
        invalid(
            "target_layers",
            "target_layers must be strictly increasing".to_string(),
        );
    }
    if let Some(layer) = target_layers.iter().find(|&&layer| layer >= layer_count) {
        invalid(
            "target_layers",
            format!("target layer {layer} is out of range for layer_count {layer_count}"),
        );
    }
    if !(2..=DFLASH_MAX_DRAFT_TOKENS + 1).contains(&block_size) {
        invalid(
            "block_size",
            format!(
                "block_size must be between 2 and {}",
                DFLASH_MAX_DRAFT_TOKENS + 1
            ),
        );
    }
    if max_draft_tokens == 0 || max_draft_tokens >= block_size {
        invalid(
            "max_draft_tokens",
            "max_draft_tokens must satisfy 1 <= max_draft_tokens < block_size".to_string(),
        );
    }
}

/// A DFlash draft must name one file in a Hugging Face repo at an immutable
/// commit, with its size and SHA-256.
fn validate_dflash_draft(base: &str, draft: &DFlashDraft, issues: &mut Vec<ValidationIssue>) {
    let mut invalid = |field: &str, message: &str| {
        push_issue(
            issues,
            ValidationCode::InvalidGenerationValue,
            format!("{base}.{field}"),
            message,
        );
    };
    let repo_part = |part: &str| {
        !matches!(part, "" | "." | "..")
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    };
    let repo_parts = draft.repo.split('/').collect::<Vec<_>>();
    if repo_parts.len() != 2 || !repo_parts.iter().all(|part| repo_part(part)) {
        invalid(
            "repo",
            "draft repo must be a Hugging Face repo id, namespace/name",
        );
    }
    if draft.revision.len() != 40
        || !draft
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        invalid(
            "revision",
            "draft revision must be an immutable 40-character lowercase Git commit SHA",
        );
    }
    // No `\` or `:`, so no drive prefix or backslash path escapes the cache on
    // Windows.
    if !draft.file.ends_with(".gguf")
        || draft.file.contains(['\\', ':'])
        || draft
            .file
            .split('/')
            .any(|part| matches!(part, "" | "." | ".."))
    {
        invalid("file", "draft file must be a relative path to a .gguf file");
    }
    if draft.bytes == 0 {
        invalid("bytes", "draft bytes must be greater than zero");
    }
    validate_sha256(&format!("{base}.sha256"), &draft.sha256, issues);
}

/// A DFlash strategy names at least one proposer, each once. That each names a
/// declared `dflash` proposer is checked with the other strategies.
pub(crate) fn validate_dflash_strategy(
    name: &str,
    proposers: &[String],
    issues: &mut Vec<ValidationIssue>,
) {
    let base = format!("generation.speculative_decoding.strategies[{name}]");
    if proposers.is_empty() {
        push_issue(
            issues,
            ValidationCode::InvalidGenerationValue,
            format!("{base}.proposers"),
            "proposers must not be empty",
        );
    }
    for (index, proposer_name) in proposers.iter().enumerate() {
        if proposers[..index].contains(proposer_name) {
            push_issue(
                issues,
                ValidationCode::InvalidGenerationValue,
                format!("{base}.proposers[{index}]"),
                format!("proposer {proposer_name:?} is listed twice"),
            );
        }
    }
}
