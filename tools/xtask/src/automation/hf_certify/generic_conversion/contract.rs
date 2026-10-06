use crate::{automation::hf_certify::admission::Artifact, command::DynResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
fn source() -> PathBuf {
    "/mnt/checkpoint".into()
}
fn work() -> PathBuf {
    "/data/skippy-convert".into()
}
fn prefix() -> String {
    "BF16".into()
}
fn splits() -> usize {
    1
}
fn split_size() -> String {
    "50G".into()
}
fn memory() -> String {
    "24G".into()
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u32,
    pub source_repo: String,
    pub target_repo: String,
    pub mesh_revision: String,
    pub output_basename: String,
    pub binary: Option<Artifact>,
    #[serde(default)]
    pub source_files: Vec<Artifact>,
    #[serde(default = "source")]
    pub source: PathBuf,
    #[serde(default = "work")]
    pub work_directory: PathBuf,
    #[serde(default = "prefix")]
    pub target_prefix: String,
    #[serde(default = "splits")]
    pub expected_splits: usize,
    #[serde(default = "split_size")]
    pub split_max_size: String,
    #[serde(default = "memory")]
    pub max_memory: String,
    #[serde(default)]
    pub upload_only: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub publish_confirmed: bool,
    pub helper: Option<Artifact>,
    pub helper_source: Option<Artifact>,
    pub model_publisher: Option<Artifact>,
    pub model_publisher_source: Option<Artifact>,
    pub credential_file: Option<PathBuf>,
    pub timeout_seconds: u64,
}
pub(super) fn hex(value: &str, n: usize) -> bool {
    value.len() == n
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
}
pub(super) fn leaf(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && !matches!(value, "." | "..")
}
fn repo(value: &str) -> bool {
    let parts: Vec<_> = value.split('/').collect();
    parts.len() == 2 && parts.iter().all(|s| leaf(s))
}
fn size(value: &str) -> bool {
    if value.len() > 64 {
        return false;
    }
    let trimmed = value.trim();
    let suffix_start = trimmed
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, suffix) = trimmed.split_at(suffix_start);
    let Ok(number) = digits.parse::<u64>() else {
        return false;
    };
    let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        "t" | "tb" | "tib" => 1024_u64.pow(4),
        _ => return false,
    };
    number.checked_mul(multiplier).is_some()
}

impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        self.validate_inner(true)
    }
    pub(super) fn validate_for_bootstrap(&self) -> DynResult<()> {
        self.validate_inner(false)
    }
    fn validate_inner(&self, require_binary: bool) -> DynResult<()> {
        if self.schema_version != 1
            || !repo(&self.source_repo)
            || !repo(&self.target_repo)
            || !hex(&self.mesh_revision, 40)
            || !leaf(&self.target_prefix)
            || !leaf(&self.output_basename)
            || !(1..=1024).contains(&self.expected_splits)
            || !size(&self.split_max_size)
            || !size(&self.max_memory)
            || !(5..=259200).contains(&self.timeout_seconds)
            || !self.work_directory.is_absolute()
            || !self.source.is_absolute()
            || self.source_files.len() > 1024
        {
            return Err("generic conversion schema/default/profile bounds refused".into());
        }
        if !self.upload_only
            && ((require_binary && self.binary.is_none()) || self.source_files.is_empty())
        {
            return Err(
                "conversion requires observed supplied native binary and complete pinned source"
                    .into(),
            );
        }
        if self.dry_run && (self.upload_only || self.publish_confirmed) {
            return Err("dry-run cannot upload or claim upload-only".into());
        }
        if self
            .source_files
            .iter()
            .map(|p| &p.path)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != self.source_files.len()
        {
            return Err("generic source pins must be unique".into());
        }
        for pin in self
            .source_files
            .iter()
            .chain(self.binary.iter())
            .chain(self.helper.iter())
            .chain(self.helper_source.iter())
            .chain(self.model_publisher.iter())
            .chain(self.model_publisher_source.iter())
        {
            if !pin.path.is_absolute() || !hex(&pin.sha256, 64) {
                return Err("generic conversion absolute artifact byte pin required".into());
            }
        }
        if self.publish_confirmed
            && (self.helper.is_none()
                || self.helper_source.is_none()
                || self.model_publisher.is_none()
                || self.model_publisher_source.is_none()
                || self.expected_splits > 128
                || !self
                    .credential_file
                    .as_ref()
                    .is_some_and(|p| p.is_absolute()))
        {
            return Err(
                "publication requires pinned isolated helper/source and explicit credential file"
                    .into(),
            );
        }
        Ok(())
    }
    pub(super) fn artifact_directory(&self) -> PathBuf {
        self.work_directory.join("target").join(&self.target_prefix)
    }
    pub(super) fn shards(&self, effective: usize) -> Vec<String> {
        (1..=effective)
            .map(|n| {
                if effective == 1 {
                    format!("{}.gguf", self.output_basename)
                } else {
                    format!("{}-{n:05}-of-{:05}.gguf", self.output_basename, effective)
                }
            })
            .collect()
    }
    pub(super) fn convert_args(&self) -> Vec<String> {
        let w = &self.work_directory;
        let pairs = [
            ("--source", self.source.display().to_string()),
            ("--target", w.join("target").display().to_string()),
            ("--target-prefix", self.target_prefix.clone()),
            ("--output-basename", self.output_basename.clone()),
            ("--output-type", "bf16".into()),
            ("--expected-splits", self.expected_splits.to_string()),
            ("--window-size", "1".into()),
            (
                "--manifest",
                w.join("convert-manifest.json").display().to_string(),
            ),
            ("--split-max-size", self.split_max_size.clone()),
            ("--max-memory", self.max_memory.clone()),
            ("--stream-buffer-bytes", "8388608".into()),
            ("--spool-dir", w.join("spool").display().to_string()),
            ("--record-dir", w.join("records").display().to_string()),
            (
                "--json-event-file",
                w.join("status.json").display().to_string(),
            ),
            ("--json-event-interval-seconds", "60".into()),
            ("--json-event-window", "8".into()),
            ("--watchdog-seconds", "300".into()),
        ];
        let mut args = vec!["convert-job".into(), "--mtp".into()];
        for (k, v) in pairs {
            args.extend([k.into(), v]);
        }
        if self.dry_run {
            args.push("--dry-run".into());
        }
        args
    }
    pub(super) fn card(&self) -> Vec<u8> {
        format!("---\nlicense: apache-2.0\nbase_model: {}\ntags:\n- gguf\n- beta\n- skippy\n- mtp\n---\n\n# Inkling MTP sidecar (beta)\n\nThis is a public beta artifact for Skippy compatibility testing. It contains\nInkling's multi-token-prediction depths plus the shared embedding/output\ncontext needed by distributed final stages. It is not a standalone chat model\nand is not a promoted mesh-llm catalog entry.\n\nRequested mesh-llm revision: `{}`. Conversion used a supplied byte-pinned native `skippy-quantize`; source build attribution requires the separate observed G3 bootstrap receipt.\n",self.source_repo,self.mesh_revision).into_bytes()
    }
}
