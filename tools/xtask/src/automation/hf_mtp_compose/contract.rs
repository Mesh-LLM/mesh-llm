use crate::automation::hf_certify::admission::Artifact;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub binary: Artifact,
    pub target_parts: Vec<Artifact>,
    pub mtp_gguf: Artifact,
    pub target_basename: String,
    pub composite_basename: String,
    pub expected_parts: usize,
    pub mtp_block: u32,
    pub supplied_mesh_revision: String,
    pub native_profile: String,
    pub timeout_secs: u64,
    pub composite_repo: String,
}
fn basename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 180
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && value != "."
        && value != ".."
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.native_profile != "standalone-static-skippy-quantize-cpu"
            || !(5..=86400).contains(&self.timeout_secs)
            || !(2..=1024).contains(&self.expected_parts)
            || self.expected_parts != self.target_parts.len()
            || self.mtp_block == 0
            || self.mtp_block > 65535
            || !basename(&self.target_basename)
            || !basename(&self.composite_basename)
            || self.target_basename == self.composite_basename
            || self.supplied_mesh_revision.len() != 40
            || !self
                .supplied_mesh_revision
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid bounded compose profile/roster/basename/revision".into());
        }
        let repo = self.composite_repo.split('/').collect::<Vec<_>>();
        if repo.len() != 2 || repo.iter().any(|s| !basename(s)) {
            return Err("invalid composite model repository".into());
        }
        let mut unique = std::collections::BTreeSet::new();
        for artifact in std::iter::once(&self.binary)
            .chain(std::iter::once(&self.mtp_gguf))
            .chain(self.target_parts.iter())
        {
            if !artifact.path.is_absolute()
                || !unique.insert(&artifact.path)
                || artifact.sha256.len() != 64
                || !artifact
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return Err(
                    "compose artifacts require unique absolute paths and complete SHA256 pins"
                        .into(),
                );
            }
        }
        for (index, part) in self.target_parts.iter().enumerate() {
            if part.path.file_name().and_then(|s| s.to_str())
                != Some(
                    format!(
                        "{}-{:05}-of-{:05}.gguf",
                        self.target_basename,
                        index + 1,
                        self.expected_parts
                    )
                    .as_str(),
                )
            {
                return Err(
                    "compose target must be complete ordered canonical split roster".into(),
                );
            }
        }
        Ok(())
    }
    pub(super) fn remote_name(&self, index: usize) -> String {
        format!(
            "{}-{:05}-of-{:05}.gguf",
            self.composite_basename,
            index + 1,
            self.expected_parts
        )
    }
}
#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Identity {
    pub schema_version: u64,
    pub request_sha256: String,
    pub admitted: Input,
    pub outputs: Vec<Artifact>,
}
