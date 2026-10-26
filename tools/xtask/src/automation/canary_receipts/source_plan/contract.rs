use crate::automation::canary_receipts::Digest;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub(super) controller_root: ControllerRoot,
    pub(super) source_root: SelectedSourceRoot,
    pub(super) controller_revision: Revision,
    pub(super) selected_revision: Revision,
    pub(super) manifest: SelectedManifest,
    pub(super) output: PlanOutput,
    pub(super) cache: CachePolicy,
}

#[derive(Deserialize)]
#[serde(transparent)]
pub(super) struct ControllerRoot(PathBuf);
#[derive(Deserialize)]
#[serde(transparent)]
pub(super) struct SelectedSourceRoot(PathBuf);
#[derive(Deserialize)]
#[serde(transparent)]
pub(super) struct SelectedManifest(PathBuf);
#[derive(Deserialize)]
#[serde(transparent)]
pub(super) struct PlanOutput(PathBuf);

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(try_from = "String")]
pub(super) struct Revision(String);

impl TryFrom<String> for Revision {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 40
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value))
        } else {
            Err("expected lowercase 40-hex revision")
        }
    }
}

impl Revision {
    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum CachePolicy {
    NotChecked,
    BlobIdentity { root: PathBuf },
    GgufMetadata { root: PathBuf },
}

#[derive(Clone, Copy, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CacheAdmission {
    NotChecked,
    BlobIdentity,
    GgufMetadata,
}

#[derive(Clone, Copy, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GgufAdmission {
    Pending,
    MetadataAdmitted,
}

impl CacheAdmission {
    pub(super) fn gguf_admission(self) -> GgufAdmission {
        match self {
            Self::NotChecked | Self::BlobIdentity => GgufAdmission::Pending,
            Self::GgufMetadata => GgufAdmission::MetadataAdmitted,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PlanIdentity {
    pub(super) schema: u8,
    pub(super) controller_revision: Revision,
    pub(super) selected_revision: Revision,
    pub(super) manifest_sha256: Digest,
    pub(super) plan_sha256: Digest,
    pub(super) cache_admission: CacheAdmission,
    pub(super) gguf_admission: GgufAdmission,
}

pub(super) struct Resolved {
    pub(super) controller: PathBuf,
    pub(super) source: PathBuf,
    pub(super) manifest: PathBuf,
    pub(super) output: PathBuf,
}

impl Input {
    pub(super) fn resolve(&self) -> DynResult<Resolved> {
        let controller = absolute_directory(&self.controller_root.0)?;
        let source = absolute_directory(&self.source_root.0)?;
        if self.manifest.0.as_os_str().is_empty()
            || self
                .manifest
                .0
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err("manifest must be a selected-source relative path".into());
        }
        let manifest = source.join(&self.manifest.0).canonicalize()?;
        if !manifest.starts_with(&source) || !manifest.is_file() {
            return Err("manifest escapes selected source or is not a regular file".into());
        }
        let output = &self.output.0;
        if !output.is_absolute()
            || output
                .components()
                .any(|part| matches!(part, Component::ParentDir))
        {
            return Err("output must be an absolute normalized path".into());
        }
        let parent = absolute_directory(output.parent().ok_or("output has no parent")?)?;
        let output = parent.join(output.file_name().ok_or("output has no basename")?);
        if output.starts_with(&source) || output.starts_with(&controller) {
            return Err("output must be outside controller and selected source checkouts".into());
        }
        Ok(Resolved {
            controller,
            source,
            manifest,
            output,
        })
    }
}

fn absolute_directory(path: &Path) -> DynResult<PathBuf> {
    if !path.is_absolute() || !path.is_dir() {
        return Err("checkout/cache directory must exist and be absolute".into());
    }
    Ok(path.canonicalize()?)
}
