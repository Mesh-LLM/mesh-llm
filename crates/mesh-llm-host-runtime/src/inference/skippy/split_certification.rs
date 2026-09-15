use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::{SkippyPackageIdentity, is_content_addressed_gguf_ref};

const ROSTER_JSON: &str = include_str!("split-certified.json");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SplitCertificationAdmission {
    Certified,
    UncertifiedOverride,
}

impl SplitCertificationAdmission {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::UncertifiedOverride => "uncertified_override",
        }
    }
}

#[derive(Debug, Deserialize)]
struct SplitCertificationRoster {
    schema_version: u32,
    native_recipe: NativeRecipe,
    models: Vec<CertifiedModel>,
}

#[derive(Debug, Deserialize)]
struct NativeRecipe {
    llama_upstream_sha: String,
    skippy_abi: String,
    patch_queue_sha256: String,
}

#[derive(Debug, Deserialize)]
struct CertifiedModel {
    family: String,
    package_kind: String,
    source_model_sha256: String,
    #[serde(default)]
    manifest_sha256: Option<String>,
}

fn roster() -> Result<&'static SplitCertificationRoster> {
    static ROSTER: OnceLock<Result<SplitCertificationRoster, String>> = OnceLock::new();
    ROSTER
        .get_or_init(|| {
            serde_json::from_str(ROSTER_JSON)
                .map_err(|error| format!("parse bundled split certification roster: {error}"))
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!(error.clone()))
}

fn compiled_skippy_abi() -> String {
    format!(
        "{}.{}.{}",
        skippy_ffi::ABI_VERSION_MAJOR,
        skippy_ffi::ABI_VERSION_MINOR,
        skippy_ffi::ABI_VERSION_PATCH
    )
}

fn validate_recipe(roster: &SplitCertificationRoster) -> Result<()> {
    anyhow::ensure!(
        roster.schema_version == 1,
        "unsupported split certification roster schema {}",
        roster.schema_version
    );
    anyhow::ensure!(
        roster.native_recipe.llama_upstream_sha == env!("MESH_LLAMA_UPSTREAM_SHA"),
        "split certification roster targets llama.cpp {}, but this build uses {}",
        roster.native_recipe.llama_upstream_sha,
        env!("MESH_LLAMA_UPSTREAM_SHA")
    );
    anyhow::ensure!(
        roster.native_recipe.patch_queue_sha256 == env!("MESH_SKIPPY_PATCH_QUEUE_SHA256"),
        "split certification roster does not match this build's Skippy patch queue"
    );
    anyhow::ensure!(
        roster.native_recipe.skippy_abi == compiled_skippy_abi(),
        "split certification roster targets Skippy ABI {}, but this build uses {}",
        roster.native_recipe.skippy_abi,
        compiled_skippy_abi()
    );
    Ok(())
}

fn certified_model<'a>(
    roster: &'a SplitCertificationRoster,
    package: &SkippyPackageIdentity,
) -> Option<&'a CertifiedModel> {
    roster.models.iter().find(|model| {
        model.source_model_sha256 == package.source_model_sha256
            && match model.package_kind.as_str() {
                "content-addressed-direct-gguf" => {
                    model.manifest_sha256.is_none()
                        && is_content_addressed_gguf_ref(&package.package_ref)
                }
                "package-v2" => model
                    .manifest_sha256
                    .as_deref()
                    .is_some_and(|digest| digest == package.manifest_sha256),
                _ => false,
            }
    })
}

fn certified_family(package: &SkippyPackageIdentity) -> Result<Option<&'static str>> {
    let roster = roster().context("load split certification roster")?;
    validate_recipe(roster).context("validate split certification roster recipe")?;
    Ok(certified_model(roster, package).map(|model| model.family.as_str()))
}

pub(crate) fn require_split_certification(
    package: &SkippyPackageIdentity,
    allow_uncertified: bool,
) -> Result<SplitCertificationAdmission> {
    let family = match certified_family(package) {
        Ok(Some(family)) => family,
        Ok(None) => return handle_uncertified(package, allow_uncertified, None),
        Err(error) => return handle_uncertified(package, allow_uncertified, Some(error)),
    };
    tracing::info!(
        family,
        package_ref = package.package_ref,
        source_model_sha256 = package.source_model_sha256,
        manifest_sha256 = package.manifest_sha256,
        split_certification = "certified",
        "admitted split-serving model from exact-artifact certification roster"
    );
    Ok(SplitCertificationAdmission::Certified)
}

fn handle_uncertified(
    package: &SkippyPackageIdentity,
    allow_uncertified: bool,
    roster_error: Option<anyhow::Error>,
) -> Result<SplitCertificationAdmission> {
    if !allow_uncertified {
        if let Some(error) = roster_error {
            return Err(error).context(format!(
                "split serving certification cannot be established for {}; retry explicitly with --split --allow-uncertified-split to run it experimentally",
                package.package_ref
            ));
        }
        anyhow::bail!(
            "split serving is not certified for model artifact {} (source SHA-256 {}, manifest SHA-256 {}); retry explicitly with --split --allow-uncertified-split to run it experimentally",
            package.package_ref,
            package.source_model_sha256,
            package.manifest_sha256
        );
    }
    if let Some(error) = roster_error {
        tracing::warn!(
            package_ref = package.package_ref,
            source_model_sha256 = package.source_model_sha256,
            manifest_sha256 = package.manifest_sha256,
            error = %error,
            split_certification = "uncertified_override",
            "unsafe override admitted a split-serving model without a valid certification roster"
        );
    } else {
        tracing::warn!(
            package_ref = package.package_ref,
            source_model_sha256 = package.source_model_sha256,
            manifest_sha256 = package.manifest_sha256,
            split_certification = "uncertified_override",
            "unsafe override admitted an uncertified split-serving model"
        );
    }
    Ok(SplitCertificationAdmission::UncertifiedOverride)
}

pub(crate) fn split_certification_label(
    package_ref: Option<&str>,
    source_model_sha256: Option<&str>,
    manifest_sha256: Option<&str>,
) -> Option<&'static str> {
    let (package_ref, source_model_sha256, manifest_sha256) =
        (package_ref?, source_model_sha256?, manifest_sha256?);
    let package = SkippyPackageIdentity {
        package_ref: package_ref.to_string(),
        manifest_sha256: manifest_sha256.to_string(),
        source_model_path: Default::default(),
        source_model_sha256: source_model_sha256.to_string(),
        source_model_bytes: 0,
        source_files: Vec::new(),
        layer_weight_bytes: Vec::new(),
        layer_count: 0,
        activation_width: 0,
        tensor_count: 0,
        generation: None,
    };
    Some(
        if certified_family(&package).ok().flatten().is_some() {
            SplitCertificationAdmission::Certified
        } else {
            SplitCertificationAdmission::UncertifiedOverride
        }
        .as_str(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(source_model_sha256: &str) -> SkippyPackageIdentity {
        SkippyPackageIdentity {
            package_ref: format!("local-gguf://sha256/{source_model_sha256}"),
            manifest_sha256: "1".repeat(64),
            source_model_path: Default::default(),
            source_model_sha256: source_model_sha256.to_string(),
            source_model_bytes: 1,
            source_files: Vec::new(),
            layer_weight_bytes: Vec::new(),
            layer_count: 1,
            activation_width: 1,
            tensor_count: 1,
            generation: None,
        }
    }

    #[test]
    fn admits_exact_certified_artifact() {
        let roster = roster().unwrap();
        validate_recipe(roster).unwrap();
        let certified = package(&roster.models[0].source_model_sha256);
        assert_eq!(
            require_split_certification(&certified, false).unwrap(),
            SplitCertificationAdmission::Certified
        );
    }

    #[test]
    fn rejects_uncertified_artifact_with_actionable_override() {
        let error = require_split_certification(&package(&"f".repeat(64)), false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("--split --allow-uncertified-split")
        );
    }

    #[test]
    fn explicit_override_admits_uncertified_artifact() {
        assert_eq!(
            require_split_certification(&package(&"f".repeat(64)), true).unwrap(),
            SplitCertificationAdmission::UncertifiedOverride
        );
    }

    #[test]
    fn package_v2_needs_an_exact_manifest_entry() {
        let roster = roster().unwrap();
        let mut package = package(&roster.models[0].source_model_sha256);
        package.package_ref = "hf://example/package".to_string();
        assert!(require_split_certification(&package, false).is_err());
    }
}
