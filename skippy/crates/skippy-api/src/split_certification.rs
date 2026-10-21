use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::package::SkippyPackageIdentity;

const ROSTER_JSON: &str = include_str!("split-certified.json");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitCertificationAdmission {
    Certified,
    UncertifiedOverride,
}

impl SplitCertificationAdmission {
    pub const fn as_str(self) -> &'static str {
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
    architectures: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct NativeRecipe {
    llama_upstream_sha: String,
    skippy_abi: String,
    patch_queue_sha256: String,
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
        roster.schema_version == 2,
        "unsupported split certification roster schema {}",
        roster.schema_version
    );
    anyhow::ensure!(
        roster.native_recipe.llama_upstream_sha == env!("SKIPPY_LLAMA_UPSTREAM_SHA"),
        "split certification roster targets llama.cpp {}, but this build uses {}",
        roster.native_recipe.llama_upstream_sha,
        env!("SKIPPY_LLAMA_UPSTREAM_SHA")
    );
    anyhow::ensure!(
        roster.native_recipe.patch_queue_sha256 == env!("SKIPPY_PATCH_QUEUE_SHA256"),
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

fn architecture_is_certified(architecture: &str) -> Result<bool> {
    let roster = roster().context("load split certification roster")?;
    architecture_is_certified_by(roster, architecture)
}

fn architecture_is_certified_by(
    roster: &SplitCertificationRoster,
    architecture: &str,
) -> Result<bool> {
    validate_recipe(roster).context("validate split certification roster recipe")?;
    Ok(roster
        .architectures
        .iter()
        .any(|certified| certified == architecture))
}

pub fn require_split_certification(
    package: &SkippyPackageIdentity,
    architecture: &str,
    allow_uncertified: bool,
) -> Result<SplitCertificationAdmission> {
    admit_split_certification(
        package,
        architecture,
        allow_uncertified,
        architecture_is_certified(architecture),
    )
}

fn admit_split_certification(
    package: &SkippyPackageIdentity,
    architecture: &str,
    allow_uncertified: bool,
    certification: Result<bool>,
) -> Result<SplitCertificationAdmission> {
    match certification {
        Ok(true) => {}
        Ok(false) => return handle_uncertified(package, architecture, allow_uncertified, None),
        Err(error) => {
            return handle_uncertified(package, architecture, allow_uncertified, Some(error));
        }
    };
    tracing::info!(
        architecture,
        package_ref = package.package_ref,
        source_model_sha256 = package.source_model_sha256,
        manifest_sha256 = package.manifest_sha256,
        split_certification = "certified",
        "admitted split-serving model from architecture certification roster"
    );
    Ok(SplitCertificationAdmission::Certified)
}

fn handle_uncertified(
    package: &SkippyPackageIdentity,
    architecture: &str,
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
            "split serving architecture {architecture} is not certified for model artifact {} (source SHA-256 {}, manifest SHA-256 {}); retry explicitly with --split --allow-uncertified-split to run it experimentally",
            package.package_ref,
            package.source_model_sha256,
            package.manifest_sha256
        );
    }
    if let Some(error) = roster_error {
        tracing::warn!(
            package_ref = package.package_ref,
            architecture,
            source_model_sha256 = package.source_model_sha256,
            manifest_sha256 = package.manifest_sha256,
            error = %error,
            split_certification = "uncertified_override",
            "unsafe override admitted a split-serving model without a valid certification roster"
        );
    } else {
        tracing::warn!(
            package_ref = package.package_ref,
            architecture,
            source_model_sha256 = package.source_model_sha256,
            manifest_sha256 = package.manifest_sha256,
            split_certification = "uncertified_override",
            "unsafe override admitted an uncertified split-serving model"
        );
    }
    Ok(SplitCertificationAdmission::UncertifiedOverride)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Admission behavior is tested with an explicit fixture. The bundled roster
    // remains bound to its certified recipe, even in experimental native builds.
    fn certified_fixture() -> SplitCertificationRoster {
        SplitCertificationRoster {
            schema_version: 2,
            native_recipe: NativeRecipe {
                llama_upstream_sha: env!("SKIPPY_LLAMA_UPSTREAM_SHA").to_string(),
                skippy_abi: compiled_skippy_abi(),
                patch_queue_sha256: env!("SKIPPY_PATCH_QUEUE_SHA256").to_string(),
            },
            architectures: vec!["fixture-architecture".to_string()],
        }
    }

    fn require_with_roster(
        roster: &SplitCertificationRoster,
        package: &SkippyPackageIdentity,
        architecture: &str,
        allow_uncertified: bool,
    ) -> Result<SplitCertificationAdmission> {
        admit_split_certification(
            package,
            architecture,
            allow_uncertified,
            architecture_is_certified_by(roster, architecture),
        )
    }

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
            publisher_defaults: None,
        }
    }

    #[test]
    fn admits_different_artifact_of_certified_architecture() {
        let roster = certified_fixture();
        validate_recipe(&roster).unwrap();
        let architecture = &roster.architectures[0];
        assert_eq!(
            require_with_roster(&roster, &package(&"f".repeat(64)), architecture, false).unwrap(),
            SplitCertificationAdmission::Certified
        );
    }

    #[test]
    fn rejects_uncertified_architecture_with_actionable_override() {
        let error = require_with_roster(
            &certified_fixture(),
            &package(&"f".repeat(64)),
            "unknown",
            false,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("architecture unknown is not certified")
        );
        assert!(
            error
                .to_string()
                .contains("--split --allow-uncertified-split")
        );
    }

    #[test]
    fn explicit_override_admits_uncertified_architecture() {
        assert_eq!(
            require_with_roster(
                &certified_fixture(),
                &package(&"f".repeat(64)),
                "unknown",
                true,
            )
            .unwrap(),
            SplitCertificationAdmission::UncertifiedOverride
        );
    }

    #[test]
    fn package_v2_uses_the_same_architecture_admission() {
        let mut package = package(&"f".repeat(64));
        package.package_ref = "hf://example/package".to_string();
        assert_eq!(
            require_with_roster(
                &certified_fixture(),
                &package,
                "fixture-architecture",
                false
            )
            .unwrap(),
            SplitCertificationAdmission::Certified
        );
    }

    #[test]
    fn mismatched_recipes_reject_even_listed_architectures_without_override() {
        for mismatch in ["schema", "upstream", "patch_queue", "abi"] {
            let mut roster = certified_fixture();
            let expected_error = match mismatch {
                "schema" => {
                    roster.schema_version = 0;
                    "unsupported split certification roster schema"
                }
                "upstream" => {
                    roster.native_recipe.llama_upstream_sha = "uncertified-upstream".to_string();
                    "split certification roster targets llama.cpp"
                }
                "patch_queue" => {
                    roster.native_recipe.patch_queue_sha256 = "uncertified-patches".to_string();
                    "split certification roster does not match this build's Skippy patch queue"
                }
                _ => {
                    roster.native_recipe.skippy_abi = "uncertified-abi".to_string();
                    "split certification roster targets Skippy ABI"
                }
            };
            let package = package(&"f".repeat(64));
            let error =
                require_with_roster(&roster, &package, "fixture-architecture", false).unwrap_err();
            assert!(format!("{error:#}").contains(expected_error), "{mismatch}");
            assert_eq!(
                require_with_roster(&roster, &package, "fixture-architecture", true).unwrap(),
                SplitCertificationAdmission::UncertifiedOverride,
                "{mismatch}"
            );
        }
    }

    #[test]
    fn bundled_roster_never_certifies_a_different_native_recipe() {
        let roster = roster().unwrap();
        let architecture = roster.architectures.first().unwrap();
        let package = package(&"f".repeat(64));
        let admission = require_split_certification(&package, architecture, false);
        if let Err(recipe_error) = validate_recipe(roster) {
            let error = admission.unwrap_err();
            assert!(format!("{error:#}").contains(&recipe_error.to_string()));
            assert_eq!(
                require_split_certification(&package, architecture, true).unwrap(),
                SplitCertificationAdmission::UncertifiedOverride
            );
        } else {
            assert_eq!(admission.unwrap(), SplitCertificationAdmission::Certified);
        }
    }
}
