//! The catalog pin a named install must honor.
//!
//! A catalog entry is the operator's selection: the plugin, the release it was
//! reviewed at, and the digest of that release's archive for each platform. A
//! named install is bound to all three. GitHub's release metadata is not
//! immutable — whoever controls the repository can replace a release, its
//! assets, and the digests it reports — so it is a cross-check here and never
//! the authority. An entry that pins no version, or no digest for the platform
//! being installed for, is refused rather than resolved to a moving release.

use anyhow::{Context, Result, bail};

use crate::{catalog::CatalogEntry, source_ref::PluginVersion};

/// The exact release and archive digest a catalog install must fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CatalogPin {
    version: PluginVersion,
    sha256: String,
}

impl CatalogPin {
    /// The pin `entry` imposes on an install for `target`. Fails closed when
    /// the entry pins no version or no digest for this platform: without both
    /// there is nothing immutable for the install to be bound to.
    pub(crate) fn for_entry(entry: &CatalogEntry, target: &str) -> Result<Self> {
        let pin = entry.pinned_release(target)?;
        Ok(Self {
            version: PluginVersion::new(pin.version.to_string())
                .with_context(|| format!("catalog entry '{}'", entry.name))?,
            sha256: pin.sha256.to_string(),
        })
    }

    pub(crate) fn version(&self) -> &PluginVersion {
        &self.version
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Refuse an `@version` the operator asked for that is not the pinned
    /// release, so a name cannot be talked off its pin.
    pub(crate) fn ensure_matches_request(&self, requested: Option<&PluginVersion>) -> Result<()> {
        let Some(requested) = requested else {
            return Ok(());
        };
        if self.covers(requested.as_str()) {
            return Ok(());
        }
        bail!(
            "the catalog pins {}, not {requested}; refusing a release the catalog does not pin",
            self.version
        )
    }

    /// Refuse a release whose tag is not the pinned one. `releases/latest`
    /// answers with the same shape as a tag lookup, so the tag is what keeps a
    /// mutable latest from standing in for the pin.
    pub(crate) fn ensure_release_is_pinned(&self, release_tag: &str) -> Result<()> {
        if self.covers(release_tag) {
            return Ok(());
        }
        bail!(
            "GitHub returned release {release_tag}, not the catalog-pinned {}; refusing it",
            self.version
        )
    }

    /// GitHub's own digest for the selected asset must still describe the
    /// pinned archive. It is reported by the same mutable release object as
    /// the asset itself, so agreement is a cross-check, not the verification.
    pub(crate) fn ensure_reported_digest_agrees(
        &self,
        asset_name: &str,
        reported: Option<&str>,
    ) -> Result<()> {
        let Some(reported) = reported else {
            return Ok(());
        };
        if reported == self.sha256 {
            return Ok(());
        }
        bail!(
            "the release asset {asset_name} now reports {reported}, not the catalog-pinned {}; \
             refusing it",
            self.sha256
        )
    }

    /// The tag this pin accepts, in either the `v`-prefixed or bare spelling.
    fn covers(&self, tag: &str) -> bool {
        self.version
            .matching_segments()
            .iter()
            .any(|segment| segment == tag)
    }
}

#[cfg(test)]
mod tests {
    use crate::catalog::PluginCatalog;

    use super::*;

    const TARGET: &str = "aarch64-apple-darwin";
    const PINNED_SHA256: &str = "abababababababababababababababababababababababababababababababab";

    fn entry(json: &str) -> CatalogEntry {
        PluginCatalog::parse_jsonl(json)
            .unwrap()
            .find_exact("demo")
            .unwrap()
            .clone()
    }

    fn pinned() -> CatalogPin {
        CatalogPin::for_entry(&entry(PINNED_ENTRY), TARGET).unwrap()
    }

    const PINNED_ENTRY: &str = r#"{"name":"demo","description":"Demo","github_url":"https://github.com/mesh-llm/demo","author_email":"dev@example.com","author_name":"Mesh LLM","version":"v1.0.0","sha256":{"aarch64-apple-darwin":"abababababababababababababababababababababababababababababababab"}}"#;

    const UNPINNED_ENTRY: &str = r#"{"name":"demo","description":"Demo","github_url":"https://github.com/mesh-llm/demo","author_email":"dev@example.com","author_name":"Mesh LLM"}"#;

    #[test]
    fn a_pinned_entry_binds_its_version_and_its_target_digest() {
        let pin = pinned();
        assert_eq!(pin.version().as_str(), "v1.0.0");
        assert_eq!(pin.sha256(), PINNED_SHA256);
    }

    #[test]
    fn an_entry_that_pins_no_version_is_refused() {
        let error = CatalogPin::for_entry(&entry(UNPINNED_ENTRY), TARGET).unwrap_err();
        assert!(
            format!("{error:#}").contains("pins no version"),
            "{error:#}"
        );
    }

    #[test]
    fn an_entry_without_a_digest_for_the_target_is_refused() {
        let error = CatalogPin::for_entry(&entry(PINNED_ENTRY), "x86_64-unknown-linux-gnu")
            .expect_err("a pinned version with no digest for this target installs nothing");
        let message = format!("{error:#}");
        assert!(
            message.contains("no sha256 for x86_64-unknown-linux-gnu"),
            "{message}"
        );
    }

    #[test]
    fn a_requested_version_off_the_pin_is_refused() {
        let pin = pinned();
        let error = pin
            .ensure_matches_request(Some(&PluginVersion::new("v9.9.9").unwrap()))
            .expect_err("an @version must not move off the catalog pin");
        assert!(format!("{error:#}").contains("pins v1.0.0"), "{error:#}");

        pin.ensure_matches_request(None).unwrap();
        pin.ensure_matches_request(Some(&PluginVersion::new("v1.0.0").unwrap()))
            .unwrap();
        pin.ensure_matches_request(Some(&PluginVersion::new("1.0.0").unwrap()))
            .unwrap();
    }

    #[test]
    fn a_mutable_latest_cannot_stand_in_for_the_pin() {
        let pin = pinned();
        let error = pin
            .ensure_release_is_pinned("0.1.3")
            .expect_err("the repo's current latest is not the pinned release");
        let message = format!("{error:#}");
        assert!(
            message.contains("GitHub returned release 0.1.3, not the catalog-pinned v1.0.0"),
            "{message}"
        );

        pin.ensure_release_is_pinned("v1.0.0").unwrap();
        pin.ensure_release_is_pinned("1.0.0").unwrap();
    }

    #[test]
    fn a_release_digest_that_disagrees_with_the_pin_is_refused() {
        let pin = pinned();
        let replaced = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
        let error = pin
            .ensure_reported_digest_agrees(
                "demo-v1.0.0-aarch64-apple-darwin.tar.gz",
                Some(replaced),
            )
            .expect_err("a release rewritten away from the pinned archive is refused");
        let message = format!("{error:#}");
        assert!(message.contains("now reports ffffffff"), "{message}");
        assert!(message.contains(PINNED_SHA256), "{message}");

        pin.ensure_reported_digest_agrees("demo.tar.gz", Some(PINNED_SHA256))
            .unwrap();
        pin.ensure_reported_digest_agrees("demo.tar.gz", None)
            .unwrap();
    }
}
