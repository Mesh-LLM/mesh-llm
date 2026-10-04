use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{source_ref::PluginVersion, target::PluginTarget};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub description: String,
    pub github_url: String,
    pub author_email: String,
    pub author_name: String,
    /// The release tag this entry pins, such as `v0.1.0`. Optional for an
    /// ordinary install; required for a default install.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// SHA-256 of that release's archive, keyed by target triple, as
    /// lowercase hex. A default install refuses an archive that doesn't match.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sha256: BTreeMap<String, String>,
}

/// The exact release a default install must fetch for one platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedRelease<'a> {
    pub version: &'a str,
    pub sha256: &'a str,
}

impl CatalogEntry {
    /// The pin a default install of this entry uses on `target`. A default is
    /// installed without anyone choosing it, so an entry that pins no version,
    /// or no checksum for this platform, is refused rather than trusted.
    pub fn pinned_release(&self, target: &str) -> Result<PinnedRelease<'_>> {
        let Some(version) = self.version.as_deref() else {
            bail!(
                "catalog entry '{}' pins no version; a default install needs one",
                self.name
            );
        };
        let Some(sha256) = self.sha256.get(target) else {
            bail!(
                "catalog entry '{}' has no sha256 for {target}; refusing a default install",
                self.name
            );
        };
        Ok(PinnedRelease { version, sha256 })
    }

    fn validate_pins(&self) -> Result<()> {
        if let Some(version) = &self.version {
            PluginVersion::new(version.clone()).with_context(|| {
                format!(
                    "catalog entry '{}' has an invalid pinned version",
                    self.name
                )
            })?;
        }
        if !self.sha256.is_empty() && self.version.is_none() {
            bail!(
                "catalog entry '{}' lists sha256 digests but no version",
                self.name
            );
        }
        for (target, digest) in &self.sha256 {
            if !PluginTarget::is_supported_triple(target) {
                bail!(
                    "catalog entry '{}' has an unsupported target triple {target}",
                    self.name
                );
            }
            if !is_sha256_hex(digest) {
                bail!(
                    "catalog entry '{}' has an invalid sha256 for {target}",
                    self.name
                );
            }
        }
        Ok(())
    }
}

pub(crate) fn is_sha256_hex(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .chars()
            .all(|ch| ch.is_ascii_digit() || ('a'..='f').contains(&ch))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginCatalog {
    entries: Vec<CatalogEntry>,
}

impl PluginCatalog {
    pub fn parse_jsonl(input: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for (index, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: CatalogEntry = serde_json::from_str(line)
                .with_context(|| format!("parse plugins.jsonl line {}", index + 1))?;
            entry
                .validate_pins()
                .with_context(|| format!("plugins.jsonl line {}", index + 1))?;
            entries.push(entry);
        }
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        ensure_unique_names(&entries)?;
        Ok(Self { entries })
    }

    pub async fn fetch(client: &Client, url: &str) -> Result<Self> {
        let response = client
            .get(url)
            .header(reqwest::header::USER_AGENT, crate::github::USER_AGENT)
            .send()
            .await
            .with_context(|| format!("fetch plugin catalog {url}"))?;
        let status = response.status();
        if !status.is_success() {
            bail!("plugin catalog request failed: {status} {url}");
        }
        let body = response
            .text()
            .await
            .with_context(|| format!("read plugin catalog {url}"))?;
        Self::parse_jsonl(&body)
    }

    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn find_exact(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    pub fn search(&self, query: Option<&str>) -> Vec<&CatalogEntry> {
        let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) else {
            return self.entries.iter().collect();
        };
        let query = query.to_ascii_lowercase();
        self.entries
            .iter()
            .filter(|entry| {
                entry.name.to_ascii_lowercase().contains(&query)
                    || entry.description.to_ascii_lowercase().contains(&query)
                    || entry.author_name.to_ascii_lowercase().contains(&query)
            })
            .collect()
    }
}

fn ensure_unique_names(entries: &[CatalogEntry]) -> Result<()> {
    for pair in entries.windows(2) {
        if pair[0].name == pair[1].name {
            bail!("duplicate plugin catalog entry '{}'", pair[0].name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_searches_catalog_jsonl() {
        let catalog = PluginCatalog::parse_jsonl(
            r#"{"name":"blackboard","description":"Shared notes","github_url":"https://github.com/mesh-llm/blackboard","author_email":"maintainers@meshllm.cloud","author_name":"Mesh LLM"}
{"name":"notes","description":"Team notes","github_url":"https://github.com/acme/notes","author_email":"dev@example.com","author_name":"Acme"}
"#,
        )
        .unwrap();
        assert_eq!(catalog.entries().len(), 2);
        assert_eq!(
            catalog.find_exact("blackboard").unwrap().author_name,
            "Mesh LLM"
        );
        assert_eq!(catalog.search(Some("team"))[0].name, "notes");
    }

    const PINNED: &str = r#"{"name":"notes","description":"Team notes","github_url":"https://github.com/acme/notes","author_email":"dev@example.com","author_name":"Acme","version":"v1.0.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}"#;

    #[test]
    fn an_entry_without_pins_still_parses() {
        let catalog = PluginCatalog::parse_jsonl(
            r#"{"name":"notes","description":"Team notes","github_url":"https://github.com/acme/notes","author_email":"dev@example.com","author_name":"Acme"}"#,
        )
        .unwrap();
        let entry = catalog.find_exact("notes").unwrap();
        assert_eq!(entry.version, None);
        assert!(entry.sha256.is_empty());
    }

    #[test]
    fn a_pinned_entry_yields_its_release_for_a_pinned_target() {
        let catalog = PluginCatalog::parse_jsonl(PINNED).unwrap();
        let pin = catalog
            .find_exact("notes")
            .unwrap()
            .pinned_release("x86_64-unknown-linux-gnu")
            .unwrap();
        assert_eq!(pin.version, "v1.0.0");
        assert_eq!(
            pin.sha256,
            "abababababababababababababababababababababababababababababababab"
        );
    }

    #[test]
    fn a_default_install_is_refused_without_a_checksum_or_a_version() {
        let catalog = PluginCatalog::parse_jsonl(PINNED).unwrap();
        let entry = catalog.find_exact("notes").unwrap();
        let no_checksum = entry.pinned_release("aarch64-apple-darwin").unwrap_err();
        assert!(
            no_checksum
                .to_string()
                .contains("no sha256 for aarch64-apple-darwin"),
            "{no_checksum}"
        );

        let unpinned = CatalogEntry {
            version: None,
            sha256: BTreeMap::new(),
            ..entry.clone()
        };
        let no_version = unpinned
            .pinned_release("x86_64-unknown-linux-gnu")
            .unwrap_err();
        assert!(
            no_version.to_string().contains("pins no version"),
            "{no_version}"
        );
    }

    #[test]
    fn rejects_malformed_pins() {
        let uppercase = PINNED.replace("abab", "ABAB");
        let error = PluginCatalog::parse_jsonl(&uppercase).unwrap_err();
        assert!(format!("{error:#}").contains("invalid sha256"), "{error:#}");

        let no_version = PINNED.replace(r#""version":"v1.0.0","#, "");
        let error = PluginCatalog::parse_jsonl(&no_version).unwrap_err();
        assert!(format!("{error:#}").contains("but no version"), "{error:#}");

        let unknown_target = PINNED.replace("x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gun");
        let error = PluginCatalog::parse_jsonl(&unknown_target).unwrap_err();
        assert!(
            format!("{error:#}").contains("unsupported target triple"),
            "{error:#}"
        );

        for invalid in ["", "v1/2", "v1 2", "v1\\\\2"] {
            let bad_version = PINNED.replace("v1.0.0", invalid);
            let error = PluginCatalog::parse_jsonl(&bad_version).unwrap_err();
            assert!(
                format!("{error:#}").contains("invalid pinned version"),
                "{error:#}"
            );
        }
    }

    #[test]
    fn rejects_duplicate_names() {
        let err = PluginCatalog::parse_jsonl(
            r#"{"name":"blackboard","description":"A","github_url":"https://github.com/mesh-llm/blackboard","author_email":"a@example.com","author_name":"A"}
{"name":"blackboard","description":"B","github_url":"https://github.com/mesh-llm/blackboard2","author_email":"b@example.com","author_name":"B"}
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }
}
