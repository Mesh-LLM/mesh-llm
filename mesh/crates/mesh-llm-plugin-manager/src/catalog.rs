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
    /// The release tag this entry pins, such as `v0.1.0`. Required to install
    /// the entry by name; only search and discovery work without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// SHA-256 of that release's archive, keyed by target triple, as
    /// lowercase hex. An install refuses an archive that doesn't match.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sha256: BTreeMap<String, String>,
}

/// The exact release an install of a catalog entry must fetch for one platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedRelease<'a> {
    pub version: &'a str,
    pub sha256: &'a str,
}

impl CatalogEntry {
    /// The pin every install of this entry uses on `target`. Nothing else
    /// fixes what a name means: a release and its assets are rewritable by
    /// whoever holds the repository, so an entry that pins no version, or no
    /// checksum for this platform, is refused rather than trusted.
    pub fn pinned_release(&self, target: &str) -> Result<PinnedRelease<'_>> {
        let Some(version) = self.version.as_deref() else {
            bail!(
                "catalog entry '{}' pins no version; a catalog install needs one",
                self.name
            );
        };
        let Some(sha256) = self.sha256.get(target) else {
            bail!(
                "catalog entry '{}' has no sha256 for {target}; refusing the install",
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

/// A named, curated group of catalog plugins, such as `recommended`. A set
/// carries no releases or checksums: each member's own catalog entry pin stays
/// the authority for what installing it means.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSet {
    pub name: String,
    pub description: String,
    pub plugins: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginCatalog {
    entries: Vec<CatalogEntry>,
    sets: Vec<CatalogSet>,
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
        Ok(Self {
            entries,
            sets: Vec::new(),
        })
    }

    /// Parse `plugin-sets.jsonl`: named set records only. They live in their
    /// own file so a catalog that publishes sets stays parseable for hosts
    /// built before sets existed. Whether members exist is checked by
    /// [`Self::with_sets`].
    pub fn parse_sets_jsonl(input: &str) -> Result<Vec<CatalogSet>> {
        let mut sets = Vec::new();
        for (index, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let context = format!("plugin-sets.jsonl line {}", index + 1);
            let set: CatalogSet = serde_json::from_str(line).with_context(|| context.clone())?;
            validate_set(&set).with_context(|| context)?;
            sets.push(set);
        }
        sets.sort_by(|left, right| left.name.cmp(&right.name));
        for pair in sets.windows(2) {
            if pair[0].name == pair[1].name {
                bail!("duplicate plugin set '{}'", pair[0].name);
            }
        }
        Ok(sets)
    }

    /// Attach named sets, refusing sets that name a plugin this catalog does
    /// not list.
    pub fn with_sets(mut self, sets: Vec<CatalogSet>) -> Result<Self> {
        for set in &sets {
            for member in &set.plugins {
                if self.find_exact(member).is_none() {
                    bail!(
                        "plugin set '{}' names '{member}', which is not in the plugin catalog",
                        set.name
                    );
                }
            }
        }
        self.sets = sets;
        Ok(self)
    }

    /// Fetch the plugin entries, then the sets from `sets_url`. Anything wrong
    /// on the sets side (absent, unreachable, malformed, naming unknown
    /// plugins) yields a catalog with no sets and the reason: a broken sets
    /// file is never a broken plugin catalog.
    pub async fn fetch_with_sets(
        client: &Client,
        plugins_url: &str,
        sets_url: Option<&str>,
    ) -> Result<(Self, Option<String>)> {
        let catalog = Self::fetch(client, plugins_url).await?;
        let Some(sets_url) = sets_url else {
            return Ok((
                catalog,
                Some(format!(
                    "no plugin-sets.jsonl beside the catalog URL {plugins_url}"
                )),
            ));
        };
        match Self::fetch_sets(client, sets_url).await {
            Ok(sets) => match catalog.clone().with_sets(sets) {
                Ok(with_sets) => Ok((with_sets, None)),
                Err(error) => Ok((catalog, Some(format!("{error:#}")))),
            },
            Err(error) => Ok((catalog, Some(format!("{error:#}")))),
        }
    }

    async fn fetch_sets(client: &Client, sets_url: &str) -> Result<Vec<CatalogSet>> {
        let response = client
            .get(sets_url)
            .header(reqwest::header::USER_AGENT, crate::github::USER_AGENT)
            .send()
            .await
            .with_context(|| format!("fetch plugin sets {sets_url}"))?;
        let status = response.status();
        if !status.is_success() {
            bail!("plugin sets request failed: {status} {sets_url}");
        }
        let body = response
            .text()
            .await
            .with_context(|| format!("read plugin sets {sets_url}"))?;
        Self::parse_sets_jsonl(&body)
    }

    pub fn sets(&self) -> &[CatalogSet] {
        &self.sets
    }

    pub fn find_set(&self, name: &str) -> Option<&CatalogSet> {
        self.sets.iter().find(|set| set.name == name)
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

/// Where the sets for the catalog at `catalog_url` live: the same location
/// with the final `plugins.jsonl` replaced by `plugin-sets.jsonl`. `None` for
/// a catalog URL that does not end that way (a custom catalog), which has no
/// sibling file by convention.
pub fn sets_url_for(catalog_url: &str) -> Option<String> {
    catalog_url
        .strip_suffix("/plugins.jsonl")
        .map(|base| format!("{base}/plugin-sets.jsonl"))
}

fn validate_set(set: &CatalogSet) -> Result<()> {
    if set.name.trim().is_empty() {
        bail!("plugin set has an empty name");
    }
    let mut seen = std::collections::BTreeSet::new();
    for member in &set.plugins {
        if !seen.insert(member.as_str()) {
            bail!("plugin set '{}' lists '{member}' twice", set.name);
        }
    }
    Ok(())
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

    const SET_ENTRIES: &str = r#"{"name":"capsules","description":"Audit records","github_url":"https://github.com/Mesh-LLM/capsules","author_email":"m@meshllm.cloud","author_name":"Mesh LLM"}
{"name":"notes","description":"Team notes","github_url":"https://github.com/acme/notes","author_email":"dev@example.com","author_name":"Acme"}"#;

    #[test]
    fn parses_sets_and_attaches_them_to_the_catalog() {
        let sets = PluginCatalog::parse_sets_jsonl(
            r#"{"name":"recommended","description":"Start here","plugins":["capsules"]}
{"name":"extras","description":"More","plugins":["notes","capsules"]}
"#,
        )
        .unwrap();
        let catalog = PluginCatalog::parse_jsonl(SET_ENTRIES)
            .unwrap()
            .with_sets(sets)
            .unwrap();
        assert_eq!(
            catalog
                .sets()
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["extras", "recommended"]
        );
        assert_eq!(
            catalog.find_set("recommended").unwrap().plugins,
            ["capsules"]
        );
        assert!(catalog.find_set("missing").is_none());
    }

    #[test]
    fn rejects_invalid_sets() {
        let duplicate_name = PluginCatalog::parse_sets_jsonl(
            "{\"name\":\"a\",\"description\":\"\",\"plugins\":[]}\n{\"name\":\"a\",\"description\":\"\",\"plugins\":[]}",
        )
        .unwrap_err();
        assert!(duplicate_name.to_string().contains("duplicate plugin set"));

        let duplicate_member =
            PluginCatalog::parse_sets_jsonl(r#"{"name":"a","description":"","plugins":["x","x"]}"#)
                .unwrap_err();
        assert!(format!("{duplicate_member:#}").contains("twice"));

        let unknown_member = PluginCatalog::parse_jsonl(SET_ENTRIES)
            .unwrap()
            .with_sets(
                PluginCatalog::parse_sets_jsonl(
                    r#"{"name":"a","description":"","plugins":["ghost"]}"#,
                )
                .unwrap(),
            )
            .unwrap_err();
        assert!(
            unknown_member
                .to_string()
                .contains("not in the plugin catalog")
        );
    }

    #[test]
    fn sets_live_beside_the_default_catalog_file_only() {
        assert_eq!(
            sets_url_for(
                "https://huggingface.co/datasets/meshllm/plugin-catalog/resolve/main/plugins.jsonl"
            )
            .as_deref(),
            Some(
                "https://huggingface.co/datasets/meshllm/plugin-catalog/resolve/main/plugin-sets.jsonl"
            )
        );
        assert_eq!(sets_url_for("http://127.0.0.1:1/custom-catalog.json"), None);
        assert_eq!(sets_url_for("plugins.jsonl"), None);
    }
}
