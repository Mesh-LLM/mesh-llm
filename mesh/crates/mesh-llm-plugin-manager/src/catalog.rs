use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::source_ref::is_valid_name;
use crate::{source_ref::PluginVersion, target::PluginTarget};

/// A named group of catalog plugin names, such as `recommended`.
///
/// The catalog owns set membership; the host keeps no list of its own, so a
/// membership change needs no host rebuild. A set carries no release pins:
/// each referenced plugin entry owns its own versions, checksums, and
/// compatibility metadata, and a set never grants permissions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSet {
    pub name: String,
    pub description: String,
    /// The catalog plugin names in this set, each unique.
    pub plugins: Vec<String>,
}

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
    /// Which MeshLLM host versions the pinned release works with. Absent for
    /// legacy entries: compatibility is then unknown, not excluded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility: Option<crate::compatibility::HostCompatibility>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginCatalog {
    entries: Vec<CatalogEntry>,
    sets: Vec<CatalogSet>,
}

impl PluginCatalog {
    /// Parse `plugins.jsonl`: plugin entries only. Set records live in their
    /// own file (`parse_sets_jsonl`) so a catalog that publishes sets stays
    /// readable for hosts built before sets existed.
    pub fn parse_jsonl(input: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for (index, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let context = format!("plugins.jsonl line {}", index + 1);
            let entry: CatalogEntry = serde_json::from_str(line)
                .with_context(|| format!("parse plugin entry in {context}"))?;
            entry.validate_pins().with_context(|| context.clone())?;
            entry
                .compatibility
                .as_ref()
                .map(crate::compatibility::HostCompatibility::validate)
                .transpose()
                .with_context(|| context.clone())?;
            entries.push(entry);
        }
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        ensure_unique_names(&entries)?;
        Ok(Self {
            entries,
            sets: Vec::new(),
        })
    }

    /// Parse `plugin-sets.jsonl`: named set records. Membership validation
    /// against the plugin entries happens in [`Self::with_sets`].
    pub fn parse_sets_jsonl(input: &str) -> Result<Vec<CatalogSet>> {
        let mut sets = Vec::new();
        for (index, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let context = format!("plugin-sets.jsonl line {}", index + 1);
            let set: CatalogSet = serde_json::from_str(line)
                .with_context(|| format!("parse set record in {context}"))?;
            validate_set(&set).with_context(|| context.clone())?;
            sets.push(set);
        }
        ensure_unique_set_names(&sets)?;
        sets.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(sets)
    }

    /// Attach named sets to this catalog, refusing sets that name plugins the
    /// catalog does not carry: an unknown member reference is malformed
    /// membership, not an empty set.
    pub fn with_sets(mut self, mut sets: Vec<CatalogSet>) -> Result<Self> {
        ensure_members_exist(&sets, &self.entries)?;
        sets.sort_by(|left, right| left.name.cmp(&right.name));
        self.sets = sets;
        Ok(self)
    }

    /// Fetch the plugin entries, then the catalog's named sets from
    /// `sets_url`. A sets file that cannot be fetched yields no sets — an
    /// absent sets catalog is not an absent plugin catalog, and plugin
    /// install, search, and repair must keep working without it.
    pub async fn fetch_with_sets(
        client: &Client,
        plugins_url: &str,
        sets_url: Option<&str>,
    ) -> Result<Self> {
        let catalog = Self::fetch(client, plugins_url).await?;
        let Some(sets_url) = sets_url else {
            return Ok(catalog);
        };
        match client
            .get(sets_url)
            .header(reqwest::header::USER_AGENT, crate::github::USER_AGENT)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                let body = response
                    .text()
                    .await
                    .with_context(|| format!("read plugin sets catalog {sets_url}"))?;
                let sets = Self::parse_sets_jsonl(&body)?;
                catalog.with_sets(sets)
            }
            // A missing or unreadable sets file means no sets to offer, not a
            // failed catalog read.
            _ => Ok(catalog),
        }
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

    /// The catalog's named sets, sorted by name.
    pub fn sets(&self) -> &[CatalogSet] {
        &self.sets
    }

    pub fn find_set(&self, name: &str) -> Option<&CatalogSet> {
        self.sets.iter().find(|set| set.name == name)
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

/// Where the named-set records for the catalog at `catalog_url` live: the
/// same location with `plugins.jsonl` replaced by `plugin-sets.jsonl`, so a
/// catalog publisher only ever manages one directory. `None` when the
/// plugins URL is not a `plugins.jsonl` file (a test fixture or a custom
/// catalog), where no set file can be derived.
pub fn sets_url_for(catalog_url: &str) -> Option<String> {
    let file = catalog_url.rsplit('/').next()?;
    (file == "plugins.jsonl").then(|| {
        format!(
            "{}plugin-sets.jsonl",
            &catalog_url[..catalog_url.len() - file.len()]
        )
    })
}

fn validate_set(set: &CatalogSet) -> Result<()> {
    if !is_valid_name(&set.name) {
        bail!(
            "plugin set name {:?} must use lowercase ASCII letters, numbers, and single '-' separators",
            set.name
        );
    }
    if set.description.trim().is_empty() {
        bail!("plugin set '{}' has an empty description", set.name);
    }
    let mut seen = std::collections::BTreeSet::new();
    for member in &set.plugins {
        if !is_valid_name(member) {
            bail!(
                "plugin set '{}' names an invalid plugin {:?}; members must be catalog plugin names",
                set.name,
                member
            );
        }
        if !seen.insert(member.as_str()) {
            bail!(
                "plugin set '{}' lists '{}' more than once",
                set.name,
                member
            );
        }
    }
    Ok(())
}

fn ensure_unique_set_names(sets: &[CatalogSet]) -> Result<()> {
    for pair in sets.windows(2) {
        if pair[0].name == pair[1].name {
            bail!("duplicate plugin set '{}'", pair[0].name);
        }
    }
    Ok(())
}

fn ensure_members_exist(sets: &[CatalogSet], entries: &[CatalogEntry]) -> Result<()> {
    let names: std::collections::BTreeSet<&str> =
        entries.iter().map(|entry| entry.name.as_str()).collect();
    for set in sets {
        for member in &set.plugins {
            if !names.contains(member.as_str()) {
                bail!(
                    "plugin set '{}' names '{}', which is not a plugin entry in the catalog",
                    set.name,
                    member
                );
            }
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
        let uppercase = PINNED.replace(
            "abababababababababababababababababababababababababababababababab",
            "ABAB",
        );
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

    const SET_ENTRIES: &str = r#"{"name":"capsules","description":"Audit and provenance records","github_url":"https://github.com/Mesh-LLM/capsules","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v0.2.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"},"compatibility":{"min":"0.81.0"}}
{"name":"blackboard","description":"Shared notes","github_url":"https://github.com/mesh-llm/blackboard","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v1.1.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}
"#;

    const RECOMMENDED_SET: &str = r#"{"name":"recommended","description":"Recommended plugins for MeshLLM","plugins":["capsules"]}"#;

    fn catalog_with_sets(sets: &str) -> PluginCatalog {
        PluginCatalog::parse_jsonl(SET_ENTRIES)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(sets).unwrap())
            .unwrap()
    }

    #[test]
    fn sets_parse_from_their_own_file_and_validate_against_entries() {
        let catalog = catalog_with_sets(&format!("{RECOMMENDED_SET}\n"));
        assert_eq!(catalog.entries().len(), 2);
        assert_eq!(catalog.sets().len(), 1);
        let set = catalog.find_set("recommended").unwrap();
        assert_eq!(set.description, "Recommended plugins for MeshLLM");
        assert_eq!(set.plugins, ["capsules"]);
        assert!(catalog.find_set("nope").is_none());
        // Existing entry behavior is untouched: pins and compatibility parse.
        let entry = catalog.find_exact("capsules").unwrap();
        assert_eq!(
            entry
                .pinned_release("x86_64-unknown-linux-gnu")
                .unwrap()
                .version,
            "v0.2.0"
        );
        assert_eq!(
            entry.compatibility,
            Some(crate::compatibility::HostCompatibility {
                min: Some("0.81.0".into()),
                max: None,
            })
        );
    }

    #[test]
    fn a_pre_change_host_still_parses_a_plugins_file_with_sets_published() {
        // Set records never appear in plugins.jsonl; the plugins file keeps
        // parsing with the parser this code replaced, which knew no sets.
        let catalog = PluginCatalog::parse_jsonl(SET_ENTRIES).unwrap();
        assert!(catalog.sets().is_empty());
        assert_eq!(catalog.entries().len(), 2);
    }

    #[test]
    fn sets_url_derivation_replaces_only_the_plugins_file() {
        assert_eq!(
            sets_url_for("https://example.com/dir/plugins.jsonl"),
            Some("https://example.com/dir/plugin-sets.jsonl".to_string())
        );
        assert_eq!(
            sets_url_for("https://example.com/plugins.jsonl"),
            Some("https://example.com/plugin-sets.jsonl".to_string())
        );
        assert_eq!(sets_url_for("https://example.com/custom.jsonl"), None);
        assert_eq!(
            sets_url_for("plugins.jsonl"),
            Some("plugin-sets.jsonl".to_string())
        );
    }

    #[test]
    fn set_records_validate_their_names_membership_and_references() {
        let unknown_member = RECOMMENDED_SET.replace(r#"["capsules"]"#, r#"["ghost"]"#);
        let error = PluginCatalog::parse_jsonl(SET_ENTRIES)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(&unknown_member).unwrap())
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("set 'recommended' names 'ghost'"),
            "{error:#}"
        );

        let duplicate_member =
            RECOMMENDED_SET.replace(r#"["capsules"]"#, r#"["capsules","capsules"]"#);
        let error = PluginCatalog::parse_sets_jsonl(&duplicate_member).unwrap_err();
        assert!(format!("{error:#}").contains("more than once"), "{error:#}");

        let bad_name = RECOMMENDED_SET.replace("\"recommended\"", "\"Not A Set!\"");
        let error = PluginCatalog::parse_sets_jsonl(&bad_name).unwrap_err();
        assert!(
            format!("{error:#}").contains("plugin set name"),
            "{error:#}"
        );

        let empty_description = RECOMMENDED_SET.replace("Recommended plugins for MeshLLM", "   ");
        let error = PluginCatalog::parse_sets_jsonl(&empty_description).unwrap_err();
        assert!(
            format!("{error:#}").contains("empty description"),
            "{error:#}"
        );

        let duplicate_set = format!(
            "{RECOMMENDED_SET}\n{{\"name\":\"recommended\",\"description\":\"Again\",\"plugins\":[\"blackboard\"]}}"
        );
        let error = PluginCatalog::parse_sets_jsonl(&duplicate_set).unwrap_err();
        assert!(
            format!("{error:#}").contains("duplicate plugin set"),
            "{error:#}"
        );

        let no_members = RECOMMENDED_SET.replace(r#"["capsules"]"#, r#"[]"#);
        let catalog = catalog_with_sets(&no_members);
        assert!(catalog.find_set("recommended").unwrap().plugins.is_empty());

        // A set may not nest another set: members are plugin names only.
        let nested = format!(
            "{RECOMMENDED_SET}\n{{\"name\":\"outer\",\"description\":\"Tries to nest\",\"plugins\":[\"recommended\"]}}"
        );
        let error = PluginCatalog::parse_jsonl(SET_ENTRIES)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(&nested).unwrap())
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("set 'outer' names 'recommended'"),
            "{error:#}"
        );
    }

    #[test]
    fn a_malformed_set_record_names_the_file_it_came_from() {
        let not_a_set = r#"{"name":"recommended","description":"No members key"}"#;
        let error = PluginCatalog::parse_sets_jsonl(not_a_set).unwrap_err();
        assert!(
            format!("{error:#}").contains("parse set record in plugin-sets.jsonl line 1"),
            "{error:#}"
        );
    }

    #[test]
    fn an_entry_with_malformed_compatibility_is_refused() {
        let bad = SET_ENTRIES.replace(
            r#""compatibility":{"min":"0.81.0"}"#,
            r#""compatibility":{"min":"soon"}"#,
        );
        let error = PluginCatalog::parse_jsonl(&bad).unwrap_err();
        assert!(
            format!("{error:#}").contains("invalid host compatibility bound"),
            "{error:#}"
        );
    }
}
