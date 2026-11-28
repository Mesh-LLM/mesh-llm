//! Compile-time parity checks between the built-in config schema and the
//! canonical public website configuration reference.
//!
//! These tests fail the build when the published documentation drifts away
//! from `crate::built_in_config_settings()`: a missing field, a duplicated
//! canonical entry, or a stale navigation link. They read the website
//! Markdown source directly (the same `include_str!` technique used by
//! `documented_matrix_key_paths` in
//! `mesh-llm-host-runtime/src/plugin/config.rs`), so no separate script or
//! CI job is required to catch drift.

#[cfg(test)]
#[path = "website_docs_parity/metadata.rs"]
mod metadata_tests;

#[cfg(test)]
mod tests {
    use crate::{
        CANONICAL_MODEL_REF_SEGMENT, CANONICAL_PLUGIN_NAME_SEGMENT, ConfigValueSchema,
        built_in_config_settings,
    };
    use std::collections::BTreeMap;

    const CONFIG_TOML_MD: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../website/src/docs/pages/config-toml.md"
    ));
    const CONFIG_DEFAULTS_MD: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../website/src/docs/pages/config-defaults.md"
    ));
    const CONFIG_MODELS_MD: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../website/src/docs/pages/config-models.md"
    ));
    const CONFIG_REFERENCE_MD: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../website/src/docs/pages/config-reference.md"
    ));
    const DOCS_NAV_JS: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../website/src/_data/docs.js"
    ));
    const INSTALL_MD: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../install.md"));

    const CONFIG_PAGE_ROUTES: [&str; 4] = [
        "/docs/pages/config-toml/",
        "/docs/pages/config-defaults/",
        "/docs/pages/config-models/",
        "/docs/pages/config-reference/",
    ];

    const RUNTIME_LIFECYCLE_ANCHORS: [&str; 2] = [
        "/docs/pages/runtime-lifecycle/#runtime-modes",
        "/docs/pages/runtime-lifecycle/#activity-aware-admission",
    ];

    /// A stale published route observed in the wild before this PR: the
    /// public site's actual page lives under `/docs/pages/config-reference/`,
    /// not `/docs/config-reference/`.
    const KNOWN_STALE_URL_FRAGMENTS: [&str; 4] = [
        "meshllm.cloud/docs/config-reference/",
        "meshllm.cloud/docs/config-toml/",
        "meshllm.cloud/docs/config-defaults/",
        "meshllm.cloud/docs/config-models/",
    ];

    fn combined_config_pages() -> String {
        format!("{CONFIG_TOML_MD}\n{CONFIG_DEFAULTS_MD}\n{CONFIG_MODELS_MD}\n{CONFIG_REFERENCE_MD}")
    }

    /// The built-in schema includes top-level fields with no dot in
    /// their canonical path: the schema-version field and the required
    /// `[[models]]` entry's model reference and managed-provider controls.
    const DOTLESS_TOP_LEVEL_FIELDS: [&str; 8] = [
        "version",
        "model",
        "provider",
        "variant",
        "options",
        "request_defaults",
        "lifecycle",
        "source",
    ];

    fn looks_like_config_path(candidate: &str) -> bool {
        if DOTLESS_TOP_LEVEL_FIELDS.contains(&candidate) {
            return true;
        }
        !candidate.is_empty()
            && candidate.contains('.')
            && candidate
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            && candidate
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '<' | '>' | '-'))
    }

    /// Extract every backtick-quoted, dotted key-path-shaped token from the
    /// *first column* of every Markdown table row across the combined
    /// website configuration pages. Restricting extraction to the first
    /// (canonical "Key path") column, rather than the whole page, means an
    /// incidental prose mention of a field elsewhere in a description column
    /// does not register as a second canonical entry. Cells that pair
    /// multiple paths with `<br>` (either inside or outside the backticks)
    /// are split into their individual paths.
    fn documented_short_paths() -> Vec<String> {
        let mut paths = Vec::new();
        for line in combined_config_pages().lines() {
            let Some(line) = line.strip_prefix('|') else {
                continue;
            };
            let Some(first_column) = line.split('|').next() else {
                continue;
            };
            let mut rest = first_column;
            while let Some(start) = rest.find('`') {
                let after_tick = &rest[start + 1..];
                let Some(end) = after_tick.find('`') else {
                    break;
                };
                let inner = &after_tick[..end];
                if looks_like_config_path(inner) {
                    for part in inner.split("<br>") {
                        paths.push(part.trim().to_string());
                    }
                }
                rest = &after_tick[end + 1..];
            }
        }
        paths
    }

    /// Convert a schema-rendered canonical path (e.g.
    /// `models.<model-ref>.model_fit.ctx_size` or
    /// `defaults.model_fit.ctx_size`) into the short form the website
    /// reference documents once per field (e.g. `model_fit.ctx_size`).
    fn normalize_schema_path(rendered: &str) -> String {
        let model_prefix = format!("models.{CANONICAL_MODEL_REF_SEGMENT}.");
        let plugin_prefix = format!("plugin.{CANONICAL_PLUGIN_NAME_SEGMENT}.");

        if let Some(rest) = rendered.strip_prefix("defaults.") {
            return rest.to_string();
        }
        if let Some(rest) = rendered.strip_prefix(&model_prefix) {
            return rest.to_string();
        }
        if let Some(rest) = rendered.strip_prefix(&plugin_prefix) {
            return format!("plugin.<name>.{rest}");
        }
        rendered.to_string()
    }

    fn schema_short_paths() -> Vec<String> {
        built_in_config_settings()
            .iter()
            .map(|setting| normalize_schema_path(&setting.path.render()))
            .collect()
    }

    #[test]
    fn website_config_reference_covers_every_schema_field() {
        let documented: Vec<String> = documented_short_paths();
        let schema_paths = schema_short_paths();

        let missing: Vec<&String> = schema_paths
            .iter()
            .filter(|path| !documented.contains(path))
            .collect();

        assert!(
            missing.is_empty(),
            "built-in config schema fields missing from the website configuration \
             reference (website/src/docs/pages/config-{{toml,defaults,models,reference}}.md): \
             {missing:#?}\n\
             Every field returned by mesh_llm_config::built_in_config_settings() must have \
             exactly one canonical, backtick-quoted entry across the four config pages."
        );
    }

    #[test]
    fn website_config_reference_has_no_duplicate_canonical_entries() {
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        for path in documented_short_paths() {
            *counts.entry(path).or_insert(0) += 1;
        }

        let duplicates: Vec<(&String, &u32)> =
            counts.iter().filter(|(_, count)| **count > 1).collect();

        assert!(
            duplicates.is_empty(),
            "the website configuration reference documents these key paths more than once, \
             violating the one-canonical-entry-per-field rule: {duplicates:#?}"
        );
    }

    #[test]
    fn website_split_mode_values_match_the_exported_schema() {
        let settings = built_in_config_settings();
        let setting = settings
            .iter()
            .find(|setting| setting.path.render() == "defaults.hardware.split_mode")
            .expect("split mode schema setting");
        let ConfigValueSchema::Enum { values } = &setting.value_schema else {
            panic!("split mode must remain an enum");
        };

        assert!(values.iter().any(|value| value == "tensor"));
        let row = CONFIG_REFERENCE_MD
            .lines()
            .find(|line| line.starts_with("| `hardware.split_mode` |"))
            .expect("website split mode row");
        for value in values {
            assert!(
                row.contains(&format!("`{value}`")),
                "website split mode row is missing schema value {value:?}: {row}"
            );
        }
    }

    #[test]
    fn config_navigation_covers_all_four_stable_routes() {
        for route in CONFIG_PAGE_ROUTES {
            assert!(
                DOCS_NAV_JS.contains(route),
                "website/src/_data/docs.js is missing a navigation entry for stable route \
                 `{route}`"
            );
        }
    }

    #[test]
    fn config_navigation_covers_runtime_lifecycle_anchors() {
        let haystack = format!("{DOCS_NAV_JS}\n{}", combined_config_pages());
        for anchor in RUNTIME_LIFECYCLE_ANCHORS {
            assert!(
                haystack.contains(anchor),
                "no configuration doc or nav entry links to runtime-lifecycle anchor `{anchor}`"
            );
        }
    }

    #[test]
    fn no_stale_meshllm_cloud_config_urls() {
        for source in [
            ("install.md", INSTALL_MD),
            ("website/src/_data/docs.js", DOCS_NAV_JS),
            ("website/src/docs/pages/config-toml.md", CONFIG_TOML_MD),
            (
                "website/src/docs/pages/config-defaults.md",
                CONFIG_DEFAULTS_MD,
            ),
            ("website/src/docs/pages/config-models.md", CONFIG_MODELS_MD),
            (
                "website/src/docs/pages/config-reference.md",
                CONFIG_REFERENCE_MD,
            ),
        ] {
            let (file, content) = source;
            for stale in KNOWN_STALE_URL_FRAGMENTS {
                assert!(
                    !content.contains(stale),
                    "{file} contains a stale published URL `{stale}`; the live route is under \
                     `/docs/pages/...`"
                );
            }
        }
    }
}
