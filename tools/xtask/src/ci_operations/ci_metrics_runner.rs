//! `runner_dimensions`, `_canonical_os`, `_label_dimension` and
//! `is_comparison_executor` of `collect-ci-metrics.py`: provider, OS,
//! architecture, role and Depot size derived from runner labels.

use crate::ci_operations::ci_metrics_value::{Value, object};
use crate::repository::python_text::strip;
use std::collections::BTreeSet;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Dimensions {
    pub(crate) provider: String,
    pub(crate) operating_system: Option<String>,
    pub(crate) architecture: Option<String>,
    pub(crate) runner_role: Option<String>,
    pub(crate) runner_size: Option<String>,
}

impl Dimensions {
    pub(crate) fn to_value(&self) -> Value {
        object([
            ("provider", Value::text(&self.provider)),
            (
                "architecture",
                Value::opt_text(self.architecture.as_deref()),
            ),
            ("runner_size", Value::opt_text(self.runner_size.as_deref())),
            (
                "operating_system",
                Value::opt_text(self.operating_system.as_deref()),
            ),
            ("runner_role", Value::opt_text(self.runner_role.as_deref())),
        ])
    }

    /// A dimension by report key; empty strings count as absent (falsy).
    pub(crate) fn field(&self, key: &str) -> Option<&str> {
        let value = match key {
            "provider" => Some(self.provider.as_str()),
            "operating_system" => self.operating_system.as_deref(),
            "architecture" => self.architecture.as_deref(),
            _ => self.runner_role.as_deref(),
        };
        value.filter(|text| !text.is_empty())
    }
}

/// Python `str.lower()`, including full Unicode case mappings.
fn lower(text: &str) -> String {
    text.to_lowercase()
}

fn canonical_os(value: &Value) -> Option<String> {
    let Value::Str(text) = value else {
        return None;
    };
    let value = lower(strip(text));
    if value.is_empty() {
        return None;
    }
    let canonical = if matches!(value.as_str(), "linux" | "ubuntu") || value.starts_with("linux-") {
        "linux"
    } else if matches!(value.as_str(), "macos" | "mac" | "darwin") || value.starts_with("macos-") {
        "macos"
    } else if matches!(value.as_str(), "windows" | "win") || value.starts_with("windows-") {
        "windows"
    } else {
        return Some(value);
    };
    Some(canonical.to_owned())
}

fn label_dimension(labels: &BTreeSet<String>, prefixes: &[&str]) -> Option<String> {
    labels.iter().find_map(|label| {
        prefixes.iter().find_map(|prefix| {
            label
                .strip_prefix(prefix)
                .filter(|rest| !rest.is_empty())
                .map(str::to_owned)
        })
    })
}

/// Python sorts `str` by code point, which matches UTF-8 byte order.
pub(crate) fn runner_dimensions(
    labels: &[String],
    runner_role: &Value,
    operating_system: &Value,
) -> Dimensions {
    let normalized: BTreeSet<String> = labels.iter().map(|label| lower(label)).collect();
    let role = match runner_role {
        Value::Str(text) if !strip(text).is_empty() => Some(strip(text).to_owned()),
        _ => label_dimension(&normalized, &["runner-role:", "role:"]),
    };
    if let Some(depot) = normalized.iter().find(|label| label.starts_with("depot-")) {
        return depot_dimensions(depot, role, operating_system);
    }
    let has = |label: &str| normalized.contains(label);
    let any_prefix = |prefixes: &[&str]| {
        normalized
            .iter()
            .any(|label| prefixes.iter().any(|prefix| label.starts_with(prefix)))
    };
    let architecture = if has("mesh-llm-arm64") || has("ubuntu-24.04-arm") {
        Some("arm64")
    } else if has("macos-15-intel") {
        Some("amd64")
    } else if has("macos-15") {
        Some("arm64")
    } else if has("windows-2022") || has("ubuntu-24.04") || has("x64") || has("amd64") {
        Some("amd64")
    } else {
        None
    };
    let provider = if has("self-hosted") || any_prefix(&["mesh-llm-"]) {
        "self-hosted"
    } else if architecture.is_some() || any_prefix(&["ubuntu-", "macos-", "windows-"]) {
        "github-hosted"
    } else {
        "unknown"
    };
    let label_os = if any_prefix(&["ubuntu-"]) {
        Some("linux")
    } else if any_prefix(&["macos-"]) {
        Some("macos")
    } else if any_prefix(&["windows-"]) {
        Some("windows")
    } else {
        None
    };
    Dimensions {
        provider: provider.to_owned(),
        operating_system: canonical_os(operating_system).or(label_os.map(str::to_owned)),
        architecture: architecture.map(str::to_owned),
        runner_role: role,
        runner_size: None,
    }
}

fn depot_dimensions(label: &str, role: Option<String>, operating_system: &Value) -> Dimensions {
    let suffix = label.rsplit('-').next().unwrap_or(label);
    let platform = &label["depot-".len()..];
    let depot_os = if platform.starts_with("macos-") || platform.starts_with("mac-") {
        "macos"
    } else if platform.starts_with("windows-") || platform.starts_with("win-") {
        "windows"
    } else {
        "linux"
    };
    let architecture = if label.contains("-arm") || depot_os == "macos" {
        "arm64"
    } else {
        "amd64"
    };
    let size = if matches!(suffix, "4" | "8" | "16" | "32" | "64") {
        suffix
    } else {
        "default"
    };
    Dimensions {
        provider: "depot".to_owned(),
        operating_system: Some(
            canonical_os(operating_system).unwrap_or_else(|| depot_os.to_owned()),
        ),
        architecture: Some(architecture.to_owned()),
        runner_role: role,
        runner_size: Some(size.to_owned()),
    }
}

/// Whether a job belongs to the provider comparison cohort.
pub(crate) fn is_comparison_executor(name: &str) -> bool {
    let normalized = lower(strip(name));
    let text = normalized.as_str();
    if matches!(text, "changes" | "summary")
        || text.starts_with("plan ")
        || text.starts_with("pr /")
    {
        return false;
    }
    let orchestration = text.contains("/ ci /")
        || text.contains("lane plan")
        || (text.contains("select ") && text.contains(" runner"))
        || text.contains("runner and cache contract");
    !orchestration && !text.contains("smoke")
}
