//! Loading the JSON-compatible YAML catalogs (`ci/ownership.yml`,
//! `ci/slices.yml`, schema version 1) and validating path/crate ownership.
//! Slice-catalog validation lives in [`crate::ci_plan::slice_catalog`].

use crate::ci_plan::diagnostics::{
    PlanError, PlanResult, fail, nonempty_string, repr, string_list,
};
use crate::ci_plan::document::Json;
use crate::repository::python_text;
use std::collections::BTreeSet;
use std::path::Path;

/// A domain and the glob patterns (paths or crate names) that select it.
pub(super) struct OwnershipRule {
    pub(super) domain: String,
    pub(super) patterns: Vec<String>,
}

/// The validated `ci/ownership.yml`.
pub(super) struct Ownership {
    pub(super) domains: Vec<String>,
    pub(super) path_rules: Vec<OwnershipRule>,
    pub(super) crate_rules: Vec<OwnershipRule>,
}

impl Ownership {
    pub(super) fn domain_set(&self) -> BTreeSet<&str> {
        self.domains.iter().map(String::as_str).collect()
    }
}

/// `str(Path(text))` for a POSIX path: repeated separators and `.` parts
/// collapse and a trailing separator disappears.
pub(crate) fn python_path_display(path: &Path) -> String {
    let text = path.to_string_lossy();
    let parts = text
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>()
        .join("/");
    match (text.starts_with('/'), parts.is_empty()) {
        (true, _) => format!("/{parts}"),
        (false, true) => ".".to_owned(),
        (false, false) => parts,
    }
}

/// Python's `OSError.__str__` for a failed file read.
pub(crate) fn os_error_text(error: &std::io::Error, path: &str) -> String {
    let rendered = error.to_string();
    match error.raw_os_error() {
        Some(code) => {
            let suffix = format!(" (os error {code})");
            let reason = rendered.strip_suffix(&suffix).unwrap_or(&rendered);
            format!("[Errno {code}] {reason}: {}", repr(path))
        }
        None => rendered,
    }
}

/// `_load_manifest`: parse, require an object, require schema version 1.
pub(super) fn load(path: &Path) -> PlanResult<Json> {
    let shown = python_path_display(path);
    let bytes = std::fs::read(path).map_err(|error| {
        PlanError(format!(
            "unable to load {shown}: {}",
            os_error_text(&error, &shown)
        ))
    })?;
    let text = String::from_utf8_lossy(&bytes);
    let value = Json::parse(text.as_bytes())
        .map_err(|error| PlanError(format!("unable to load {shown}: {error}")))?;
    if value.as_object().is_none() {
        return fail(format!("{shown} must contain an object"));
    }
    if !value.get("schema_version").is_some_and(Json::equals_one) {
        return fail(format!("{shown} has unsupported schema_version"));
    }
    Ok(value)
}

/// The ownership half of `_validate_manifests`, in legacy check order.
pub(super) fn validate_ownership(ownership: &Json) -> PlanResult<Ownership> {
    let domains = string_list(ownership.get("domains"), "ownership.domains")?;
    let known = domains.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if ownership.get("unknown_path_policy").and_then(Json::as_str) != Some("fail") {
        return fail("ownership.unknown_path_policy must be 'fail'");
    }
    let path_rules = rules(ownership, RuleKind::Path, &known)?;
    let crate_rules = rules(ownership, RuleKind::Crate, &known)?;
    Ok(Ownership {
        domains,
        path_rules,
        crate_rules,
    })
}

#[derive(Clone, Copy)]
enum RuleKind {
    Path,
    Crate,
}

impl RuleKind {
    fn field(self) -> &'static str {
        match self {
            RuleKind::Path => "path_rules",
            RuleKind::Crate => "crate_rules",
        }
    }

    fn noun(self) -> &'static str {
        match self {
            RuleKind::Path => "path rule",
            RuleKind::Crate => "crate rule",
        }
    }

    fn list_key(self) -> &'static str {
        match self {
            RuleKind::Path => "patterns",
            RuleKind::Crate => "crates",
        }
    }
}

fn rules(
    ownership: &Json,
    kind: RuleKind,
    known: &BTreeSet<&str>,
) -> PlanResult<Vec<OwnershipRule>> {
    let field = kind.field();
    let Some(items) = ownership.get(field).and_then(Json::as_array) else {
        return fail(format!("ownership.{field} must be an array"));
    };
    let mut parsed = Vec::with_capacity(items.len());
    for (index, rule) in items.iter().enumerate() {
        if rule.as_object().is_none() {
            return fail(format!("ownership.{field}[{index}] must be an object"));
        }
        let domain = nonempty_string(rule.get("domain"), &format!("{field}[{index}].domain"))?;
        if !known.contains(domain.as_str()) {
            return fail(format!(
                "{} references unknown domain {}",
                kind.noun(),
                repr(&domain)
            ));
        }
        let key = kind.list_key();
        let patterns = string_list(rule.get(key), &format!("{field}[{index}].{key}"))?;
        if matches!(kind, RuleKind::Path) && patterns.is_empty() {
            return fail(format!("path rule {} has no patterns", repr(&domain)));
        }
        parsed.push(OwnershipRule { domain, patterns });
    }
    Ok(parsed)
}

/// `str.strip()` for subprocess diagnostics.
pub(super) fn stripped(text: &str) -> &str {
    python_text::strip(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_plan_path_display_matches_pathlib() {
        assert_eq!(python_path_display(Path::new("a//b/./c/")), "a/b/c");
        assert_eq!(python_path_display(Path::new("/tmp//x")), "/tmp/x");
        assert_eq!(python_path_display(Path::new("./")), ".");
    }

    #[test]
    fn migration_ci_plan_missing_catalog_reports_python_oserror() {
        let path = Path::new("/nonexistent-ci-plan-root/ci/ownership.yml");
        let error = load(path).expect_err("missing file");
        assert_eq!(
            error.0,
            "unable to load /nonexistent-ci-plan-root/ci/ownership.yml: [Errno 2] No such file or directory: '/nonexistent-ci-plan-root/ci/ownership.yml'"
        );
    }

    #[test]
    fn migration_ci_plan_rejects_unowned_rule_domains() {
        let document = Json::parse(
            br#"{"domains":["docs"],"unknown_path_policy":"fail","path_rules":[{"domain":"x","patterns":["*"]}],"crate_rules":[]}"#,
        )
        .expect("valid JSON");
        let error = validate_ownership(&document).err().expect("unknown domain");
        assert_eq!(error.0, "path rule references unknown domain 'x'");
    }
}
