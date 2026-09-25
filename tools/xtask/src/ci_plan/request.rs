//! The planner input contract: closed fields, profile/event pairing, source
//! identity and repository-relative changed paths.

use crate::ci_plan::diagnostics::{PlanResult, fail, nonempty_string, repr, string_list};
use crate::ci_plan::document::Json;

pub(super) const FORCE_ALL: &str = "__force_all__";

const ALLOWED_FIELDS: &[&str] = &[
    "profile",
    "event_name",
    "source_sha",
    "base_sha",
    "changed_files",
    "affected_crates",
    "workspace_packages",
];
const REQUIRED_FIELDS: &[&str] = &[
    "profile",
    "event_name",
    "source_sha",
    "base_sha",
    "changed_files",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Profile {
    PrDraft,
    PrReady,
    Main,
    ManualFull,
}

impl Profile {
    pub(super) const ALL: [Profile; 4] = [
        Profile::PrDraft,
        Profile::PrReady,
        Profile::Main,
        Profile::ManualFull,
    ];

    pub(super) fn name(self) -> &'static str {
        match self {
            Profile::PrDraft => "pr-draft",
            Profile::PrReady => "pr-ready",
            Profile::Main => "main",
            Profile::ManualFull => "manual-full",
        }
    }

    pub(super) fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|profile| profile.name() == name)
    }

    /// Main and manual-full test the whole workspace.
    pub(super) fn exhaustive(self) -> bool {
        matches!(self, Profile::Main | Profile::ManualFull)
    }
}

/// A validated request. `affected_crates` and `workspace_packages` stay raw:
/// the legacy planner validates them later, after the catalogs load.
pub(super) struct Request {
    pub(super) profile: Profile,
    pub(super) event_name: String,
    pub(super) source_sha: String,
    pub(super) base_sha: String,
    pub(super) changed_files: Json,
    pub(super) affected_crates: Option<Json>,
    pub(super) workspace_packages: Option<Json>,
}

fn is_sha(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

pub(super) fn parse(payload: &Json) -> PlanResult<Request> {
    let Some(fields) = payload.as_object() else {
        return fail("planner input must be an object");
    };
    let mut unknown = fields
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| !ALLOWED_FIELDS.contains(name))
        .collect::<Vec<_>>();
    unknown.sort_unstable();
    if !unknown.is_empty() {
        return fail(format!(
            "planner input has unknown fields: {}",
            unknown.join(", ")
        ));
    }
    if let Some(missing) = REQUIRED_FIELDS
        .iter()
        .find(|field| payload.get(field).is_none())
    {
        return fail(format!("planner input is missing {missing}"));
    }
    let profile_name = nonempty_string(payload.get("profile"), "profile")?;
    let Some(profile) = Profile::parse(&profile_name) else {
        return fail(format!("unsupported profile {}", repr(&profile_name)));
    };
    let event_name = nonempty_string(payload.get("event_name"), "event_name")?;
    if !["pull_request", "push", "workflow_dispatch"].contains(&event_name.as_str()) {
        return fail(format!("unsupported event_name {}", repr(&event_name)));
    }
    let source_sha = nonempty_string(payload.get("source_sha"), "source_sha")?;
    if !is_sha(&source_sha) {
        return fail("source_sha must be a lowercase 40-character SHA");
    }
    let base_sha = match payload.get("base_sha").and_then(Json::as_str) {
        Some(text) if text.is_empty() || is_sha(text) => text.to_owned(),
        _ => return fail("base_sha must be empty or a lowercase 40-character SHA"),
    };
    check_event(profile, &event_name)?;
    Ok(Request {
        profile,
        event_name,
        source_sha,
        base_sha,
        changed_files: payload.get("changed_files").cloned().unwrap_or(Json::Null),
        affected_crates: payload.get_present("affected_crates").cloned(),
        workspace_packages: payload.get_present("workspace_packages").cloned(),
    })
}

fn check_event(profile: Profile, event_name: &str) -> PlanResult<()> {
    match profile {
        Profile::PrDraft | Profile::PrReady if event_name != "pull_request" => {
            fail(format!("profile {} requires pull_request", profile.name()))
        }
        Profile::Main if !matches!(event_name, "push" | "workflow_dispatch") => {
            fail("profile main requires push or workflow_dispatch")
        }
        Profile::ManualFull if event_name != "workflow_dispatch" => {
            fail("profile manual-full requires workflow_dispatch")
        }
        _ => Ok(()),
    }
}

/// Strips one `./`, rejects absolute, backslash, parent and empty paths,
/// then removes repeats while keeping first positions.
pub(super) fn normalise_changed_files(raw: &Json) -> PlanResult<Vec<String>> {
    let files = string_list(Some(raw), "changed_files")?;
    let mut normalised: Vec<String> = Vec::new();
    for path in files {
        let candidate = path.strip_prefix("./").unwrap_or(&path);
        if candidate != FORCE_ALL {
            check_repository_path(&path, candidate)?;
        }
        if !normalised.iter().any(|seen| seen == candidate) {
            normalised.push(candidate.to_owned());
        }
    }
    Ok(normalised)
}

fn check_repository_path(original: &str, candidate: &str) -> PlanResult<()> {
    if candidate.starts_with('/') || candidate.contains('\\') {
        return fail(format!(
            "changed file is not a repository-relative POSIX path: {}",
            repr(original)
        ));
    }
    // `pathlib.PurePosixPath.parts` drops empty and `.` components.
    let parent = candidate.split('/').any(|part| part == "..");
    if parent || candidate.is_empty() || candidate == "." {
        return fail(format!(
            "changed file is not a normal repository path: {}",
            repr(original)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(text: &str) -> PlanResult<Vec<String>> {
        normalise_changed_files(&Json::parse(text.as_bytes()).expect("valid JSON"))
    }

    #[test]
    fn migration_ci_plan_changed_files_normalise_like_legacy() {
        assert_eq!(
            files(r#"["./docs/a.md","docs/a.md","docs//b.md","./__force_all__"]"#),
            Ok(vec![
                "docs/a.md".to_owned(),
                "docs//b.md".to_owned(),
                FORCE_ALL.to_owned()
            ])
        );
        assert_eq!(
            files(r#"["./"]"#),
            fail("changed file is not a normal repository path: './'")
        );
        assert_eq!(
            files(r#"["a/../b"]"#),
            fail("changed file is not a normal repository path: 'a/../b'")
        );
    }

    #[test]
    fn migration_ci_plan_sha_accepts_only_lowercase_hex() {
        assert!(is_sha(&"0f".repeat(20)));
        assert!(!is_sha(&"0F".repeat(20)));
        assert!(!is_sha(&"0f".repeat(19)));
    }
}
