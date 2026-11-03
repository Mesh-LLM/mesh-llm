use super::ownership::{canonical_patterns, effective_owners, validate};
use super::policy;

fn terminal_block() -> String {
    canonical_patterns(&policy::fixture())
        .iter()
        .map(|pattern| format!("{pattern} @Mesh-LLM/ci-policy-reviewers\n"))
        .collect()
}

#[test]
fn checked_in_ownership_protects_every_policy_pattern() {
    let codeowners = include_str!("../../../../.github/CODEOWNERS");
    validate(&policy::fixture(), codeowners).unwrap();
    for path in [
        ".github/workflows/pr_windows.yml",
        "mesh/deploy/windows/setup.ps1",
        "tools/xtask/src/ci_governance/windows_guard.rs",
        "scripts/windows.ps1",
        ".github/CODEOWNERS",
        "ci/governance/policy.json",
        ".agents/skills/manage-ci/SKILL.md",
    ] {
        assert_eq!(
            effective_owners(codeowners, path).unwrap(),
            ["@Mesh-LLM/ci-policy-reviewers"]
        );
    }
    assert!(
        !effective_owners(codeowners, "mesh/crates/mesh-llm-ui/src/main.rs")
            .unwrap()
            .contains(&"@Mesh-LLM/ci-policy-reviewers".into())
    );
}

#[test]
fn later_windows_override_is_denied_and_final_team_block_restores_protection() {
    let windows = "/.github/workflows/*windows*.yml @windows-owner\n";
    let invalid = format!("{}{}", terminal_block(), windows);
    assert_eq!(
        effective_owners(&invalid, ".github/workflows/pr_windows.yml").unwrap(),
        ["@windows-owner"]
    );
    assert!(validate(&policy::fixture(), &invalid).is_err());
    validate(
        &policy::fixture(),
        &format!("{windows}{}", terminal_block()),
    )
    .unwrap();
}

#[test]
fn any_missing_control_or_additional_owner_fails() {
    let block = terminal_block();
    for pattern in canonical_patterns(&policy::fixture()) {
        let missing = block.replace(&format!("{pattern} @Mesh-LLM/ci-policy-reviewers\n"), "");
        assert!(validate(&policy::fixture(), &missing).is_err(), "{pattern}");
    }
    let shared = block.replace(
        "/ci/ @Mesh-LLM/ci-policy-reviewers",
        "/ci/ @Mesh-LLM/ci-policy-reviewers @other",
    );
    assert!(validate(&policy::fixture(), &shared).is_err());
    assert!(validate(&policy::fixture(), &(block + "/ci/\n")).is_err());
}

#[test]
fn component_stars_do_not_cross_slashes_and_double_star_can() {
    let rules = "* @fallback\n/ci/*.json @flat\n/tools/**/windows* @windows\n";
    assert_eq!(
        effective_owners(rules, "ci/policy.json").unwrap(),
        ["@flat"]
    );
    assert_eq!(
        effective_owners(rules, "ci/governance/policy.json").unwrap(),
        ["@fallback"]
    );
    assert_eq!(
        effective_owners(rules, "tools/windows.rs").unwrap(),
        ["@windows"]
    );
    assert_eq!(
        effective_owners(rules, "tools/xtask/src/windows/mod.rs").unwrap(),
        ["@windows"]
    );
}

#[test]
fn anchored_directory_and_basename_patterns_follow_codeowners_rules() {
    let rules = "* @fallback\n/docs/ @root-docs\n*.md @markdown\n";
    assert_eq!(
        effective_owners(rules, "docs/nested/source.rs").unwrap(),
        ["@root-docs"]
    );
    assert_eq!(
        effective_owners(rules, "mesh/docs/source.rs").unwrap(),
        ["@fallback"]
    );
    assert_eq!(
        effective_owners(rules, "mesh/docs/README.md").unwrap(),
        ["@markdown"]
    );
    assert_eq!(
        effective_owners(rules, "docs-other/source.rs").unwrap(),
        ["@fallback"]
    );
}

#[test]
fn unsupported_patterns_fail_instead_of_approximating() {
    for pattern in [
        "!ci/",
        "/ci/[ab]",
        "/ci/file?",
        "/ci/a\\ b",
        "/ci/a**b",
        "/ci/***",
        "/ci/#gate",
    ] {
        assert!(
            effective_owners(&format!("{pattern} @owner\n"), "ci/a").is_err(),
            "{pattern}"
        );
    }
}
