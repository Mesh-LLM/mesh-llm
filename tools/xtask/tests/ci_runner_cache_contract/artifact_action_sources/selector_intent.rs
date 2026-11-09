//! Execute the maintained selector action with the existing bounded fake date.
use super::super::support::Fixture;
use serde_json::Value;
fn projection(values: &Value, depot: bool, native: bool) {
    assert_eq!(values["depot_enabled"], depot.to_string());
    assert_eq!(values["allow_depot_remote_cache"], "false");
    assert_eq!(values["allow_native_github_cache"], native.to_string());
    assert_eq!(values["allow_trusted_sccache_seed"], (!depot).to_string());
    for (key, hosted) in [
        ("runner", "ubuntu-24.04"),
        ("runner_arm", "ubuntu-24.04-arm"),
        ("runner_macos", "macos-15"),
        ("runner_windows", "windows-2022"),
    ] {
        assert_eq!(
            values[key],
            if depot {
                format!("depot-{hosted}")
            } else {
                hosted.to_owned()
            }
        );
    }
    for size in [4, 8, 16] {
        for (key, hosted) in [
            ("runner", "ubuntu-24.04"),
            ("runner_arm", "ubuntu-24.04-arm"),
        ] {
            assert_eq!(
                values[format!("{key}_{size}")],
                if depot {
                    format!("depot-{hosted}-{size}")
                } else {
                    hosted.to_owned()
                }
            );
        }
    }
}
fn selected(env: &[(&str, &str)], depot: bool, native: bool) {
    let fixture = Fixture::new();
    let output = fixture.selector(env);
    assert!(output.status.success(), "{env:?}: {output:?}");
    projection(&fixture.outputs(), depot, native);
    fixture.0.close().unwrap();
}
#[test]
fn shared_selector_preserves_legacy_approval_independence_and_dispatch_branch_controls() {
    for (reference, sha) in [
        (
            "refs/pull/12/merge",
            "fedcba9876543210fedcba9876543210fedcba98",
        ),
        (
            "refs/pull/13/merge",
            "0123456789abcdef0123456789abcdef01234567",
        ),
    ] {
        selected(
            &[
                ("INPUT_DEPOT_PR_ENABLED", "true"),
                ("INPUT_PR_APPROVED_REF", reference),
                ("INPUT_PR_APPROVED_SHA", sha),
            ],
            true,
            true,
        );
    }
    // The global bounded exception has expired, so stale approval data cannot
    // grant a provider selection after the deadline.
    selected(
        &[
            ("INPUT_DEPOT_PR_ENABLED", "true"),
            ("INPUT_PR_APPROVED_REF", "refs/pull/12/merge"),
            (
                "INPUT_PR_APPROVED_SHA",
                "0123456789abcdef0123456789abcdef01234567",
            ),
            ("FIXTURE_DATE", "2026-09-14"),
        ],
        false,
        true,
    );
    for (main, manual, origin, provider) in [
        ("true", "false", "", true),
        ("true", "true", "push", true),
        ("false", "true", "push", false),
        ("false", "false", "", false),
    ] {
        selected(
            &[
                ("INPUT_EVENT_NAME", "workflow_dispatch"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_DEPOT_MAIN_ENABLED", main),
                ("INPUT_DEPOT_PR_ENABLED", "true"),
                ("INPUT_MANUAL_USE_DEPOT", manual),
                ("INPUT_ORIGINAL_EVENT_NAME", origin),
            ],
            provider,
            true,
        );
    }
    selected(
        &[
            ("INPUT_EVENT_NAME", "workflow_dispatch"),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_DEPOT_PR_ENABLED", "true"),
            ("FIXTURE_DATE", "2026-09-14"),
        ],
        true,
        false,
    );
    // Without the global opt-in, main provider selection does not grant
    // native-cache access even before the exception deadline.
    selected(
        &[
            ("INPUT_EVENT_NAME", "workflow_dispatch"),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_DEPOT_PR_ENABLED", "false"),
        ],
        true,
        false,
    );
    selected(
        &[
            ("INPUT_EVENT_NAME", "workflow_dispatch"),
            ("INPUT_REF", "refs/heads/feature"),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_MANUAL_USE_DEPOT", "true"),
        ],
        false,
        true,
    );
}
#[test]
fn shared_selector_exact_canary_cannot_bypass_repository_event_or_hosted_controls() {
    let cases: &[&[(&str, &str)]] = &[
        &[("INPUT_PR_CANARY_REF", "refs/pull/13/merge")],
        &[("INPUT_PR_CANARY_REF", "")],
        &[
            ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
            ("INPUT_HEAD_REPOSITORY", "attacker/mesh-llm"),
        ],
        &[
            ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
            ("INPUT_REPOSITORY", "attacker/mesh-llm"),
        ],
        &[
            ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
            ("INPUT_EVENT_NAME", "pull_request_target"),
        ],
        &[
            ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
            ("INPUT_FORCE_HOSTED", "true"),
        ],
        &[
            ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
            ("INPUT_EVENT_NAME", "workflow_dispatch"),
            ("INPUT_REF", "refs/heads/main"),
        ],
    ];
    for env in cases {
        selected(env, false, true);
    }
    // Exact canary remains admitted after the broad exception expires, while
    // its native and remote cache write authority remains denied.
    selected(
        &[
            ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
            ("FIXTURE_DATE", "2026-10-02"),
        ],
        true,
        false,
    );
}
