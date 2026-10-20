//! Signing credentials never enter the download/load consumer or ordinary lanes.
use super::{document, input, job, named, source, steps, text};

#[test]
fn signing_diagnostic_is_manual_isolated_and_uses_immutable_public_artifacts() {
    let controller = document("ci-control.yml");
    assert_eq!(controller.get("on").unwrap().entries().len(), 1);
    assert!(
        controller
            .get("on")
            .unwrap()
            .get("workflow_dispatch")
            .is_some()
    );
    let plan = job(&controller, "plan");
    assert!(
        text(plan, "if")
            .unwrap()
            .contains("!inputs.runtime_signing_diagnostic")
    );
    let caller = job(&controller, "runtime_signing_diagnostic");
    assert!(
        text(caller, "if")
            .unwrap()
            .contains("inputs.runtime_signing_diagnostic")
    );
    assert_eq!(
        text(caller, "uses"),
        Some("./.github/workflows/runtime-signing-diagnostic.yml")
    );
    assert_eq!(caller.get("secrets").unwrap().entries().len(), 2);
    let diagnostic = document("runtime-signing-diagnostic.yml");
    assert_eq!(diagnostic.get("on").unwrap().entries().len(), 1);
    assert!(diagnostic.get("on").unwrap().get("workflow_call").is_some());
    let sign = job(&diagnostic, "sign");
    let consumer = job(&diagnostic, "consume");
    for job in [sign, consumer] {
        assert_eq!(text(job, "runs-on"), Some("ubuntu-24.04"));
        assert!(job.get("timeout-minutes").is_some());
        assert!(job.get("environment").is_none());
        assert!(job.get("env").is_none());
    }
    assert_eq!(text(consumer, "needs"), Some("sign"));
    let setup = steps(sign)
        .iter()
        .find(|s| text(s, "uses") == Some("./.github/actions/prepare-automation"))
        .unwrap();
    assert_eq!(input(setup, "reuse-hosted-binary"), Some("false"));
    assert_eq!(input(setup, "runner-profile"), Some("hosted-bare"));
    let secrets = named(sign, "Sign with repository credentials")
        .get("env")
        .unwrap();
    assert_eq!(secrets.entries().len(), 2);
    for step in steps(consumer) {
        assert!(!format!("{step:?}").contains("secrets."));
    }
    let download = steps(consumer)
        .iter()
        .find(|s| {
            text(s, "uses")
                .unwrap_or("")
                .starts_with("actions/download-artifact@")
        })
        .unwrap();
    assert_eq!(
        input(download, "artifact-ids"),
        Some("${{ needs.sign.outputs.artifact_id }}")
    );
    let test = named(
        consumer,
        "Strict HTTP install, negative cases, and real native load",
    );
    assert_eq!(
        text(test, "run"),
        Some("cargo test --locked -p skippy-runtime-install -- --include-ignored")
    );
    let source = source("runtime-signing-diagnostic.yml");
    assert!(!source.contains("contents: write"));
    assert!(!source.contains("secrets: inherit"));
    assert!(!source.contains("action-gh-release"));
}

#[test]
fn release_signing_uses_repository_secrets_and_publishes_the_signature() {
    let release = document("release.yml");
    let publish = job(&release, "publish");
    assert!(publish.get("environment").is_none());
    let sign = named(publish, "Sign native runtime release catalog");
    assert_eq!(sign.get("env").unwrap().entries().len(), 2);
    assert!(
        text(sign, "run")
            .unwrap()
            .contains("native-runtimes.json.minisig")
    );
    assert_eq!(
        input(named(publish, "Publish GitHub release"), "files"),
        Some("release-artifacts/*")
    );
}
