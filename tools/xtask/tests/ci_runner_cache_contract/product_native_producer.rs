//! Product-source automation producer: same-run, credential-free, and never an authority artifact.
use super::{
    support::{self, Fixture},
    workflow_yaml::{self, Node},
};
use std::{fs, process::Command};

const PRODUCER: &str = ".github/workflows/product-automation-artifact.yml";
const RUNNERS: &str = "${{ inputs.platform == 'macos-arm64' && 'macos-15' || inputs.platform == 'windows-x64' && 'windows-2022' || 'ubuntu-24.04' }}";
const CONSUMERS: &[&str] = &[
    "ci-linux-host-slice.yml",
    "ci-linux-runtime-slice.yml",
    "ci-linux-product-slice.yml",
    "ci-macos-host-slice.yml",
    "ci-macos-runtime-slice.yml",
    "ci-macos-product-slice.yml",
    "ci-windows-host-slice.yml",
    "ci-windows-runtime-slice.yml",
    "ci-windows-product-slice.yml",
    "ci-rust-tests-slice.yml",
    "static-abi-artifact.yml",
];
/// Slices that forward the identity to a reusable workflow which restores it.
const NESTED_CONSUMERS: &[(&str, &str)] = &[("ci-macos-sdk-slice.yml", "sdk-smoke.yml")];

fn read(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(support::root().join(path)).unwrap()).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn steps(document: &Node) -> &[Node] {
    let producer = document.get("jobs").unwrap().get("producer").unwrap();
    let Node::Seq(steps) = producer.get("steps").unwrap() else {
        panic!("steps")
    };
    steps
}

#[test]
fn producer_is_hosted_read_only_secret_free_and_compiles_the_exact_product_source() {
    let document = read(PRODUCER);
    let producer = document.get("jobs").unwrap().get("producer").unwrap();
    assert_eq!(text(producer, "runs-on"), RUNNERS);
    let permissions = producer.get("permissions").unwrap();
    assert_eq!(permissions.entries().len(), 1);
    assert_eq!(text(permissions, "contents"), "read");
    let serialized = fs::read_to_string(support::root().join(PRODUCER)).unwrap();
    for forbidden in [
        "secrets",
        "environment:",
        "id-token",
        "self-hosted",
        "depot",
    ] {
        assert!(!serialized.contains(forbidden), "{forbidden}");
    }
    let inputs = document
        .get("on")
        .unwrap()
        .get("workflow_call")
        .unwrap()
        .get("inputs")
        .unwrap();
    assert_eq!(inputs.entries().len(), 3);
    let steps = steps(&document);
    assert_eq!(steps.len(), 3);
    assert_eq!(text(&steps[0], "id"), "identity");
    let checkout = steps[1].get("with").unwrap();
    assert_eq!(text(checkout, "ref"), "${{ inputs.product_source_sha }}");
    assert_eq!(text(checkout, "persist-credentials"), "false");
    assert_eq!(
        text(&steps[2], "uses"),
        "./.github/actions/upload-automation"
    );
    let upload = steps[2].get("with").unwrap();
    for (key, value) in [
        ("source_sha", "${{ inputs.product_source_sha }}"),
        ("source-profile", "product-clean"),
        ("ci-lane", "${{ inputs.lane }}"),
        (
            "runner-profile",
            "${{ runner.os == 'Linux' && 'hosted-bare' || 'image' }}",
        ),
    ] {
        assert_eq!(text(upload, key), value, "{key}");
    }
}

#[test]
fn product_artifacts_cannot_collide_with_or_masquerade_as_protected_authority() {
    let action =
        fs::read_to_string(support::root().join(".github/actions/upload-automation/action.yml"))
            .unwrap();
    assert!(action.contains(
        "automation-${{ inputs.source-profile == 'product-clean' && 'product-' || '' }}"
    ));
    let protected = read(".github/workflows/protected-automation-artifact.yml");
    let protected = protected_as_producer(&protected);
    let upload = steps(&protected)[3].get("with").unwrap();
    assert_eq!(text(upload, "source-profile"), "protected-clean");
    let prepare = read(".github/actions/prepare-automation/action.yml");
    let Node::Seq(prepare_steps) = prepare.get("runs").unwrap().get("steps").unwrap() else {
        panic!()
    };
    for step in prepare_steps {
        if text(step, "name") == "Restore hosted automation binary"
            || text(step, "name") == "Save hosted automation binary"
        {
            assert!(text(step, "if").contains("inputs.reuse-hosted-binary == 'true'"));
        }
    }
    let upload_action = read(".github/actions/upload-automation/action.yml");
    let Node::Seq(upload_steps) = upload_action.get("runs").unwrap().get("steps").unwrap() else {
        panic!()
    };
    let prepare_step = upload_steps
        .iter()
        .find(|step| text(step, "id") == "prepare")
        .unwrap();
    assert_eq!(
        text(prepare_step.get("with").unwrap(), "reuse-hosted-binary"),
        "${{ inputs.source-profile != 'protected-clean' && 'true' || 'false' }}"
    );
}

fn protected_as_producer(document: &Node) -> Node {
    Node::Map(vec![(
        "jobs".into(),
        Node::Map(vec![(
            "producer".into(),
            document
                .get("jobs")
                .unwrap()
                .get("producer")
                .unwrap()
                .clone(),
        )]),
    )])
}

#[test]
fn every_lane_producer_is_linked_by_needs_and_every_consumer_restores_without_compiling() {
    for (lane, platform, producer) in [
        (
            "ci-linux-lane.yml",
            "linux-x64",
            "product_automation_linux_x64",
        ),
        (
            "ci-macos-lane.yml",
            "macos-arm64",
            "product_automation_macos_arm64",
        ),
        (
            "ci-windows-lane.yml",
            "windows-x64",
            "product_automation_windows_x64",
        ),
        (
            "ci-pr-canary-lane.yml",
            "linux-x64",
            "product_automation_linux_x64",
        ),
    ] {
        let document = read(&format!(".github/workflows/{lane}"));
        let jobs = document.get("jobs").unwrap();
        let job = jobs.get(producer).unwrap();
        assert_eq!(
            text(job, "uses"),
            "./.github/workflows/product-automation-artifact.yml"
        );
        assert_eq!(text(job.get("with").unwrap(), "platform"), platform);
        assert!(job.get("secrets").is_none());
        let mut consumers = 0;
        for (name, candidate) in jobs.entries() {
            let Some(callee) = text(candidate, "uses").strip_prefix("./.github/workflows/") else {
                continue;
            };
            let Some(inputs) = candidate.get("with") else {
                continue;
            };
            if inputs.get("product_automation").is_none() {
                continue;
            }
            assert!(
                CONSUMERS.contains(&callee)
                    || NESTED_CONSUMERS.iter().any(|(slice, _)| *slice == callee),
                "{lane}/{name}: {callee}"
            );
            assert!(
                candidate.get("needs").unwrap().list().contains(&producer),
                "{lane}/{name}"
            );
            let expected = format!("${{{{ needs.{producer}.outputs.identity_json }}}}");
            assert_eq!(text(inputs, "product_automation"), expected);
            consumers += 1;
        }
        assert!(consumers > 0, "{lane}");
        let summary = jobs.get("summary").unwrap().get("needs").unwrap().list();
        assert!(summary.contains(&producer), "{lane} summary");
    }
}

#[test]
fn consumer_slices_restore_the_same_run_binary_and_never_fall_back_to_a_build() {
    for name in CONSUMERS {
        let source =
            fs::read_to_string(support::root().join(".github/workflows").join(name)).unwrap();
        let document = workflow_yaml::parse(&source).unwrap();
        let input = document
            .get("on")
            .unwrap()
            .get("workflow_call")
            .unwrap()
            .get("inputs")
            .unwrap()
            .get("product_automation")
            .unwrap();
        assert_eq!(text(input, "type"), "string", "{name}");
        assert!(
            source.contains("uses: ./.github/actions/restore-automation"),
            "{name}"
        );
        assert!(
            source.contains("artifact_id: ${{ fromJson(inputs.product_automation).artifact_id }}"),
            "{name}"
        );
        assert!(
            source.contains(
                "binary_sha256: ${{ fromJson(inputs.product_automation).binary_sha256 }}"
            ),
            "{name}"
        );
        assert!(
            source.contains("producer_os: ${{ fromJson(inputs.product_automation).producer_os }}"),
            "{name}"
        );
        if *name != "static-abi-artifact.yml" {
            assert_eq!(text(input, "required"), "true", "{name}");
            assert!(
                !source.contains("uses: ./.github/actions/prepare-automation"),
                "{name}"
            );
            assert!(!source.contains("cargo xtool"), "{name}");
        }
    }
}

#[test]
fn nested_consumer_slices_forward_the_identity_to_a_restoring_workflow() {
    for (slice, workflow) in NESTED_CONSUMERS {
        let source =
            fs::read_to_string(support::root().join(".github/workflows").join(slice)).unwrap();
        let document = workflow_yaml::parse(&source).unwrap();
        let input = document
            .get("on")
            .unwrap()
            .get("workflow_call")
            .unwrap()
            .get("inputs")
            .unwrap()
            .get("product_automation")
            .unwrap();
        assert_eq!(text(input, "type"), "string", "{slice}");
        assert_eq!(text(input, "required"), "true", "{slice}");
        assert!(
            source.contains(&format!("uses: ./.github/workflows/{workflow}")),
            "{slice}"
        );
        assert!(
            source.contains("product_automation: ${{ inputs.product_automation }}"),
            "{slice}"
        );
        let callee =
            fs::read_to_string(support::root().join(".github/workflows").join(workflow)).unwrap();
        for expected in [
            "uses: ./.github/actions/restore-automation",
            "artifact_id: ${{ fromJson(inputs.product_automation).artifact_id }}",
            "binary_sha256: ${{ fromJson(inputs.product_automation).binary_sha256 }}",
            "producer_os: ${{ fromJson(inputs.product_automation).producer_os }}",
        ] {
            assert!(callee.contains(expected), "{workflow}: {expected}");
        }
        for forbidden in ["uses: ./.github/actions/prepare-automation", "cargo xtool"] {
            assert!(!callee.contains(forbidden), "{workflow}: {forbidden}");
        }
    }
}

#[test]
fn restore_rejects_a_producer_on_a_different_runner_before_any_download() {
    let action = read(".github/actions/restore-automation/action.yml");
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!()
    };
    let admission = text(&steps[0], "run");
    for (os, arch, runner_os, runner_arch, accepted) in [
        ("Linux", "X64", "Linux", "X64", true),
        ("Linux", "X64", "macOS", "ARM64", false),
        ("macOS", "ARM64", "macOS", "X64", false),
        ("", "", "Linux", "X64", true),
    ] {
        let fixture = Fixture::new();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .args(["-c", admission])
            .env("PATH", "/usr/bin:/bin")
            .env("RUNNER_TEMP", fixture.path())
            .env("RUNNER_OS", runner_os)
            .env("RUNNER_ARCH", runner_arch)
            .env("AUTOMATION_PRODUCER_OS", os)
            .env("AUTOMATION_PRODUCER_ARCH", arch)
            .env("AUTOMATION_ARTIFACT_ID", "")
            .env("AUTOMATION_BINARY_SHA256", "")
            .env(
                "AUTOMATION_SOURCE_SHA",
                "0123456789012345678901234567890123456789",
            );
        assert_eq!(
            fixture.run(command).status.success(),
            accepted,
            "{os}/{arch} on {runner_os}/{runner_arch}"
        );
    }
}
