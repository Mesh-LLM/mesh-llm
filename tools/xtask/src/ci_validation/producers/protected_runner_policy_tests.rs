use super::*;
use std::{fs, path::Path};
fn source(name: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".github/workflows")
            .join(name),
    )
    .unwrap()
}
fn producers() -> [(&'static str, Producer); 2] {
    [
        ("native-sdk-artifact.yml", Producer::NativeSdk),
        ("static-abi-artifact.yml", Producer::StaticAbi),
    ]
}
#[test]
fn checked_in_sdk_and_static_abi_producers_bind_protected_runner_nodes() {
    for (name, producer) in producers() {
        check(&source(name), name, producer).unwrap();
    }
}
#[test]
fn comments_and_unrelated_nodes_cannot_satisfy_sdk_runner_authority() {
    for (name, producer) in producers() {
        let original = source(name);
        for (valid, invalid) in [
            (
                "ref: ${{ github.event.repository.default_branch }}",
                "ref: ${{ inputs.source_sha }}",
            ),
            (
                "repository: ${{ github.repository }}",
                "repository: attacker/mesh-llm",
            ),
            (
                "head_sha: ${{ github.event.pull_request.head.sha || github.sha }}",
                "head_sha: ${{ github.sha }}",
            ),
            (
                "manual_use_depot: ${{ inputs.use_depot }}",
                "manual_use_depot: true",
            ),
            (
                "TARGET: ${{ inputs.target }}",
                "TARGET: x86_64-unknown-linux-gnu",
            ),
            (
                "RUNNER_16: ${{ steps.policy.outputs.runner_16 }}",
                "RUNNER_16: ${{ steps.policy.outputs.runner_4 }}",
            ),
            (
                "runner: ${{ steps.resolve.outputs.runner }}",
                "runner: ${{ steps.policy.outputs.runner }}",
            ),
            (
                "runs-on: ${{ needs.runner_policy.outputs.runner }}",
                "runs-on: ubuntu-24.04",
            ),
        ] {
            assert!(
                original.contains(valid),
                "{name}: missing test anchor {valid}"
            );
            let changed = format!(
                "{}\n# retained spelling: {valid}\n",
                original.replace(valid, invalid)
            );
            assert!(
                check(&changed, name, producer).is_err(),
                "{name}: admitted {invalid}"
            );
        }
    }
}
#[test]
fn sdk_runner_input_defaults_and_resolver_presence_are_required() {
    for (name, producer) in producers() {
        let original = source(name);
        for (valid, invalid) in [
            ("default: '8'", "default: '64'"),
            ("id: resolve", "id: unused_resolver"),
            ("id: policy", "id: unused_policy"),
        ] {
            assert!(original.contains(valid));
            assert!(check(&original.replace(valid, invalid), name, producer).is_err());
        }
    }
}
