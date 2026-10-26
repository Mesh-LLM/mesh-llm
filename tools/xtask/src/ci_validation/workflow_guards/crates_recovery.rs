//! Recovery publishes only a verified stable release through the trusted controller.
use super::{Node, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let workflow = workflows
        .get("resume-crates-release.yml")
        .ok_or("crate recovery workflow missing")?;
    let events = h::member(workflow, "on")?;
    if !matches!(events, Node::Map(entries) if entries.len() == 1 && entries[0].0 == "workflow_dispatch")
    {
        return Err("crate recovery must remain manual only".into());
    }
    h::binding(h::member(workflow, "permissions")?, "contents", "read")?;
    let job = h::job(workflow, "publish")?;
    h::condition(
        job,
        "github.repository == 'Mesh-LLM/mesh-llm' && github.ref == 'refs/heads/main'",
    )?;
    h::binding(job, "runs-on", "ubuntu-24.04")?;
    h::binding(h::member(job, "env")?, "SCCACHE_GHA_ENABLED", "false")?;
    let steps = h::steps(job)?;
    let (_, cache) = h::step(
        steps,
        "uses",
        "./controller/.github/actions/configure-sccache-gha",
    )?;
    h::binding(
        h::member(cache, "with")?,
        "allow_depot_remote_cache",
        "false",
    )?;
    h::binding(
        h::member(cache, "with")?,
        "allow_native_github_cache",
        "false",
    )?;
    let (controller, checkout) = h::step(steps, "name", "Check out trusted recovery controller")?;
    h::binding(h::member(checkout, "with")?, "ref", "${{ github.sha }}")?;
    h::binding(h::member(checkout, "with")?, "path", "controller")?;
    h::binding(h::member(checkout, "with")?, "persist-credentials", "false")?;
    let (source, checkout) = h::step(steps, "name", "Check out immutable release source")?;
    h::binding(
        h::member(checkout, "with")?,
        "ref",
        "${{ inputs.release_tag }}",
    )?;
    h::binding(h::member(checkout, "with")?, "path", "release-source")?;
    h::binding(h::member(checkout, "with")?, "persist-credentials", "false")?;
    let (verify, validation) = h::step(steps, "name", "Verify exact stable release source")?;
    h::before(controller, verify)?;
    h::before(source, verify)?;
    h::binding(
        h::member(validation, "env")?,
        "RELEASE_TAG",
        "${{ inputs.release_tag }}",
    )?;
    h::binding(
        h::member(validation, "env")?,
        "EXPECTED_SOURCE_SHA",
        "${{ inputs.expected_source_sha }}",
    )?;
    let body = super::field(validation, "run").ok_or("release source validation missing")?;
    for required in [
        "^v[0-9]+\\.[0-9]+\\.[0-9]+$",
        "^[0-9a-f]{40}$",
        "\"refs/tags/${RELEASE_TAG}^{}\"",
        "checked_out_sha=\"$(git -C release-source rev-parse HEAD)\"",
        "\"$remote_sha\" != \"$EXPECTED_SOURCE_SHA\"",
        "\"$checked_out_sha\" != \"$EXPECTED_SOURCE_SHA\"",
        "exit 1",
    ] {
        if !body.contains(required) {
            return Err(format!("release source validation lost {required}").into());
        }
    }
    let (consistency, check) = h::step(steps, "name", "Check crates.io publish-chain consistency")?;
    h::before(verify, consistency)?;
    h::binding(check, "working-directory", "release-source")?;
    h::command(
        check,
        &[
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--locked",
            "|",
            "just",
            "--justfile",
            "../controller/Justfile",
            "automation-run",
            "repository",
            "publish-order",
            "--selected-script",
            "\"$PWD/scripts/publish-crates.sh\"",
            "--source-root",
            "\"$PWD\"",
        ],
        &[],
    )?;
    let (runtime, restore) = h::step(steps, "id", "runtime")?;
    h::before(consistency, runtime)?;
    h::binding(
        h::member(restore, "env")?,
        "RELEASE_TAG",
        "${{ inputs.release_tag }}",
    )?;
    h::command(
        restore,
        &["sha256sum", "--check", "\"$archive.sha256\""],
        &[],
    )?;
    let body = super::field(restore, "run").ok_or("release runtime restore missing")?;
    for required in [
        "mesh-llm-${RELEASE_TAG}-x86_64-unknown-linux-gnu.tar.gz",
        "libmtmd.so libllama-common.so libllama.so",
        "\"$lib_dir/$library\"",
        "lib_dir=%s",
        "$GITHUB_OUTPUT",
    ] {
        if !body.contains(required) {
            return Err(format!("release runtime identity lost {required}").into());
        }
    }
    let (publish, command) = h::step(steps, "name", "Resume crates.io package chain")?;
    h::before(runtime, publish)?;
    h::binding(command, "working-directory", "release-source")?;
    h::command(
        command,
        &["../controller/scripts/publish-crates.sh", "--resume"],
        &[],
    )?;
    let environment = h::member(command, "env")?;
    h::binding(
        environment,
        "CARGO_REGISTRY_TOKEN",
        "${{ secrets.CARGO_REGISTRY_TOKEN }}",
    )?;
    h::binding(
        environment,
        "LLAMA_STAGE_LIB_DIR",
        "${{ steps.runtime.outputs.lib_dir }}",
    )?;
    if secret_references(workflow) != 1 {
        return Err("registry token must be confined to the publishing step".into());
    }
    Ok(())
}

fn secret_references(node: &Node) -> usize {
    match node {
        Node::Scalar(text) => text.matches("secrets.").count(),
        Node::Seq(values) => values.iter().map(secret_references).sum(),
        Node::Map(values) => values
            .iter()
            .map(|(_, value)| secret_references(value))
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::workflow_yaml;
    use super::*;
    const SOURCE: &str = include_str!("../../../../../.github/workflows/resume-crates-release.yml");
    fn documents(source: &str) -> BTreeMap<String, Node> {
        BTreeMap::from([(
            "resume-crates-release.yml".into(),
            workflow_yaml::parse(source).unwrap(),
        )])
    }
    #[test]
    fn crates_recovery_preserves_main_admission_exact_source_and_scoped_publication() {
        check(&documents(SOURCE)).unwrap();
    }
    #[test]
    fn crates_recovery_rejects_changed_source_checks_or_unverified_runtime() {
        for (from, to) in [
            (
                "github.ref == 'refs/heads/main'",
                "github.ref == 'refs/heads/foreign'",
            ),
            ("ref: ${{ github.sha }}", "ref: ${{ inputs.release_tag }}"),
            ("--no-deps --locked", "--no-deps"),
            ("--source-root \"$PWD\"", ""),
            (
                "allow_depot_remote_cache: \"false\"",
                "allow_depot_remote_cache: \"true\"",
            ),
            (
                "allow_native_github_cache: \"false\"",
                "allow_native_github_cache: \"true\"",
            ),
            (
                "../controller/Justfile automation-run",
                "Justfile automation-run",
            ),
            (
                "--selected-script \"$PWD/scripts/publish-crates.sh\"",
                "--selected-script ../controller/scripts/publish-crates.sh",
            ),
            ("^[0-9a-f]{40}$", "^[0-9a-f]{7}$"),
            (
                "$remote_sha\" != \"$EXPECTED_SOURCE_SHA",
                "$remote_sha\" == \"$EXPECTED_SOURCE_SHA",
            ),
            (
                "$checked_out_sha\" != \"$EXPECTED_SOURCE_SHA",
                "$checked_out_sha\" == \"$EXPECTED_SOURCE_SHA",
            ),
            ("sha256sum --check", "echo unverified"),
            ("libmtmd.so libllama-common.so libllama.so", "libllama.so"),
            ("${{ steps.runtime.outputs.lib_dir }}", "/unverified/lib"),
            (
                "../controller/scripts/publish-crates.sh --resume",
                "scripts/publish-crates.sh --resume",
            ),
        ] {
            assert!(SOURCE.contains(from));
            assert!(
                check(&documents(&SOURCE.replace(from, to))).is_err(),
                "accepted recovery mutation: {from}"
            );
        }
    }
    #[test]
    fn crates_recovery_rejects_credential_persistence_and_token_exposure_before_publish() {
        assert!(
            check(&documents(&SOURCE.replace(
                "persist-credentials: false",
                "persist-credentials: true"
            )))
            .is_err()
        );
        let leaked = SOURCE.replace("SCCACHE_GHA_ENABLED: \"false\"", "SCCACHE_GHA_ENABLED: \"false\"\n      CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}");
        assert!(check(&documents(&leaked)).is_err());
    }
}
