//! Manual registry sampling retains main-owned admission and native runner credentials.
use super::{Node, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let workflow = workflows
        .get("depot-registry-canary.yml")
        .ok_or("registry canary missing")?;
    let events = h::member(workflow, "on")?;
    h::member(events, "workflow_dispatch")?;
    if !matches!(events, Node::Map(entries) if entries.len() == 1 && entries[0].0 == "workflow_dispatch")
    {
        return Err("registry diagnostic must remain manual only".into());
    }
    read_only(h::member(workflow, "permissions")?)?;
    let jobs = h::member(workflow, "jobs")?;
    if !matches!(jobs, Node::Map(entries) if entries.len() == 3 && entries.iter().all(|(name, _)| matches!(name.as_str(), "policy" | "pull" | "summary")))
    {
        return Err("registry canary job roster changed".into());
    }
    let policy = h::job(workflow, "policy")?;
    for name in ["policy", "pull", "summary"] {
        if let Some(permissions) = h::job(workflow, name)?.get("permissions") {
            read_only(permissions)?;
        }
    }
    h::binding(policy, "runs-on", "ubuntu-24.04")?;
    let (_, validate) = h::step(h::steps(policy)?, "id", "validate")?;
    let outputs = h::member(policy, "outputs")?;
    h::binding(outputs, "digest", "${{ steps.validate.outputs.digest }}")?;
    h::binding(
        outputs,
        "depot_image",
        "${{ steps.validate.outputs.depot_image }}",
    )?;
    let inputs = h::member(validate, "env")?;
    h::binding(inputs, "UPSTREAM_IMAGE", "${{ inputs.upstream_image }}")?;
    h::binding(inputs, "DEPOT_REPOSITORY", "${{ inputs.depot_repository }}")?;
    h::binding(
        inputs,
        "DEPOT_REGISTRY_HOST",
        "${{ vars.DEPOT_REGISTRY_HOST }}",
    )?;
    let body = super::field(validate, "run").ok_or("registry admission missing")?;
    for required in [
        "$GITHUB_REPOSITORY\" != \"Mesh-LLM/mesh-llm",
        "$GITHUB_EVENT_NAME\" != \"workflow_dispatch",
        "$GITHUB_REF\" != \"refs/heads/main",
        "@sha256:([a-f0-9]{64})$",
        "upstream_image must be pinned by sha256 digest",
        "DEPOT_REGISTRY_HOST",
        "DEPOT_REPOSITORY",
    ] {
        if !body.contains(required) {
            return Err(format!("registry admission lost {required}").into());
        }
    }
    if !body.contains("digest=\"sha256:${BASH_REMATCH[1]}\"") || !body.contains("exit 1") {
        return Err("registry digest binding or rejection missing".into());
    }
    let pull = h::job(workflow, "pull")?;
    h::binding(pull, "needs", "policy")?;
    read_only(h::member(pull, "permissions")?)?;
    h::binding(pull, "runs-on", "depot-ubuntu-24.04")?;
    let strategy = h::member(pull, "strategy")?;
    h::binding(strategy, "fail-fast", "false")?;
    let matrix = h::member(strategy, "matrix")?;
    exact_list(matrix, "source", &["upstream", "depot"])?;
    exact_list(matrix, "sample", &["1", "2", "3", "4", "5"])?;
    let steps = h::steps(pull)?;
    let (_, execute) = h::step(steps, "name", "Pull exact image on a fresh runner")?;
    h::command(execute, &["docker", "pull", "\"$image\""], &[])?;
    let body = super::field(execute, "run").ok_or("registry pull missing")?;
    for required in [
        "\"${DEPOT_ORG_ID:-}\" != \"1ntz5vlngn\"",
        "pre-authenticated Mesh-LLM Depot runner",
        "\"$resolved_digest\" != \"$EXPECTED_DIGEST\"",
        "digest mismatch",
        "observation.json",
    ] {
        if !body.contains(required) {
            return Err(format!("registry sample lost {required}").into());
        }
    }
    let environment = h::member(execute, "env")?;
    h::binding(environment, "SOURCE", "${{ matrix.source }}")?;
    h::binding(environment, "SAMPLE", "${{ matrix.sample }}")?;
    h::binding(
        environment,
        "UPSTREAM_IMAGE",
        "${{ inputs.upstream_image }}",
    )?;
    h::binding(
        environment,
        "EXPECTED_DIGEST",
        "${{ needs.policy.outputs.digest }}",
    )?;
    h::binding(
        environment,
        "DEPOT_IMAGE",
        "${{ needs.policy.outputs.depot_image }}",
    )?;
    let (_, upload) = h::step(steps, "name", "Upload pull observation")?;
    h::binding(
        upload,
        "uses",
        "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f",
    )?;
    h::binding(
        h::member(upload, "with")?,
        "name",
        "depot-registry-pull-${{ matrix.source }}-${{ matrix.sample }}",
    )?;
    h::binding(h::member(upload, "with")?, "path", "observation.json")?;
    let summary = h::job(workflow, "summary")?;
    exact_list(summary, "needs", &["policy", "pull"])?;
    h::binding(summary, "runs-on", "ubuntu-24.04")?;
    h::condition(summary, "${{ !cancelled() }}")?;
    let summary_steps = h::steps(summary)?;
    let (download_index, download) = h::step(summary_steps, "name", "Download pull observations")?;
    let (prepare_index, prepare) = h::step(
        summary_steps,
        "uses",
        "./.github/actions/prepare-automation",
    )?;
    h::binding(h::member(prepare, "with")?, "runner-profile", "hosted-bare")?;
    let (evaluate_index, evaluate) = h::step(summary_steps, "name", "Evaluate adoption gate")?;
    h::before(prepare_index, evaluate_index)?;
    h::before(download_index, evaluate_index)?;
    h::command(
        evaluate,
        &["\"$MESH_LLM_AUTOMATION_BIN\"", "ci-ops", "registry-pulls"],
        &[],
    )?;
    h::binding(
        h::member(evaluate, "env")?,
        "ENFORCE_THRESHOLD",
        "${{ inputs.enforce_threshold }}",
    )?;
    h::binding(
        download,
        "uses",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    )?;
    h::binding(
        h::member(download, "with")?,
        "pattern",
        "depot-registry-pull-*",
    )?;
    for forbidden in [
        "secrets.",
        "id-token",
        "depot pull-token",
        "docker login",
        "DEPOT_PROJECT_ID",
        "DEPOT_REGISTRY_PULL_TOKEN",
        "printenv",
    ] {
        if mentions(workflow, forbidden) {
            return Err(format!("registry diagnostic must not introduce {forbidden}").into());
        }
    }
    Ok(())
}

fn read_only(node: &Node) -> DynResult<()> {
    if !matches!(node, Node::Map(entries) if entries.len() == 1 && entries[0].0 == "contents") {
        return Err("registry permissions must be contents read only".into());
    }
    h::binding(node, "contents", "read")
}

fn exact_list(node: &Node, key: &str, expected: &[&str]) -> DynResult<()> {
    if h::member(node, key)?.list().as_slice() == expected {
        Ok(())
    } else {
        Err(format!("registry {key} must retain {expected:?}").into())
    }
}

fn mentions(node: &Node, fragment: &str) -> bool {
    match node {
        Node::Scalar(text) => text.contains(fragment),
        Node::Seq(values) => values.iter().any(|value| mentions(value, fragment)),
        Node::Map(values) => values
            .iter()
            .any(|(name, value)| name.contains(fragment) || mentions(value, fragment)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::workflow_yaml;
    use super::*;
    const SOURCE: &str = include_str!("../../../../../.github/workflows/depot-registry-canary.yml");
    fn documents(source: &str) -> BTreeMap<String, Node> {
        BTreeMap::from([(
            "depot-registry-canary.yml".into(),
            workflow_yaml::parse(source).unwrap(),
        )])
    }
    #[test]
    fn registry_canary_preserves_manual_main_digest_and_native_credentials() {
        check(&documents(SOURCE)).unwrap();
    }
    #[test]
    fn registry_canary_rejects_weakened_admission_and_digest_observations() {
        for (from, to) in [
            ("\"refs/heads/main\"", "\"refs/heads/foreign\""),
            ("${DEPOT_ORG_ID:-}", "${OTHER_ORG_ID:-}"),
            ("[upstream, depot]", "[depot]"),
            ("${{ steps.validate.outputs.digest }}", "sha256:unbound"),
            ("${{ needs.policy.outputs.digest }}", "sha256:unbound"),
            ("${{ matrix.sample }}", "1"),
            ("[policy, pull]", "[policy]"),
            ("contents: read", "contents: write"),
            (
                "  policy:\n",
                "  policy:\n    permissions:\n      contents: write\n",
            ),
            (
                "  summary:\n",
                "  summary:\n    permissions:\n      contents: write\n",
            ),
            ("runner-profile: hosted-bare", "runner-profile: image"),
            ("[1, 2, 3, 4, 5]", "[1]"),
            (
                "$resolved_digest\" != \"$EXPECTED_DIGEST",
                "$resolved_digest\" == \"$EXPECTED_DIGEST",
            ),
            ("@b7c566a772e6b6bfb58ed0dc250532a479d7789f", "@main"),
            ("@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", "@main"),
        ] {
            assert!(SOURCE.contains(from));
            assert!(
                check(&documents(&SOURCE.replace(from, to))).is_err(),
                "accepted registry mutation: {from}"
            );
        }
    }
    #[test]
    fn registry_canary_rejects_interactive_registry_auth_or_repository_secrets() {
        for injection in [
            "docker login",
            "depot pull-token",
            "printenv",
            "${{ secrets.REGISTRY_TOKEN }}",
        ] {
            let changed = SOURCE.replace(
                "          docker pull",
                &format!("          {injection}\n          docker pull"),
            );
            assert!(check(&documents(&changed)).is_err());
        }
    }
}
