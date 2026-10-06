//! Local source contracts for manual-main authority delivery and reusable runner boundaries.
use super::Node;
pub(super) fn field<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
use std::collections::BTreeMap;
fn exact(node: &Node, key: &str, expected: &str) -> Result<(), String> {
    if field(node, key) == Some(expected) {
        Ok(())
    } else {
        Err(format!("authority source {key} changed"))
    }
}
fn compact(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn steps(job: &Node) -> Result<&[Node], String> {
    let Some(Node::Seq(steps)) = job.get("steps") else {
        return Err("authority steps absent".into());
    };
    Ok(steps)
}
fn named<'a>(steps: &'a [Node], name: &str) -> Result<(usize, &'a Node), String> {
    let found: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| field(s, "name") == Some(name))
        .collect();
    let [found] = found.as_slice() else {
        return Err(format!("unique authority step required: {name}"));
    };
    Ok(*found)
}
pub(super) fn manual(document: &Node) -> Result<(), String> {
    let on = document
        .get("on")
        .ok_or("manual authority trigger absent")?;
    if on.entries().len() != 1 || on.get("workflow_dispatch").is_none() {
        return Err("authority remains manual only".into());
    }
    let jobs = document.get("jobs").ok_or("manual authority jobs absent")?;
    let producer = jobs
        .get("authority_automation")
        .ok_or("same-commit authority producer absent")?;
    let gate = "github.repository == 'Mesh-LLM/mesh-llm' && github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main' && (github.event.inputs.mode == 'seed' || github.event.inputs.mode == 'verify-pr-write')";
    if field(producer, "if").map(compact).as_deref() != Some(gate) {
        return Err("authority producer manual-main admission changed".into());
    }
    exact(producer, "runs-on", "ubuntu-24.04")?;
    let permissions = producer
        .get("permissions")
        .ok_or("producer permissions absent")?;
    if permissions.entries().len() != 1 {
        return Err("producer least privilege changed".into());
    }
    exact(permissions, "contents", "read")?;
    let p = steps(producer)?;
    if p.len() != 4 {
        return Err("authority producer must reuse checkout/freeze/origin/upload graph".into());
    }
    exact(
        &p[0],
        "uses",
        "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
    )?;
    let with = p[0].get("with").ok_or("producer checkout inputs absent")?;
    exact(with, "ref", "${{ github.sha }}")?;
    exact(with, "persist-credentials", "false")?;
    exact(&p[1], "id", "authority_source")?;
    exact(&p[2], "id", "authority_origin")?;
    exact(&p[3], "id", "authority_upload")?;
    exact(&p[3], "uses", "./.github/actions/upload-automation")?;
    let upload = p[3].get("with").ok_or("producer upload inputs absent")?;
    exact(
        upload,
        "source_sha",
        "${{ steps.authority_source.outputs.sha }}",
    )?;
    exact(upload, "runner-profile", "hosted-bare")?;
    exact(upload, "source-profile", "protected-clean")?;
    let outputs = producer
        .get("outputs")
        .ok_or("producer identity outputs absent")?;
    for (key, value) in [
        (
            "authority_automation_source_sha",
            "${{ steps.authority_source.outputs.sha }}",
        ),
        (
            "authority_automation_artifact_id",
            "${{ steps.authority_upload.outputs.artifact_id }}",
        ),
        (
            "authority_automation_binary_sha256",
            "${{ steps.authority_upload.outputs.binary_sha256 }}",
        ),
    ] {
        exact(outputs, key, value)?;
    }
    for (job, mode) in [
        ("seed_authority_marker", "seed"),
        ("verify_pr_write", "verify-pr-write"),
    ] {
        let job = jobs.get(job).ok_or("manual marker job absent")?;
        exact(job, "needs", "authority_automation")?;
        exact(job, "runs-on", "depot-ubuntu-24.04")?;
        let expected = format!(
            "github.repository == 'Mesh-LLM/mesh-llm' && github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main' && github.event.inputs.mode == '{mode}'"
        );
        if field(job, "if").map(compact) != Some(expected) {
            return Err("manual marker mode/ref admission changed".into());
        }
        if !matches!(job.get("permissions"),Some(Node::Map(m)) if m.is_empty())
            || job.get("secrets").is_some()
        {
            return Err("manual marker permissions/secrets changed".into());
        }
        let all = steps(job)?;
        if all.iter().any(|s| {
            field(s, "uses")
                .is_some_and(|u| u.starts_with("actions/checkout@") || u.starts_with("./"))
        }) {
            return Err("marker runner cannot checkout or execute local action".into());
        }
        let (identity, _) = named(all, "Validate bounded sentinel identity")?;
        let (admit, admission) = named(all, "Admit protected automation producer identity")?;
        let (download, transport) = named(all, "Download protected authority automation")?;
        let (verify, verification) = named(
            all,
            "Verify protected authority automation without executing it",
        )?;
        let (attest, endpoint) = named(all, "Attest provider-injected cache backend")?;
        if !(identity < admit && admit < download && download < verify && verify < attest) {
            return Err("identity/delivery/verification/attestation order changed".into());
        }
        exact(
            transport,
            "uses",
            "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        )?;
        let with = transport.get("with").ok_or("download identity absent")?;
        exact(
            with,
            "artifact-ids",
            "${{ needs.authority_automation.outputs.authority_automation_artifact_id }}",
        )?;
        exact(
            with,
            "path",
            "${{ runner.temp }}/immutable-automation-restored",
        )?;
        for s in [admission, verification] {
            exact(s, "shell", "bash")?;
            for key in ["if", "continue-on-error"] {
                if s.get(key).is_some() {
                    return Err("authority verification cannot be bypassed".into());
                }
            }
            let env = s.get("env").ok_or("authority receipt identity absent")?;
            for (key, output) in [
                ("AUTOMATION_SOURCE_SHA", "authority_automation_source_sha"),
                ("AUTOMATION_ARTIFACT_ID", "authority_automation_artifact_id"),
                (
                    "AUTOMATION_BINARY_SHA256",
                    "authority_automation_binary_sha256",
                ),
            ] {
                exact(
                    env,
                    key,
                    &format!("${{{{ needs.authority_automation.outputs.{output} }}}}"),
                )?;
            }
        }
        exact(
            admission.get("env").unwrap(),
            "EXPECTED_SOURCE_SHA",
            "${{ github.sha }}",
        )?;
        super::authority_delivery_body::check(&p[1], admission, verification)?;
        super::authority_delivery_body::origin(&p[2], producer, job, admission, transport)?;
        exact(endpoint, "shell", "bash")?;
        if endpoint.get("if").is_some()
            || endpoint.get("env").is_some()
            || endpoint.get("continue-on-error").is_some()
        {
            return Err("typed attestation cannot be overridden or bypassed".into());
        }
        let commands = field(endpoint, "run")
            .ok_or("endpoint commands absent")?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>();
        if commands
            != [
                "set -euo pipefail",
                "\"$MESH_LLM_AUTOMATION_BIN\" ci-ops authority-audit endpoint ACTIONS_CACHE_URL",
                "\"$MESH_LLM_AUTOMATION_BIN\" ci-ops authority-audit endpoint ACTIONS_RESULTS_URL",
            ]
        {
            return Err("actual manual typed endpoint calls changed".into());
        }
        if all.iter().enumerate().any(|(i, s)| {
            field(s, "uses").is_some_and(|u| u.starts_with("actions/cache/")) && i <= attest
        }) {
            return Err("cache phase must follow typed attestation".into());
        }
    }
    Ok(())
}
pub(super) fn reusable(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    for (name, document) in workflows {
        if let Some(inputs) = document
            .get("on")
            .and_then(|n| n.get("workflow_call"))
            .and_then(|n| n.get("inputs"))
            && (inputs.get("runs_on").is_some() || inputs.get("runs-on").is_some())
        {
            return Err(format!("{name}: raw runner input forbidden"));
        }
        let Some(jobs) = document.get("jobs") else {
            continue;
        };
        for (_, job) in jobs.entries() {
            if field(job, "runs-on")
                .is_some_and(|s| s.contains("inputs.runs_on") || s.contains("inputs.runs-on"))
            {
                return Err("reusable raw runner expression forbidden".into());
            }
            if (name.starts_with("ci-")
                || name.starts_with("main_")
                || name.starts_with("pr_")
                || name == "docker-precheck.yml"
                || name == "static-abi-artifact.yml")
                && let Some(Node::Seq(steps)) = job.get("steps")
            {
                for step in steps.iter().filter(|s| {
                    field(s, "uses").is_some_and(|u| u.starts_with("actions/checkout@"))
                }) {
                    exact(
                        step.get("with")
                            .ok_or("checkout credential policy absent")?,
                        "persist-credentials",
                        "false",
                    )?;
                }
            }
        }
    }
    for (name, jobs) in [
        (
            "ci-linux-sdk-slice.yml",
            &[("rust_smoke", "rust"), ("kotlin_smoke", "kotlin")][..],
        ),
        ("ci-macos-sdk-slice.yml", &[("swift_smoke", "swift")][..]),
    ] {
        let document = workflows.get(name).ok_or("SDK slice absent")?;
        for (job, kind) in jobs {
            let job = document
                .get("jobs")
                .and_then(|n| n.get(job))
                .ok_or("SDK row consumer absent")?;
            exact(
                job,
                "if",
                &format!("${{{{ contains(fromJson(inputs.sdk_matrix).*.id, '{kind}') }}}}"),
            )?;
        }
    }
    Ok(())
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    manual(
        workflows
            .get("depot-canary.yml")
            .ok_or("manual canary absent")?,
    )?;
    credential_smokes(workflows)?;
    reusable(workflows)
}
pub(super) fn documentation(source: &str) -> Result<(), String> {
    let (_, rest) = source
        .split_once("The current main allowlist is:")
        .ok_or("documented main allowlist absent")?;
    let (_, rest) = rest
        .split_once("```text\n")
        .ok_or("allowlist fence absent")?;
    let (list, _) = rest.split_once("```").ok_or("allowlist fence incomplete")?;
    let lines: std::collections::BTreeSet<_> = list.lines().map(str::trim).collect();
    for name in [
        "ci-linux-lane.yml",
        "ci-linux-runtime-slice.yml",
        "depot-canary.yml",
        "release.yml",
        "ci-quality-slice.yml",
        "ci-web-slice.yml",
        "ci-ui-artifact-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-linux-product-slice.yml",
        "ci-rust-tests-slice.yml",
        "ci-macos-host-slice.yml",
        "ci-macos-runtime-slice.yml",
        "ci-macos-product-slice.yml",
        "ci-windows-host-slice.yml",
        "ci-windows-runtime-slice.yml",
        "ci-windows-product-slice.yml",
        "ci-platform-checks-slice.yml",
        "native-sdk-artifact.yml",
        "static-abi-artifact.yml",
        "swift-sdk-artifact.yml",
    ] {
        if !lines.contains(
            format!("Mesh-LLM/mesh-llm/.github/workflows/{name}@refs/heads/main").as_str(),
        ) {
            return Err(format!("documented allowlist missing {name}"));
        }
    }
    if lines.iter().any(|s| {
        [
            "hf-download-smoke.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "sdk-smoke.yml",
        ]
        .iter()
        .any(|name| s.ends_with(&format!("/{name}@refs/heads/main")))
    }) {
        return Err("credential smoke cannot enter documented Depot allowlist".into());
    }
    let (_, future) = source
        .split_once("## Future protected PR Depot executor\n")
        .ok_or("future executor section absent")?;
    let future = future.split("\n## ").next().unwrap_or("");
    for required in [
        "`permissions: contents: read`",
        "`persist-credentials: false`",
        "receive no repository secrets or registry credentials",
    ] {
        if !future.contains(required) {
            return Err("future executor least privilege document changed".into());
        }
    }
    Ok(())
}

const CREDENTIAL_RUNNERS: &str = "[[\"hf-download-smoke.yml\",\"hf_download_smoke\",\"ubuntu-24.04\"],[\"smoke.yml\",\"smoke_tests\",\"${{ inputs.runner == 'gpu-nvidia' && fromJSON('[\\\"self-hosted\\\",\\\"Linux\\\",\\\"X64\\\",\\\"amd64\\\",\\\"gpu-nvidia\\\",\\\"mesh-llm-amd64\\\",\\\"mesh-llm\\\"]') || 'ubuntu-24.04' }}\"],[\"smoke.yml\",\"smoke_tests_macos\",\"macos-15\"],[\"scripted-binary-smoke.yml\",\"scripted_binary_smoke\",\"ubuntu-24.04\"],[\"sdk-smoke.yml\",\"sdk_smoke\",\"${{ inputs.sdk_kind == 'swift' && 'macos-15' || inputs.sdk_kind == 'kotlin' && inputs.kotlin_artifact_target == 'aarch64-unknown-linux-gnu' && 'ubuntu-24.04-arm' || 'ubuntu-24.04' }}\"]]";
fn credential_smokes(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let rows: Vec<(String, String, String)> = serde_json::from_str(CREDENTIAL_RUNNERS)
        .map_err(|_| "credential runner declaration malformed")?;
    for (name, job, expected) in rows {
        let job = workflows
            .get(&name)
            .and_then(|d| d.get("jobs"))
            .and_then(|d| d.get(&job))
            .ok_or("credential smoke runner absent")?;
        exact(job, "runs-on", &expected)?;
        for step in steps(job)?
            .iter()
            .filter(|s| field(s, "uses").is_some_and(|u| u.starts_with("actions/checkout@")))
        {
            exact(
                step.get("with").ok_or("smoke checkout policy absent")?,
                "persist-credentials",
                "false",
            )?;
        }
    }
    Ok(())
}
