//! Structural bounds of the deliberate no-checkout cache authority exemption.
use super::Node;

fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn normalized(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn sentinel(job: &Node) -> Result<(), String> {
    protected_source(job)?;
    if !job
        .get("needs")
        .is_some_and(|needs| needs.list().contains(&"runner_policy"))
    {
        return Err("sentinel must depend on protected runner policy producer".into());
    }
    let expected = "needs.runner_policy.outputs.authority_sentinel_depot_enabled == 'true' && inputs.original_event_name == 'pull_request' && github.event_name == 'pull_request' && github.ref == vars.DEPOT_PR_SENTINEL_REF";
    if normalized(text(job, "if").ok_or("sentinel condition missing")?) != expected {
        return Err("sentinel must retain direct-PR exact-ref admission".into());
    }
    if !matches!(job.get("permissions"),Some(Node::Map(entries)) if entries.is_empty()) {
        return Err("sentinel must have permissions {}".into());
    }
    if text(job, "runs-on") != Some("${{ needs.runner_policy.outputs.authority_sentinel_runner }}")
    {
        return Err("sentinel must use protected dedicated runner output".into());
    }
    let Node::Seq(steps) = job.get("steps").ok_or("sentinel steps missing")? else {
        return Err("sentinel steps malformed".into());
    };
    let attest = steps
        .iter()
        .position(|s| text(s, "name") == Some("Attest provider-injected cache backend"))
        .ok_or("sentinel attestation missing")?;
    let validate = steps
        .iter()
        .position(|s| text(s, "id") == Some("validate"))
        .ok_or("sentinel identity validation missing")?;
    protected_attestation(steps, attest)?;
    super::authority_delivery_body::quality_consumer(job, steps, attest)?;
    if validate >= attest {
        return Err("sentinel identity must precede attestation".into());
    }
    let mut caches = 0;
    for (index, step) in steps.iter().enumerate() {
        if let Some(action) = text(step, "uses") {
            if action.starts_with("actions/checkout@")
                || action.starts_with("./")
                || action.starts_with("audit-depot-pr-isolation@")
            {
                return Err("sentinel no-checkout boundary excludes checkout/local audits".into());
            }
            if action.starts_with("actions/cache/") {
                if !matches!(
                    action,
                    "actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25"
                        | "actions/cache/save@caa296126883cff596d87d8935842f9db880ef25"
                ) {
                    return Err("sentinel cache action must remain immutable pinned".into());
                }
                if index <= attest {
                    return Err("sentinel cache probe must follow authority attestation".into());
                }
                caches += 1;
            }
        }
        for value in [text(step, "if"), text(step, "run")].into_iter().flatten() {
            if value.contains("allow_native_github_cache")
                || value.contains("allow_depot_remote_cache")
            {
                return Err(
                    "sentinel diagnostic exception cannot be gated by ordinary cache authorization"
                        .into(),
                );
            }
        }
    }
    if caches == 0 {
        return Err("sentinel cache diagnostic missing".into());
    }
    Ok(())
}

fn protected_attestation(steps: &[Node], attest: usize) -> Result<(), String> {
    let download = steps
        .iter()
        .position(|s| text(s, "name") == Some("Download protected authority automation"))
        .ok_or("protected authority download missing")?;
    let verify = steps
        .iter()
        .position(|s| {
            text(s, "name") == Some("Verify protected authority automation without executing it")
        })
        .ok_or("protected authority verification missing")?;
    if !(download < verify && verify < attest) {
        return Err("protected download/verification must precede attestation".into());
    }
    if text(&steps[download], "uses")
        != Some("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c")
    {
        return Err("protected artifact transport must remain pinned".into());
    }
    let inputs = steps[download]
        .get("with")
        .ok_or("protected download inputs missing")?;
    if text(inputs, "artifact-ids")
        != Some("${{ needs.runner_policy.outputs.authority_automation_artifact_id }}")
    {
        return Err("protected download must use immutable controller artifact identity".into());
    }
    let environment = steps[verify]
        .get("env")
        .ok_or("protected verifier identity inputs missing")?;
    for (key, output) in [
        ("AUTOMATION_SOURCE_SHA", "authority_automation_source_sha"),
        ("AUTOMATION_ARTIFACT_ID", "authority_automation_artifact_id"),
        (
            "AUTOMATION_BINARY_SHA256",
            "authority_automation_binary_sha256",
        ),
    ] {
        let expected = format!("${{{{ needs.runner_policy.outputs.{output} }}}}");
        if text(environment, key) != Some(expected.as_str()) {
            return Err(format!(
                "protected verifier {key} must bind immutable controller output"
            ));
        }
    }
    if text(&steps[verify], "run").is_none_or(|run| run.trim().is_empty()) {
        return Err("protected verifier executable admission body missing".into());
    }
    let expected = [
        "set -euo pipefail",
        "\"$MESH_LLM_AUTOMATION_BIN\" ci-ops authority-audit endpoint ACTIONS_CACHE_URL",
        "\"$MESH_LLM_AUTOMATION_BIN\" ci-ops authority-audit endpoint ACTIONS_RESULTS_URL",
    ];
    let commands = text(&steps[attest], "run")
        .ok_or("attestation commands missing")?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();
    if commands != expected {
        return Err(
            "attestation must execute both typed endpoint checks without values or bypasses".into(),
        );
    }
    if steps[attest]
        .get("env")
        .is_some_and(|env| env.get("MESH_LLM_AUTOMATION_BIN").is_some())
    {
        return Err("attestation cannot override the verified protected executable".into());
    }
    if text(&steps[attest], "shell") != Some("bash") {
        return Err("attestation must retain fail-fast Bash execution".into());
    }
    Ok(())
}

/// Source policy only; actual upload/restore fixtures own artifact admission behavior.
pub(super) fn producer(job: &Node) -> Result<(), String> {
    let outputs = job
        .get("outputs")
        .ok_or("authority producer outputs missing")?;
    for (key, value) in [
        (
            "authority_sentinel_runner",
            "${{ steps.sentinel_policy.outputs.runner }}",
        ),
        (
            "authority_sentinel_depot_enabled",
            "${{ steps.sentinel_policy.outputs.depot_enabled }}",
        ),
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
        if text(outputs, key) != Some(value) {
            return Err(format!("authority producer {key} projection changed"));
        }
    }
    let Some(Node::Seq(steps)) = job.get("steps") else {
        return Err("authority producer steps missing".into());
    };
    let checkout = steps
        .iter()
        .position(|s| text(s, "uses").is_some_and(|v| v.starts_with("actions/checkout@")))
        .ok_or("authority protected checkout missing")?;
    let checkout_inputs = steps[checkout]
        .get("with")
        .ok_or("protected checkout inputs missing")?;
    if text(checkout_inputs, "ref") != Some("${{ github.event.repository.default_branch }}")
        || text(checkout_inputs, "persist-credentials") != Some("false")
    {
        return Err(
            "authority producer must checkout protected default branch without credentials".into(),
        );
    }
    let freeze = steps
        .iter()
        .position(|s| text(s, "id") == Some("authority_source"))
        .ok_or("protected source freeze missing")?;
    let upload = steps
        .iter()
        .position(|s| text(s, "id") == Some("authority_upload"))
        .ok_or("protected authority upload missing")?;
    super::authority_delivery_body::quality_producer(job, steps, freeze, upload)?;
    if !(checkout < freeze && freeze < upload) {
        return Err("protected checkout/freeze/upload order changed".into());
    }
    if text(&steps[freeze], "run").is_none_or(|run| run.trim().is_empty()) {
        return Err("protected source freeze admission missing".into());
    }
    if text(&steps[upload], "uses") != Some("./.github/actions/upload-automation") {
        return Err("protected authority upload owner changed".into());
    }
    let inputs = steps[upload]
        .get("with")
        .ok_or("protected upload inputs missing")?;
    for (key, value) in [
        ("source_sha", "${{ steps.authority_source.outputs.sha }}"),
        ("runner-profile", "hosted-bare"),
        ("source-profile", "protected-clean"),
    ] {
        if text(inputs, key) != Some(value) {
            return Err(format!("protected upload {key} binding changed"));
        }
    }
    Ok(())
}

// Only the protected no-checkout diagnostic has this stricter source boundary.
fn protected_source(node: &Node) -> Result<(), String> {
    match node {
        Node::Scalar(value) => {
            for forbidden in [
                "secrets.",
                "inputs.source_sha",
                "github.event.pull_request.head",
                "audit-depot-pr-isolation@",
            ] {
                if value.contains(forbidden) {
                    return Err(
                        "sentinel cannot project credentials or PR-controlled source".into(),
                    );
                }
            }
            for line in value.lines() {
                let mut words = line.split_whitespace();
                let first = words.next();
                if matches!(first, Some("cargo" | "rustc"))
                    || (first == Some("git") && words.next() == Some("checkout"))
                {
                    return Err("sentinel cannot compile or checkout source".into());
                }
            }
        }
        Node::Seq(nodes) => {
            for child in nodes {
                protected_source(child)?;
            }
        }
        Node::Map(entries) => {
            for (key, child) in entries {
                if key == "secrets" {
                    return Err("sentinel cannot map credentials".into());
                }
                protected_source(child)?;
            }
        }
    }
    Ok(())
}
