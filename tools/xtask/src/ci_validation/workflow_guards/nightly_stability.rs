//! Preserve native nightly evidence and independent KV execution.
use super::{Node, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;
const ADMITTED: &str = "${{ steps.preflight.outputs.run == 'true' }}";
const FINISH: &str = "${{ !cancelled() && steps.preflight.outputs.run == 'true' }}";

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    wrapper(
        workflows
            .get("nightly-stability.yml")
            .ok_or("nightly wrapper missing")?,
    )?;
    ownership(
        workflows
            .get("nightly-kv-coverage.yml")
            .ok_or("KV ownership schedule missing")?,
    )?;
    let workflow = workflows
        .get("nightly-stability-run.yml")
        .ok_or("nightly stability workflow missing")?;
    let job = h::job(workflow, "stability")?;
    h::binding(job, "runs-on", "ubuntu-24.04")?;
    let steps = h::steps(job)?;
    let (preflight, _) = h::step(steps, "id", "preflight")?;
    let (prepare, automation) = h::step(steps, "id", "automation")?;
    h::binding(automation, "uses", "./.github/actions/prepare-automation")?;
    h::condition(automation, ADMITTED)?;
    h::before(preflight, prepare)?;
    let (execute, general) = h::step(steps, "id", "stability_harness")?;
    h::before(prepare, execute)?;
    h::condition(general, ADMITTED)?;
    h::binding(general, "continue-on-error", "true")?;
    h::command(
        general,
        &[
            "\"${MESH_LLM_AUTOMATION_BIN:?automation",
            "preparation",
            "required}\"",
            "automation",
            "stability",
            "nightly",
            "\"${args[@]}\"",
        ],
        &[],
    )?;
    let (kv_execute, kv) = h::step(steps, "id", "kv_tool_loop")?;
    h::condition(
        kv,
        "${{ !cancelled() && steps.preflight.outputs.run == 'true' && inputs.kv_tool_loop }}",
    )?;
    h::binding(kv, "continue-on-error", "true")?;
    h::before(execute, kv_execute)?;
    h::command(
        kv,
        &[
            "\"${MESH_LLM_AUTOMATION_BIN:?automation",
            "preparation",
            "required}\"",
            "automation",
            "stability",
            "kv-tool-loop",
        ],
        &[
            ("--base-url", "\"$MESH_STABILITY_BASE_URL\""),
            ("--models", "\"$MESH_KV_MODELS\""),
            ("--attempts", "\"$MESH_KV_ATTEMPTS\""),
            ("--pressure-turns", "\"$MESH_KV_PRESSURE_TURNS\""),
            ("--overlap-requests", "\"$MESH_KV_OVERLAP_REQUESTS\""),
            ("--timeout", "\"$MESH_STABILITY_TIMEOUT\""),
            ("--min-cached-tokens", "\"$MESH_KV_MIN_CACHED_TOKENS\""),
            (
                "--suffix-prefill-limit",
                "\"$MESH_KV_SUFFIX_PREFILL_LIMIT\"",
            ),
            (
                "--output-dir",
                "\"$MESH_STABILITY_OUTPUT_DIR/kv-tool-loop\"",
            ),
        ],
    )?;
    let (summary_index, summary) = h::step(steps, "name", "Publish run summary")?;
    let (upload_index, upload) = h::step(steps, "name", "Upload stability evidence")?;
    let (fail_index, fail) = h::step(steps, "name", "Fail on stability regression")?;
    for step in [summary, upload] {
        h::condition(step, FINISH)?;
    }
    for (first, second) in [
        (kv_execute, summary_index),
        (summary_index, upload_index),
        (upload_index, fail_index),
    ] {
        h::before(first, second)?;
    }
    h::condition(
        fail,
        "${{ !cancelled() && steps.preflight.outputs.run == 'true' && (steps.stability_harness.outcome == 'failure' || steps.kv_tool_loop.outcome == 'failure') }}",
    )?;
    h::command(fail, &["exit", "1"], &[])?;
    let upload_inputs = h::member(upload, "with")?;
    h::binding(upload_inputs, "path", "${{ inputs.output_dir }}/")?;
    h::binding(upload_inputs, "retention-days", "30")?;
    if !super::field(upload, "uses")
        .is_some_and(|value| value.starts_with("actions/upload-artifact@"))
    {
        return Err("nightly evidence upload action missing".into());
    }
    let summary_body = super::field(summary, "run").ok_or("nightly summary body missing")?;
    for required in [
        "$MESH_STABILITY_OUTPUT_DIR/summary.md",
        "$MESH_STABILITY_OUTPUT_DIR/kv-tool-loop/summary.md",
        "$GITHUB_STEP_SUMMARY",
    ] {
        if !summary_body.contains(required) {
            return Err(format!("nightly summary must consume {required}").into());
        }
    }
    Ok(())
}

fn wrapper(document: &Node) -> DynResult<()> {
    let job = h::job(document, "stability")?;
    h::binding(job, "uses", "./.github/workflows/nightly-stability-run.yml")?;
    let inputs = h::member(
        h::member(h::member(document, "on")?, "workflow_dispatch")?,
        "inputs",
    )?;
    for name in [
        "base_url",
        "models",
        "attempts",
        "agent_smokes",
        "skip_streaming",
        "timeout",
        "output_dir",
    ] {
        h::member(inputs, name)?;
    }
    if inputs.get("runs_on").is_some() || h::member(job, "with")?.get("runs_on").is_some() {
        return Err("nightly wrapper must not select privileged runner labels".into());
    }
    Ok(())
}

fn ownership(document: &Node) -> DynResult<()> {
    let events = h::member(document, "on")?;
    h::member(events, "workflow_dispatch")?;
    if events.get("pull_request").is_some() || events.get("push").is_some() {
        return Err("KV ownership certification must remain schedule/manual only".into());
    }
    let Node::Seq(schedule) = h::member(events, "schedule")? else {
        return Err("KV ownership schedule missing".into());
    };
    if schedule.len() != 1 {
        return Err("KV ownership schedule is ambiguous".into());
    }
    h::binding(&schedule[0], "cron", "23 6 * * *")?;
    let job = h::job(document, "ownership-state-machines")?;
    h::binding(job, "runs-on", "ubuntu-24.04")?;
    let steps = h::steps(job)?;
    let checkout = h::checkout(steps, "main", None)?;
    let (verify, image) = h::step(steps, "name", "Verify prebuilt test environment")?;
    h::command(image, &["verify-runner-image", "public", "cpu"], &[])?;
    let (run, tests) = h::step(steps, "id", "state_machines")?;
    h::command(
        tests,
        &["cargo", "test", "--locked", "-p", "skippy-cache"],
        &[],
    )?;
    h::before(checkout, verify)?;
    h::before(verify, run)?;
    let environment = h::member(job, "env")?;
    for name in [
        "SKIPPY_CACHE_STATE_MACHINE_SEEDS",
        "SKIPPY_CACHE_STATE_MACHINE_STEPS",
    ] {
        h::member(environment, name)?;
    }
    if super::field(environment, "RUSTC_WRAPPER") == Some("") || mentions(document, "secrets.") {
        return Err(
            "KV ownership must preserve compiler wrapper and credential-free execution".into(),
        );
    }
    Ok(())
}

fn mentions(node: &Node, fragment: &str) -> bool {
    match node {
        Node::Scalar(text) => text.contains(fragment),
        Node::Seq(values) => values.iter().any(|value| mentions(value, fragment)),
        Node::Map(values) => values.iter().any(|(_, value)| mentions(value, fragment)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::workflow_yaml;
    use super::*;
    const SOURCE: &str = include_str!("../../../../../.github/workflows/nightly-stability-run.yml");
    fn documents(source: &str) -> BTreeMap<String, Node> {
        BTreeMap::from([
            (
                "nightly-stability-run.yml".into(),
                workflow_yaml::parse(source).unwrap(),
            ),
            (
                "nightly-stability.yml".into(),
                workflow_yaml::parse(include_str!(
                    "../../../../../.github/workflows/nightly-stability.yml"
                ))
                .unwrap(),
            ),
            (
                "nightly-kv-coverage.yml".into(),
                workflow_yaml::parse(include_str!(
                    "../../../../../.github/workflows/nightly-kv-coverage.yml"
                ))
                .unwrap(),
            ),
        ])
    }
    #[test]
    fn nightly_native_owner_preserves_independent_failures_and_uploaded_evidence() {
        check(&documents(SOURCE)).unwrap();
    }
    #[test]
    fn nightly_kv_ownership_rejects_untrusted_or_incomplete_certification() {
        const KV: &str = include_str!("../../../../../.github/workflows/nightly-kv-coverage.yml");
        for (from, to) in [
            ("ref: main", "ref: foreign"),
            ("runs-on: ubuntu-24.04", "runs-on: self-hosted"),
            ("verify-runner-image public cpu", "echo unverified"),
            ("SKIPPY_CACHE_STATE_MACHINE_SEEDS:", "UNUSED_SEEDS:"),
            ("RUST_BACKTRACE: \"1\"", "RUSTC_WRAPPER: \"\""),
            (
                "RUST_BACKTRACE: \"1\"",
                "RUST_BACKTRACE: \"${{ secrets.FIXTURE }}\"",
            ),
        ] {
            assert!(KV.contains(from));
            let mut workflows = documents(SOURCE);
            workflows.insert(
                "nightly-kv-coverage.yml".into(),
                workflow_yaml::parse(&KV.replace(from, to)).unwrap(),
            );
            assert!(
                check(&workflows).is_err(),
                "accepted broken ownership policy: {from}"
            );
        }
    }

    #[test]
    fn nightly_native_owner_rejects_lost_failure_paths_and_evidence() {
        for (from, to) in [
            ("continue-on-error: true", "continue-on-error: false"),
            (
                "steps.kv_tool_loop.outcome == 'failure'",
                "steps.kv_tool_loop.outcome == 'success'",
            ),
            (
                "$MESH_STABILITY_OUTPUT_DIR/kv-tool-loop/summary.md",
                "$MESH_STABILITY_OUTPUT_DIR/other/summary.md",
            ),
            ("${{ inputs.output_dir }}/", "foreign-evidence/"),
            (
                "uses: ./.github/actions/prepare-automation",
                "uses: ./.github/actions/prepare-ui",
            ),
            (
                "automation stability nightly",
                "automation stability tool-call",
            ),
        ] {
            assert!(SOURCE.contains(from));
            assert!(
                check(&documents(&SOURCE.replace(from, to))).is_err(),
                "accepted broken nightly handoff: {from}"
            );
        }
    }

    #[test]
    fn nightly_kv_native_owner_rejects_missing_command_or_changed_probe_inputs() {
        for (from, to) in [
            (
                "automation stability kv-tool-loop",
                "automation stability nightly",
            ),
            ("--pressure-turns", "--other-turns"),
            ("--overlap-requests", "--other-requests"),
            ("--min-cached-tokens", "--other-tokens"),
            ("--suffix-prefill-limit", "--other-limit"),
            ("\"$MESH_KV_MODELS\"", "\"$MESH_STABILITY_MODELS\""),
        ] {
            assert!(SOURCE.contains(from));
            assert!(
                check(&documents(&SOURCE.replace(from, to))).is_err(),
                "accepted broken KV caller: {from}"
            );
        }
    }
}
