//! Native reporter qualification remains an executed CPU lane with immutable model admission.
use super::{Node, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;
const CADENCE: &str = "${{ (inputs.original_event_name == 'pull_request' || inputs.original_event_name == 'pull_request_target') && 'pull-request' || inputs.original_event_name == 'push' && 'main' || 'manual' }}";
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let workflow = workflows
        .get("ci-linux-runtime-slice.yml")
        .ok_or("runtime slice missing")?;
    let steps = h::steps(h::job(workflow, "linux_runtime")?)?;
    let (prepare, product) = h::step(steps, "name", "Prepare immutable Linux native runtime")?;
    h::binding(product, "id", "native_runtime")?;
    h::condition(
        product,
        "${{ steps.runtime_cache.outputs.cache-hit != 'true' }}",
    )?;
    let cached = cached_runtime(steps)?;
    let (restore, model) = h::step(steps, "name", "Restore runtime-event gate model")?;
    h::binding(model, "id", "gate_model")?;
    h::binding(model, "uses", "./.github/actions/restore-test-model")?;
    let inputs = h::member(model, "with")?;
    h::binding(
        inputs,
        "model_manifest",
        "ci/model-artifacts/manifests/skippy-ci-smoke.json",
    )?;
    h::binding(inputs, "model_artifact_id", "family-qwen3-dense")?;
    h::binding(inputs, "model_cadence", CADENCE)?;
    let (execute, gate) = h::step(steps, "name", "Run native runtime-event gate")?;
    let (upload, evidence) = h::step(steps, "name", "Upload native runtime-event gate evidence")?;
    for step in [model, gate] {
        h::condition(step, "${{ matrix.runtime.backend == 'cpu' }}")?;
    }
    h::condition(
        evidence,
        "${{ !cancelled() && matrix.runtime.backend == 'cpu' }}",
    )?;
    h::before(prepare, execute)?;
    h::before(cached, execute)?;
    let env = h::member(gate, "env")?;
    h::binding(
        env,
        "RUNTIME_DIR",
        "${{ steps.cached_runtime.outputs.runtime_dir || steps.native_runtime.outputs.runtime_dir }}",
    )?;
    h::binding(
        env,
        "MODEL_PATH",
        "${{ steps.gate_model.outputs.model_path }}",
    )?;
    h::before(restore, execute)?;
    h::before(execute, upload)?;
    gate_arguments(gate)?;
    if !super::field(evidence, "uses")
        .is_some_and(|value| value.starts_with("actions/upload-artifact@"))
    {
        return Err("runtime-event evidence must be uploaded".into());
    }
    let inputs = h::member(evidence, "with")?;
    h::binding(inputs, "path", "runtime-events-native-evidence.txt")?;
    h::binding(inputs, "if-no-files-found", "error")
}
fn cached_runtime(steps: &[Node]) -> DynResult<usize> {
    let (cached, verification) = h::step(steps, "name", "Verify restored Linux CPU runtime")?;
    h::binding(verification, "id", "cached_runtime")?;
    h::binding(
        h::member(verification, "env")?,
        "EXPECTED_BACKEND",
        "${{ matrix.runtime.backend }}",
    )?;
    h::binding(
        h::member(verification, "env")?,
        "EXPECTED_TARGET",
        "${{ matrix.runtime.target }}",
    )?;
    h::condition(
        verification,
        "${{ steps.runtime_cache.outputs.cache-hit == 'true' }}",
    )?;
    h::command(
        verification,
        &["scripts/verify-native-runtime-package.sh"],
        &[
            ("--expected-backend", "\"$EXPECTED_BACKEND\""),
            ("--expected-target", "\"$EXPECTED_TARGET\""),
            ("\"$EXPECTED_TARGET\"", "\"$runtime_dir\""),
            ("\"$runtime_dir\"", "\"$runtime_dir.tar.gz\""),
        ],
    )?;
    h::command(
        verification,
        &["if", "!", "tar"],
        &[
            ("-xOzf", "\"$runtime_dir.tar.gz\""),
            ("cmp", "-s"),
            ("-s", "\"$runtime_dir/manifest.json\""),
        ],
    )?;
    h::command(
        verification,
        &[
            "echo",
            "\"runtime_dir=$PWD/$runtime_dir\"",
            ">>",
            "\"$GITHUB_OUTPUT\"",
        ],
        &[],
    )?;
    Ok(cached)
}
fn gate_arguments(step: &Node) -> DynResult<()> {
    let run = super::field(step, "run").ok_or("native reporter gate command missing")?;
    let physical: Vec<_> = run
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    for (index, line) in physical.iter().enumerate() {
        if line.ends_with('\\') != (index + 1 < physical.len()) {
            return Err("native reporter arguments require direct command continuations".into());
        }
    }
    let lines: Vec<_> = physical
        .iter()
        .map(|line| line.trim().trim_end_matches('\\').trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let Some((command, arguments)) = lines.split_first() else {
        return Err("native reporter gate command missing".into());
    };
    if *command != "scripts/ci-runtime-events-native-gate.sh" {
        return Err("native reporter gate must execute directly".into());
    }
    let mut values = BTreeMap::new();
    for argument in arguments {
        let (key, value) = argument
            .split_once(' ')
            .ok_or("native reporter argument missing")?;
        if values.insert(key, value.trim()).is_some() {
            return Err("duplicate native reporter argument".into());
        }
    }
    let expected = BTreeMap::from([
        ("--bundle-dir", "\"$(dirname \"$RUNTIME_DIR\")\""),
        ("--model", "\"$MODEL_PATH\""),
        ("--evidence", "runtime-events-native-evidence.txt"),
    ]);
    if values != expected {
        return Err(
            "native reporter gate must bind bundle parent, restored model and uploaded evidence"
                .into(),
        );
    }
    Ok(())
}
#[cfg(test)]
#[path = "runtime_events_tests.rs"]
mod tests;
