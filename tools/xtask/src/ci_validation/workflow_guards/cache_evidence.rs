//! Parsed wiring for cache capture, explicit caller policy and job-local evidence.
use super::{Node, cache_predicate::requires};
use std::collections::{BTreeMap, BTreeSet};
const CONFIGURE: &str = "./.github/actions/configure-sccache-gha";
const CAPTURE: &str = "./.github/actions/capture-sccache-stats";
pub(super) fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
pub(super) fn steps(node: &Node) -> &[Node] {
    match node.get("steps") {
        Some(Node::Seq(steps)) => steps,
        _ => &[],
    }
}
fn equal(node: &Node, key: &str, expected: &str) -> Result<(), String> {
    if text(node, key) != expected {
        return Err(format!("cache evidence {key} must bind {expected}"));
    }
    Ok(())
}
fn required_step(steps: &[Node], matches: impl Fn(&Node) -> bool) -> Result<usize, String> {
    let mut selected = steps.iter().enumerate().filter(|(_, step)| matches(step));
    let index = selected
        .next()
        .ok_or("required cache action step missing")?
        .0;
    if selected.next().is_some() {
        return Err("required cache action step duplicated".into());
    }
    Ok(index)
}
fn logging_only(step: &Node) -> bool {
    // This admits literal echo diagnostics, not arbitrary additional shell or action authority.
    if text(step, "shell") != "bash"
        || step
            .entries()
            .iter()
            .any(|(key, _)| !matches!(key.as_str(), "name" | "shell" | "run"))
    {
        return false;
    }
    let run = text(step, "run").trim();
    let Some(message) = run.strip_prefix("echo ") else {
        return false;
    };
    let message = message
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
        .or_else(|| {
            message
                .strip_prefix('\'')
                .and_then(|text| text.strip_suffix('\''))
        })
        .unwrap_or(message);
    !message.is_empty()
        && message
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || " _-.,:!?/".contains(c))
}
fn ancillary(steps: &[Node], required: &[usize]) -> Result<(), String> {
    for (index, step) in steps.iter().enumerate() {
        if !required.contains(&index) && !logging_only(step) {
            return Err("additional cache action step must be literal logging only".into());
        }
    }
    Ok(())
}
pub(super) fn capture_action(action: &Node) -> Result<(), String> {
    equal(
        action
            .get("inputs")
            .and_then(|v| v.get("artifact_name"))
            .ok_or("artifact input missing")?,
        "required",
        "true",
    )?;
    let steps = steps(action.get("runs").ok_or("capture runs missing")?);
    let capture_index = required_step(steps, |s| text(s, "id") == "capture")?;
    let upload_index = required_step(steps, |s| {
        text(s, "uses").starts_with("actions/upload-artifact@")
    })?;
    let threshold_index = required_step(steps, |s| {
        text(s, "if").contains("steps.capture.outputs.cache_passed")
    })?;
    if !(capture_index < upload_index && upload_index < threshold_index) {
        return Err("capture must precede upload and threshold enforcement".into());
    }
    ancillary(steps, &[capture_index, upload_index, threshold_index])?;
    let (capture, upload, threshold) = (
        &steps[capture_index],
        &steps[upload_index],
        &steps[threshold_index],
    );
    equal(capture, "id", "capture")?;
    equal(capture, "shell", "bash")?;
    let env = capture.get("env").ok_or("capture environment missing")?;
    for (key, value) in [
        ("SCCACHE_STATS_ARTIFACT_NAME", "${{ inputs.artifact_name }}"),
        (
            "SCCACHE_STATS_OUTPUT_DIR",
            "${{ runner.temp }}/mesh-llm-sccache-evidence",
        ),
        (
            "SCCACHE_CACHE_EXPECTATION",
            "${{ inputs.cache_expectation }}",
        ),
        ("SCCACHE_MINIMUM_HIT_RATE", "${{ inputs.minimum_hit_rate }}"),
    ] {
        equal(env, key, value)?;
    }
    equal(
        upload,
        "uses",
        "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f",
    )?;
    let inputs = upload.get("with").ok_or("evidence upload inputs missing")?;
    for (key, value) in [
        ("name", "${{ inputs.artifact_name }}"),
        ("path", "${{ steps.capture.outputs.stats_file }}"),
        ("retention-days", "14"),
        ("if-no-files-found", "error"),
    ] {
        equal(inputs, key, value)?;
    }
    equal(
        threshold,
        "if",
        "${{ steps.capture.outputs.cache_passed != 'true' }}",
    )?;
    equal(threshold, "shell", "bash")?;
    let outputs = action.get("outputs").ok_or("capture outputs missing")?;
    for key in [
        "stats_file",
        "compile_requests",
        "requests_executed",
        "cache_hits",
        "cache_misses",
        "cache_writes",
        "cache_read_errors",
        "cache_write_errors",
        "hit_rate",
        "cache_classification",
        "cache_passed",
    ] {
        equal(
            outputs.get(key).ok_or("required capture output missing")?,
            "value",
            &format!("${{{{ steps.capture.outputs.{key} }}}}"),
        )?;
    }
    Ok(())
}
pub(super) fn configure_action(action: &Node) -> Result<(), String> {
    let inputs = action.get("inputs").ok_or("configure inputs missing")?;
    for key in ["allow_depot_remote_cache", "allow_native_github_cache"] {
        equal(
            inputs.get(key).ok_or("configure authority input missing")?,
            "default",
            "false",
        )?;
    }
    let steps = steps(action.get("runs").ok_or("configure runs missing")?);
    let configure_index = required_step(steps, |s| {
        s.get("env")
            .and_then(|e| e.get("INPUT_ALLOW_NATIVE_GITHUB_CACHE"))
            .is_some()
    })?;
    ancillary(steps, &[configure_index])?;
    let configure = &steps[configure_index];
    let env = configure
        .get("env")
        .ok_or("configure input projection missing")?;
    for (key, value) in [
        (
            "INPUT_ALLOW_DEPOT_REMOTE_CACHE",
            "${{ inputs.allow_depot_remote_cache }}",
        ),
        (
            "INPUT_ALLOW_NATIVE_GITHUB_CACHE",
            "${{ inputs.allow_native_github_cache }}",
        ),
        (
            "DISPATCH_ORIGINAL_EVENT_NAME",
            "${{ github.event.inputs.original_event_name || '' }}",
        ),
    ] {
        equal(env, key, value)?;
    }
    Ok(())
}
// The CI slices and original evidence callers own direct-install adjacency.
// Release initialization is checked by its release owner before compilation;
// every explicit configure caller here still declares cache authority.
fn owns_direct_installation(name: &str) -> bool {
    name.starts_with("ci-")
        || matches!(
            name,
            "ci.yml"
                | "static-abi-artifact.yml"
                | "swift-sdk-artifact.yml"
                | "hf-download-smoke.yml"
                | "native-sdk-artifact.yml"
        )
}
fn callers(name: &str, steps: &[Node]) -> Result<(), String> {
    for (index, step) in steps.iter().enumerate() {
        if owns_direct_installation(name)
            && text(step, "uses").starts_with("mozilla-actions/sccache-action@")
            && steps.get(index + 1).map(|n| text(n, "uses")) != Some(CONFIGURE)
        {
            return Err(
                "direct sccache installation must immediately apply explicit cache policy".into(),
            );
        }
        if text(step, "uses") == CONFIGURE {
            let inputs = step
                .get("with")
                .ok_or("explicit configure inputs missing")?;
            if text(inputs, "allow_native_github_cache").is_empty() {
                return Err("configure caller must declare native cache authority".into());
            }
        }
    }
    Ok(())
}
fn explicit_native_policy(name: &str, job: &str, steps: &[Node]) -> Result<(), String> {
    // These are intentional caller exceptions to centrally projected CI policy.
    // Other CI/provider projections remain owned by cache_consumers/boundaries.
    let expected = match (name, job) {
        ("cache-warm-sccache.yml", "warm") | ("depot-canary.yml", "runtime_seed") => Some("false"),
        ("hf-download-smoke.yml", "hf_download_smoke") => Some("true"),
        ("node-sdk-addon-artifact.yml", "linux_addon" | "macos_addon" | "windows_addon") => {
            Some("true")
        }
        ("release.yml", "metadata" | "publish" | "publish_crates_preflight" | "publish_crates") => {
            Some("false")
        }
        (
            "release.yml",
            "build"
            | "build_native_runtime_linux_aarch64_cuda"
            | "build_native_runtime_linux_x86_64_cuda",
        ) => Some("true"),
        (
            "release.yml",
            "build_native_runtime_linux_x86_64_rocm" | "build_native_runtime_linux_x86_64_vulkan",
        ) => Some(
            "${{ startsWith(needs.metadata.outputs.runner_16, 'depot-') && 'false' || 'true' }}",
        ),
        _ => None,
    };
    if let Some(expected) = expected {
        let callers: Vec<_> = steps
            .iter()
            .filter(|s| text(s, "uses") == CONFIGURE)
            .collect();
        if callers.is_empty() {
            return Err(format!(
                "{name}/{job}: explicit compiler cache policy missing"
            ));
        }
        for caller in callers {
            equal(
                caller.get("with").ok_or("configure inputs missing")?,
                "allow_native_github_cache",
                expected,
            )?;
        }
    }
    Ok(())
}
fn instrumented(name: &str, workflow: &Node) -> Result<(), String> {
    let mut names = BTreeSet::new();
    for (_, job) in workflow
        .get("jobs")
        .ok_or("instrumented jobs missing")?
        .entries()
    {
        for step in steps(job) {
            if text(step, "uses") != CAPTURE {
                continue;
            }
            let inputs = step.get("with").ok_or("capture inputs missing")?;
            let artifact = text(inputs, "artifact_name");
            if !artifact.starts_with("sccache-")
                || !artifact.contains("${{ github.run_attempt }}")
                || !names.insert(artifact)
            {
                return Err(format!(
                    "{name}: capture artifacts must be unique and distinguish run attempts"
                ));
            }
            if !requires(text(step, "if"), "!cancelled()") {
                return Err(format!("{name}: cache evidence must exclude cancellation"));
            }
        }
    }
    if names.is_empty() {
        return Err(format!("{name}: cache capture missing"));
    }
    Ok(())
}
fn safetensors(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let smoke = workflows
        .get("ci-rust-tests-slice.yml")
        .and_then(|w| w.get("jobs"))
        .and_then(|j| j.get("safetensors_runtime_smoke"))
        .ok_or("SafeTensors cache job missing")?;
    let steps = steps(smoke);
    let build = steps
        .iter()
        .position(|s| text(s, "id") == "safetensors_smoke_test")
        .ok_or("SafeTensors build missing")?;
    if steps.get(build + 1).map(|s| text(s, "uses")) != Some(CAPTURE) {
        return Err("SafeTensors capture must immediately follow compilation".into());
    }
    let exercise = steps
        .iter()
        .position(|s| {
            s.get("env")
                .and_then(|e| e.get("SAFETENSORS_SMOKE_TEST_BINARY"))
                .is_some()
        })
        .ok_or("SafeTensors runtime loop missing")?;
    if exercise <= build + 1 {
        return Err("SafeTensors exercise must follow cache capture".into());
    }
    Ok(())
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    for (name, workflow) in workflows {
        for (job_name, job) in workflow.get("jobs").ok_or("jobs missing")?.entries() {
            explicit_native_policy(name, job_name, steps(job))?;
            callers(name, steps(job)).map_err(|e| format!("{name}: {e}"))?;
        }
    }
    for name in [
        "ci-quality-slice.yml",
        "ci-rust-tests-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-linux-runtime-slice.yml",
        "static-abi-artifact.yml",
        "swift-sdk-artifact.yml",
    ] {
        let workflow = workflows.get(name).ok_or("instrumented workflow missing")?;
        instrumented(name, workflow)?;
    }
    safetensors(workflows)
}
