//! Restored Laya product/model inputs and planned hardware-device coverage.
use super::{Node, handoffs as h};
use crate::command::DynResult;
use std::{collections::BTreeMap, path::Path};

pub(super) fn check(root: &Path, workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let action = super::workflow_yaml::parse(&std::fs::read_to_string(
        root.join(".github/actions/run-laya-product-smoke/action.yml"),
    )?)?;
    action_handoffs(&action)?;
    platforms(workflows)
}

fn action_handoffs(action: &Node) -> DynResult<()> {
    let inputs = h::member(action, "inputs")?;
    for key in ["startup_timeout_seconds", "read_timeout_seconds"] {
        h::binding(h::member(inputs, key)?, "default", "300")?;
    }
    let steps = h::steps(h::member(action, "runs")?)?;
    let (restore_index, restore) = h::step(steps, "id", "restore")?;
    let (read_index, read) = h::step(steps, "name", "Run Laya startup and golden reads")?;
    h::before(restore_index, read_index)?;
    let gate = "${{ inputs.device != 'Vulkan0' || inputs.enable_vulkan_inference == 'true' }}";
    h::condition(restore, gate)?;
    h::condition(read, gate)?;
    h::binding(restore, "uses", "./.github/actions/restore-smoke-inputs")?;
    let restored = h::member(restore, "with")?;
    for (key, value) in [
        ("artifact_name", "${{ inputs.artifact_name }}"),
        ("artifact_path", "${{ inputs.artifact_path }}"),
        ("staged_binary_path", "${{ inputs.staged_binary_path }}"),
        ("expected_backend", "${{ inputs.expected_backend }}"),
        (
            "model_manifest",
            "ci/model-artifacts/manifests/product-smoke.json",
        ),
        ("model_artifact_id", "family-laya-multilingual"),
        ("model_cache_scope", "laya-multilingual-f16"),
    ] {
        h::binding(restored, key, value)?;
    }
    let env = h::member(read, "env")?;
    for (key, value) in [
        ("MESH_BINARY", "${{ inputs.staged_binary_path }}"),
        ("MODEL_PATH", "${{ steps.restore.outputs.model_path }}"),
        ("DEVICE", "${{ inputs.device }}"),
        (
            "STARTUP_TIMEOUT_SECONDS",
            "${{ inputs.startup_timeout_seconds }}",
        ),
        ("READ_TIMEOUT_SECONDS", "${{ inputs.read_timeout_seconds }}"),
    ] {
        h::binding(env, key, value)?;
    }
    h::command(
        read,
        &["mesh_automation", "automation", "laya", "product"],
        &[
            ("--mesh-binary", "\"$MESH_BINARY\""),
            ("--model", "\"$MODEL_PATH\""),
            ("--device", "\"$DEVICE\""),
            ("--startup-timeout", "\"$STARTUP_TIMEOUT_SECONDS\""),
            ("--read-timeout", "\"$READ_TIMEOUT_SECONDS\""),
        ],
    )?;
    let (_, skip) = h::step(steps, "name", "Report uncertified Vulkan smoke skip")?;
    h::condition(
        skip,
        "${{ inputs.device == 'Vulkan0' && inputs.enable_vulkan_inference != 'true' }}",
    )?;
    h::command(skip, &["echo"], &[])
}

fn platforms(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let linux = workflows
        .get("ci-linux-product-smoke-slice.yml")
        .ok_or("missing Linux Laya slice")?;
    for (name, backend, device, plan, smoke) in [
        ("laya_cpu", "cpu", "CPU", "linux-cpu", "core"),
        ("laya_cuda", "cuda", "CUDA0", "linux-cuda", "core-cuda"),
        ("laya_vulkan", "vulkan", "Vulkan0", "linux-vulkan", "core"),
        ("laya_rocm", "rocm", "ROCm0", "linux-rocm", "core"),
    ] {
        let job = h::job(linux, name)?;
        row(job, backend, device, plan, smoke)?;
        if name != "laya_cpu" {
            cleanup(job)?;
        }
        if name == "laya_vulkan" {
            let inputs = h::member(laya_step(job)?, "with")?;
            h::binding(
                inputs,
                "enable_vulkan_inference",
                "${{ vars.MESH_VULKAN_INFERENCE_RUNNER_ENABLED == 'true' }}",
            )?;
            h::binding(h::member(job, "env")?, "MESH_LLM_VULKAN_AVAILABLE", "1")?;
        }
    }
    for (workflow, name, backend, device, plan, smoke) in [
        (
            "ci-macos-product-smoke-slice.yml",
            "laya_metal",
            "metal",
            "MTL0",
            "macos-metal",
            "metal-model-load",
        ),
        (
            "ci-windows-product-smoke-slice.yml",
            "laya_cpu",
            "cpu",
            "CPU",
            "windows-cpu",
            "core",
        ),
    ] {
        row(
            h::job(
                workflows
                    .get(workflow)
                    .ok_or("missing platform Laya slice")?,
                name,
            )?,
            backend,
            device,
            plan,
            smoke,
        )?;
    }
    Ok(())
}
fn laya_step(job: &Node) -> DynResult<&Node> {
    Ok(h::step(
        h::steps(job)?,
        "uses",
        "./.github/actions/run-laya-product-smoke",
    )?
    .1)
}
fn row(job: &Node, backend: &str, device: &str, plan: &str, smoke: &str) -> DynResult<()> {
    let qualification = match device {
        "Vulkan0" => " && vars.MESH_VULKAN_INFERENCE_RUNNER_ENABLED == 'true'",
        "ROCm0" => " && vars.MESH_ROCM_INFERENCE_RUNNER_ENABLED == 'true'",
        _ => "",
    };
    let gate = format!(
        "${{{{ contains(fromJson(inputs.runtime_matrix).*.id, '{plan}') && contains(fromJson(inputs.smoke_matrix).*.id, '{smoke}'){qualification} }}}}"
    );
    h::condition(job, &gate)?;
    h::binding(job, "timeout-minutes", "${{ inputs.timeout_minutes }}")?;
    let runner = h::member(job, "runs-on")?.list();
    let required = match device {
        "CUDA0" | "Vulkan0" => vec!["self-hosted", "gpu-nvidia"],
        "ROCm0" => vec!["self-hosted", "gpu-amd"],
        "MTL0" => vec!["macos-15"],
        _ if plan.starts_with("windows") => vec!["windows-2022"],
        _ => vec!["ubuntu-24.04"],
    };
    if !required.iter().all(|label| runner.contains(label)) {
        return Err("Laya device lacks its declared hardware runner".into());
    }
    let steps = h::steps(job)?;
    let checkout = h::checkout(steps, "${{ inputs.source_sha }}", None)?;
    let (invoke, action) = h::step(steps, "uses", "./.github/actions/run-laya-product-smoke")?;
    h::before(checkout, invoke)?;
    let inputs = h::member(action, "with")?;
    h::binding(inputs, "expected_backend", backend)?;
    h::binding(inputs, "device", device)?;
    let artifact = if plan == "macos-metal" {
        "ci-product-macos-${{ inputs.architecture }}-metal".to_owned()
    } else {
        format!(
            "ci-product-{}-amd64-{backend}",
            if plan.starts_with("windows") {
                "windows"
            } else {
                "linux"
            }
        )
    };
    h::binding(inputs, "artifact_name", &artifact)
}
fn cleanup(job: &Node) -> DynResult<()> {
    let steps = h::steps(job)?;
    let (invoke, action) = h::step(steps, "uses", "./.github/actions/run-laya-product-smoke")?;
    let (index, cleanup) = h::step(steps, "name", "Clean self-hosted Laya smoke outputs")?;
    h::before(invoke, index)?;
    h::condition(cleanup, "${{ success() || failure() || cancelled() }}")?;
    h::binding(cleanup, "timeout-minutes", "5")?;
    let env = h::member(cleanup, "env")?;
    let inputs = h::member(action, "with")?;
    for (environment, input) in [
        ("CLEANUP_ARTIFACT_PATH", "artifact_path"),
        ("CLEANUP_BINARY_PATH", "staged_binary_path"),
    ] {
        h::binding(
            env,
            environment,
            super::field(inputs, input).ok_or("missing Laya cleanup output binding")?,
        )?;
    }
    h::command(
        cleanup,
        &["\"$MESH_LLM_AUTOMATION_BIN\"", "ci-ops", "runner-cleanup"],
        &[("--job", "smoke"), ("--evidence-uploaded", "false")],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn action() -> Node {
        super::super::workflow_yaml::parse(include_str!(
            "../../../../../.github/actions/run-laya-product-smoke/action.yml"
        ))
        .unwrap()
    }
    fn mutable_step<'a>(action: &'a mut Node, name: &str) -> &'a mut Node {
        let Node::Seq(steps) = h::mutable(h::mutable(action, "runs"), "steps") else {
            unreachable!()
        };
        steps
            .iter_mut()
            .find(|step| super::super::field(step, "name") == Some(name))
            .unwrap()
    }
    #[test]
    fn restored_action_handoffs_are_admitted() {
        action_handoffs(&action()).unwrap();
    }
    #[test]
    fn wrong_model_deadline_and_unqualified_vulkan_are_rejected() {
        for (path, value) in [
            (vec!["env", "MODEL_PATH"], "${{ inputs.model }}"),
            (vec!["env", "MESH_BINARY"], "mesh-llm"),
            (vec!["env", "READ_TIMEOUT_SECONDS"], "0"),
            (vec!["if"], "true"),
        ] {
            let mut action = action();
            h::replace(
                mutable_step(&mut action, "Run Laya startup and golden reads"),
                &path,
                Node::Scalar(value.into()),
            );
            assert!(action_handoffs(&action).is_err(), "{path:?}");
        }
    }
    #[test]
    fn read_cannot_precede_restore_or_use_a_comment_as_command() {
        let mut node = action();
        let Node::Seq(steps) = h::mutable(h::mutable(&mut node, "runs"), "steps") else {
            unreachable!()
        };
        steps.swap(1, 2);
        assert!(action_handoffs(&node).is_err());
        let mut node = action();
        h::replace(
            mutable_step(&mut node, "Run Laya startup and golden reads"),
            &["run"],
            Node::Scalar(
                "# mesh_automation automation laya product --model \"$MODEL_PATH\"\necho skipped"
                    .into(),
            ),
        );
        assert!(action_handoffs(&node).is_err());
    }
    fn workflows() -> BTreeMap<String, Node> {
        [
            (
                "ci-linux-product-smoke-slice.yml",
                include_str!("../../../../../.github/workflows/ci-linux-product-smoke-slice.yml"),
            ),
            (
                "ci-macos-product-smoke-slice.yml",
                include_str!("../../../../../.github/workflows/ci-macos-product-smoke-slice.yml"),
            ),
            (
                "ci-windows-product-smoke-slice.yml",
                include_str!("../../../../../.github/workflows/ci-windows-product-smoke-slice.yml"),
            ),
        ]
        .into_iter()
        .map(|(name, source)| {
            (
                name.into(),
                super::super::workflow_yaml::parse(source).unwrap(),
            )
        })
        .collect()
    }
    #[test]
    fn planned_platform_rows_are_admitted_and_device_gate_swaps_fail() {
        platforms(&workflows()).unwrap();
        for path in [
            vec!["jobs", "laya_vulkan", "if"],
            vec!["jobs", "laya_rocm", "if"],
        ] {
            let mut nodes = workflows();
            h::replace(
                nodes.get_mut("ci-linux-product-smoke-slice.yml").unwrap(),
                &path,
                Node::Scalar("true".into()),
            );
            assert!(platforms(&nodes).is_err());
        }
        let mut nodes = workflows();
        let job = h::mutable(
            h::mutable(
                nodes.get_mut("ci-linux-product-smoke-slice.yml").unwrap(),
                "jobs",
            ),
            "laya_cuda",
        );
        let Node::Seq(steps) = h::mutable(job, "steps") else {
            unreachable!()
        };
        let step = steps
            .iter_mut()
            .find(|step| {
                super::super::field(step, "uses")
                    == Some("./.github/actions/run-laya-product-smoke")
            })
            .unwrap();
        h::replace(step, &["with", "device"], Node::Scalar("CPU".into()));
        assert!(platforms(&nodes).is_err());
    }
}
