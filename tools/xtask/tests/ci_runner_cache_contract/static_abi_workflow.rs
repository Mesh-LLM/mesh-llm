//! Bind the portable ABI cache and immutable producer handoff to active YAML.
use super::{
    support::root,
    workflow_yaml::{self, Node},
};
use std::fs;
fn source() -> String {
    fs::read_to_string(root().join(".github/workflows/static-abi-artifact.yml")).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn input(node: &Node, key: &str, value: &str) -> bool {
    node.get("with").and_then(|n| text(n, key)) == Some(value)
}
fn step<'a>(node: &'a Node, name: &str) -> (usize, &'a Node) {
    let Node::Seq(steps) = node.get("steps").unwrap() else {
        panic!("step sequence required")
    };
    let matches = steps
        .iter()
        .enumerate()
        .filter(|(_, node)| text(node, "name") == Some(name))
        .collect::<Vec<_>>();
    let [(index, node)] = matches.as_slice() else {
        panic!("one {name} required")
    };
    (*index, *node)
}
fn graph(document: &Node) -> bool {
    let producer = document
        .get("jobs")
        .unwrap()
        .get("static_abi_artifact")
        .unwrap();
    let (epoch_index, epoch) = step(producer, "Resolve static ABI toolchain epoch");
    let (prepare_index, _) = step(producer, "Prepare patched llama.cpp checkout");
    let (patched_index, patched) = step(producer, "Identify patched llama.cpp for cache reuse");
    let (cache_index, cache) = step(producer, "Cache portable static ABI input");
    let (restore_index, restore) = step(producer, "Restore and verify cached portable static ABI");
    let (build_index, build) = step(producer, "Build and archive immutable static ABI input");
    let (upload_index, upload) = step(producer, "Upload immutable static ABI input");
    let Some(key) = cache.get("with").and_then(|node| text(node, "key")) else {
        return false;
    };
    let hash_inputs = [
        "'scripts/build-llama.sh'",
        "'skippy/scripts/build-llama.sh'",
        "'scripts/prepare-llama.sh'",
        "'skippy/scripts/prepare-llama.sh'",
        "'scripts/restore-static-abi-input.sh'",
        "'tools/xtask/src/artifact/**'",
        "'tools/xtask/src/prepared_input/**'",
        "'tools/xtask/src/repository/python_text.rs'",
        "'tools/xtask/src/ci_plan/document.rs'",
        "'Cargo.lock'",
        "'tools/xtask/Cargo.toml'",
        "'.github/actions/prepare-static-abi-input/action.yml'",
        "'.github/actions/resolve-native-toolchain-epoch/action.yml'",
        "'skippy/llama_cpp/upstream.txt'",
        "'skippy/llama_cpp/patches/**'",
        "'third_party/llama.cpp/upstream.txt'",
        "'third_party/llama.cpp/patches/**'",
        "'Justfile'",
        "'just/**'",
        "'.github/cache-version.txt'",
    ];
    let image = producer
        .get("container")
        .and_then(|node| text(node, "image"));
    let pinned = epoch
        .get("with")
        .and_then(|node| text(node, "pinned_epoch"));
    let epoch_matches = image.zip(pinned).is_some_and(|(image, pinned)| {
        image
            .strip_prefix("ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:")
            .zip(pinned.strip_prefix("mesh-llm-cuda-runner-sha256-"))
            .is_some_and(|(left, right)| left == right && left.len() == 64)
    });
    epoch_index<cache_index && prepare_index<patched_index && patched_index<cache_index && cache_index<restore_index
        && restore_index<build_index && build_index<upload_index && epoch_matches
        && document.get("env").and_then(|env|text(env,"CACHE_NAMESPACE"))==Some("mesh-llm")
        && text(epoch,"id")==Some("native_toolchain")
        && text(epoch,"uses")==Some("./.github/actions/resolve-native-toolchain-epoch")
        && producer.get("outputs").and_then(|outputs|text(outputs,"toolchain_epoch"))==Some("${{ steps.native_toolchain.outputs.epoch }}")
        && text(cache,"id")==Some("static_abi_cache")
        && input(cache,"path","static-abi-artifact-output")
        && cache.get("with").unwrap().get("restore-keys").is_none()
        && text(cache,"if")==Some("${{ needs.runner_policy.outputs.allow_native_github_cache == 'true' }}")
        && text(patched,"id")==Some("patched_llama")
        && text(patched,"shell")==Some("bash")
        && text(patched,"run").is_some_and(patched_identity)
        && key.starts_with("${{ format('{0}-{1}-skippy-abi-{2}-{3}-{4}-{5}-{6}', env.CACHE_NAMESPACE, runner.os, inputs.backend, inputs.target, steps.native_toolchain.outputs.epoch, steps.patched_llama.outputs.sha, hashFiles(")
        && hash_inputs.into_iter().all(|value|key.contains(value))
        && text(restore,"if")==Some("${{ steps.static_abi_cache.outputs.cache-hit == 'true' }}")
        && text(build,"if")==Some("${{ steps.static_abi_cache.outputs.cache-hit != 'true' }}")
        && text(build,"uses")==Some("./.github/actions/prepare-static-abi-input")
        && input(build,"backend","${{ inputs.backend }}") && input(build,"target","${{ inputs.target }}") && input(build,"build","true")
        && input(upload,"name","${{ inputs.artifact_name }}") && input(upload,"path","static-abi-artifact-output/*")
        && input(upload,"if-no-files-found","error") && upload.get("if").is_none()
}
fn patched_identity(run: &str) -> bool {
    run.lines().map(str::trim).collect::<Vec<_>>()
        == [
            "set -euo pipefail",
            "patched_sha=\"$(tr -d '[:space:]' < .deps/llama.cpp/.mesh-llm-patched-sha)\"",
            "[[ \"$patched_sha\" =~ ^[0-9a-f]{40}$ ]]",
            "echo \"sha=$patched_sha\" >> \"$GITHUB_OUTPUT\"",
        ]
}
#[test]
fn static_abi_workflow_binds_portable_exact_cache_identity_prepare_restore_and_immutable_upload() {
    assert!(graph(&workflow_yaml::parse(&source()).unwrap()));
    for (valid, invalid) in [
        (
            "path: static-abi-artifact-output\n",
            "path: .deps/llama.cpp/build-stage-abi-static\n",
        ),
        (
            "steps.native_toolchain.outputs.epoch, steps.patched_llama.outputs.sha, hashFiles(",
            "'unknown-epoch', steps.patched_llama.outputs.sha, hashFiles(",
        ),
        ("'Justfile', 'just/**'", "'Justfile'"),
        (
            "path: static-abi-artifact-output/*",
            "path: unrelated-build/*",
        ),
        (
            "steps.static_abi_cache.outputs.cache-hit != 'true'",
            "steps.static_abi_cache.outputs.cache-hit == 'true'",
        ),
    ] {
        let original = source();
        assert!(original.contains(valid));
        let changed = format!("{}\n# {valid}\n", original.replace(valid, invalid));
        assert!(!graph(&workflow_yaml::parse(&changed).unwrap()));
    }
}

#[test]
fn static_abi_cache_refuses_any_missing_identity_dimension_or_unbound_patched_source() {
    for dimension in [
        "env.CACHE_NAMESPACE",
        "runner.os",
        "inputs.backend",
        "inputs.target",
        "steps.native_toolchain.outputs.epoch",
        "steps.patched_llama.outputs.sha",
        "hashFiles(",
    ] {
        let original = source();
        let anchor = format!(", {dimension}");
        assert!(original.contains(&anchor));
        let changed = format!(
            "{}\n# {anchor}\n",
            original.replacen(&anchor, ", 'unbound'", 1)
        );
        assert!(
            !graph(&workflow_yaml::parse(&changed).unwrap()),
            "{dimension}"
        );
    }
    for (valid, invalid) in [
        ("id: patched_llama", "id: unrelated_source"),
        (
            "< .deps/llama.cpp/.mesh-llm-patched-sha",
            "< ci/unverified-source.txt",
        ),
        (
            "[[ \"$patched_sha\" =~ ^[0-9a-f]{40}$ ]]",
            "true # unchecked source",
        ),
        (
            "echo \"sha=$patched_sha\" >> \"$GITHUB_OUTPUT\"",
            "echo \"sha=unverified\" >> \"$GITHUB_OUTPUT\"",
        ),
    ] {
        let original = source();
        assert!(original.contains(valid));
        let changed = format!("{}\n# {valid}\n", original.replacen(valid, invalid, 1));
        assert!(!graph(&workflow_yaml::parse(&changed).unwrap()), "{valid}");
    }
}
