//! Native SDK producer graph and actual ABI restore step contract fixtures.
use super::{
    support::{Fixture, root},
    workflow_yaml::{self, Node},
};
use std::{fs, process::Command};

fn source() -> String {
    fs::read_to_string(root().join(".github/workflows/native-sdk-artifact.yml")).unwrap()
}
fn document() -> Node {
    workflow_yaml::parse(&source()).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn job<'a>(node: &'a Node, name: &str) -> &'a Node {
    node.get("jobs").unwrap().get(name).unwrap()
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(items) = node.get("steps").unwrap() else {
        panic!("steps must be a sequence")
    };
    items
}
fn named<'a>(node: &'a Node, name: &str) -> (usize, &'a Node) {
    let values = steps(node)
        .iter()
        .enumerate()
        .filter(|(_, step)| text(step, "name") == Some(name))
        .collect::<Vec<_>>();
    let [(index, step)] = values.as_slice() else {
        panic!("one {name} step required")
    };
    (*index, *step)
}
fn input(node: &Node, key: &str, value: &str) -> bool {
    node.get("with").and_then(|node| text(node, key)) == Some(value)
}
fn immutable_job(node: &Node, linux: bool) -> bool {
    let checkouts = steps(node)
        .iter()
        .enumerate()
        .filter(|(_, step)| {
            text(step, "uses").is_some_and(|value| value.starts_with("actions/checkout@"))
        })
        .collect::<Vec<_>>();
    let [(checkout_index, checkout)] = checkouts.as_slice() else {
        return false;
    };
    let (release_index, _) = named(node, "Prepare dispatched release version");
    let (prepare_index, prepare) = named(node, "Prepare immutable native SDK input");
    let (upload_index, upload) = named(node, "Upload immutable native SDK input");
    *checkout_index < release_index
        && release_index < prepare_index
        && prepare_index < upload_index
        && input(checkout, "ref", "${{ inputs.source_sha || github.sha }}")
        && input(checkout, "persist-credentials", "false")
        && text(prepare, "id") == Some("native-sdk")
        && text(prepare, "uses") == Some("./.github/actions/prepare-native-sdk-input")
        && ["backend", "target", "profile", "include_runtime_crate"]
            .into_iter()
            .all(|key| input(prepare, key, &format!("${{{{ inputs.{key} }}}}")))
        && (!linux
            || input(
                prepare,
                "require_prebuilt_static_abi",
                "${{ inputs.static_abi_artifact_name != '' }}",
            ))
        && input(upload, "name", "${{ inputs.artifact_name }}")
        && input(
            upload,
            "path",
            "${{ steps.native-sdk.outputs.upload_path }}",
        )
        && input(upload, "if-no-files-found", "error")
        && input(upload, "retention-days", "${{ inputs.retention_days }}")
        && text(upload, "uses").is_some_and(|value| value.starts_with("actions/upload-artifact@"))
        && upload.get("if").is_none()
}
fn abi_reuse(document: &Node) -> bool {
    let producer = job(document, "produce_linux_static_abi");
    let linux = job(document, "linux_native_sdk_artifact");
    let (checkout_index, _) = named(linux, "Prepare patched llama.cpp checkout for ABI reuse");
    let (download_index, download) = named(linux, "Download immutable static ABI input");
    let (restore_index, restore) = named(linux, "Restore immutable static ABI input");
    let (prepare_index, _) = named(linux, "Prepare immutable native SDK input");
    let mut needs = linux.get("needs").unwrap().list();
    needs.sort_unstable();
    needs == ["produce_linux_static_abi", "runner_policy"]
        && text(producer, "uses") == Some("./.github/workflows/static-abi-artifact.yml")
        && text(producer, "needs") == Some("runner_policy")
        && text(producer, "if")
            == Some(
                "${{ inputs.produce_static_abi && endsWith(inputs.target, '-unknown-linux-gnu') }}",
            )
        && input(
            producer,
            "artifact_name",
            "${{ inputs.static_abi_artifact_name }}",
        )
        && [
            "target",
            "backend",
            "runner_size",
            "original_event_name",
            "force_hosted",
            "use_depot",
        ]
        .into_iter()
        .all(|key| input(producer, key, &format!("${{{{ inputs.{key} }}}}")))
        && text(linux, "if")
            == Some(
                "${{ !cancelled() && needs.runner_policy.result == 'success' && endsWith(inputs.target, '-unknown-linux-gnu') && (needs.produce_linux_static_abi.result == 'success' || (needs.produce_linux_static_abi.result == 'skipped' && !inputs.produce_static_abi)) }}",
            )
        && checkout_index < download_index
        && download_index < restore_index
        && restore_index < prepare_index
        && [download, restore]
            .into_iter()
            .all(|step| text(step, "if") == Some("${{ inputs.static_abi_artifact_name != '' }}"))
        && input(download, "name", "${{ inputs.static_abi_artifact_name }}")
        && input(download, "path", "${{ inputs.static_abi_artifact_path }}")
        && text(download, "uses")
            .is_some_and(|value| value.starts_with("actions/download-artifact@"))
        && linux
            .get("env")
            .and_then(|env| text(env, "LLAMA_STAGE_BUILD_DIR"))
            == Some(".deps/llama.cpp/build-stage-abi-static")
}
#[test]
fn native_sdk_workflow_binds_immutable_source_preparation_and_uploads_on_both_platforms() {
    let check = |document: &Node| {
        immutable_job(job(document, "linux_native_sdk_artifact"), true)
            && immutable_job(job(document, "macos_native_sdk_artifact"), false)
    };
    assert!(check(&document()));
    for (valid, invalid) in [
        ("ref: ${{ inputs.source_sha || github.sha }}", "ref: main"),
        (
            "path: ${{ steps.native-sdk.outputs.upload_path }}",
            "path: dist/unverified",
        ),
        (
            "include_runtime_crate: ${{ inputs.include_runtime_crate }}",
            "include_runtime_crate: false",
        ),
        (
            "require_prebuilt_static_abi: ${{ inputs.static_abi_artifact_name != '' }}",
            "require_prebuilt_static_abi: false",
        ),
    ] {
        let original = source();
        assert!(original.contains(valid));
        let changed = format!("{}\n# {valid}\n", original.replace(valid, invalid));
        assert!(!check(&workflow_yaml::parse(&changed).unwrap()));
    }
}
#[test]
fn native_sdk_workflow_requires_matching_abi_producer_and_ordered_restore_without_rebuild() {
    assert!(abi_reuse(&document()));
    for (valid, invalid) in [
        (
            "needs: [runner_policy, produce_linux_static_abi]",
            "needs: [runner_policy]",
        ),
        (
            "artifact_name: ${{ inputs.static_abi_artifact_name }}",
            "artifact_name: unrelated-abi",
        ),
        (
            "path: ${{ inputs.static_abi_artifact_path }}",
            "path: unrelated-directory",
        ),
        (
            "LLAMA_STAGE_BUILD_DIR: .deps/llama.cpp/build-stage-abi-static",
            "LLAMA_STAGE_BUILD_DIR: unrelated-build",
        ),
    ] {
        let original = source();
        assert!(original.contains(valid));
        let changed = format!("{}\n# {valid}\n", original.replace(valid, invalid));
        assert!(!abi_reuse(&workflow_yaml::parse(&changed).unwrap()));
    }
    let document = document();
    let (_, restore) = named(
        job(&document, "linux_native_sdk_artifact"),
        "Restore immutable static ABI input",
    );
    for failure in ["false", "true"] {
        let fixture = Fixture::new();
        fs::create_dir_all(fixture.path().join("scripts")).unwrap();
        fixture.executable("restore-observer",r#"[[ "$#" == 4 && "$1" == 'ABI input' && "$2" == 'ABI output' && "$3" == aarch64-unknown-linux-gnu && "$4" == cpu ]] || exit 97
printf 'restore\n' > events
[[ "$FAIL_RESTORE" == false ]] || exit 97
"#);
        fs::copy(
            fixture.path().join("bin/restore-observer"),
            fixture.path().join("scripts/restore-static-abi-input.sh"),
        )
        .unwrap();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .current_dir(fixture.path())
            .args(["-c", text(restore, "run").unwrap()]);
        for (key, value) in [
            ("STATIC_ABI_ARTIFACT_PATH", "ABI input"),
            ("LLAMA_STAGE_BUILD_DIR", "ABI output"),
            ("STATIC_ABI_TARGET", "aarch64-unknown-linux-gnu"),
            ("STATIC_ABI_BACKEND", "cpu"),
            ("FAIL_RESTORE", failure),
        ] {
            command.env(key, value);
        }
        let result = fixture.run(command);
        assert_eq!(result.status.success(), failure == "false");
        assert_eq!(
            fs::read_to_string(fixture.path().join("events")).unwrap(),
            "restore\n"
        );
    }
}
