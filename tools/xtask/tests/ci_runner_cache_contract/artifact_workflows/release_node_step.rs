//! Actual release checksum consumers with finite assets; no addon/native build.
use super::{Node, document, input, job, named, source, support, text, workflow_yaml};
use sha2::{Digest as _, Sha256};
use std::{
    fs,
    process::{Command, Output},
};

const NODE: &str = "Verify complete Node SDK addon release matrix";
const SKIPPY: &str = "Verify standalone Skippy CLI release assets";
const PREPARE: &str = "Prepare release artifact verification";
const TAG: &str = "v1.2.3";
const NATIVE_VERIFY: &str =
    "\"${MESH_LLM_AUTOMATION_BIN:?}\" artifact verify-checksum \"release-artifacts/$archive\"";

fn assets(step: &str) -> Vec<String> {
    let (targets, prefix, suffix) = if step == NODE {
        (
            ["darwin-arm64", "linux-arm64", "linux-x64", "win32-x64"],
            "mesh-llm-node-sdk-addon-1.2.3-",
            ".tar.gz",
        )
    } else {
        (
            [
                "darwin-aarch64",
                "linux-x86_64",
                "linux-aarch64",
                "windows-x86_64",
            ],
            "skippy-v1.2.3-",
            "-cli.tar.gz",
        )
    };
    targets
        .iter()
        .map(|target| format!("{prefix}{target}{suffix}"))
        .collect()
}
fn fixture(step: &str) -> support::Fixture {
    let fixture = support::Fixture::new();
    fs::create_dir(fixture.path().join("release-artifacts")).unwrap();
    for name in assets(step) {
        let payload = format!("finite checksum payload {name}");
        fs::write(
            fixture.path().join("release-artifacts").join(&name),
            &payload,
        )
        .unwrap();
        sidecar(&fixture, &name, payload.as_bytes(), &name);
    }
    fixture
}
fn sidecar(fixture: &support::Fixture, archive: &str, payload: &[u8], recorded: &str) {
    fs::write(
        fixture
            .path()
            .join("release-artifacts")
            .join(format!("{archive}.sha256")),
        format!("{}  {recorded}\n", hex::encode(Sha256::digest(payload))),
    )
    .unwrap();
}
fn execute(fixture: &support::Fixture, step: &str, admitted: bool) -> Output {
    let doc = document("release.yml");
    let run = text(named(job(&doc, "publish"), step), "run").unwrap();
    let mut command = Command::new("/bin/bash");
    command
        .args(["-c", run])
        .current_dir(fixture.path())
        .env("RELEASE_TAG", TAG);
    if admitted {
        command.env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"));
    } else {
        command.env_remove("MESH_LLM_AUTOMATION_BIN");
    }
    fixture.run(command)
}
fn refused(output: Output, reason: &str) {
    assert!(!output.status.success(), "{reason}: {output:?}");
}
#[test]
fn actual_release_checksum_steps_accept_all_four_assets_only_with_prepared_owner() {
    for step in [NODE, SKIPPY] {
        let fixture = fixture(step);
        let output = execute(&fixture, step, true);
        assert!(output.status.success(), "{step}: {output:?}");
        refused(execute(&fixture, step, false), "unprepared native owner");
    }
}
#[test]
fn actual_release_checksum_steps_refuse_each_missing_empty_or_tampered_asset() {
    for step in [NODE, SKIPPY] {
        for name in assets(step) {
            for change in ["missing", "empty", "tampered"] {
                let fixture = fixture(step);
                let archive = fixture.path().join("release-artifacts").join(&name);
                if change == "missing" {
                    fs::remove_file(archive).unwrap();
                } else {
                    fs::write(
                        archive,
                        if change == "empty" {
                            &b""[..]
                        } else {
                            &b"tampered"[..]
                        },
                    )
                    .unwrap();
                }
                refused(
                    execute(&fixture, step, true),
                    &format!("{step}/{name}/{change}"),
                );
            }
        }
    }
}
#[test]
fn actual_release_checksum_steps_refuse_each_missing_empty_malformed_or_extra_sidecar() {
    for step in [NODE, SKIPPY] {
        for name in assets(step) {
            for change in ["missing", "empty", "malformed", "extra-line"] {
                let fixture = fixture(step);
                let checksum = fixture
                    .path()
                    .join("release-artifacts")
                    .join(format!("{name}.sha256"));
                match change {
                    "missing" => fs::remove_file(checksum).unwrap(),
                    "empty" => fs::write(checksum, "").unwrap(),
                    "malformed" => fs::write(checksum, format!("not-a-digest  {name}\n")).unwrap(),
                    _ => {
                        let valid = fs::read_to_string(&checksum).unwrap();
                        fs::write(checksum, format!("{valid}{valid}")).unwrap();
                    }
                }
                refused(
                    execute(&fixture, step, true),
                    &format!("{step}/{name}/{change}"),
                );
            }
        }
    }
}
#[test]
fn actual_release_checksum_steps_refuse_correct_decoy_digest_for_tampered_expected_archive() {
    for step in [NODE, SKIPPY] {
        for name in assets(step) {
            let fixture = fixture(step);
            // The old Node regex treated filename dots as wildcards; Skippy
            // previously delegated the sidecar's arbitrary filename to shasum.
            let decoy = name.replace('.', "x");
            let payload = b"correctly hashed unrelated decoy";
            fs::write(
                fixture.path().join("release-artifacts").join(&decoy),
                payload,
            )
            .unwrap();
            fs::write(
                fixture.path().join("release-artifacts").join(&name),
                b"tampered expected archive",
            )
            .unwrap();
            sidecar(&fixture, &name, payload, &decoy);
            let output = execute(&fixture, step, true);
            assert!(
                String::from_utf8_lossy(&output.stderr).contains("checksum sidecar names"),
                "{output:?}"
            );
            refused(output, &format!("{step}/{name}/decoy"));
        }
    }
}

fn selected<'a>(steps: &'a [Node], name: &str) -> Option<(usize, &'a Node)> {
    let found: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| text(step, "name") == Some(name))
        .collect();
    match found.as_slice() {
        [entry] => Some(*entry),
        _ => None,
    }
}
fn executable_lines(run: &str) -> Vec<&str> {
    run.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}
fn admitted_graph(doc: &Node) -> bool {
    let Some(publish) = doc.get("jobs").and_then(|jobs| jobs.get("publish")) else {
        return false;
    };
    let Some(Node::Seq(publish_steps)) = publish.get("steps") else {
        return false;
    };
    let Some((prepare_index, prepare)) = selected(publish_steps, PREPARE) else {
        return false;
    };
    let Some((release_index, _)) = selected(publish_steps, "Publish GitHub release") else {
        return false;
    };
    if text(prepare, "uses") != Some("./.github/actions/prepare-automation")
        || prepare.get("if").is_some()
        || input(prepare, "runner-profile") != Some("hosted-bare")
        || input(prepare, "allow_depot_remote_cache") != Some("false")
        || input(prepare, "allow_native_github_cache") != Some("false")
    {
        return false;
    }
    let Some(checkout) = publish_steps.iter().position(|step| {
        text(step, "uses").is_some_and(|value| value.starts_with("actions/checkout@"))
    }) else {
        return false;
    };
    if checkout >= prepare_index
        || input(&publish_steps[checkout], "ref")
            != Some("${{ needs.metadata.outputs.source_sha }}")
        || input(&publish_steps[checkout], "persist-credentials") != Some("false")
    {
        return false;
    }
    if !publish
        .get("needs")
        .is_some_and(|needs| needs.list().contains(&"build_node_sdk_addon"))
        || !text(publish, "if").is_some_and(|condition| {
            condition.contains("needs.build_node_sdk_addon.result == 'success'")
        })
    {
        return false;
    }
    [NODE, SKIPPY]
        .into_iter()
        .all(|name| admitted_verifier(publish_steps, name, prepare_index, release_index))
        && admitted_node_producer(doc)
}
fn admitted_verifier(rows: &[Node], name: &str, prepare: usize, release: usize) -> bool {
    let Some((index, verify)) = selected(rows, name) else {
        return false;
    };
    if !(prepare < index && index < release)
        || verify.get("if").is_some()
        || verify
            .get("continue-on-error")
            .is_some_and(|value| value.text() != Some("false"))
        || verify.get("env").and_then(|env| text(env, "RELEASE_TAG"))
            != Some("${{ needs.metadata.outputs.tag }}")
    {
        return false;
    }
    let Some(run) = text(verify, "run") else {
        return false;
    };
    // This is the closed checksum gate, not a general shell interpreter.
    // Comments/blank lines are inert; extra executable statements must be
    // reviewed rather than disabling errexit or masking the native refusal.
    let expected: &[&str] = if name == NODE {
        &[
            "set -euo pipefail",
            "version=\"${RELEASE_TAG#v}\"",
            "for target in darwin-arm64 linux-arm64 linux-x64 win32-x64; do",
            "archive=\"mesh-llm-node-sdk-addon-$version-$target.tar.gz\"",
            "sidecar=\"$archive.sha256\"",
            "test -s \"release-artifacts/$archive\" || {",
            "echo \"Missing Node SDK addon release asset: $archive\" >&2",
            "exit 1",
            "}",
            "test -s \"release-artifacts/$sidecar\" || {",
            "echo \"Missing Node SDK addon checksum: $sidecar\" >&2",
            "exit 1",
            "}",
            NATIVE_VERIFY,
            "done",
        ]
    } else {
        &[
            "set -euo pipefail",
            "for target in darwin-aarch64 linux-x86_64 linux-aarch64 windows-x86_64; do",
            "archive=\"skippy-${RELEASE_TAG}-${target}-cli.tar.gz\"",
            "test -s \"release-artifacts/$archive\"",
            "test -s \"release-artifacts/$archive.sha256\"",
            NATIVE_VERIFY,
            "done",
        ]
    };
    executable_lines(run) == expected
}
fn admitted_node_producer(doc: &Node) -> bool {
    let Some(producer) = doc
        .get("jobs")
        .and_then(|jobs| jobs.get("build_node_sdk_addon"))
    else {
        return false;
    };
    if text(producer, "uses") != Some("./.github/workflows/node-sdk-addon-artifact.yml")
        || !producer
            .get("needs")
            .is_some_and(|needs| needs.list() == ["metadata"])
    {
        return false;
    }
    let Some(Node::Seq(rows)) = producer
        .get("strategy")
        .and_then(|strategy| strategy.get("matrix"))
        .and_then(|matrix| matrix.get("include"))
    else {
        return false;
    };
    if rows
        .iter()
        .filter_map(|row| text(row, "target"))
        .collect::<Vec<_>>()
        != ["darwin-arm64", "linux-arm64", "linux-x64", "win32-x64"]
    {
        return false;
    }
    [
        ("target", "${{ matrix.target }}"),
        (
            "artifact_name",
            "release-node-sdk-addon-${{ matrix.target }}",
        ),
        ("release_tag", "${{ needs.metadata.outputs.tag }}"),
        (
            "prepare_release_version",
            "${{ github.event_name == 'workflow_dispatch' }}",
        ),
    ]
    .into_iter()
    .all(|(key, value)| input(producer, key) == Some(value))
}
#[test]
fn release_artifact_graph_binds_producer_fanin_tag_uncached_preparation_and_verify_before_publish()
{
    let doc = document("release.yml");
    assert!(admitted_graph(&doc));
    // These local contract mutations change executable source/bindings, never
    // the checked-in workflow or its protected catalogs.
    let original = source("release.yml");
    for (before, after) in [
        (
            "name: Prepare release artifact verification",
            "name: Detached release artifact verification",
        ),
        ("runner-profile: hosted-bare", "runner-profile: image"),
        (
            "allow_depot_remote_cache: \"false\"",
            "allow_depot_remote_cache: \"true\"",
        ),
        (
            "allow_native_github_cache: \"false\"",
            "allow_native_github_cache: \"true\"",
        ),
        (
            "artifact_name: release-node-sdk-addon-${{ matrix.target }}",
            "artifact_name: unrelated-${{ matrix.target }}",
        ),
        ("needs.build_node_sdk_addon.result == 'success'", "true"),
        ("      - build_node_sdk_addon\n", ""),
        (
            "RELEASE_TAG: ${{ needs.metadata.outputs.tag }}",
            "RELEASE_TAG: v0.0.0",
        ),
        (
            NATIVE_VERIFY,
            "# checksum owner is mentioned but not executed",
        ),
        (
            NATIVE_VERIFY,
            "shasum -a 256 -c \"release-artifacts/$archive.sha256\"",
        ),
        (
            "for target in darwin-arm64 linux-arm64 linux-x64 win32-x64; do",
            "for target in darwin-arm64 linux-x64 win32-x64; do",
        ),
        ("target: darwin-arm64", "target: darwin-x64"),
        (
            NATIVE_VERIFY,
            "\"${UNRELATED_AUTOMATION_BIN:?}\" artifact verify-checksum \"release-artifacts/$archive\"",
        ),
        (
            "ref: ${{ needs.metadata.outputs.source_sha }}",
            "ref: ${{ github.sha }}",
        ),
    ] {
        assert!(original.contains(before), "{before}");
        let changed = workflow_yaml::parse(&original.replace(before, after)).unwrap();
        assert!(!admitted_graph(&changed), "{before}");
    }
    let mut reordered = doc.clone();
    let Node::Map(root) = &mut reordered else {
        panic!("root")
    };
    let Node::Map(jobs) = &mut root.iter_mut().find(|(key, _)| key == "jobs").unwrap().1 else {
        panic!("jobs")
    };
    let Node::Map(publish) = &mut jobs.iter_mut().find(|(key, _)| key == "publish").unwrap().1
    else {
        panic!("publish")
    };
    let Node::Seq(rows) = &mut publish
        .iter_mut()
        .find(|(key, _)| key == "steps")
        .unwrap()
        .1
    else {
        panic!("steps")
    };
    let prepare = rows
        .iter()
        .position(|step| text(step, "name") == Some(PREPARE))
        .unwrap();
    let verify = rows
        .iter()
        .position(|step| text(step, "name") == Some(NODE))
        .unwrap();
    rows.swap(prepare, verify);
    assert!(!admitted_graph(&reordered));
}

fn mutate_checksum_step(doc: &mut Node, name: &str, field: &str, value: String) {
    let Node::Map(root) = doc else { panic!("root") };
    let Node::Map(jobs) = &mut root.iter_mut().find(|(key, _)| key == "jobs").unwrap().1 else {
        panic!("jobs")
    };
    let Node::Map(publish) = &mut jobs.iter_mut().find(|(key, _)| key == "publish").unwrap().1
    else {
        panic!("publish")
    };
    let Node::Seq(rows) = &mut publish
        .iter_mut()
        .find(|(key, _)| key == "steps")
        .unwrap()
        .1
    else {
        panic!("steps")
    };
    let step = rows
        .iter_mut()
        .find(|step| text(step, "name") == Some(name))
        .unwrap();
    let Node::Map(fields) = step else {
        panic!("step")
    };
    if let Some((_, existing)) = fields.iter_mut().find(|(key, _)| key == field) {
        *existing = Node::Scalar(value);
    } else {
        fields.push((field.into(), Node::Scalar(value)));
    }
}
#[test]
fn both_release_checksum_gates_refuse_masked_errors_and_keep_inert_comments() {
    let original = document("release.yml");
    for name in [NODE, SKIPPY] {
        let run = text(named(job(&original, "publish"), name), "run").unwrap();
        for value in ["true", "${{ always() }}"] {
            let mut changed = original.clone();
            mutate_checksum_step(&mut changed, name, "continue-on-error", value.into());
            assert!(!admitted_graph(&changed), "{name}/{value}");
        }
        for changed_run in [
            run.replace("set -euo pipefail", "set -uo pipefail"),
            run.replace(NATIVE_VERIFY, &format!("set +e\n{NATIVE_VERIFY}")),
            run.replace(NATIVE_VERIFY, &format!("set +o errexit\n{NATIVE_VERIFY}")),
            run.replace(NATIVE_VERIFY, &format!("{NATIVE_VERIFY} || true")),
        ] {
            let mut changed = original.clone();
            mutate_checksum_step(&mut changed, name, "run", changed_run);
            assert!(!admitted_graph(&changed), "{name}");
        }
        let mut inert = original.clone();
        mutate_checksum_step(&mut inert, name, "continue-on-error", "false".into());
        mutate_checksum_step(&mut inert, name, "run", format!("# inert comment\n\n{run}"));
        assert!(admitted_graph(&inert), "{name}");
    }
}
