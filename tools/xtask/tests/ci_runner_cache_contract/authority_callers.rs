//! Same-run protected native authority delivery through reusable workflow calls.
use super::{
    support,
    workflow_yaml::{self, Node},
};
use std::{collections::BTreeMap, fs};

const VARIANTS: &[&str] = &["linux_x64", "linux_arm64", "macos_arm64", "windows_x64"];
const ROOTS: &[(&str, &[&str])] = &[
    ("ci-quality-lane.yml", &["linux_x64"]),
    ("ci-website-lane.yml", &["linux_x64"]),
    ("ci-linux-lane.yml", &["linux_x64"]),
    ("ci-macos-lane.yml", &["linux_x64", "macos_arm64"]),
    ("ci-windows-lane.yml", &["linux_x64", "windows_x64"]),
    ("ci-pr-canary-lane.yml", &["linux_x64"]),
    ("release.yml", &["linux_x64", "linux_arm64", "macos_arm64"]),
];
const LIBRARIES: &[&str] = &[
    "ci-linux-host-slice.yml",
    "ci-linux-runtime-slice.yml",
    "ci-linux-product-slice.yml",
    "ci-macos-host-slice.yml",
    "ci-macos-runtime-slice.yml",
    "ci-macos-product-slice.yml",
    "ci-windows-host-slice.yml",
    "ci-windows-runtime-slice.yml",
    "ci-windows-product-slice.yml",
    "ci-platform-checks-slice.yml",
    "ci-quality-slice.yml",
    "ci-rust-tests-slice.yml",
    "ci-ui-artifact-slice.yml",
    "ci-web-slice.yml",
    "native-sdk-artifact.yml",
    "static-abi-artifact.yml",
    "swift-sdk-artifact.yml",
];
fn read(name: &str) -> Node {
    workflow_yaml::parse(
        &fs::read_to_string(support::root().join(".github/workflows").join(name)).unwrap(),
    )
    .unwrap()
}
fn member<'a>(node: &'a Node, key: &str) -> Result<&'a Node, String> {
    node.get(key).ok_or_else(|| format!("missing {key}"))
}
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn check_root(document: &Node, variants: &[&str]) -> Result<(), String> {
    let jobs = member(document, "jobs")?;
    let source = member(jobs, "authority_source")?;
    if text(source, "runs-on") != "ubuntu-24.04" {
        return Err("protected source hosted broker".into());
    }
    let Node::Seq(steps) = member(source, "steps")? else {
        return Err("source steps".into());
    };
    let checkout = steps
        .iter()
        .find(|step| text(step, "uses").starts_with("actions/checkout@"))
        .ok_or("source checkout")?;
    let inputs = member(checkout, "with")?;
    if text(inputs, "ref") != "${{ github.event.repository.default_branch }}"
        || text(inputs, "persist-credentials") != "false"
    {
        return Err("protected source without credentials".into());
    }
    let actual: Vec<_> = jobs
        .entries()
        .iter()
        .filter_map(|(name, _)| {
            name.strip_prefix("authority_")
                .filter(|name| *name != "source")
        })
        .collect();
    if actual != variants {
        return Err("deduplicated native roster".into());
    }
    for variant in variants {
        let producer = member(jobs, &format!("authority_{variant}"))?;
        if text(producer, "uses") != "./.github/workflows/protected-automation-artifact.yml"
            || member(producer, "needs")?.list() != ["authority_source"]
        {
            return Err("native factory depends on frozen source".into());
        }
        let permissions = member(producer, "permissions")?;
        if permissions.entries().len() != 1
            || text(permissions, "contents") != "read"
            || producer.get("secrets").is_some()
        {
            return Err("read-only native factory".into());
        }
        let inputs = member(producer, "with")?;
        if text(inputs, "platform") != variant.replace('_', "-")
            || text(inputs, "protected_source_sha")
                != "${{ needs.authority_source.outputs.source_sha }}"
        {
            return Err("native factory source and platform".into());
        }
    }
    Ok(())
}
fn expected_variants(root: &str, target: &str) -> &'static [&'static str] {
    if target == "native-sdk-artifact.yml" && root == "release.yml" {
        return &["linux_x64", "linux_arm64", "macos_arm64"];
    }
    if target.starts_with("ci-macos-")
        || target == "swift-sdk-artifact.yml"
        || (target == "ci-platform-checks-slice.yml" && root == "ci-macos-lane.yml")
    {
        return &["macos_arm64"];
    }
    if target.starts_with("ci-windows-") || target == "ci-platform-checks-slice.yml" {
        return &["windows_x64"];
    }
    &["linux_x64"]
}
fn check_call(root: &str, target: &str, job: &Node) -> Result<(), String> {
    let inputs = member(job, "with")?;
    let expected = if root == "native-sdk-artifact.yml" {
        VARIANTS
    } else {
        expected_variants(root, target)
    };
    for variant in expected {
        let key = format!("authority_{variant}");
        let binding = if root == "native-sdk-artifact.yml" {
            format!("${{{{ inputs.{key} }}}}")
        } else {
            if !member(job, "needs")?.list().contains(&key.as_str()) {
                return Err("caller requires native factory".into());
            }
            format!("${{{{ needs.{key}.outputs.identity_json }}}}")
        };
        if text(inputs, &key) != binding {
            return Err("caller forwards dependency-provided identity".into());
        }
    }
    Ok(())
}
#[test]
fn each_topic_deduplicates_native_producers_and_freezes_protected_source_once() {
    for (root, variants) in ROOTS {
        check_root(&read(root), variants).unwrap();
    }
    assert_eq!(
        ROOTS
            .iter()
            .map(|(_, variants)| variants.len())
            .sum::<usize>(),
        11
    );
}
#[test]
fn every_audited_reusable_call_receives_its_native_identity_including_nested_sdk() {
    let mut edges = BTreeMap::new();
    for entry in fs::read_dir(support::root().join(".github/workflows")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("yml") {
            continue;
        }
        let root = path.file_name().unwrap().to_str().unwrap();
        let document = read(root);
        for (name, job) in member(&document, "jobs").unwrap().entries() {
            let Some(target) = text(job, "uses").strip_prefix("./.github/workflows/") else {
                continue;
            };
            if LIBRARIES.contains(&target) {
                check_call(root, target, job)
                    .unwrap_or_else(|error| panic!("{root}/{name}: {error}"));
                edges.insert(format!("{root}/{name}"), target.to_owned());
            }
        }
    }
    assert_eq!(edges.len(), 28);
    for library in LIBRARIES {
        let document = read(library);
        let inputs = member(
            member(member(&document, "on").unwrap(), "workflow_call").unwrap(),
            "inputs",
        )
        .unwrap();
        for variant in VARIANTS {
            let input = member(inputs, &format!("authority_{variant}")).unwrap();
            assert_eq!(text(input, "type"), "string");
            assert_eq!(text(input, "default"), "");
            assert_eq!(text(input, "required"), "false");
        }
    }
}
#[test]
fn caller_and_factory_mutations_cannot_replace_provenance_or_disconnect_dependencies() {
    let source =
        fs::read_to_string(support::root().join(".github/workflows/ci-website-lane.yml")).unwrap();
    for (from, to, factory) in [
        ("needs: [authority_linux_x64]", "needs: []", false),
        (
            "${{ needs.authority_linux_x64.outputs.identity_json }}",
            "${{ inputs.source_sha }}",
            false,
        ),
        (
            "${{ needs.authority_source.outputs.source_sha }}",
            "${{ inputs.source_sha }}",
            true,
        ),
        ("platform: linux-x64", "platform: windows-x64", true),
    ] {
        assert!(source.contains(from), "mutation anchor {from}");
        let document = workflow_yaml::parse(&source.replace(from, to)).unwrap();
        let result = if factory {
            check_root(&document, &["linux_x64"])
        } else {
            check_call(
                "ci-website-lane.yml",
                "ci-web-slice.yml",
                member(member(&document, "jobs").unwrap(), "web").unwrap(),
            )
        };
        assert!(result.is_err(), "mutation {from}");
    }
}
