//! Real addon producer version/path boundary with an inert npm executable.
use super::workflows::{bash, field, named, steps, tool, workflow};
use std::{fs, os::unix::fs::PermissionsExt, path::Path};

fn npm_boundary(root: &Path) -> String {
    let bin = root.join("bin");
    fs::create_dir(&bin).unwrap();
    for name in ["node", "mkdir"] {
        std::os::unix::fs::symlink(tool(name), bin.join(name)).unwrap();
    }
    let npm = bin.join("npm");
    fs::write(
        &npm,
        r#"#!/bin/sh
set -eu
case "$1" in
  test) [ "$#" = 3 ] && [ "$2" = --prefix ] || exit 90; prefix="$3" ;;
  run) [ "$#" = 4 ] && [ "$2" = build:native ] && [ "$3" = --prefix ] || exit 90; prefix="$4" ;;
  *) exit 91 ;;
esac
[ "$prefix" = "$SDK_DIR/node" ] && [ -f "$prefix/package.json" ] || exit 92
printf '%s\n' "$*" >> "$NPM_CALLS"
if [ "$1" = run ]; then
  mkdir -p "$prefix/native/$NODE_SDK_TARGET"
  printf fixture > "$prefix/native/$NODE_SDK_TARGET/mesh_llm_nodejs.node"
fi
"#,
    )
    .unwrap();
    fs::set_permissions(npm, fs::Permissions::from_mode(0o700)).unwrap();
    bin.to_str().unwrap().to_owned()
}

fn check_layout(script: &str, target: &str, prefix: &str) {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("Node source with spaces");
    super::populate(&root, prefix);
    let (ok, paths, error) = super::invoke(&root);
    assert!(ok, "{error}");
    let node = root.join(&paths["sdk_dir"]).join("node");
    fs::create_dir(&node).unwrap();
    fs::write(node.join("package.json"), r#"{"version":"1.2.3"}"#).unwrap();
    let path = npm_boundary(&root);
    let calls = root.join("npm.calls");
    let values = [
        ("PATH", path.as_str()),
        ("SDK_DIR", paths["sdk_dir"].as_str()),
        ("NODE_SDK_TARGET", target),
        ("RELEASE_TAG", "v1.2.3"),
        ("PREPARE_RELEASE_VERSION", "false"),
        ("NPM_CALLS", calls.to_str().unwrap()),
    ];
    let (ok, _, error) = bash(&root, script, &values);
    assert!(ok, "{target}/{prefix}: {error}");
    assert_eq!(
        fs::read_to_string(&calls).unwrap(),
        format!(
            "test --prefix {}/node\nrun build:native --prefix {}/node\n",
            paths["sdk_dir"], paths["sdk_dir"]
        )
    );
    assert!(
        node.join("native")
            .join(target)
            .join("mesh_llm_nodejs.node")
            .is_file()
    );
    let mut invalid = values;
    invalid[3].1 = "v1.2.4";
    let (ok, _, error) = bash(&root, script, &invalid);
    assert!(
        !ok && error.contains("Node SDK version mismatch"),
        "{error}"
    );
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 2);
}

#[test]
fn all_three_addon_producers_bind_sdk_layout_before_version_checks_and_native_staging() {
    let tree = workflow("node-sdk-addon-artifact.yml");
    let mut checked = 0;
    for (job, _) in tree.get("jobs").unwrap().entries() {
        let steps = steps(&tree, job);
        if !steps.iter().any(|s| {
            s.get("name").and_then(crate::workflow_yaml::Node::text)
                == Some("Build, smoke, and stage immutable addon")
        }) {
            continue;
        }
        let build = named(steps, "Build, smoke, and stage immutable addon");
        assert_eq!(
            field(build.get("env").unwrap(), "SDK_DIR"),
            "${{ steps.layout.outputs.sdk_dir }}"
        );
        let layout = steps
            .iter()
            .position(|s| s.get("id").and_then(crate::workflow_yaml::Node::text) == Some("layout"))
            .unwrap();
        assert!(layout < steps.iter().position(|s| std::ptr::eq(s, build)).unwrap());
        let script = field(build, "run");
        let prefix = script
            .split_once("smoke_root=")
            .expect("native source lookup boundary")
            .0;
        let target = match job.as_str() {
            "linux_addon" => "linux-x64",
            "macos_addon" => "darwin-arm64",
            "windows_addon" => "win32-x64",
            _ => panic!("unreviewed addon producer {job}"),
        };
        for layout in ["", "mesh"] {
            check_layout(prefix, target, layout);
        }
        checked += 1;
    }
    assert_eq!(checked, 3);
}
