//! The host's raw JSON contract must not rely on another crate's feature union.

#[test]
fn host_declares_its_raw_json_dependency_feature() {
    let manifest: toml::Value = toml::from_str(include_str!("../Cargo.toml")).unwrap();
    let features = manifest["dependencies"]["serde_json"]
        .get("features")
        .and_then(toml::Value::as_array);

    assert!(features.is_some_and(|features| {
        features
            .iter()
            .any(|feature| feature.as_str() == Some("raw_value"))
    }));
}

#[test]
fn host_normal_dependencies_exclude_cli_model_commands_and_packaging() {
    let rustc = std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("read rustc host target");
    assert!(
        rustc.status.success(),
        "{}",
        String::from_utf8_lossy(&rustc.stderr)
    );
    let version = String::from_utf8(rustc.stdout).unwrap();
    let host = version
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("rustc reports its host target");
    let output = std::process::Command::new(env!("CARGO"))
        .args(["metadata", "--locked", "--offline", "--format-version", "1"])
        .args(["--filter-platform", host])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("read resolved Cargo graph");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let packages = metadata["packages"].as_array().unwrap();
    let root = packages
        .iter()
        .find(|package| package["name"] == "mesh-llm-host-runtime")
        .unwrap()["id"]
        .as_str()
        .unwrap();
    let nodes = metadata["resolve"]["nodes"].as_array().unwrap();
    let mut pending = vec![root];
    let mut reached = std::collections::BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !reached.insert(id) {
            continue;
        }
        let node = nodes.iter().find(|node| node["id"] == id).unwrap();
        for dependency in node["deps"].as_array().unwrap() {
            if dependency["dep_kinds"]
                .as_array()
                .unwrap()
                .iter()
                .any(|kind| kind["kind"].is_null())
            {
                pending.push(dependency["pkg"].as_str().unwrap());
            }
        }
    }
    for package in packages {
        if matches!(
            package["name"].as_str(),
            Some("skippy-commands" | "skippy-model-package")
        ) {
            assert!(
                !reached.contains(package["id"].as_str().unwrap()),
                "host normal dependency graph reaches {}",
                package["name"]
            );
        }
    }
}
