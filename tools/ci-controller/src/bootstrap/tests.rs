use super::dependency_boundary::{self, BoundaryViolation, ResolvedMetadata};
use super::{BootstrapTool, CONTROLLER, tool_binary};
use crate::DynResult;
use crate::temp_dir::unique_temp_dir;
use serde_json::{Value, json};
use std::fs;

/// An xtask-shaped descriptor: the tool plus two admitted local crates.
const TOOL: BootstrapTool = BootstrapTool {
    package: "xtask",
    local_packages: &["xtask", "mesh-llm-release-footer", "trajectory-reader"],
    usage: "usage: cargo xtool automation bootstrap",
    recipe: "just automation-bootstrap",
};

fn package(name: &str, local: bool, links: Option<&str>) -> Value {
    json!({
        "id": format!("{name}-id"),
        "name": name,
        "source": if local { Value::Null } else { json!("registry+https://github.com/rust-lang/crates.io-index") },
        "links": links,
    })
}

fn edge(to: &str, kind: Option<&str>) -> Value {
    json!({"pkg": format!("{to}-id"), "dep_kinds": [{"kind": kind, "target": null}]})
}

/// Given: xtask -> footer -> sha2 (normal edges) and xtask -> ui (dev edge).
fn bootstrap_graph() -> Value {
    json!({
        "target_directory": "/unused",
        "packages": [
            package("xtask", true, None),
            package("mesh-llm-release-footer", true, None),
            package("sha2", false, None),
            package("mesh-llm-ui", true, None),
        ],
        "resolve": {"nodes": [
            {"id": "xtask-id", "deps": [edge("mesh-llm-release-footer", None), edge("mesh-llm-ui", Some("dev"))]},
            {"id": "mesh-llm-release-footer-id", "deps": [edge("sha2", None)]},
            {"id": "sha2-id", "deps": []},
            {"id": "mesh-llm-ui-id", "deps": []},
        ]},
    })
}

fn inject(graph: &mut Value, from: &str, dependency: Value, dep_kind: Option<&str>) {
    let name = dependency["name"].as_str().unwrap_or_default().to_owned();
    graph["packages"].as_array_mut().unwrap().push(dependency);
    let nodes = graph["resolve"]["nodes"].as_array_mut().unwrap();
    nodes.push(json!({"id": format!("{name}-id"), "deps": []}));
    let parent = nodes
        .iter_mut()
        .find(|node| node["id"] == format!("{from}-id"))
        .unwrap();
    parent["deps"]
        .as_array_mut()
        .unwrap()
        .push(edge(&name, dep_kind));
}

fn parse(graph: Value) -> ResolvedMetadata {
    serde_json::from_value(graph).unwrap()
}

#[test]
fn migration_bootstrap_boundary_accepts_tool_closure_and_ignores_dev_edges() -> DynResult<()> {
    // When the compiled closure holds only xtask, the footer, and registry crates.
    let result = dependency_boundary::violations(&parse(bootstrap_graph()), &TOOL)?;
    // Then nothing is rejected, including the dev-only UI edge.
    assert_eq!(result, Vec::new());
    Ok(())
}

#[test]
fn migration_bootstrap_boundary_rejects_transitive_product_crate() -> DynResult<()> {
    // Given a native workspace crate injected behind the footer's normal edge.
    let mut graph = bootstrap_graph();
    inject(
        &mut graph,
        "mesh-llm-release-footer",
        package("skippy-ffi", true, None),
        None,
    );
    // When the boundary is evaluated.
    let result = dependency_boundary::violations(&parse(graph), &TOOL)?;
    // Then the transitive product crate is named.
    assert_eq!(
        result,
        vec![BoundaryViolation::WorkspaceCrate {
            package: "skippy-ffi".into()
        }]
    );
    Ok(())
}

#[test]
fn migration_bootstrap_boundary_rejects_registry_native_link_via_build_edge() {
    // Given a registry crate linking a native library through a build edge.
    let mut graph = bootstrap_graph();
    inject(
        &mut graph,
        "xtask",
        package("aws-lc-sys", false, Some("aws_lc")),
        Some("build"),
    );
    // When the command-level check runs.
    let error = dependency_boundary::check(&parse(graph), &TOOL).unwrap_err();
    // Then the diagnostic names the link and its package.
    assert!(
        error
            .to_string()
            .contains("native library link `aws_lc` via `aws-lc-sys`"),
        "{error}"
    );
}

#[test]
fn migration_bootstrap_boundary_requires_resolution() {
    // Given metadata produced with --no-deps (no resolve graph).
    let mut graph = bootstrap_graph();
    graph["resolve"] = Value::Null;
    // When the boundary is evaluated, then it fails closed.
    let error = dependency_boundary::violations(&parse(graph), &TOOL).unwrap_err();
    assert!(
        error.to_string().contains("dependency resolution"),
        "{error}"
    );
}

#[test]
fn migration_bootstrap_boundary_accepts_portable_trajectory_selection_contracts() -> DynResult<()> {
    let mut graph = bootstrap_graph();
    inject(
        &mut graph,
        "xtask",
        package("trajectory-reader", true, None),
        None,
    );
    dependency_boundary::check(&parse(graph), &TOOL)
}

#[test]
fn migration_bootstrap_boundary_still_rejects_codec_link_when_reader_feature_is_enabled() {
    let mut graph = bootstrap_graph();
    inject(
        &mut graph,
        "xtask",
        package("trajectory-reader", true, None),
        None,
    );
    inject(
        &mut graph,
        "trajectory-reader",
        package("zstd-sys", false, Some("zstd")),
        None,
    );
    let error = dependency_boundary::check(&parse(graph), &TOOL).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("native library link `zstd` via `zstd-sys`"),
        "{error}"
    );
}

#[test]
fn migration_bootstrap_tool_binary_accepts_profile_output() -> DynResult<()> {
    // Given a clean target root containing <target>/debug/xtask.
    let target = unique_temp_dir("xtask-bootstrap-target");
    fs::create_dir_all(target.join("debug"))?;
    let binary = target.join("debug/xtask");
    fs::write(&binary, b"tool")?;
    // When the running executable is that output.
    let resolved = tool_binary(&target, &binary, TOOL.recipe)?;
    // Then its absolute canonical path is returned.
    assert_eq!(resolved, binary.canonicalize()?);
    assert!(resolved.is_absolute());
    fs::remove_dir_all(target)?;
    Ok(())
}

#[test]
fn migration_bootstrap_tool_binary_rejects_copy_outside_target() -> DynResult<()> {
    // Given a built target root and a copied tool elsewhere.
    let target = unique_temp_dir("xtask-bootstrap-target");
    let elsewhere = unique_temp_dir("xtask-bootstrap-copy");
    fs::create_dir_all(target.join("debug"))?;
    fs::create_dir_all(&elsewhere)?;
    let copy = elsewhere.join("xtask");
    fs::write(&copy, b"tool")?;
    // When the copy claims to be the bootstrap output.
    let error = tool_binary(&target, &copy, TOOL.recipe).unwrap_err();
    // Then it is rejected with the rebuild instruction.
    assert!(
        error.to_string().contains("just automation-bootstrap"),
        "{error}"
    );
    fs::remove_dir_all(target)?;
    fs::remove_dir_all(elsewhere)?;
    Ok(())
}

#[test]
fn migration_bootstrap_tool_binary_rejects_unbuilt_target_root() {
    // Given a target directory that does not exist yet.
    let target = unique_temp_dir("xtask-bootstrap-unbuilt");
    // When a bootstrap path is requested, then the error says to build first.
    let error = tool_binary(&target, &target.join("debug/xtask"), TOOL.recipe).unwrap_err();
    assert!(error.to_string().contains("is not built"), "{error}");
}

#[test]
fn controller_boundary_admits_only_its_own_package() -> DynResult<()> {
    // Given ci-controller -> serde (registry) and ci-controller -> footer (local).
    let graph = json!({
        "packages": [
            package("ci-controller", true, None),
            package("serde", false, None),
            package("mesh-llm-release-footer", true, None),
        ],
        "resolve": {"nodes": [
            {"id": "ci-controller-id", "deps": [edge("serde", None), edge("mesh-llm-release-footer", None)]},
            {"id": "serde-id", "deps": []},
            {"id": "mesh-llm-release-footer-id", "deps": []},
        ]},
        "target_directory": "/tmp/target",
    });
    // When the controller descriptor checks that closure.
    let result = dependency_boundary::violations(&parse(graph), &CONTROLLER)?;
    // Then the local crate xtask may compile is still foreign to the controller.
    assert_eq!(
        result,
        vec![BoundaryViolation::WorkspaceCrate {
            package: "mesh-llm-release-footer".into()
        }]
    );
    Ok(())
}
