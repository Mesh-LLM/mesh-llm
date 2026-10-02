use super::*;
use crate::ci_validation::lane_results::workflow_yaml;
use std::{fs, path::Path};
fn actual() -> BTreeMap<String, Node> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    BTreeMap::from([(
        "llama-upstream-canary.yml".into(),
        workflow_yaml::parse(
            &fs::read_to_string(root.join(".github/workflows/llama-upstream-canary.yml")).unwrap(),
        )
        .unwrap(),
    )])
}
fn field_mut<'a>(node: &'a mut Node, key: &str) -> &'a mut Node {
    let Node::Map(fields) = node else {
        panic!("mapping required")
    };
    &mut fields.iter_mut().find(|(name, _)| name == key).unwrap().1
}
fn job(workflows: &mut BTreeMap<String, Node>) -> &mut Node {
    field_mut(
        field_mut(
            workflows.get_mut("llama-upstream-canary.yml").unwrap(),
            "jobs",
        ),
        "resolve",
    )
}
fn steps(workflows: &mut BTreeMap<String, Node>) -> &mut Vec<Node> {
    let Node::Seq(steps) = field_mut(job(workflows), "steps") else {
        panic!("steps required")
    };
    steps
}
fn resolve(workflows: &mut BTreeMap<String, Node>) -> &mut Node {
    steps(workflows)
        .iter_mut()
        .find(|step| field(step, "id") == Some("resolve"))
        .unwrap()
}
#[test]
fn actual_protected_controller_bootstrap_and_selected_output_order_are_admitted() {
    check(&actual()).unwrap();
}
#[test]
fn unprotected_checkout_missing_or_late_bootstrap_and_selected_prepublication_are_rejected() {
    let mut workflows = actual();
    *field_mut(job(&mut workflows), "if") = Node::Scalar("true".into());
    assert!(check(&workflows).is_err());
    let mut workflows = actual();
    let step_nodes = steps(&mut workflows);
    let bootstrap = step_nodes
        .iter()
        .position(|step| field(step, "uses") == Some("./.github/actions/prepare-automation"))
        .unwrap();
    let resolver = step_nodes
        .iter()
        .position(|step| field(step, "id") == Some("resolve"))
        .unwrap();
    step_nodes.swap(bootstrap, resolver);
    assert!(check(&workflows).is_err());
    let mut workflows = actual();
    steps(&mut workflows)
        .retain(|step| field(step, "uses") != Some("./.github/actions/prepare-automation"));
    assert!(check(&workflows).is_err());
    for output in ["mesh_source=fake", "certify=true", "mode=pinned-build"] {
        let mut workflows = actual();
        let step = resolve(&mut workflows);
        let run = format!(
            "echo '{output}' >> \"$GITHUB_OUTPUT\"\n{}",
            field(step, "run").unwrap()
        );
        *field_mut(step, "run") = Node::Scalar(run);
        assert!(check(&workflows).is_err());
    }
}
#[test]
fn selected_compiler_bootstrap_authority_and_masked_owner_execution_are_rejected() {
    let mut workflows = actual();
    let checkout = steps(&mut workflows)
        .iter_mut()
        .find(|step| {
            field(step, "uses").is_some_and(|action| action.starts_with("actions/checkout@"))
        })
        .unwrap();
    *field_mut(field_mut(checkout, "with"), "ref") = Node::Scalar("${{ inputs.mesh_ref }}".into());
    assert!(check(&workflows).is_err());
    let mut workflows = actual();
    let step = resolve(&mut workflows);
    let run = field(step, "run").unwrap().replace(
        "--summary \"$GITHUB_STEP_SUMMARY\"",
        "--summary \"$GITHUB_STEP_SUMMARY\" || true",
    );
    *field_mut(step, "run") = Node::Scalar(run);
    assert!(check(&workflows).is_err());
    let mut workflows = actual();
    let step = resolve(&mut workflows);
    let run = field(step, "run").unwrap().replace(
        "--expected-origin \"$GITHUB_SERVER_URL/$GITHUB_REPOSITORY\"",
        "--expected-origin \"$MESH_REF\"",
    );
    *field_mut(step, "run") = Node::Scalar(run);
    assert!(check(&workflows).is_err());
}
