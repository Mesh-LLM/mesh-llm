//! Producer reachability, platform adapters and smoke input contracts.
mod graph;
mod planner;
mod platform;
mod smoke;
use super::{
    support,
    workflow_yaml::{self, Node},
};
use std::fs;

fn source(file: &str) -> String {
    fs::read_to_string(support::root().join(format!(".github/workflows/{file}"))).unwrap()
}
fn document(file: &str) -> Node {
    workflow_yaml::parse(&source(file)).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn job<'a>(node: &'a Node, key: &str) -> &'a Node {
    node.get("jobs").unwrap().get(key).unwrap()
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(steps) = node.get("steps").unwrap() else {
        panic!("step sequence")
    };
    steps
}
fn named<'a>(node: &'a Node, name: &str) -> &'a Node {
    let matches = steps(node)
        .iter()
        .filter(|step| text(step, "name") == Some(name))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one {name} required");
    matches[0]
}
fn input<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get("with").and_then(|inputs| text(inputs, key))
}
fn assert_mutation_rejected(file: &str, from: &str, to: &str, check: impl Fn(&Node) -> bool) {
    let original = source(file);
    assert!(original.contains(from), "mutation anchor {from}");
    let changed = format!("{}\n# {from}\n", original.replace(from, to));
    assert!(!check(&workflow_yaml::parse(&changed).unwrap()));
}

mod cache_flow_intent;
mod declared_graph;
mod lane_policy_intent;
mod release_flow_intent;
mod release_ui_step;
mod release_windows_intent;
mod reporter_intent;
