//! Remaining local authority assertions: source projection and actual retained adapters.
use crate::{
    support,
    workflow_yaml::{self, Node},
};
use std::{collections::BTreeMap, fs};
fn workflows() -> BTreeMap<String, Node> {
    fs::read_dir(support::root().join(".github/workflows"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "yml" || e == "yaml"))
        .map(|p| {
            (
                p.file_name().unwrap().to_str().unwrap().to_owned(),
                workflow_yaml::parse(&fs::read_to_string(p).unwrap()).unwrap(),
            )
        })
        .collect()
}
fn job<'a>(documents: &'a BTreeMap<String, Node>, workflow: &str, name: &str) -> &'a Node {
    documents[workflow].get("jobs").unwrap().get(name).unwrap()
}
fn steps(job: &Node) -> &[Node] {
    let Node::Seq(steps) = job.get("steps").unwrap() else {
        panic!("steps")
    };
    steps
}
fn body(workflow: &str, name: &str, step: &str) -> String {
    let documents = workflows();
    steps(job(&documents, workflow, name))
        .iter()
        .find(|s| s.get("name").and_then(Node::text) == Some(step))
        .unwrap()
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .to_owned()
}
fn replace(node: &mut Node, path: &[&str], value: Node) {
    let Node::Map(entries) = node else {
        panic!("mapping")
    };
    let (_, child) = entries.iter_mut().find(|(k, _)| k == path[0]).unwrap();
    if path.len() == 1 {
        *child = value
    } else {
        replace(child, &path[1..], value)
    }
}
#[path = "phase_endpoints.rs"]
mod phase_endpoints;
#[path = "resource_audit.rs"]
mod resource_audit;
#[path = "selector.rs"]
mod selector;
