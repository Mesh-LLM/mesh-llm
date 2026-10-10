use super::*;
use std::{fs, path::Path};

#[test]
fn lane_summary_cardinality_owns_parsed_and_constructed_graph_admission() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = fs::read_to_string(root.join(".github/workflows/ci-quality-lane.yml")).unwrap();
    let tree = super::super::workflow_yaml::parse(&source).unwrap();
    assert!(LaneGraph::parse(&tree, Lane::Quality).is_ok());
    let jobs = tree.get("jobs").unwrap().entries();
    let summary = jobs.iter().find(|(key, _)| key == SUMMARY).unwrap().clone();
    for (count, mut entries) in [
        (
            0,
            jobs.iter()
                .filter(|(key, _)| key != SUMMARY)
                .cloned()
                .collect::<Vec<_>>(),
        ),
        (2, jobs.to_vec()),
    ] {
        if count == 2 {
            entries.push(summary.clone());
        }
        let graph = Node::Map(vec![("jobs".into(), Node::Map(entries))]);
        let error = LaneGraph::parse(&graph, Lane::Quality).err().unwrap();
        assert!(error.contains("exactly one summary job"), "{error}");
        assert!(error.contains(&format!("found {count}")), "{error}");
    }
}
