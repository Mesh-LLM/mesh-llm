//! The diagnostic may execute protected downloaded automation, never PR code or secrets.
use crate::{
    cache_authority, support,
    workflow_yaml::{self, Node},
};
#[test]
fn authority_sentinel_refuses_hidden_secret_source_and_compiler_projections() {
    let document = workflow_yaml::parse(
        &std::fs::read_to_string(support::root().join(".github/workflows/ci-quality-slice.yml"))
            .unwrap(),
    )
    .unwrap();
    let original = document
        .get("jobs")
        .unwrap()
        .get("authority_sentinel")
        .unwrap();
    cache_authority::sentinel(original).unwrap();
    for (key, value) in [
        ("env", "${{ secrets.PRIVATE_FIXTURE }}"),
        ("env", "${{ inputs.source_sha }}"),
        ("env", "${{ github.event.pull_request.head.sha }}"),
        ("with", "${{ secrets.PRIVATE_FIXTURE }}"),
        ("run", "set -euo pipefail\ncargo build --release"),
        ("run", "rustc source.rs"),
        ("run", "git checkout private-fixture-head"),
        (
            "uses",
            "Mesh-LLM/mesh-llm/.github/actions/audit-depot-pr-isolation@private-fixture",
        ),
    ] {
        let mut changed = original.clone();
        let Node::Map(job) = &mut changed else {
            panic!("job")
        };
        let (_, Node::Seq(steps)) = job.iter_mut().find(|(k, _)| k == "steps").unwrap() else {
            panic!("steps")
        };
        let payload = if matches!(key, "env" | "with") {
            Node::Map(vec![("EXTRA".into(), Node::Scalar(value.into()))])
        } else {
            Node::Scalar(value.into())
        };
        steps.push(Node::Map(vec![(key.into(), payload)]));
        assert!(
            cache_authority::sentinel(&changed).is_err(),
            "{key}/{value}"
        );
    }
}
