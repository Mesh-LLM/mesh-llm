use super::{Node, handoffs as h};
use crate::command::DynResult;
use std::collections::{BTreeMap, BTreeSet};

const PLACEMENTS: [(&str, &str, &str); 4] = [
    (
        "llama-upstream-canary.yml",
        "preflight",
        "accelerator-memory-128plus",
    ),
    (
        "llama-canary-family-pass.yml",
        "build",
        "accelerator-memory-128plus",
    ),
    (
        "llama-canary-family-pass.yml",
        "family",
        "${{ matrix.memory_tier }}",
    ),
    (
        "llama-canary-family-pass.yml",
        "retry_family",
        "${{ matrix.memory_tier }}",
    ),
];

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    for (workflow, job, tier) in PLACEMENTS {
        let document = workflows.get(workflow).ok_or("missing canary workflow")?;
        let placement = h::member(h::job(document, job)?, "runs-on")?;
        let Node::Seq(labels) = placement else {
            return Err(
                format!("{workflow}/{job}: canary placement must use managed labels").into(),
            );
        };
        let expected = [
            "self-hosted",
            "macOS",
            "ARM64",
            "jy-managed",
            "family-certify",
            tier,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let actual = labels
            .iter()
            .filter_map(Node::text)
            .collect::<BTreeSet<_>>();
        if actual != expected || labels.len() != expected.len() {
            return Err(format!("{workflow}/{job}: canary placement must be {expected:?}").into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_validation::lane_results::workflow_yaml;

    const UPSTREAM: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/workflows/llama-upstream-canary.yml"
    ));
    const FAMILY: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/workflows/llama-canary-family-pass.yml"
    ));

    fn member_mut<'a>(node: &'a mut Node, name: &str) -> &'a mut Node {
        let Node::Map(entries) = node else {
            panic!("expected mapping")
        };
        &mut entries.iter_mut().find(|(key, _)| key == name).unwrap().1
    }

    #[test]
    fn managed_placement_rejects_pool_fallbacks_and_node_pinning() {
        let workflows = BTreeMap::from([
            (
                "llama-upstream-canary.yml".into(),
                workflow_yaml::parse_resolved_aliases(UPSTREAM).unwrap(),
            ),
            (
                "llama-canary-family-pass.yml".into(),
                workflow_yaml::parse(FAMILY).unwrap(),
            ),
        ]);
        check(&workflows).unwrap();
        for (workflow, job, tier) in PLACEMENTS {
            for (label, replacement) in [
                ("jy-managed", "other-pool"),
                ("macOS", "Linux"),
                ("ARM64", "X64"),
                ("family-certify", "named-node"),
                (tier, "accelerator-memory-64plus"),
            ] {
                let mut changed = workflows.clone();
                let jobs = member_mut(changed.get_mut(workflow).unwrap(), "jobs");
                let Node::Seq(labels) = member_mut(member_mut(jobs, job), "runs-on") else {
                    panic!("expected runner labels");
                };
                *labels
                    .iter_mut()
                    .find(|value| value.text() == Some(label))
                    .unwrap() = Node::Scalar(replacement.into());
                assert!(check(&changed).is_err(), "{workflow}/{job}: {label}");
            }
        }
    }
}
