//! Repository identity and both cache decisions must survive caller projection.
use super::Node;
use std::collections::BTreeMap;
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn expected(node: &Node, key: &str, value: &str) -> Result<(), String> {
    if text(node, key) != value {
        return Err(format!("{key} must bind {value}"));
    }
    Ok(())
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let mut selectors = 0;
    for (name, workflow) in workflows {
        for (_, job) in workflow.get("jobs").ok_or("jobs missing")?.entries() {
            let Some(Node::Seq(steps)) = job.get("steps") else {
                continue;
            };
            for step in steps {
                if text(step, "uses") != "./.github/actions/select-ci-runners" {
                    continue;
                }
                selectors += 1;
                let inputs = step.get("with").ok_or("runner selector inputs missing")?;
                for (key, value) in [
                    ("event_name", "${{ github.event_name }}"),
                    ("repository", "${{ github.repository }}"),
                    (
                        "head_repository",
                        "${{ github.event.pull_request.head.repo.full_name }}",
                    ),
                    ("ref", "${{ github.ref }}"),
                ] {
                    expected(inputs, key, value).map_err(|e| format!("{name}: {e}"))?;
                }
                let sha = if name == "release.yml" {
                    "${{ github.sha }}"
                } else {
                    "${{ github.event.pull_request.head.sha || github.sha }}"
                };
                expected(inputs, "head_sha", sha).map_err(|e| format!("{name}: {e}"))?;
            }
        }
    }
    if selectors == 0 {
        return Err("runner selector callers missing".into());
    }
    Ok(())
}
pub(super) fn nested_windows(action: &Node) -> Result<(), String> {
    let input = action
        .get("inputs")
        .and_then(|v| v.get("allow-native-github-cache"))
        .ok_or("nested native cache authority input missing")?;
    expected(input, "default", "false")?;
    let Some(Node::Seq(steps)) = action.get("runs").and_then(|v| v.get("steps")) else {
        return Err("nested cache steps missing".into());
    };
    let mut count = 0;
    for step in steps {
        if text(step, "uses").starts_with("actions/cache") {
            if !super::cache_predicate::requires(
                text(step, "if"),
                "inputs.allow-native-github-cache == 'true'",
            ) {
                return Err(
                    "nested Windows cache must require explicit native authorization".into(),
                );
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err("nested cache operation missing".into());
    }
    Ok(())
}
