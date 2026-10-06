//! Repository identity and both cache decisions must survive caller projection.
use super::Node;
use std::collections::BTreeMap;
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn expected(node: &Node, key: &str, value: &str) -> Result<(), String> {
    if node.get(key).and_then(Node::text) != Some(value) {
        return Err(format!("{key} must bind {value}"));
    }
    Ok(())
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    eligible(workflows)?;
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
const ELIGIBLE: &[&str] = &[
    "ci-quality-slice.yml",
    "ci-web-slice.yml",
    "ci-ui-artifact-slice.yml",
    "ci-rust-tests-slice.yml",
    "ci-linux-host-slice.yml",
    "ci-linux-runtime-slice.yml",
    "ci-linux-product-slice.yml",
    "static-abi-artifact.yml",
    "ci-macos-host-slice.yml",
    "ci-macos-runtime-slice.yml",
    "ci-macos-product-slice.yml",
    "swift-sdk-artifact.yml",
    "ci-platform-checks-slice.yml",
    "ci-windows-host-slice.yml",
    "ci-windows-runtime-slice.yml",
    "ci-windows-product-slice.yml",
    "native-sdk-artifact.yml",
];
fn eligible(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    for name in ELIGIBLE {
        let job = workflows
            .get(*name)
            .and_then(|w| w.get("jobs"))
            .and_then(|j| j.get("runner_policy"))
            .ok_or_else(|| format!("{name}: bounded runner policy missing"))?;
        let Some(Node::Seq(steps)) = job.get("steps") else {
            return Err(format!("{name}: runner policy steps absent"));
        };
        let selectors: Vec<_> = steps
            .iter()
            .filter(|s| text(s, "uses") == "./.github/actions/select-ci-runners")
            .collect();
        let expected_count = if *name == "ci-quality-slice.yml" {
            2
        } else {
            1
        };
        if selectors.len() != expected_count {
            return Err(format!("{name}: bounded policy selector census changed"));
        }
        if selectors
            .iter()
            .filter(|s| text(s, "id") == "policy")
            .count()
            != 1
            || selectors
                .iter()
                .filter(|s| text(s, "id") == "sentinel_policy")
                .count()
                != expected_count - 1
        {
            return Err(format!(
                "{name}: ordinary/sentinel selector identity changed"
            ));
        }
        for selector in selectors {
            bounded_inputs(name, selector)?;
        }
    }
    Ok(())
}
fn bounded_inputs(name: &str, selector: &Node) -> Result<(), String> {
    let inputs = selector
        .get("with")
        .ok_or("bounded selector inputs absent")?;
    for (key, value) in [
        ("event_name", "${{ github.event_name }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        ("repository", "${{ github.repository }}"),
        (
            "head_repository",
            "${{ github.event.pull_request.head.repo.full_name }}",
        ),
        (
            "head_sha",
            "${{ github.event.pull_request.head.sha || github.sha }}",
        ),
        ("ref", "${{ github.ref }}"),
        ("force_hosted", "${{ inputs.force_hosted }}"),
    ] {
        expected(inputs, key, value)?;
    }
    let sentinel = text(selector, "id") == "sentinel_policy";
    if sentinel {
        if name != "ci-quality-slice.yml" {
            return Err("dedicated sentinel cannot replace ordinary policy".into());
        }
        for (key, value) in [
            ("depot_main_enabled", "false"),
            ("depot_pr_enabled", "false"),
            ("pr_canary_ref", "${{ vars.DEPOT_PR_SENTINEL_REF }}"),
            ("pr_approved_ref", ""),
            ("pr_approved_sha", ""),
            ("manual_use_depot", "false"),
        ] {
            expected(inputs, key, value)?;
        }
    } else {
        expected(selector, "id", "policy")?;
        for (key, value) in [
            (
                "depot_pr_enabled",
                "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            ),
            ("pr_canary_ref", "${{ vars.DEPOT_PR_CANARY_REF }}"),
            ("pr_approved_ref", "${{ vars.DEPOT_PR_APPROVED_REF }}"),
            ("pr_approved_sha", "${{ vars.DEPOT_PR_APPROVED_SHA }}"),
        ] {
            expected(inputs, key, value)?;
        }
        if !["false", "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}"]
            .contains(&text(inputs, "depot_main_enabled"))
        {
            return Err("ordinary main admission must be false or repository gate".into());
        }
        if inputs.get("manual_use_depot").is_some() {
            expected(inputs, "manual_use_depot", "${{ inputs.use_depot }}")?;
        }
    }
    if selector.get("if").is_some() || selector.get("continue-on-error").is_some() {
        return Err("bounded selector cannot be bypassed".into());
    }
    Ok(())
}
