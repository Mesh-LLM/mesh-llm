//! Parsed contracts of the event entrypoints: five native PR workflows that
//! call the protected lane, five native main workflows that call the
//! same-commit lane, and a dispatch-only manual controller that verifies the
//! plan digest before dispatching.

use super::Checked;
use super::results::Lane;
use super::workflow_yaml::{self, Node};
use std::fs;
use std::path::Path;

#[derive(Clone, Copy)]
enum Entry {
    Pr,
    Main,
}

impl Entry {
    fn prefix(self) -> &'static str {
        match self {
            Entry::Pr => "pr",
            Entry::Main => "main",
        }
    }

    fn check_label(self) -> &'static str {
        match self {
            Entry::Pr => "PR",
            Entry::Main => "Main",
        }
    }

    fn lane_call(self, lane: Lane) -> String {
        let name = lane.name();
        match self {
            Entry::Pr => format!("Mesh-LLM/mesh-llm/.github/workflows/ci-{name}-lane.yml@main"),
            Entry::Main => format!("./.github/workflows/ci-{name}-lane.yml"),
        }
    }
}

fn load(workflows: &Path, file: &str) -> Checked<Node> {
    let source =
        fs::read_to_string(workflows.join(file)).map_err(|error| format!("{file}: {error}"))?;
    workflow_yaml::parse(&source).map_err(|error| format!("{file}: {error}"))
}

pub(super) fn validate(workflows: &Path) -> Checked<()> {
    for entry in [Entry::Pr, Entry::Main] {
        for lane in Lane::ALL {
            let file = format!("{}_{}.yml", entry.prefix(), lane.name());
            let tree = load(workflows, &file)?;
            entrypoint(&tree, entry, lane).map_err(|error| format!("{file}: {error}"))?;
        }
    }
    let controller = load(workflows, "ci-control.yml")?;
    self::controller(&controller).map_err(|error| format!("ci-control.yml: {error}"))
}

fn job<'a>(tree: &'a Node, name: &str) -> Checked<&'a Node> {
    tree.get("jobs")
        .and_then(|jobs| jobs.get(name))
        .ok_or_else(|| format!("missing job {name}"))
}

fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}

fn trigger(tree: &Node, entry: Entry) -> Checked<()> {
    let on = tree.get("on").map(Node::entries).unwrap_or_default();
    let names: Vec<&str> = on.iter().map(|(name, _)| name.as_str()).collect();
    match entry {
        Entry::Pr => {
            if names != ["pull_request"] {
                return Err("must be triggered only by pull_request".to_owned());
            }
            let event = &on[0].1;
            if event.get("paths").is_some() || event.get("paths-ignore").is_some() {
                return Err("pull_request trigger must not filter paths".to_owned());
            }
            let concurrency = tree.get("concurrency");
            let cancels = concurrency.map(|c| text(c, "cancel-in-progress")) == Some("true");
            let per_pr = concurrency.is_some_and(|c| {
                text(c, "group").ends_with("${{ github.event.pull_request.number }}")
            });
            if !(cancels && per_pr) {
                return Err("concurrency must cancel superseded runs of the same PR".to_owned());
            }
        }
        Entry::Main => {
            let branches = on
                .iter()
                .find(|(name, _)| name == "push")
                .and_then(|(_, push)| push.get("branches"))
                .map(Node::list);
            if names != ["push"] || branches != Some(vec!["main"]) {
                return Err("must be triggered only by pushes to main".to_owned());
            }
            if tree.get("concurrency").is_some() {
                return Err("main entrypoints must not declare concurrency".to_owned());
            }
        }
    }
    Ok(())
}

fn entrypoint(tree: &Node, entry: Entry, lane: Lane) -> Checked<()> {
    trigger(tree, entry)?;
    let jobs: Vec<&str> = tree
        .get("jobs")
        .map(Node::entries)
        .unwrap_or_default()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    if jobs != ["plan", "lane", "required"] {
        return Err("jobs must be exactly plan, lane and required".to_owned());
    }
    let plan = job(tree, "plan")?;
    let expected = format!("${{{{ steps.plan.outputs.{}_lane_plan }}}}", lane.name());
    let outputs = plan.get("outputs");
    if outputs.map(|o| text(o, "lane_plan")) != Some(expected.as_str()) {
        return Err(format!("plan output lane_plan must be {expected}"));
    }
    if outputs.map(|o| text(o, "plan_digest")) != Some("${{ steps.plan.outputs.plan_digest }}") {
        return Err("plan output plan_digest must come from the canonical planner".to_owned());
    }
    let call = job(tree, "lane")?;
    if text(call, "uses") != entry.lane_call(lane) {
        return Err(format!("lane job must call {}", entry.lane_call(lane)));
    }
    lane_secrets(call.get("secrets"), entry)?;
    let with = call.get("with");
    let passes = |key: &str, value: &str| with.map(|w| text(w, key)) == Some(value);
    if !passes("lane_plan_json", "${{ needs.plan.outputs.lane_plan }}")
        || !passes("plan_digest", "${{ needs.plan.outputs.plan_digest }}")
    {
        return Err("lane job must receive the digest-bound lane plan".to_owned());
    }
    required(job(tree, "required")?, entry, lane)
}

/// PR lanes receive no secrets. Trusted main passes each secret by name to
/// the same-commit lane, never `inherit`.
fn lane_secrets(secrets: Option<&Node>, entry: Entry) -> Checked<()> {
    match (secrets, entry) {
        (None, _) => Ok(()),
        (Some(_), Entry::Pr) => Err("PR lane job must not pass secrets".to_owned()),
        (Some(Node::Map(entries)), Entry::Main) => {
            match entries.iter().find(|(name, value)| {
                value.text() != Some(format!("${{{{ secrets.{name} }}}}").as_str())
            }) {
                Some((name, _)) => Err(format!("lane job must pass secret {name} by its own name")),
                None => Ok(()),
            }
        }
        (Some(_), Entry::Main) => Err("lane job must pass named secrets, never inherit".to_owned()),
    }
}

fn required(job: &Node, entry: Entry, lane: Lane) -> Checked<()> {
    let name = format!("{} / {}", entry.check_label(), lane.label());
    if text(job, "name") != name {
        return Err(format!("required job must be named '{name}'"));
    }
    let needs = job.get("needs").map(Node::list).unwrap_or_default();
    if needs != ["plan", "lane"] || text(job, "if") != "${{ !cancelled() }}" {
        return Err("required job must need plan and lane and run unless cancelled".to_owned());
    }
    Ok(())
}

fn controller(tree: &Node) -> Checked<()> {
    let on = tree.get("on").map(Node::entries).unwrap_or_default();
    if on.len() != 1 || on[0].0 != "workflow_dispatch" {
        return Err("controller must be triggered only by workflow_dispatch".to_owned());
    }
    let dispatch = job(tree, "dispatch")?;
    let script = match dispatch.get("steps") {
        Some(Node::Seq(steps)) => steps
            .iter()
            .filter_map(|step| step.get("with"))
            .map(|with| text(with, "script"))
            .find(|script| script.contains("createWorkflowDispatch"))
            .unwrap_or(""),
        _ => "",
    };
    let verify = script.find("digest !== process.env.PLAN_DIGEST");
    let dispatch_at = script.find("createWorkflowDispatch");
    match (verify, dispatch_at) {
        (Some(verify), Some(dispatch)) if verify < dispatch => {}
        _ => return Err("controller must verify the plan digest before dispatch".to_owned()),
    }
    for lane in Lane::ALL {
        if !script.contains(&format!("'ci-{}-lane.yml'", lane.name())) {
            return Err(format!(
                "controller must dispatch ci-{}-lane.yml",
                lane.name()
            ));
        }
    }
    Ok(())
}
