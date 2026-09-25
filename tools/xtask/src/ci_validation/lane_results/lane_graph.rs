//! Parsed contracts of one reusable lane workflow: native triggers, PR-only
//! supersession and fail-fast, push-only secrets, one stable summary that
//! needs every slice call and reports through the correlated action, and a
//! producer/consumer graph in which every read of `needs.X` and every
//! consumed artifact name is reachable through declared `needs`.

use super::Checked;
use super::results::Lane;
use super::workflow_yaml::Node;
use std::collections::BTreeSet;

pub(super) const PR_ONLY: &str = "${{ inputs.original_event_name == 'pull_request' }}";
const SUMMARY: &str = "summary";
const REPORTER: &str = "./.github/actions/report-ci-lane";

pub(super) struct LaneGraph<'a> {
    jobs: &'a [(String, Node)],
}

impl<'a> LaneGraph<'a> {
    /// Checks everything that does not depend on a plan.
    pub(super) fn parse(tree: &'a Node, lane: Lane) -> Checked<Self> {
        let graph = Self {
            jobs: tree.get("jobs").map(Node::entries).unwrap_or_default(),
        };
        graph.identity(lane)?;
        triggers(tree)?;
        concurrency(tree)?;
        graph.summary()?;
        for (name, job) in graph.slices() {
            graph.job(name, job)?;
        }
        Ok(graph)
    }

    fn job_named(&self, name: &str) -> Option<&'a Node> {
        self.jobs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, job)| job)
    }

    fn slices(&self) -> impl Iterator<Item = (&'a str, &'a Node)> {
        self.jobs
            .iter()
            .filter(|(name, _)| name != SUMMARY)
            .map(|(name, job)| (name.as_str(), job))
    }

    fn identity(&self, lane: Lane) -> Checked<()> {
        let expected = format!("CI / {}", lane.label());
        let actual = self
            .job_named(SUMMARY)
            .and_then(|job| job.get("name"))
            .and_then(Node::text)
            .unwrap_or("");
        if actual == expected {
            return Ok(());
        }
        Err(format!(
            "lane workflow summary is '{actual}', expected '{expected}'"
        ))
    }

    fn summary(&self) -> Checked<()> {
        let summary = self
            .job_named(SUMMARY)
            .ok_or("lane workflow has no summary job")?;
        if summary.get("if").and_then(Node::text) != Some("${{ !cancelled() }}") {
            return Err("summary must run with if: ${{ !cancelled() }}".to_owned());
        }
        let needs = needs(summary);
        if let Some((name, _)) = self.slices().find(|(name, _)| !needs.contains(name)) {
            return Err(format!("summary does not need lane job '{name}'"));
        }
        let reports = steps(summary).any(|step| {
            step.get("uses").and_then(Node::text) == Some(REPORTER)
                && with(step, "plan_digest") == Some("${{ inputs.plan_digest }}")
        });
        if !reports {
            return Err(format!(
                "summary must report the plan digest through {REPORTER}"
            ));
        }
        Ok(())
    }

    fn job(&self, name: &str, job: &Node) -> Checked<()> {
        let declared = needs(job);
        if let Some(read) = reads(job).into_iter().find(|read| !declared.contains(read)) {
            return Err(format!(
                "job '{name}' reads needs.{read} without declaring it"
            ));
        }
        self.artifacts(name, job)?;
        if with(job, "fail_fast").is_some_and(|value| value != PR_ONLY) {
            return Err(format!(
                "job '{name}' fail_fast must be enabled only for pull requests"
            ));
        }
        secrets(name, job)
    }

    /// Every `*_artifact_name` input names a lane producer's
    /// `artifact_name` that is reachable through `needs`.
    fn artifacts(&self, name: &str, job: &Node) -> Checked<()> {
        let inputs = job.get("with").map(Node::entries).unwrap_or_default();
        for (input, value) in inputs
            .iter()
            .filter(|(key, _)| key.ends_with("_artifact_name"))
        {
            let producer = self
                .slices()
                .find(|(_, candidate)| with(candidate, "artifact_name") == value.text())
                .map(|(producer, _)| producer)
                .ok_or_else(|| format!("consumer '{name}' input {input}: no lane producer"))?;
            if !self.ancestors(name).contains(producer) {
                return Err(format!(
                    "consumer '{name}' input {input}: producer '{producer}' is not on its needs path"
                ));
            }
        }
        Ok(())
    }

    fn ancestors(&self, name: &str) -> BTreeSet<&'a str> {
        let mut seen = BTreeSet::new();
        let mut queue = self.job_named(name).map(needs).unwrap_or_default();
        while let Some(next) = queue.pop() {
            if let Some((key, job)) = self.jobs.iter().find(|(key, _)| key == next)
                && seen.insert(key.as_str())
            {
                queue.extend(needs(job));
            }
        }
        seen
    }

    /// Planned jobs exist, and each hard `needs.X.result == 'success'`
    /// gate of a planned job names a planned producer.
    pub(super) fn check_planned(&self, planned: &[&str]) -> Checked<()> {
        if let Some(job) = planned.iter().find(|job| self.job_named(job).is_none()) {
            return Err(format!(
                "planned job '{job}' is not in the lane workflow graph"
            ));
        }
        for job in planned {
            let node = self.job_named(job).ok_or("planned job disappeared")?;
            let gate = node.get("if").and_then(Node::text).unwrap_or("");
            for producer in needs(node) {
                let hard = gate.contains(&format!("needs.{producer}.result == 'success'"))
                    && !gate.contains(&format!("|| needs.{producer}.result"));
                if hard && !planned.contains(&producer) {
                    return Err(format!(
                        "planned job '{job}' needs producer '{producer}', which is not planned"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn needs(job: &Node) -> Vec<&str> {
    job.get("needs").map(Node::list).unwrap_or_default()
}

fn with<'n>(node: &'n Node, key: &str) -> Option<&'n str> {
    node.get("with")
        .and_then(|with| with.get(key))
        .and_then(Node::text)
}

fn steps(job: &Node) -> impl Iterator<Item = &Node> {
    match job.get("steps") {
        Some(Node::Seq(steps)) => steps.iter(),
        _ => [].iter(),
    }
}

/// Job IDs read as `needs.<id>.` in the job's `if` expression.
fn reads(job: &Node) -> BTreeSet<&str> {
    let gate = job.get("if").and_then(Node::text).unwrap_or("");
    gate.match_indices("needs.")
        .filter(|(at, _)| !gate[..*at].ends_with(|ch: char| ch.is_alphanumeric() || ch == '_'))
        .filter_map(|(at, _)| gate[at + 6..].split('.').next())
        .filter(|id| !id.is_empty())
        .collect()
}

fn triggers(tree: &Node) -> Checked<()> {
    let on = tree.get("on");
    let has = |trigger: &str| on.and_then(|on| on.get(trigger)).is_some();
    if !(has("workflow_call") && has("workflow_dispatch")) {
        return Err("lane workflow must be both reusable and dispatchable".to_owned());
    }
    let inputs = on
        .and_then(|on| on.get("workflow_call"))
        .and_then(|call| call.get("inputs"));
    for input in [
        "lane_plan_json",
        "plan_digest",
        "source_sha",
        "original_event_name",
    ] {
        if inputs.and_then(|inputs| inputs.get(input)).is_none() {
            return Err(format!("lane workflow must accept input {input}"));
        }
    }
    Ok(())
}

fn concurrency(tree: &Node) -> Checked<()> {
    let concurrency = tree.get("concurrency");
    let field = |key: &str| concurrency.and_then(|c| c.get(key)).and_then(Node::text);
    let keyed = field("group")
        .is_some_and(|group| group.contains("${{ inputs.supersession_key || inputs.source_sha }}"));
    if keyed && field("cancel-in-progress") == Some(PR_ONLY) {
        return Ok(());
    }
    Err("lane concurrency must cancel only superseded pull requests".to_owned())
}

fn secrets(name: &str, job: &Node) -> Checked<()> {
    match job.get("secrets") {
        None => Ok(()),
        Some(Node::Map(entries)) => {
            for (secret, value) in entries {
                let gated = format!(
                    "${{{{ inputs.original_event_name == 'push' && secrets.{secret} || '' }}}}"
                );
                if value.text() != Some(gated.as_str()) {
                    return Err(format!(
                        "job '{name}' passes secret {secret} outside trusted push runs"
                    ));
                }
            }
            Ok(())
        }
        Some(_) => Err(format!(
            "job '{name}' must pass named secrets, never inherit"
        )),
    }
}
