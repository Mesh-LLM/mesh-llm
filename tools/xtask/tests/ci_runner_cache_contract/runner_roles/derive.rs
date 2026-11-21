//! Runner placement facts derived from the checked-in workflows.
use super::matrix::{Profile, Provider};
use crate::{
    support,
    workflow_yaml::{self, Node},
};
use regex::Regex;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::LazyLock,
};

const SELECTOR_ACTION: &str = "./.github/actions/select-ci-runners";
const GITHUB_HOSTED: [&str; 5] = [
    "macos-15",
    "ubuntu-24.04",
    "ubuntu-24.04-arm",
    "ubuntu-latest",
    "windows-2022",
];
const GITHUB_HOSTED_GUARD: &str = "runner.environment == 'github-hosted'";

static NEEDS_OUTPUT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"needs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_]+)").unwrap());
static MATRIX_VALUE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"matrix\.([A-Za-z0-9_]+)").unwrap());
static CALLER_VALUE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(?:inputs|vars)\.[A-Za-z0-9_]+").unwrap());
static STEP_OUTPUT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"steps\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_]+)").unwrap());
static SELECTOR_LABEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(runner[a-z0-9_]*)=([A-Za-z0-9.-]+)\s*$").unwrap());

pub(super) struct Workflow {
    pub(super) name: String,
    pub(super) document: Node,
}

pub(super) fn checked_in_workflows() -> Vec<Workflow> {
    let directory = support::root().join(".github/workflows");
    let mut names = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".yml") || name.ends_with(".yaml"))
        .collect::<Vec<_>>();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let text = fs::read_to_string(directory.join(&name)).unwrap();
            let document = workflow_yaml::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
            Workflow { name, document }
        })
        .collect()
}

pub(super) fn provider_of(label: &str) -> Provider {
    if GITHUB_HOSTED.contains(&label) {
        Provider::GithubHosted
    } else if label.starts_with("depot-") {
        Provider::Depot
    } else {
        Provider::SelfHosted
    }
}

pub(super) struct Selector {
    pub(super) workflow: String,
    pub(super) inputs: BTreeMap<String, String>,
}

#[derive(Default)]
pub(super) struct Job {
    pub(super) workflow: String,
    /// Selector key to the selector outputs this job's `runs-on` consumes.
    pub(super) selectors: BTreeMap<String, BTreeSet<String>>,
    pub(super) labels: BTreeSet<String>,
    pub(super) dynamic: Vec<String>,
    pub(super) profiles: BTreeSet<Profile>,
}

impl Job {
    pub(super) fn needs_row(&self) -> bool {
        !self.selectors.is_empty()
            || !self.dynamic.is_empty()
            || self
                .labels
                .iter()
                .any(|label| provider_of(label) != Provider::GithubHosted)
    }
}

pub(super) struct Inventory {
    pub(super) selectors: BTreeMap<String, Selector>,
    pub(super) jobs: BTreeMap<String, Job>,
    pub(super) triggers: BTreeMap<String, BTreeSet<String>>,
    pub(super) dispatch_inputs: BTreeMap<String, BTreeMap<String, String>>,
    /// Selector runner output to its (GitHub-hosted, Depot) labels.
    pub(super) runner_labels: BTreeMap<String, (String, String)>,
}

pub(super) fn inventory(workflows: &[Workflow]) -> Inventory {
    let mut inventory = Inventory {
        selectors: BTreeMap::new(),
        jobs: BTreeMap::new(),
        triggers: BTreeMap::new(),
        dispatch_inputs: BTreeMap::new(),
        runner_labels: selector_labels(),
    };
    for workflow in workflows {
        read_workflow(workflow, &mut inventory);
    }
    inventory
}

fn read_workflow(workflow: &Workflow, inventory: &mut Inventory) {
    let name = &workflow.name;
    let on = workflow.document.get("on");
    let triggers = on.map_or_else(Vec::new, |on| {
        on.entries().iter().map(|(k, _)| k.clone()).collect()
    });
    inventory
        .triggers
        .insert(name.clone(), triggers.into_iter().collect());
    let dispatch = on
        .and_then(|on| on.get("workflow_dispatch"))
        .and_then(|d| d.get("inputs"))
        .map_or_else(BTreeMap::new, |inputs| {
            inputs
                .entries()
                .iter()
                .map(|(k, v)| (k.clone(), text(v.get("type")).to_owned()))
                .collect()
        });
    inventory.dispatch_inputs.insert(name.clone(), dispatch);
    let Some(jobs) = workflow.document.get("jobs") else {
        return;
    };
    for (job_id, job) in jobs.entries() {
        for (step_id, inputs) in selector_steps(job) {
            let selector = Selector {
                workflow: name.clone(),
                inputs,
            };
            inventory
                .selectors
                .insert(format!("{name}#{job_id}/{step_id}"), selector);
        }
        if let Some(runs_on) = job.get("runs-on") {
            let placed = placement(name, jobs, job, runs_on);
            inventory.jobs.insert(format!("{name}#{job_id}"), placed);
        }
    }
}

fn text(node: Option<&Node>) -> &str {
    node.and_then(Node::text).unwrap_or_default()
}

fn steps(node: Option<&Node>) -> &[Node] {
    match node.and_then(|n| n.get("steps")) {
        Some(Node::Seq(steps)) => steps,
        _ => &[],
    }
}

fn selector_steps(job: &Node) -> Vec<(String, BTreeMap<String, String>)> {
    steps(Some(job))
        .iter()
        .filter(|step| text(step.get("uses")) == SELECTOR_ACTION)
        .map(|step| {
            let id = text(step.get("id"));
            assert!(!id.is_empty(), "select-ci-runners step without an id");
            let inputs = step.get("with").map_or_else(BTreeMap::new, |with| {
                with.entries()
                    .iter()
                    .map(|(k, v)| (k.clone(), text(Some(v)).to_owned()))
                    .collect()
            });
            (id.to_owned(), inputs)
        })
        .collect()
}

fn selector_labels() -> BTreeMap<String, (String, String)> {
    let script = support::step(&support::action("select-ci-runners"), "run");
    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for captures in script.lines().filter_map(|l| SELECTOR_LABEL.captures(l)) {
        found
            .entry(captures[1].to_owned())
            .or_default()
            .push(captures[2].to_owned());
    }
    found
        .into_iter()
        .map(|(output, labels)| {
            let (depot, hosted): (Vec<_>, Vec<_>) =
                labels.into_iter().partition(|l| l.starts_with("depot-"));
            assert!(
                depot.len() == 1 && hosted.len() == 1,
                "selector output {output} must have one hosted and one Depot label"
            );
            (output, (hosted[0].clone(), depot[0].clone()))
        })
        .collect()
}

fn placement(workflow: &str, jobs: &Node, job: &Node, runs_on: &Node) -> Job {
    let mut placed = Job {
        workflow: workflow.to_owned(),
        profiles: profiles(steps(Some(job)), false, 0),
        ..Job::default()
    };
    match runs_on {
        Node::Seq(items) => placed
            .labels
            .extend(items.iter().filter_map(Node::text).map(str::to_owned)),
        Node::Scalar(label) if !label.contains("${{") => {
            placed.labels.insert(label.clone());
        }
        Node::Scalar(expression) => read_expression(workflow, jobs, job, expression, &mut placed),
        Node::Map(_) => placed.dynamic.push("runs-on group".to_owned()),
    }
    placed
}

fn read_expression(workflow: &str, jobs: &Node, job: &Node, expression: &str, placed: &mut Job) {
    for literal in quoted_literals(expression) {
        if literal.starts_with('[') {
            let labels: Vec<String> = serde_json::from_str(&literal).unwrap();
            placed.labels.extend(labels);
        } else {
            placed.labels.insert(literal);
        }
    }
    for (whole, producer, output) in used_references(&NEEDS_OUTPUT, expression) {
        let traced = trace(workflow, jobs, &producer, &output);
        if traced.is_empty() {
            placed.dynamic.push(whole);
        }
        for (selector, consumed) in traced {
            placed
                .selectors
                .entry(selector)
                .or_default()
                .insert(consumed);
        }
    }
    for (whole, key, _) in used_references(&MATRIX_VALUE, expression) {
        let values = matrix_values(job, &key);
        if values.is_empty() {
            placed.dynamic.push(whole);
        }
        placed.labels.extend(values);
    }
    for (whole, _, _) in used_references(&CALLER_VALUE, expression) {
        placed.dynamic.push(whole);
    }
}

/// References whose value is used, not compared or dereferenced further.
fn used_references(pattern: &Regex, expression: &str) -> Vec<(String, String, String)> {
    pattern
        .captures_iter(expression)
        .filter(|captures| {
            let rest = expression[captures.get(0).unwrap().end()..].trim_start();
            !(rest.starts_with('.') || rest.starts_with("==") || rest.starts_with("!="))
        })
        .map(|captures| {
            let group = |i| captures.get(i).map_or("", |m| m.as_str()).to_owned();
            (group(0), group(1), group(2))
        })
        .collect()
}

/// Single-quoted expression strings that are values rather than comparison operands.
pub(super) fn quoted_literals(expression: &str) -> Vec<String> {
    let mut literals = Vec::new();
    let mut rest = expression;
    let mut consumed = 0;
    while let Some(open) = rest.find('\'') {
        let Some(length) = rest[open + 1..].find('\'') else {
            break;
        };
        let start = consumed + open;
        let end = start + length + 2;
        let before = expression[..start].trim_end();
        let after = expression[end..].trim_start();
        let compared = before.ends_with("==")
            || before.ends_with("!=")
            || after.starts_with("==")
            || after.starts_with("!=");
        if !compared {
            literals.push(expression[start + 1..end - 1].to_owned());
        }
        consumed = end;
        rest = &expression[end..];
    }
    literals
}

fn matrix_values(job: &Node, key: &str) -> Vec<String> {
    let Some(matrix) = job.get("strategy").and_then(|s| s.get("matrix")) else {
        return Vec::new();
    };
    let mut values = match matrix.get(key) {
        Some(Node::Seq(items)) => items
            .iter()
            .filter_map(Node::text)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    };
    if let Some(Node::Seq(include)) = matrix.get("include") {
        values.extend(
            include
                .iter()
                .filter_map(|entry| entry.get(key).and_then(Node::text))
                .map(str::to_owned),
        );
    }
    values
}

/// Selector outputs behind a producer job output, through at most one mapping step.
fn trace(workflow: &str, jobs: &Node, producer: &str, output: &str) -> Vec<(String, String)> {
    let Some(job) = jobs.get(producer) else {
        return Vec::new();
    };
    let selector_ids: BTreeSet<String> =
        selector_steps(job).into_iter().map(|(id, _)| id).collect();
    let key = |id: &str| format!("{workflow}#{producer}/{id}");
    let expression = text(job.get("outputs").and_then(|o| o.get(output)));
    let mut found = Vec::new();
    for captures in STEP_OUTPUT.captures_iter(expression) {
        if selector_ids.contains(&captures[1]) {
            found.push((key(&captures[1]), captures[2].to_owned()));
            continue;
        }
        let mapping = steps(Some(job))
            .iter()
            .find(|step| text(step.get("id")) == &captures[1]);
        for value in mapping.into_iter().flat_map(mapping_values) {
            for inner in STEP_OUTPUT.captures_iter(value) {
                if selector_ids.contains(&inner[1]) {
                    found.push((key(&inner[1]), inner[2].to_owned()));
                }
            }
        }
    }
    found
}

fn mapping_values(step: &Node) -> Vec<&str> {
    ["env", "with"]
        .into_iter()
        .filter_map(|key| step.get(key))
        .flat_map(Node::entries)
        .filter_map(|(_, value)| value.text())
        .collect()
}

fn profiles(steps_list: &[Node], guarded: bool, depth: usize) -> BTreeSet<Profile> {
    let mut found = BTreeSet::new();
    for step in steps_list {
        let Some(name) = text(step.get("uses")).strip_prefix("./.github/actions/") else {
            continue;
        };
        let guarded = guarded || text(step.get("if")).contains(GITHUB_HOSTED_GUARD);
        match name {
            "restore-automation" => {
                found.insert(Profile::RestoreAutomation);
            }
            "prepare-ci-controller" => {
                found.insert(Profile::CiController);
            }
            "prepare-automation" | "upload-automation" => {
                found.extend(requested_profiles(step, guarded));
            }
            _ if depth < 3 => {
                if let Some(action) = local_action(name) {
                    found.extend(profiles(steps(action.get("runs")), guarded, depth + 1));
                }
            }
            _ => {}
        }
    }
    found
}

fn local_action(name: &str) -> Option<Node> {
    let path = support::root().join(format!(".github/actions/{name}/action.yml"));
    let source = fs::read_to_string(path).ok()?;
    Some(workflow_yaml::parse(&source).unwrap_or_else(|e| panic!("{name}: {e}")))
}

fn requested_profiles(step: &Node, guarded: bool) -> Vec<Profile> {
    let requested = step
        .get("with")
        .and_then(|w| w.get("runner-profile"))
        .and_then(Node::text);
    let names = match requested {
        None => vec!["image".to_owned()],
        Some(profile) if !profile.contains("${{") => vec![profile.to_owned()],
        Some(expression) => quoted_literals(expression),
    };
    let guarded = guarded || requested.is_some_and(|p| p.contains(GITHUB_HOSTED_GUARD));
    names
        .iter()
        .filter_map(|name| match name.as_str() {
            "image" => Some(Profile::Image),
            "hosted-bare" if guarded => Some(Profile::HostedBareGithubHostedOnly),
            "hosted-bare" => Some(Profile::HostedBare),
            "preinstalled-macos" => Some(Profile::PreinstalledMacos),
            "preinstalled-linux" => Some(Profile::PreinstalledLinux),
            _ => None,
        })
        .collect()
}
