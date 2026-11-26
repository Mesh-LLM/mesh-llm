//! Runner placement facts derived from the checked-in workflows.
use super::matrix::{Profile, Provider};
use crate::{
    support,
    workflow_yaml::{self, Node},
};
use regex::Regex;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::Path,
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
pub(super) const GITHUB_HOSTED_GUARD: &str = "runner.environment == 'github-hosted'";
const LINUX_GUARD: &str = "runner.os == 'Linux'";

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
    /// A hosted-bare preparation that no `runner.os == 'Linux'` guard confines to Linux.
    pub(super) hosted_bare_any_os: bool,
}

#[derive(Clone, Copy, Default)]
struct Guards {
    github_hosted: bool,
    linux: bool,
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
    inventory_in(workflows, &support::root())
}

/// The inventory with local actions read from `root`'s `.github/actions`.
pub(super) fn inventory_in(workflows: &[Workflow], root: &Path) -> Inventory {
    let mut inventory = Inventory {
        selectors: BTreeMap::new(),
        jobs: BTreeMap::new(),
        triggers: BTreeMap::new(),
        dispatch_inputs: BTreeMap::new(),
        runner_labels: selector_labels(),
    };
    for workflow in workflows {
        read_workflow(workflow, root, &mut inventory);
    }
    inventory
}

fn read_workflow(workflow: &Workflow, root: &Path, inventory: &mut Inventory) {
    let name = &workflow.name;
    let on = workflow.document.get("on");
    let triggers: Vec<String> = match on {
        Some(Node::Map(entries)) => entries.iter().map(|(k, _)| k.clone()).collect(),
        Some(on) => on.list().into_iter().map(str::to_owned).collect(),
        None => Vec::new(),
    };
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
            let placed = placement(name, root, jobs, job, runs_on);
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

fn placement(workflow: &str, root: &Path, jobs: &Node, job: &Node, runs_on: &Node) -> Job {
    let prepared = profiles(root, steps(Some(job)), Guards::default(), &mut Vec::new());
    let mut placed = Job {
        workflow: workflow.to_owned(),
        profiles: prepared.iter().map(|(profile, _)| *profile).collect(),
        hosted_bare_any_os: prepared.iter().any(|(profile, linux)| {
            matches!(
                profile,
                Profile::HostedBare | Profile::HostedBareGithubHostedOnly
            ) && !linux
        }),
        ..Job::default()
    };
    match runs_on {
        Node::Seq(items) => {
            for item in items.iter().filter_map(Node::text) {
                read_label(workflow, jobs, job, item, &mut placed);
            }
        }
        Node::Scalar(label) => read_label(workflow, jobs, job, label, &mut placed),
        Node::Map(_) => placed.dynamic.push("runs-on group".to_owned()),
    }
    placed
}

fn read_label(workflow: &str, jobs: &Node, job: &Node, label: &str, placed: &mut Job) {
    if label.contains("${{") {
        read_expression(workflow, jobs, job, label, placed);
    } else {
        placed.labels.insert(label.to_owned());
    }
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
    // A result no extractor recognizes (a dispatch payload) still needs a row,
    // even when another alternative resolves.
    placed.dynamic.extend(
        result_values(expression)
            .into_iter()
            .filter(|value| !resolved_runner(value))
            .map(str::to_owned),
    );
}

/// The value operand of each `a && 'x' || b && 'y' || 'z'` alternative.
fn result_values(expression: &str) -> Vec<&str> {
    strip_expression(expression)
        .split("||")
        .filter_map(|alternative| alternative.rsplit("&&").next())
        .map(str::trim)
        .collect()
}

fn resolved_runner(value: &str) -> bool {
    !quoted_literals(value).is_empty()
        || [&NEEDS_OUTPUT, &MATRIX_VALUE, &CALLER_VALUE]
            .iter()
            .any(|pattern| pattern.is_match(value))
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

/// The condition holds only on GitHub-hosted runners: every `||` alternative is a
/// conjunction that contains the guard.
pub(super) fn requires_github_hosted(condition: &str) -> bool {
    requires(condition, GITHUB_HOSTED_GUARD)
}

fn requires(condition: &str, guard: &str) -> bool {
    let condition = strip_expression(condition);
    !condition.is_empty()
        && condition
            .split("||")
            .all(|alternative| alternative.contains(guard))
}

/// Value literals of an `a && 'x' || b && 'y' || 'z'` selection, each paired with
/// whether the conjunction that selects it contains `guard`. A literal in an
/// alternative without the guard, including the final fallback, is not guarded.
pub(super) fn guarded_literals(expression: &str, guard: &str) -> Vec<(String, bool)> {
    strip_expression(expression)
        .split("||")
        .flat_map(|alternative| {
            let guarded = alternative.contains(guard);
            quoted_literals(alternative)
                .into_iter()
                .map(move |literal| (literal, guarded))
        })
        .collect()
}

fn strip_expression(text: &str) -> &str {
    let text = text.trim();
    text.strip_prefix("${{")
        .and_then(|rest| rest.strip_suffix("}}"))
        .unwrap_or(text)
        .trim()
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

/// Requested profiles, each paired with whether a Linux guard confines it.
fn profiles(
    root: &Path,
    steps_list: &[Node],
    guards: Guards,
    chain: &mut Vec<String>,
) -> BTreeSet<(Profile, bool)> {
    let mut found = BTreeSet::new();
    for step in steps_list {
        let Some(name) = text(step.get("uses")).strip_prefix("./.github/actions/") else {
            continue;
        };
        let condition = text(step.get("if"));
        let guards = Guards {
            github_hosted: guards.github_hosted || requires_github_hosted(condition),
            linux: guards.linux || requires(condition, LINUX_GUARD),
        };
        match name {
            "restore-automation" => {
                found.insert((Profile::RestoreAutomation, guards.linux));
            }
            "prepare-ci-controller" => {
                found.insert((Profile::CiController, guards.linux));
            }
            "prepare-automation" | "upload-automation" => {
                found.extend(requested_profiles(step, guards));
            }
            _ => {
                assert!(
                    !chain.iter().any(|n| n == name),
                    "local action cycle {chain:?} -> {name}"
                );
                let action = local_action(root, name);
                chain.push(name.to_owned());
                found.extend(profiles(root, steps(action.get("runs")), guards, chain));
                chain.pop();
            }
        }
    }
    found
}

fn local_action(root: &Path, name: &str) -> Node {
    let directory = root.join(".github/actions").join(name);
    for file in ["action.yml", "action.yaml"] {
        match fs::read_to_string(directory.join(file)) {
            Ok(source) => {
                return workflow_yaml::parse(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => panic!("{name}/{file}: {e}"),
        }
    }
    panic!("local action {name} has no action.yml or action.yaml")
}

fn requested_profiles(step: &Node, guards: Guards) -> Vec<(Profile, bool)> {
    let requested = step
        .get("with")
        .and_then(|w| w.get("runner-profile"))
        .and_then(Node::text);
    let names = match requested {
        None => vec![("image".to_owned(), guards)],
        Some(profile) if !profile.contains("${{") => vec![(profile.to_owned(), guards)],
        Some(expression) => {
            let branches: Vec<(String, Guards)> = guarded_literals(expression, GITHUB_HOSTED_GUARD)
                .into_iter()
                .zip(guarded_literals(expression, LINUX_GUARD))
                .map(|((name, hosted), (_, linux))| {
                    let branch = Guards {
                        github_hosted: guards.github_hosted || hosted,
                        linux: guards.linux || linux,
                    };
                    (name, branch)
                })
                .collect();
            assert!(
                !branches.is_empty(),
                "runner-profile {expression} names no literal profile"
            );
            for value in result_values(expression) {
                assert!(
                    value.starts_with('\'') && !quoted_literals(value).is_empty(),
                    "runner-profile {expression} has unresolved branch {value}"
                );
            }
            branches
        }
    };
    names
        .into_iter()
        .filter_map(|(name, guards)| {
            let profile = match name.as_str() {
                "image" => Profile::Image,
                "hosted-bare" if guards.github_hosted => Profile::HostedBareGithubHostedOnly,
                "hosted-bare" => Profile::HostedBare,
                "preinstalled-macos" => Profile::PreinstalledMacos,
                "preinstalled-linux" => Profile::PreinstalledLinux,
                _ => return None,
            };
            Some((profile, guards.linux))
        })
        .collect()
}
