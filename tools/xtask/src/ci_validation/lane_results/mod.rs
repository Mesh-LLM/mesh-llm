//! `ci validate-lane`: the Rust owner of `scripts/validate-ci-lane-results.py`,
//! and `ci validate-graph`: the parsed entrypoint graph contracts.
//!
//! `validate-lane --lane-plan J --needs J` keeps the legacy argv, `ERROR:`
//! messages and status 2. Optional, additive inputs extend the check without
//! changing any schema: `--workflow` checks the lane's parsed producer/
//! consumer graph against the planned jobs, and `--plan-digest` with
//! `--canonical-plan` binds the projection to the digest-bound plan.
//! Callers are unchanged; cutover belongs to a later task.

mod digest;
mod entrypoint_graph;
mod lane_graph;
mod results;
mod workflow_yaml;

use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::fs;
use std::path::Path;

type Checked<T> = Result<T, String>;

const PROGRAM: &str = "validate-ci-lane-results.py";
const USAGE: &str = "validate-ci-lane-results.py [-h] --lane-plan LANE_PLAN --needs NEEDS";
const HELP: &str = "usage: validate-ci-lane-results.py [-h] --lane-plan LANE_PLAN --needs NEEDS

options:
  -h, --help            show this help message and exit
  --lane-plan LANE_PLAN
  --needs NEEDS
";
const LANE_OPTIONS: [&str; 5] = [
    "--lane-plan",
    "--needs",
    "--workflow",
    "--plan-digest",
    "--canonical-plan",
];
const GRAPH: Grammar = Grammar {
    usage: "cargo xtool ci validate-graph --workflows DIR",
    values: &["--workflows"],
    flags: &[],
};

/// `ci validate-lane` or `ci validate-graph`, selected by `verb`.
pub(crate) fn run(verb: &str, args: &[String]) -> DynResult<()> {
    match verb {
        "validate-graph" => graph_report(args).emit(),
        _ => lane_report(args).emit(),
    }
}

fn error(message: String) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("ERROR: {message}\n"),
        code: 2,
    }
}

fn usage_error(message: &str) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("usage: {USAGE}\n{PROGRAM}: error: {message}\n"),
        code: 2,
    }
}

fn lane_report(args: &[String]) -> CheckReport {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        return CheckReport::success(HELP.to_owned());
    }
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(message) => return usage_error(&message),
    };
    let request = Request {
        lane_plan: options.value("--lane-plan"),
        needs: options.value("--needs"),
        workflow: options.get("--workflow").map(Path::new),
        digest: options
            .get("--plan-digest")
            .zip(options.get("--canonical-plan")),
    };
    match request.validate() {
        Ok(()) => CheckReport::default(),
        Err(message) => error(message),
    }
}

/// argparse order: a value-less option fails at once, then missing required
/// options, then unrecognized arguments. Later values replace earlier ones.
struct Options<'a>(Vec<(&'static str, &'a str)>);

impl<'a> Options<'a> {
    fn parse(args: &'a [String]) -> Checked<Self> {
        let mut values = Vec::new();
        let mut extras = Vec::new();
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            if arg == "--" {
                extras.extend(rest.by_ref().map(String::as_str));
                break;
            }
            let (name, inline) = match arg.split_once('=') {
                Some((name, value)) if arg.starts_with("--") => (name, Some(value)),
                _ => (arg.as_str(), None),
            };
            let Some(option) = LANE_OPTIONS.iter().find(|option| **option == name) else {
                extras.push(arg.as_str());
                continue;
            };
            let value = match inline {
                Some(value) => value,
                None => rest
                    .next()
                    .filter(|value| !looks_like_option(value))
                    .ok_or_else(|| format!("argument {option}: expected one argument"))?,
            };
            values.push((*option, value));
        }
        let options = Self(values);
        let missing: Vec<&str> = ["--lane-plan", "--needs"]
            .into_iter()
            .filter(|flag| options.get(flag).is_none())
            .collect();
        if !missing.is_empty() {
            let list = missing.join(", ");
            return Err(format!("the following arguments are required: {list}"));
        }
        if !extras.is_empty() {
            return Err(format!("unrecognized arguments: {}", extras.join(" ")));
        }
        Ok(options)
    }

    fn get(&self, name: &str) -> Option<&'a str> {
        self.0
            .iter()
            .rev()
            .find(|(option, _)| *option == name)
            .map(|(_, value)| *value)
    }

    fn value(&self, name: &str) -> &'a str {
        self.get(name).unwrap_or_default()
    }
}

/// argparse treats a dash-prefixed word without spaces that is not a
/// negative number as the next option rather than a value.
fn looks_like_option(value: &str) -> bool {
    value.len() > 1
        && value.starts_with('-')
        && !value.contains(' ')
        && value[1..].parse::<f64>().is_err()
}

struct Request<'a> {
    lane_plan: &'a str,
    needs: &'a str,
    workflow: Option<&'a Path>,
    digest: Option<(&'a str, &'a str)>,
}

impl Request<'_> {
    fn validate(&self) -> Checked<()> {
        let lane_plan = results::load_object(self.lane_plan, "lane plan")?;
        let needs = results::load_object(self.needs, "needs")?;
        let outcome = results::validate(&lane_plan, &needs)?;
        if let Some((digest, canonical)) = self.digest {
            digest::verify(&lane_plan, digest, canonical)?;
        }
        if let Some(path) = self.workflow {
            let source =
                fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
            let tree = workflow_yaml::parse(&source)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            lane_graph::LaneGraph::parse(&tree, outcome.lane)?.check_planned(&outcome.planned)?;
        }
        Ok(())
    }
}

fn graph_report(args: &[String]) -> CheckReport {
    let parsed = match GRAPH.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let Some(workflows) = parsed.last("--workflows") else {
        return GRAPH.error("the following arguments are required: --workflows");
    };
    match entrypoint_graph::validate(Path::new(workflows)) {
        Ok(()) => CheckReport::default(),
        Err(message) => error(message),
    }
}
