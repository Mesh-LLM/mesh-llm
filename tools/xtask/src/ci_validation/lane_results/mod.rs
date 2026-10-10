//! Lane result admission and parsed entrypoint graph contracts.
//! Exact named options provide JSON plans/results and optional workflow/digest
//! evidence. Usage and domain failures retain status2; domain reports retain
//! the consumed ERROR prefix.

mod authority_results;
mod digest;
mod entrypoint_graph;
mod lane_graph;
mod results;
pub(super) mod workflow_yaml;

use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::fs;
use std::path::Path;

type Checked<T> = Result<T, String>;

const USAGE: &str = "cargo xtool ci validate-lane --lane-plan JSON --needs JSON [--workflow FILE] [--plan-digest SHA256 --canonical-plan JSON]";
const HELP: &str = "Validate every planned lane job and optional immutable graph evidence.

Options:
  -h, --help
  --lane-plan JSON
  --needs JSON
  --workflow FILE
  --plan-digest SHA256
  --canonical-plan JSON
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
        stderr: format!("usage: {USAGE}\nerror: {message}\n"),
        code: 2,
    }
}

fn lane_report(args: &[String]) -> CheckReport {
    if matches!(args, [argument] if argument == "-h" || argument == "--help") {
        return CheckReport::success(format!("usage: {USAGE}\n\n{HELP}"));
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

/// This command has no positional arguments or ambiguous repeated options.
struct Options<'a>(std::collections::BTreeMap<&'static str, &'a str>);

impl<'a> Options<'a> {
    fn parse(args: &'a [String]) -> Checked<Self> {
        let mut values = std::collections::BTreeMap::new();
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            let (name, inline) = match arg.split_once('=') {
                Some((name, value)) if arg.starts_with("--") => (name, Some(value)),
                _ => (arg.as_str(), None),
            };
            let option = LANE_OPTIONS
                .iter()
                .find(|option| **option == name)
                .ok_or_else(|| format!("unknown lane option or positional argument: {arg}"))?;
            let value = match inline {
                Some(value) => value,
                None => rest
                    .next()
                    .map(String::as_str)
                    .filter(|value| !value.starts_with('-'))
                    .ok_or_else(|| format!("{option} requires a value"))?,
            };
            if value.is_empty() {
                return Err(format!("{option} requires a nonempty value"));
            }
            if values.insert(*option, value).is_some() {
                return Err(format!("{option} must be supplied once"));
            }
        }
        for required in ["--lane-plan", "--needs"] {
            if !values.contains_key(required) {
                return Err(format!("missing required option {required}"));
            }
        }
        Ok(Self(values))
    }
    fn get(&self, name: &str) -> Option<&'a str> {
        self.0.get(name).copied()
    }
    fn value(&self, name: &str) -> &'a str {
        self.get(name).unwrap_or_default()
    }
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
        return GRAPH.error("missing required option --workflows");
    };
    match entrypoint_graph::validate(Path::new(workflows)) {
        Ok(()) => CheckReport::default(),
        Err(message) => error(message),
    }
}
