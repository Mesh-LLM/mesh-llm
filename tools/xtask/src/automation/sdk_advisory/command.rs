use super::admission::{Controller, Trigger, admit};
use super::error::{Checked, Rejected};
use super::input::read;
use super::rows::Catalog;
use super::selection::{Selection, select};
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: super::USAGE,
    values: &[
        "--event-name",
        "--controller-repository",
        "--controller-ref",
        "--event",
        "--producer-run",
        "--artifacts",
    ],
    flags: &[],
};

pub(super) fn report(root: &Path, args: &[String]) -> CheckReport {
    if args == ["--help"] {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage));
    }
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("sdk-advisory accepts only named evidence inputs");
    }
    for name in GRAMMAR.values {
        if parsed.all(name).len() != 1 {
            return GRAMMAR.error(&format!("{name} must be supplied exactly once"));
        }
    }
    match execute(root, &parsed)
        .and_then(|selection| serde_json::to_string_pretty(&selection).map_err(Rejected::Input))
    {
        Ok(json) => CheckReport::success(format!("{json}\n")),
        Err(error) => CheckReport::failure(String::new(), format!("sdk advisory: {error}\n")),
    }
}

fn execute(root: &Path, args: &ParsedArgs) -> Checked<Selection> {
    let value = |name| args.last(name).unwrap_or_default();
    let controller = Controller {
        repository: value("--controller-repository"),
        reference: value("--controller-ref"),
    };
    let event = read(Path::new(value("--event")))?;
    let trigger = match value("--event-name") {
        "workflow_run" => Trigger::WorkflowRun(&event),
        "workflow_dispatch" => Trigger::Manual(&event),
        _ => return Err(Rejected::Event),
    };
    let run = read(Path::new(value("--producer-run")))?;
    let producer = admit(&controller, trigger, &run)?;
    let catalog = Catalog::parse(
        &read(&root.join("ci/ownership.yml"))?,
        &read(&root.join("ci/slices.yml"))?,
    )?;
    select(&producer, &catalog, &read(Path::new(value("--artifacts")))?)
}
