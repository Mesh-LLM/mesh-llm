use crate::prepared_input::text_io::decode_utf8;
use crate::release::command_failure::Uncaught;
use crate::release::regroup_argv;
use crate::release::regroup_body::{list_line, parse_body};
use crate::release::regroup_plan::{Stop, render, validate, validate_metadata};
use crate::release::regroup_values::{get_or_null, load_json};
use crate::repository::check_report::CheckReport;
use std::path::Path;

pub(crate) fn run(args: &[String]) -> CheckReport {
    match regroup_argv::parse(args) {
        Err(report) => report,
        Ok(parsed) => run_parsed(&parsed).unwrap_or_else(|stop| match stop {
            Stop::Exit(message) => CheckReport::failure(String::new(), format!("{message}\n")),
            Stop::Raised(error) => error.report(String::new()),
        }),
    }
}

fn run_parsed(args: &regroup_argv::Args) -> Result<CheckReport, Stop> {
    let bytes =
        std::fs::read(&args.body).map_err(|error| Uncaught::os(Path::new(&args.body), &error))?;
    let source = decode_utf8(bytes).map_err(Uncaught::decode)?;
    let body = parse_body(&source.replace("\r\n", "\n").replace('\r', "\n")).map_err(Stop::Exit)?;
    if args.list {
        let stdout = body
            .order
            .iter()
            .map(|pr| list_line(pr, &body.entries[pr]))
            .collect();
        return Ok(CheckReport {
            stdout,
            stderr: format!("\n{} entries\n", body.order.len()),
            code: 0,
        });
    }
    let Some(path) = &args.plan else {
        return Ok(regroup_argv::fail(
            "--plan is required unless --list is given",
        ));
    };
    let mut plan = load_json(path)?;
    if let Some(path) = &args.metadata_from {
        let trusted = load_json(path)?;
        let crate::ci_operations::ci_metrics_value::Value::Object(fields) = &mut plan else {
            return Err(Uncaught::new(
                "TypeError",
                "plan does not support item assignment".to_owned(),
            )
            .into());
        };
        for field in ["version", "date"] {
            let value = get_or_null(&trusted, field)?;
            match fields.iter_mut().find(|(name, _)| name == field) {
                Some((_, existing)) => *existing = value,
                None => fields.push((field.to_owned(), value)),
            }
        }
    }
    validate_metadata(&plan)?;
    let count = validate(&plan, &body)?;
    if args.check {
        return Ok(CheckReport::success(format!(
            "ok: plan covers all {count} entries exactly once\n"
        )));
    }
    let Some(path) = &args.out else {
        return Ok(regroup_argv::fail(
            "--out is required unless --check is given",
        ));
    };
    let notes = render(&plan, &body)?;
    std::fs::write(path, notes).map_err(|error| Uncaught::os(Path::new(path), &error))?;
    Ok(CheckReport::success(format!(
        "ok: regrouped {count} entries -> {path}\n"
    )))
}
