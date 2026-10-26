//! `repository cargo-census`: validate and record the Cargo packages actually
//! completed by one CI batch. The executed set must equal the selected set
//! exactly; a batch that silently skips or adds a package fails.

use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::collections::BTreeSet;
use std::path::Path;

const USAGE: &str = "repository cargo-census --kind <test|clippy> --batch-id <id> --source-sha <sha> --selected <json-array> --executed <json-array> --out <file>";

fn package_list(value: Option<&str>, label: &str) -> Result<Vec<String>, String> {
    let parsed = value
        .map(str::as_bytes)
        .map(crate::qualification::contract::parse_strict)
        .transpose()?
        .ok_or_else(|| format!("{label} is required"))?;
    let items = parsed
        .as_array()
        .ok_or_else(|| format!("{label} must contain nonempty package names"))?;
    let mut names = Vec::new();
    for item in items {
        match item.as_str() {
            Some(name) if !name.is_empty() => names.push(name.to_owned()),
            _ => return Err(format!("{label} must contain nonempty package names")),
        }
    }
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return Err(format!("{label} contains duplicate packages"));
    }
    Ok(names)
}

pub(crate) fn validate(selected: &[String], executed: &[String]) -> Result<(), String> {
    let selected_set: BTreeSet<&String> = selected.iter().collect();
    let executed_set: BTreeSet<&String> = executed.iter().collect();
    if selected_set != executed_set {
        let missing: Vec<&&String> = selected_set.difference(&executed_set).collect();
        let unexpected: Vec<&&String> = executed_set.difference(&selected_set).collect();
        return Err(format!(
            "Cargo package execution mismatch: missing={missing:?}, unexpected={unexpected:?}"
        ));
    }
    Ok(())
}

pub(crate) fn run(args: &[String]) -> crate::command::DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--kind",
            "--batch-id",
            "--source-sha",
            "--selected",
            "--executed",
            "--out",
        ],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    let report = match census(&parsed) {
        Ok(line) => CheckReport::success(line),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    };
    report.emit()
}

fn census(parsed: &crate::repository::check_args::ParsedArgs) -> Result<String, String> {
    let kind = parsed.last("--kind").unwrap_or("");
    if !matches!(kind, "test" | "clippy") {
        return Err("--kind must be test or clippy".to_owned());
    }
    for required in ["--batch-id", "--source-sha", "--out"] {
        if parsed.last(required).is_none_or(str::is_empty) {
            return Err(format!("the following arguments are required: {required}"));
        }
    }
    if !parsed.positionals.is_empty() {
        return Err(format!(
            "unrecognized arguments: {}",
            parsed.positionals.join(" ")
        ));
    }
    let selected = package_list(parsed.last("--selected"), "selected")?;
    let executed = package_list(parsed.last("--executed"), "executed")?;
    validate(&selected, &executed)?;
    let batch_id = parsed.last("--batch-id").unwrap_or_default();
    let mut selected_sorted = selected;
    let mut executed_sorted = executed;
    selected_sorted.sort();
    executed_sorted.sort();
    let receipt = serde_json::json!({
        "schema": "mesh-ci-cargo-census-v1",
        "kind": kind,
        "batch_id": batch_id,
        "source_sha": parsed.last("--source-sha").unwrap_or_default(),
        "selected": selected_sorted,
        "executed": executed_sorted,
    });
    let mut text = serde_json::to_string(&receipt).map_err(|error| error.to_string())?;
    text.push('\n');
    std::fs::write(Path::new(parsed.last("--out").unwrap_or_default()), text)
        .map_err(|error| error.to_string())?;
    Ok(format!(
        "{kind} batch {batch_id}: {} packages executed\n",
        executed_sorted.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::validate;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn accepts_actual_batch_in_any_execution_order() {
        assert!(validate(&names(&["a", "b", "c"]), &names(&["c", "a", "b"])).is_ok());
    }

    #[test]
    fn rejects_missing_or_extra_package() {
        let error = validate(&names(&["a", "b"]), &names(&["a", "z"])).unwrap_err();
        assert!(error.contains("missing"), "{error}");
        assert!(error.contains("unexpected"), "{error}");
    }
}
