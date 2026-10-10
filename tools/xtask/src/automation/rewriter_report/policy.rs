use super::types::{
    BuilderFile, BuilderKey, ConstructorName, Field, FieldError, IdempotenceReport, Report,
    ReportValue, SummaryCounts, ValidateBuilder, ValidateReport,
};
use super::{DriftGate, Mode, Options, Outcome};
use crate::repository::check_report::CheckReport;
use std::collections::HashSet;

mod builder;

const DRIFT: &str = "generated family patch drifts from checked-in patch";

pub(super) fn check(report: &Report, options: &Options) -> CheckReport {
    let (mut failures, count) = match report {
        Report::Validate(report) => (
            validate(report),
            report.builders.value().map_or(0, Vec::len),
        ),
        Report::Idempotence(report) => (
            idempotence(report),
            report.builders.value().map_or(0, Vec::len),
        ),
    };
    let mut stdout = String::new();
    match (options.patch_check, options.drift_gate) {
        (Some(Outcome::Fail), DriftGate::Warn) => stdout.push_str(&format!("warn: {DRIFT}\n")),
        (Some(Outcome::Fail), DriftGate::Fail) => failures.push(DRIFT.into()),
        _ => {}
    }
    if options.compile == Outcome::Fail {
        failures.push("transformed tree failed to compile".into());
    }
    if options.graph == Outcome::Fail {
        failures.push("no-allocation graph verifier failed on transformed tree".into());
    }
    if failures.is_empty() {
        let mode = match options.mode {
            Mode::Validate => "validate",
            Mode::Idempotence => "idempotence",
        };
        stdout.push_str(&format!(
            "ok: report valid ({mode}); {count} builders checked\n"
        ));
        CheckReport::success(stdout)
    } else {
        CheckReport::failure(
            stdout,
            failures
                .into_iter()
                .map(|failure| format!("fail: {failure}\n"))
                .collect(),
        )
    }
}

fn idempotence(report: &IdempotenceReport) -> Vec<String> {
    let mut failures = Vec::new();
    report.builders.errors("builders", false, &mut failures);
    if let Some(builders) = report.builders.value() {
        for builder in builders.iter().filter_map(Field::value) {
            if builder
                .verdict
                .value()
                .is_some_and(|verdict| verdict == "transformable")
            {
                let label = builder
                    .file
                    .value()
                    .map_or("<unknown>", |file| file.0.as_str());
                failures.push(format!(
                    "{label}: idempotence violation -- transformable on second run"
                ));
            }
        }
    }
    failures
}

fn validate(report: &ValidateReport) -> Vec<String> {
    let mut failures = Vec::new();
    match &report.schema_version {
        Field::Present(version) if version.0 != 1 => {
            failures.push(format!("schema_version {} != supported 1", version.0))
        }
        Field::Missing => failures.push("schema_version None != supported 1".into()),
        _ => report
            .schema_version
            .errors("schema_version", false, &mut failures),
    }
    for (name, field) in [
        ("llama_cpp_commit", &report.llama_cpp_commit),
        ("generator_version", &report.generator_version),
    ] {
        if matches!(field, Field::Invalid(FieldError::Duplicate)) {
            field.errors(name, false, &mut failures);
        } else if !field.value().is_some_and(|value| !value.is_empty()) {
            failures.push(format!("missing required string field '{name}'"));
        }
    }
    let Some(builders) = report.builders.value().filter(|items| !items.is_empty()) else {
        report.builders.errors("builders", false, &mut failures);
        failures.push("missing non-empty 'builders' array".into());
        return failures;
    };
    let mut seen = HashSet::new();
    for (index, candidate) in builders.iter().enumerate() {
        let path = format!("builders[{index}]");
        match candidate {
            Field::Present(record) => {
                let label = record
                    .file
                    .value()
                    .map_or_else(|| format!("<builders[{index}]>"), |file| file.0.clone());
                if let Some(key) = identity(record, &label)
                    && !seen.insert(key)
                {
                    let constructor = record
                        .constructor
                        .value()
                        .map_or("", |name| name.0.as_str());
                    failures.push(format!(
                        "duplicate builder record for {label}::{constructor}"
                    ));
                }
                builder::validate_builder(record, (&label, &path), &mut failures);
            }
            Field::Missing | Field::Null | Field::Invalid(_) => {
                candidate.errors(&path, false, &mut failures)
            }
        }
    }
    match &report.summary {
        Field::Present(summary) => summary_total(summary, builders.len(), &mut failures),
        Field::Missing => failures.push("missing 'summary' object".into()),
        Field::Null | Field::Invalid(_) => report.summary.errors("summary", false, &mut failures),
    }
    failures
}

fn identity(builder: &ValidateBuilder, label: &str) -> Option<BuilderKey> {
    let file = match &builder.file {
        Field::Missing => BuilderFile(label.into()),
        Field::Present(file) => file.clone(),
        Field::Null | Field::Invalid(_) => return None,
    };
    let constructor = match &builder.constructor {
        Field::Missing => ConstructorName(String::new()),
        Field::Present(name) => name.clone(),
        Field::Null | Field::Invalid(_) => return None,
    };
    Some(BuilderKey(file, constructor))
}

fn summary_total(summary: &SummaryCounts, count: usize, failures: &mut Vec<String>) {
    let previous = failures.len();
    summary.errors("summary", failures);
    if previous != failures.len() {
        return;
    }
    let total: u128 = [
        &summary.transformable,
        &summary.already_transformed,
        &summary.supported_auxiliary,
        &summary.supported_whole_model,
        &summary.unsupported_shape,
        &summary.error,
    ]
    .into_iter()
    .filter_map(Field::value)
    .map(|count| count.exact())
    .sum();
    if u128::try_from(count) != Ok(total) {
        failures.push(format!(
            "summary counts ({total}) != builder records ({count})"
        ));
    }
}

fn quoted_string(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut result = String::from(quote);
    for character in value.chars() {
        match character {
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\\' => result.push_str("\\\\"),
            character if character == quote => {
                result.push('\\');
                result.push(character);
            }
            character => result.push(character),
        }
    }
    result.push(quote);
    result
}
