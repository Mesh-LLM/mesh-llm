use super::input::{self, ReportInputError};
use super::types::{Field, ReportValue, SummaryCounts, Verdict};
use serde::de::{IgnoredAny, MapAccess};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub(in crate::automation) enum Error {
    #[error(transparent)]
    Input(#[from] ReportInputError),
    #[error("invalid producer report: {0:?}")]
    Contract(Vec<String>),
    #[error("rewriter emitted no builder records")]
    Empty,
    #[error("first rewriter pass received pre-transformed model builders: {0}")]
    Inherited(String),
    #[error("first rewriter pass did not transform every decoder builder: {0}")]
    Refused(String),
    #[error("second rewriter pass still has edits for: {0}")]
    Remaining(String),
}

#[derive(Clone, Copy)]
pub(in crate::automation) enum Pass {
    First,
    Second,
}

#[derive(Debug, Default)]
struct Projection {
    builders: Field<Vec<Field<Builder>>>,
    summary: Field<SummaryCounts>,
}

#[derive(Debug, Default)]
struct Builder {
    file: Field<String>,
    verdict: Field<Verdict>,
    unsupported_reason: Field<String>,
}

impl ReportValue for Projection {
    const EXPECTED: &'static str = "object";
    fn map<'de, A: MapAccess<'de>>(mut map: A) -> Result<Field<Self>, A::Error> {
        let mut report = Self::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "builders" => report.builders.read_member(&mut map)?,
                "summary" => report.summary.read_member(&mut map)?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(Field::Present(report))
    }
}

impl ReportValue for Builder {
    const EXPECTED: &'static str = "object";
    fn map<'de, A: MapAccess<'de>>(mut map: A) -> Result<Field<Self>, A::Error> {
        let mut builder = Self::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "file" => builder.file.read_member(&mut map)?,
                "verdict" => builder.verdict.read_member(&mut map)?,
                "unsupported_reason" => builder.unsupported_reason.read_member(&mut map)?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(Field::Present(builder))
    }
    fn errors(&self, path: &str, failures: &mut Vec<String>) {
        self.file.errors(&format!("{path}.file"), false, failures);
        self.verdict
            .errors(&format!("{path}.verdict"), false, failures);
        self.unsupported_reason
            .errors(&format!("{path}.unsupported_reason"), false, failures);
    }
}

pub(in crate::automation) type Summary = BTreeMap<&'static str, u128>;

pub(in crate::automation) fn load(path: &Path, pass: Pass) -> Result<Summary, Error> {
    let report: Projection = input::load_projection(path)?;
    let mut failures = Vec::new();
    report.builders.errors("builders", false, &mut failures);
    report.summary.errors("summary", false, &mut failures);
    if !failures.is_empty() {
        return Err(Error::Contract(failures));
    }
    let builders = report
        .builders
        .value()
        .filter(|items| !items.is_empty())
        .ok_or(Error::Empty)?;
    gate(builders, pass)?;
    let mut summary = BTreeMap::new();
    if let Some(counts) = report.summary.value() {
        for (key, count) in [
            ("transformable", &counts.transformable),
            ("already_transformed", &counts.already_transformed),
            ("supported_auxiliary", &counts.supported_auxiliary),
            ("supported_whole_model", &counts.supported_whole_model),
            ("unsupported_shape", &counts.unsupported_shape),
            ("error", &counts.error),
        ] {
            if let Some(count) = count.value() {
                summary.insert(key, count.exact());
            }
        }
    }
    Ok(summary)
}

fn gate(builders: &[Field<Builder>], pass: Pass) -> Result<(), Error> {
    let mut inherited = Vec::new();
    let mut refused = Vec::new();
    let mut remaining = Vec::new();
    for builder in builders.iter().filter_map(Field::value) {
        let file = builder.file.value().map_or("<unknown>", String::as_str);
        match builder.verdict.value() {
            Some(Verdict::AlreadyTransformed) => inherited.push(file),
            Some(Verdict::Transformable) => remaining.push(file),
            Some(Verdict::UnsupportedShape) => refused.push(format!(
                "{file}: {}",
                builder
                    .unsupported_reason
                    .value()
                    .map_or("unsupported_shape", String::as_str)
            )),
            Some(Verdict::Error) => refused.push(format!(
                "{file}: {}",
                builder
                    .unsupported_reason
                    .value()
                    .map_or("error", String::as_str)
            )),
            Some(Verdict::SupportedAuxiliary | Verdict::SupportedWholeModel) | None => {}
        }
    }
    match pass {
        Pass::First if !inherited.is_empty() => Err(Error::Inherited(
            inherited
                .into_iter()
                .take(10)
                .collect::<Vec<_>>()
                .join(", "),
        )),
        Pass::First if !refused.is_empty() => Err(Error::Refused(
            refused.into_iter().take(10).collect::<Vec<_>>().join(", "),
        )),
        Pass::Second if !remaining.is_empty() => Err(Error::Remaining(
            remaining
                .into_iter()
                .take(10)
                .collect::<Vec<_>>()
                .join(", "),
        )),
        Pass::First | Pass::Second => Ok(()),
    }
}

#[cfg(test)]
#[path = "generator_tests.rs"]
mod tests;
