use super::boundary::{ExitSuccess, ModelClass, Truth, object_rows_last_wins};
use super::{Error, ErrorKind, Family, FamilyModel};
use crate::repository::text::is_space;
use serde::{Deserialize, de::IgnoredAny};

#[derive(Debug, Deserialize)]
#[serde(remote = "Self")]
struct ResultRow {
    family: Option<String>,
    #[serde(default)]
    exit_code: ExitSuccess,
    split_layer: Option<IgnoredAny>,
    workload_class: Option<String>,
    #[serde(default)]
    mmproj_smoke: Truth,
    #[serde(default, deserialize_with = "object_rows_last_wins")]
    outcomes: Vec<LaneResult>,
}

impl<'de> Deserialize<'de> for ResultRow {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        super::boundary::ordered_object_last_wins(deserializer, Self::deserialize)
    }
}

#[derive(Debug, Deserialize)]
struct LaneResult {
    name: Option<String>,
    status: Option<String>,
    #[serde(default)]
    exit_code: ExitSuccess,
}

pub(crate) fn validate_results(
    bytes: &[u8],
    family: &Family,
    model: &FamilyModel,
) -> Result<(), Error> {
    let rows = read_json_documents(bytes)?;
    if rows.is_empty()
        || rows.iter().any(|row| {
            !row.exit_code.0
                || !matches!(row.family.as_deref(), Some("battery"))
                    && row.family.as_deref() != Some(family.as_str())
        })
    {
        return Err(Error::new(
            ErrorKind::Results,
            "missing, foreign, or failed results",
        ));
    }
    check_battery(&rows)?;
    let family_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.family.as_deref() == Some(family.as_str()))
        .collect();
    if family_rows.is_empty() {
        return Err(Error::new(
            ErrorKind::Results,
            "missing family-scoped results",
        ));
    }
    let selected: Vec<_> = match &model.class {
        ModelClass::Causal => family_rows
            .iter()
            .filter(|row| row.split_layer.is_some())
            .collect(),
        ModelClass::Workload(_) => family_rows
            .iter()
            .filter(|row| row.workload_class.is_some())
            .collect(),
    };
    let selected = match selected.as_slice() {
        [row] => *row,
        _ => {
            return Err(Error::new(
                ErrorKind::Certification,
                "expected one consolidated or workload certification",
            ));
        }
    };
    match &model.class {
        ModelClass::Causal => {}
        ModelClass::Workload(class) => {
            if selected.workload_class.as_deref() != Some(class.as_str()) {
                return Err(Error::new(
                    ErrorKind::WorkloadClass,
                    "workload class mismatch",
                ));
            }
        }
    }
    for lane in &model.certification_lanes {
        if !lane_passed(&selected.outcomes, lane) {
            return Err(Error::new(
                ErrorKind::RequiredLane,
                format!("required lane {lane} incomplete"),
            ));
        }
    }
    match &model.class {
        ModelClass::Causal => {
            let count = family_rows.iter().filter(|row| row.mmproj_smoke.0).count();
            if count != usize::from(model.mmproj_artifact.0) {
                return Err(Error::new(
                    ErrorKind::Multimodal,
                    "multimodal evidence incomplete",
                ));
            }
        }
        ModelClass::Workload(_) => {}
    }
    Ok(())
}

fn read_json_documents(bytes: &[u8]) -> Result<Vec<ResultRow>, Error> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| Error::new(ErrorKind::Json, error.to_string()))?;
    let mut remaining = text;
    let mut rows = Vec::new();
    loop {
        remaining = remaining.trim_start_matches(is_space);
        if remaining.is_empty() {
            return Ok(rows);
        }
        if !remaining.starts_with('{') {
            return Err(Error::new(
                ErrorKind::Json,
                "expected a stream of JSON objects",
            ));
        }
        let mut stream = serde_json::Deserializer::from_str(remaining).into_iter::<ResultRow>();
        match stream.next() {
            Some(row) => rows.push(row?),
            None => return Ok(rows),
        }
        remaining = &remaining[stream.byte_offset()..];
    }
}

fn check_battery(rows: &[ResultRow]) -> Result<(), Error> {
    let mut battery = rows
        .iter()
        .filter(|row| row.family.as_deref() == Some("battery"));
    match (battery.next(), battery.next()) {
        (None, None) => Ok(()),
        (Some(row), None) if lane_passed(&row.outcomes, "environment-preflight") => Ok(()),
        _ => Err(Error::new(
            ErrorKind::BatteryPreflight,
            "global battery preflight incomplete",
        )),
    }
}

fn lane_passed(outcomes: &[LaneResult], lane: &str) -> bool {
    let mut matches = outcomes
        .iter()
        .filter(|item| item.name.as_deref() == Some(lane));
    matches!((matches.next(), matches.next()), (Some(item), None)
        if item.status.as_deref() == Some("pass") && item.exit_code.0)
}
