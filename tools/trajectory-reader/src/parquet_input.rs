use std::{collections::BTreeMap, fs::File, path::Path};

use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
    schema::types::Type,
};

use super::selection::{Selection, Selector, Trajectory};
use crate::DynResult;

const COLUMNS: [&str; 6] = [
    "session_id",
    "source_dataset",
    "messages_json",
    "n_turns",
    "max_isl",
    "total_tokens",
];

pub fn select(path: &Path, selection: &Selection) -> DynResult<Vec<Trajectory>> {
    let mut selector = Selector::new(selection)?;
    let reader = SerializedFileReader::new(File::open(path)?)?;
    let schema = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema();
    let fields = schema
        .get_fields()
        .iter()
        .filter(|field| COLUMNS.contains(&field.name()))
        .cloned()
        .collect::<Vec<_>>();
    if fields.len() != COLUMNS.len() {
        return Err(
            "trajectory Parquet must contain all six selection columns exactly once".into(),
        );
    }
    let projection = Type::group_type_builder(schema.name())
        .with_fields(fields)
        .build()?;
    for row in reader.get_row_iter(Some(projection))? {
        if let Some(row) = trajectory(row?, selection)? {
            selector.observe(row);
        }
    }
    selector.finish()
}

pub fn trajectory(row: Row, selection: &Selection) -> DynResult<Option<Trajectory>> {
    let columns = row.into_columns().into_iter().collect::<BTreeMap<_, _>>();
    if matches!(columns.get("source_dataset"), Some(Field::Null)) {
        return Ok(None);
    }
    let source_dataset = text(&columns, "source_dataset")?;
    if !selection.sources.contains(&source_dataset) {
        return Ok(None);
    }
    let Some(n_turns) = filter_integer(&columns, "n_turns")? else {
        return Ok(None);
    };
    let Some(max_isl) = filter_integer(&columns, "max_isl")? else {
        return Ok(None);
    };
    if !selection.eligible_fields(&source_dataset, max_isl, n_turns) {
        return Ok(None);
    }
    Ok(Some(Trajectory {
        session_id: text(&columns, "session_id")?,
        source_dataset,
        messages_json: text(&columns, "messages_json")?,
        n_turns,
        max_isl,
        total_tokens: integer(&columns, "total_tokens")?,
    }))
}

fn text(columns: &BTreeMap<String, Field>, name: &str) -> DynResult<String> {
    match columns.get(name) {
        Some(Field::Str(value)) => Ok(value.clone()),
        Some(Field::Bytes(value)) => Ok(std::str::from_utf8(value.data())?.to_owned()),
        _ => Err(format!("trajectory column {name} must be UTF-8 text").into()),
    }
}

fn integer(columns: &BTreeMap<String, Field>, name: &str) -> DynResult<u64> {
    let value = match columns.get(name) {
        Some(Field::Byte(value)) => u64::try_from(*value)?,
        Some(Field::Short(value)) => u64::try_from(*value)?,
        Some(Field::Int(value)) => u64::try_from(*value)?,
        Some(Field::Long(value)) => u64::try_from(*value)?,
        Some(Field::UByte(value)) => u64::from(*value),
        Some(Field::UShort(value)) => u64::from(*value),
        Some(Field::UInt(value)) => u64::from(*value),
        Some(Field::ULong(value)) => *value,
        _ => return Err(format!("trajectory column {name} must be a nonnegative integer").into()),
    };
    Ok(value)
}

fn filter_integer(columns: &BTreeMap<String, Field>, name: &str) -> DynResult<Option<u64>> {
    match columns.get(name) {
        Some(Field::Null) => Ok(None),
        Some(Field::Byte(value)) if *value < 0 => Ok(None),
        Some(Field::Short(value)) if *value < 0 => Ok(None),
        Some(Field::Int(value)) if *value < 0 => Ok(None),
        Some(Field::Long(value)) if *value < 0 => Ok(None),
        _ => integer(columns, name).map(Some),
    }
}
