//! Two-pass Parquet admission keeps duplicate metadata, not dataset message bodies.
use super::{Cohorts, RecordedTrajectory, Selection, messages};
use crate::DynResult;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
    schema::types::Type,
};
use std::{collections::BTreeMap, fs::File, path::Path};
const COLUMNS: [&str; 8] = [
    "session_id",
    "source_dataset",
    "agent_framework",
    "recorded_model",
    "messages_json",
    "n_turns",
    "max_isl",
    "total_tokens",
];
type Columns = BTreeMap<String, Field>;
type Order = ([u8; 16], String);
#[derive(Clone)]
struct Candidate {
    index: usize,
    framework: String,
    source: String,
    model: Option<String>,
    turns: u64,
    isl: u64,
    tokens: u64,
    body_hash: [u8; 16],
}
impl Candidate {
    fn preferred(&self, prior: &Self) -> bool {
        self.isl
            .cmp(&prior.isl)
            .reverse()
            .then_with(|| self.tokens.cmp(&prior.tokens).reverse())
            .then_with(|| self.body_hash.cmp(&prior.body_hash))
            .then_with(|| self.framework.cmp(&prior.framework))
            .then_with(|| self.source.cmp(&prior.source))
            .then_with(|| self.model.cmp(&prior.model))
            .then_with(|| self.turns.cmp(&prior.turns))
            .is_lt()
    }
}
fn reader(path: &Path) -> DynResult<SerializedFileReader<File>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("dataset must be a regular Parquet file".into());
    }
    Ok(SerializedFileReader::new(file)?)
}
fn projection(reader: &SerializedFileReader<File>) -> DynResult<Type> {
    let schema = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema();
    let fields = schema
        .get_fields()
        .iter()
        .filter(|f| COLUMNS.contains(&f.name()))
        .cloned()
        .collect::<Vec<_>>();
    if fields.len() != 8
        || fields
            .iter()
            .map(|f| f.name())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != 8
    {
        return Err("trajectory Parquet requires all eight columns exactly once".into());
    }
    Ok(Type::group_type_builder(schema.name())
        .with_fields(fields)
        .build()?)
}
fn text<'a>(row: &'a Columns, key: &str) -> DynResult<&'a str> {
    match row.get(key) {
        Some(Field::Str(s)) => Ok(s),
        Some(Field::Bytes(b)) => Ok(std::str::from_utf8(b.data())?),
        _ => Err(format!("{key} must be UTF-8 text").into()),
    }
}
fn number(row: &Columns, key: &str) -> DynResult<Option<u64>> {
    Ok(match row.get(key) {
        Some(Field::Null) => None,
        Some(Field::Long(n)) => u64::try_from(*n).ok(),
        Some(Field::Int(n)) => u64::try_from(*n).ok(),
        Some(Field::Byte(n)) => u64::try_from(*n).ok(),
        Some(Field::Short(n)) => u64::try_from(*n).ok(),
        Some(Field::UByte(n)) => Some(u64::from(*n)),
        Some(Field::UShort(n)) => Some(u64::from(*n)),
        Some(Field::UInt(n)) => Some(u64::from(*n)),
        Some(Field::ULong(n)) => Some(*n),
        _ => return Err(format!("{key} must be an integer").into()),
    })
}
fn eligible(row: &Columns, s: &Selection) -> DynResult<bool> {
    if matches!(row.get("source_dataset"), Some(Field::Null)) {
        return Ok(false);
    }
    let source = text(row, "source_dataset")?;
    if !s.sources.iter().any(|v| v == source) {
        return Ok(false);
    }
    Ok(
        matches!((number(row,"max_isl")?,number(row,"n_turns")?), (Some(isl),Some(turns)) if isl>=s.min_isl&&isl<s.max_isl_exclusive&&turns>=s.min_turns),
    )
}
fn columns(row: Row) -> DynResult<Columns> {
    let fields = row.into_columns();
    let count = fields.len();
    let values = fields.into_iter().collect::<Columns>();
    if values.len() != count {
        return Err("duplicate Parquet column".into());
    }
    Ok(values)
}
fn model(row: &Columns) -> DynResult<Option<String>> {
    match row.get("recorded_model") {
        Some(Field::Null) => Ok(None),
        _ => Ok(Some(text(row, "recorded_model")?.into())),
    }
}
fn recorded(row: &Columns) -> DynResult<RecordedTrajectory> {
    let messages = messages(text(row, "messages_json")?)?;
    Ok(RecordedTrajectory {
        session_id: text(row, "session_id")?.into(),
        source_dataset: text(row, "source_dataset")?.into(),
        agent_framework: text(row, "agent_framework")?.into(),
        recorded_model: model(row)?,
        n_turns: number(row, "n_turns")?.ok_or("missing turns")?,
        max_isl: number(row, "max_isl")?.ok_or("missing ISL")?,
        total_tokens: number(row, "total_tokens")?.ok_or("missing total tokens")?,
        assistant_turns: messages.iter().filter(|m| m.role == "assistant").count(),
        messages,
    })
}
fn duplicates(
    reader: &SerializedFileReader<File>,
    s: &Selection,
) -> DynResult<BTreeMap<String, Candidate>> {
    let mut best = BTreeMap::<String, Candidate>::new();
    for (index, row) in reader.get_row_iter(Some(projection(reader)?))?.enumerate() {
        let row = columns(row?)?;
        if !eligible(&row, s)? {
            continue;
        }
        let id = text(&row, "session_id")?;
        if id.is_empty() || id.contains(['\n', '\r']) {
            return Err("session ID must be nonempty and single-line".into());
        }
        let candidate = Candidate {
            index,
            framework: text(&row, "agent_framework")?.into(),
            source: text(&row, "source_dataset")?.into(),
            model: model(&row)?,
            turns: number(&row, "n_turns")?.ok_or("missing turns")?,
            isl: number(&row, "max_isl")?.ok_or("missing ISL")?,
            tokens: number(&row, "total_tokens")?.ok_or("missing total tokens")?,
            body_hash: md5::compute(text(&row, "messages_json")?).0,
        };
        if best.get(id).is_none_or(|old| candidate.preferred(old)) {
            best.insert(id.into(), candidate);
        }
    }
    Ok(best)
}
type Bucket = BTreeMap<Order, Result<RecordedTrajectory, String>>;
fn selected(
    reader: &SerializedFileReader<File>,
    s: &Selection,
    allocation: &[usize],
    best: &BTreeMap<String, Candidate>,
) -> DynResult<BTreeMap<String, Bucket>> {
    let mut chosen = s
        .frameworks
        .iter()
        .map(|f| (f.clone(), Bucket::new()))
        .collect::<BTreeMap<_, _>>();
    for (index, row) in reader.get_row_iter(Some(projection(reader)?))?.enumerate() {
        let row = columns(row?)?;
        if !eligible(&row, s)? {
            continue;
        }
        let id = text(&row, "session_id")?;
        let candidate = best
            .get(id)
            .ok_or("dataset changed between selection passes")?;
        if candidate.index != index {
            continue;
        }
        let Some(framework_index) = s.frameworks.iter().position(|f| f == &candidate.framework)
        else {
            continue;
        };
        let admitted = recorded(&row).map_err(|e| e.to_string());
        if admitted
            .as_ref()
            .is_ok_and(|r| r.assistant_turns < usize::try_from(s.min_turns).unwrap_or(usize::MAX))
        {
            continue;
        }
        let bucket = chosen
            .get_mut(&candidate.framework)
            .ok_or("unknown admitted framework")?;
        bucket.insert((md5::compute(id).0, id.into()), admitted);
        if bucket.len() > allocation[framework_index] * s.cohorts.len() {
            bucket.pop_last();
        }
    }
    Ok(chosen)
}
pub fn select(path: &Path, s: &Selection) -> DynResult<Cohorts> {
    let allocation = s.allocation()?;
    let reader = reader(path)?;
    let best = duplicates(&reader, s)?;
    let mut chosen = selected(&reader, s, &allocation, &best)?;
    let mut cohorts = s
        .cohorts
        .iter()
        .map(|n| (n.clone(), Vec::new()))
        .collect::<Cohorts>();
    for (i, framework) in s.frameworks.iter().enumerate() {
        let rows = chosen.remove(framework).ok_or("missing framework bucket")?;
        let required = allocation[i] * s.cohorts.len();
        if rows.len() != required {
            return Err(format!(
                "framework {framework} has {} eligible trajectories; {required} required",
                rows.len()
            )
            .into());
        }
        for (index, row) in rows.into_values().enumerate() {
            cohorts
                .get_mut(&s.cohorts[index / allocation[i]])
                .ok_or("missing cohort")?
                .push(row.map_err(|e| format!("framework {framework}: {e}"))?);
        }
    }
    Ok(cohorts)
}
