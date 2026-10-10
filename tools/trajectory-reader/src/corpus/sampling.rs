use super::{
    acquisition::{Artifact, Format, inspect, regular},
    config::Source,
    digest,
};
use crate::DynResult;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read},
};

pub(super) const ALGORITHM: &str = "sha256-source-row-v1";
const ROW_BYTES: usize = 16 * 1024 * 1024;
pub(super) fn sample(
    files: &[Artifact],
    source: &Source,
    seed: u64,
    limit: usize,
) -> DynResult<Vec<Value>> {
    if limit == 0 || limit > 100_000 {
        return Err("sample limit outside 1..100000".into());
    }
    let identity = serde_json::to_string(&(
        seed,
        &source.name,
        &source.dataset,
        &source.revision,
        &source.config,
        &source.split,
    ))?;
    let mut selected = BTreeMap::new();
    let mut index = 0u64;
    let mut retained_bytes = 0usize;
    for file in files {
        let mut visit = |row: Value| -> DynResult<()> {
            if !row.is_object() {
                return Err("dataset row requires an object".into());
            }
            let rank = digest(format!("{identity}\n{index}").as_bytes());
            let key = (rank, index);
            index = index.checked_add(1).ok_or("dataset row index overflow")?;
            if selected.len() == limit
                && selected
                    .last_key_value()
                    .is_some_and(|(last, _)| last <= &key)
            {
                return Ok(());
            }
            let row_bytes = row.to_string().len();
            if row_bytes > ROW_BYTES {
                return Err("selected dataset row exceeds 16MiB".into());
            }
            retained_bytes = retained_bytes
                .checked_add(row_bytes)
                .ok_or("sample memory overflow")?;
            selected.insert(key, (row, row_bytes));
            if selected.len() > limit
                && let Some((_, (_, bytes))) = selected.pop_last()
            {
                retained_bytes -= bytes;
            }
            if retained_bytes > 128 * 1024 * 1024 {
                return Err("retained samples exceed 128MiB".into());
            }
            Ok(())
        };
        match file.format {
            Format::Parquet => {
                let reader = SerializedFileReader::new(regular(&file.path)?)?;
                for row in reader.get_row_iter(None)? {
                    visit(row?.to_json_value())?;
                }
            }
            Format::Jsonl => jsonl(file, &mut visit)?,
        }
        let after = inspect(&file.path, file.format)?;
        if after.sha256 != file.sha256 || after.bytes != file.bytes {
            return Err("dataset artifact changed during sampling".into());
        }
    }
    Ok(selected.into_values().map(|(row, _)| row).collect())
}
fn jsonl(file: &Artifact, visit: &mut impl FnMut(Value) -> DynResult<()>) -> DynResult<()> {
    let mut reader = BufReader::new(regular(&file.path)?);
    loop {
        let mut line = Vec::new();
        let count = Read::by_ref(&mut reader)
            .take((ROW_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)?;
        if count == 0 {
            break;
        }
        if count > ROW_BYTES {
            return Err("JSONL row exceeds 16MiB".into());
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        visit(serde_json::from_slice(&line)?)?;
    }
    Ok(())
}
