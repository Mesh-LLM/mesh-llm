use super::{PatchError, ShardError};
use std::collections::BTreeSet;

pub(super) struct DiffSection<'a> {
    pub(super) source: &'a str,
    pub(super) bytes: &'a [u8],
}

pub(super) fn split(diff: &[u8]) -> Result<Vec<DiffSection<'_>>, ShardError> {
    let text = std::str::from_utf8(diff).map_err(|_| PatchError::InvalidUtf8)?;
    let mut headers = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        if let Some((source, end)) = header(&text[cursor..]) {
            headers.push((cursor, source));
            cursor += end;
        }
        match text[cursor..].find('\n') {
            Some(newline) => cursor += newline + 1,
            None => break,
        }
    }
    if !matches!(headers.first(), Some((0, _))) {
        return Err(ShardError::DiffFormat);
    }
    let mut sources = BTreeSet::new();
    let mut sections = Vec::with_capacity(headers.len());
    for (index, (start, source)) in headers.iter().enumerate() {
        if !sources.insert(*source) {
            return Err(ShardError::RepeatedSource((*source).to_owned()));
        }
        let end = headers
            .get(index + 1)
            .map_or(diff.len(), |(start, _)| *start);
        sections.push(DiffSection {
            source,
            bytes: &diff[*start..end],
        });
    }
    Ok(sections)
}

fn header(text: &str) -> Option<(&str, usize)> {
    let after_prefix = text.strip_prefix("diff --git a/")?;
    let after_models = after_prefix.strip_prefix("src/models/")?;
    let space = after_models.find(' ')?;
    if space == 0 {
        return None;
    }
    let destination = after_models[space..].strip_prefix(" b/")?;
    let destination_length = destination.find('\n').unwrap_or(destination.len());
    if destination_length == 0 {
        return None;
    }
    let source_length = "src/models/".len() + space;
    let end = text.len() - destination.len() + destination_length;
    Some((&after_prefix[..source_length], end))
}
