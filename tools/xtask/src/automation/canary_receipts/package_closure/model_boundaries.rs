//! Read-only paired layer-boundary inventory for narrow agent manifest admission.
use super::process;
use crate::command::DynResult;
use std::{collections::BTreeSet, fs, path::Path};

pub(super) fn inventory(native: &Path) -> DynResult<(BTreeSet<String>, BTreeSet<String>)> {
    let root = native.canonicalize()?;
    let directory = root.join("src/models");
    if !directory.is_dir() || !directory.canonicalize()?.starts_with(&root) {
        return Err("prepared native source has no contained src/models".into());
    }
    let mut sources = BTreeSet::new();
    let mut registered = BTreeSet::new();
    for entry in fs::read_dir(&directory)? {
        process::check()?;
        let path = entry?.path();
        if path.extension().is_none_or(|extension| extension != "cpp") {
            continue;
        }
        if !fs::symlink_metadata(&path)?.is_file() || !path.canonicalize()?.starts_with(&root) {
            return Err("native model inventory member escapes prepared source".into());
        }
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or("invalid native model source name")?
            .to_owned();
        sources.insert(name.clone());
        let bytes = fs::read(path)?;
        let text = String::from_utf8_lossy(&bytes);
        let executable = mask(&text);
        if call(&executable, "begin_block") && call(&executable, "end_block") {
            registered.insert(name);
        }
    }
    Ok((sources, registered))
}
fn call(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, _)| {
        if text[..at]
            .chars()
            .next_back()
            .is_some_and(|char| char.is_alphanumeric() || char == '_')
        {
            return false;
        }
        text[at + name.len()..].trim_start().starts_with('(')
    })
}
fn ident(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}
fn raw_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start != 0 && ident(bytes[start - 1]) {
        return None;
    }
    let rest = &bytes[start..];
    let prefix = [b"u8R\"".as_slice(), b"uR\"", b"UR\"", b"LR\"", b"R\""]
        .into_iter()
        .find(|prefix| rest.starts_with(prefix))?;
    let delimiter_start = start + prefix.len();
    let extent = bytes[delimiter_start..]
        .iter()
        .position(|byte| *byte == b'(')?;
    if extent > 16
        || bytes[delimiter_start..delimiter_start + extent]
            .iter()
            .any(|byte| byte.is_ascii_whitespace() || b" ()\\".contains(byte))
    {
        return None;
    }
    let mut closing = vec![b')'];
    closing.extend_from_slice(&bytes[delimiter_start..delimiter_start + extent]);
    closing.push(b'"');
    let content = delimiter_start + extent + 1;
    Some(
        bytes[content..]
            .windows(closing.len())
            .position(|part| part == closing)
            .map_or(bytes.len(), |offset| content + offset + closing.len()),
    )
}
fn blank(mask: &mut [u8], start: usize, end: usize) {
    for byte in &mut mask[start..end] {
        if *byte != b'\n' {
            *byte = b' ';
        }
    }
}
pub(super) fn mask(text: &str) -> String {
    mask_with_preserved(text, &[])
}
pub(super) fn mask_with_preserved(text: &str, preserved: &[&str]) -> String {
    let bytes = text.as_bytes();
    let mut masked = bytes.to_vec();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at..].starts_with(b"//") {
            let end = bytes[at..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| at + offset);
            blank(&mut masked, at, end);
            at = end;
        } else if bytes[at..].starts_with(b"/*") {
            let end = bytes[at + 2..]
                .windows(2)
                .position(|pair| pair == b"*/")
                .map_or(bytes.len(), |offset| at + offset + 4);
            blank(&mut masked, at, end);
            at = end;
        } else if let Some(end) = raw_end(bytes, at) {
            blank(&mut masked, at, end);
            at = end;
        } else if matches!(bytes[at], b'\'' | b'"') {
            let quote = bytes[at];
            let mut end = at + 1;
            while end < bytes.len() {
                if bytes[end] == b'\\' {
                    end = (end + 2).min(bytes.len());
                    continue;
                }
                let terminal = bytes[end] == quote;
                end += 1;
                if terminal {
                    break;
                }
            }
            if quote != b'"' || !preserved.contains(&&text[at..end]) {
                blank(&mut masked, at, end);
            }
            at = end;
        } else {
            at += 1;
        }
    }
    String::from_utf8(masked).expect("mask replaces only complete comments and literals")
}

#[cfg(test)]
mod call_tests {
    use super::{call, mask};
    #[test]
    fn paired_call_names_reject_identifier_mentions_and_masked_literals() {
        assert!(call("stage->begin_block \n (layer);", "begin_block"));
        for source in [
            "not_begin_block(layer)",
            "begin_block_suffix(layer)",
            "begin_block;",
            "/* begin_block(layer) */",
            r#"auto text=R"fake(begin_block(layer))fake";"#,
        ] {
            assert!(!call(&mask(source), "begin_block"), "{source}");
        }
        assert!(!call("ébegin_block(layer)", "begin_block"));
    }
}
