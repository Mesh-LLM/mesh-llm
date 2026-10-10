use super::schema::{Budget, Source};
use crate::command::DynResult;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read as _, Write as _},
    path::Path,
};
const MAX_FILES: usize = 16384;
fn roster(root: &Path, budget: &Budget<'_>) -> DynResult<Vec<String>> {
    let mut pending = vec![(root.to_path_buf(), 0_usize)];
    let mut files = Vec::new();
    while let Some((directory, depth)) = pending.pop() {
        budget.check()?;
        if depth > 32 {
            return Err("source nesting bound".into());
        }
        for entry in fs::read_dir(directory)? {
            budget.check()?;
            let entry = entry?;
            let ty = entry.file_type()?;
            if ty.is_symlink() {
                return Err("snapshot symlink refused".into());
            }
            if ty.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else if ty.is_file() {
                let path = entry.path();
                if path.strip_prefix(root)?.components().any(|c| {
                    c.as_os_str()
                        .to_str()
                        .is_none_or(|n| n.contains('\\') || n.chars().any(char::is_control))
                }) {
                    return Err("snapshot component grammar".into());
                }
                let relative = path
                    .strip_prefix(root)?
                    .to_str()
                    .ok_or("snapshot path must be UTF-8")?
                    .replace('\\', "/");
                if relative.chars().any(char::is_control) || relative.contains('\\') {
                    return Err("snapshot path grammar".into());
                }
                files.push(relative);
            } else {
                return Err("nonregular snapshot entry".into());
            }
            if files.len() + pending.len() > MAX_FILES {
                return Err("source roster bound".into());
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err("empty source snapshot".into());
    }
    Ok(files)
}
fn file(
    path: &Path,
    output: Option<&Path>,
    used: &mut u64,
    maximum: u64,
    budget: &Budget<'_>,
) -> DynResult<String> {
    budget.check()?;
    if !fs::symlink_metadata(path)?.is_file() {
        return Err("regular source required".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut source = options.open(path)?;
    if !source.metadata()?.is_file() {
        return Err("opened source is not regular".into());
    }
    let mut destination = output
        .map(|p| tempfile::NamedTempFile::new_in(p.parent().unwrap()))
        .transpose()?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        budget.check()?;
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        *used = used
            .checked_add(read as u64)
            .filter(|v| *v <= maximum)
            .ok_or("source byte bound")?;
        hash.update(&buffer[..read]);
        if let Some(dest) = destination.as_mut() {
            dest.write_all(&buffer[..read])?;
        }
    }
    budget.check()?;
    if let (Some(dest), Some(path)) = (destination, output) {
        dest.as_file().sync_all()?;
        budget.check()?;
        dest.persist_noclobber(path)
            .map_err(|_| "fresh tokenizer file refused")?;
    }
    Ok(hex::encode(hash.finalize()))
}
fn tree(rows: &BTreeMap<String, String>) -> DynResult<String> {
    let mut hash = Sha256::new();
    for (name, pin) in rows {
        hash.update((name.len() as u64).to_be_bytes());
        hash.update(name.as_bytes());
        hash.update(hex::decode(pin)?);
    }
    Ok(hex::encode(hash.finalize()))
}
fn pins(source: &Source, maximum: u64, budget: &Budget<'_>) -> DynResult<BTreeMap<String, String>> {
    let mut used = 0;
    roster(&source.directory, budget)?
        .into_iter()
        .map(|name| {
            let hash = file(
                &source.directory.join(&name),
                None,
                &mut used,
                maximum,
                budget,
            )?;
            Ok((name, hash))
        })
        .collect()
}
pub(super) fn materialize(
    source: &Source,
    output: &Path,
    expected: &str,
    maximum: u64,
    budget: &Budget<'_>,
) -> DynResult<Value> {
    let before = pins(source, maximum, budget)?;
    let source_pin = tree(&before)?;
    if source_pin != source.source_tree_sha256 {
        return Err("supplied source tree pin mismatch".into());
    }
    let granite = source.key == "granite-h1-hybrid";
    let mut selected = BTreeMap::new();
    for (relative, pin) in &before {
        let basename = Path::new(relative)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("basename")?;
        let keep = if granite {
            basename != "README.md"
        } else {
            [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ]
            .contains(&basename)
                && !relative.contains('/')
        };
        if keep
            && selected
                .insert(basename.to_owned(), (relative, pin))
                .is_some()
        {
            return Err("flattened snapshot name collision".into());
        }
    }
    if selected.is_empty() || !selected.contains_key("tokenizer.json") {
        return Err("required tokenizer asset missing".into());
    }
    let projected: BTreeMap<_, _> = selected
        .iter()
        .map(|(name, (_, pin))| (name.clone(), (*pin).clone()))
        .collect();
    if tree(&projected)? != expected {
        return Err("projected tokenizer tree differs from consumed pin".into());
    }
    budget.check()?;
    fs::create_dir(output)?;
    let mut used = 0;
    for (name, (relative, expected_pin)) in selected {
        let actual = file(
            &source.directory.join(relative),
            Some(&output.join(&name)),
            &mut used,
            maximum,
            budget,
        )?;
        if &actual != expected_pin {
            return Err("source changed while materializing".into());
        }
    }
    let after = pins(source, maximum, budget)?;
    if after != before {
        return Err("source custody changed".into());
    }
    budget.check()?;
    Ok(
        json!({"key":source.key,"source_kind":source.kind,"source_tree_sha256":source_pin,"output_tree_sha256":expected,"files":projected,"source_unchanged":true,"exporter_attested":false,"semantic_qualification_performed":false}),
    )
}
#[cfg(test)]
pub(super) fn fixture_tree(root: &Path) -> String {
    crate::product::digest::tree_sha256(root).unwrap_or_else(|_| panic!("fixture digest"))
}
