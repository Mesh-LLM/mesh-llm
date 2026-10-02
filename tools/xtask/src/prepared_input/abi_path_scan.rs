use super::{Checked, Rejected, text_io};
use crate::repository::text;
use std::path::{Path, PathBuf};

struct ScanOperands<'a> {
    stage: PathBuf,
    forbidden: Vec<&'a str>,
}

pub(super) fn run(args: &[String]) -> Checked<String> {
    let [stage, values @ ..] = args else {
        return Err(Rejected("expected stage argument".to_owned()));
    };
    let mut forbidden = Vec::new();
    for value in values {
        if !value.is_empty() && !forbidden.contains(&value.as_str()) {
            forbidden.push(value.as_str());
        }
    }
    scan(&ScanOperands {
        stage: PathBuf::from(stage),
        forbidden,
    })?;
    Ok(String::new())
}

fn scan(operands: &ScanOperands<'_>) -> Checked<()> {
    let mut entries = Vec::new();
    collect(&operands.stage, Path::new(""), &mut entries);
    entries.sort();
    for relative in entries {
        let path = operands.stage.join(&relative);
        if !path.is_file() {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|error| text_io::os_error(&path, &error))?;
        for value in &operands.forbidden {
            if bytes
                .windows(value.len())
                .any(|window| window == value.as_bytes())
            {
                return Err(Rejected(format!(
                    "portable static ABI retained producer-local path {} in {}",
                    text::repr(value),
                    relative.display()
                )));
            }
        }
    }
    Ok(())
}

fn collect(directory: &Path, prefix: &Path, entries: &mut Vec<PathBuf>) {
    let Ok(children) = std::fs::read_dir(directory) else {
        return;
    };
    for child in children.flatten() {
        let relative = prefix.join(child.file_name());
        entries.push(relative.clone());
        if child.file_type().is_ok_and(|kind| kind.is_dir()) {
            collect(&child.path(), &relative, entries);
        }
    }
}

#[cfg(test)]
#[path = "abi_path_scan_tests.rs"]
mod tests;
