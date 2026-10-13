//! Maintained Mesh and Skippy crate metadata and repository-local README links.
use crate::command::DynResult;
use std::fs;
use std::path::Path;

pub(super) fn check(root: &Path) -> DynResult<()> {
    let errors = problems(root)?;
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}

fn problems(root: &Path) -> DynResult<Vec<String>> {
    let root = root.canonicalize()?;
    let mut errors = Vec::new();
    for product in ["mesh", "skippy"] {
        let crates = root.join(product).join("crates");
        if !crates.is_dir() {
            errors.push(format!(
                "{}: missing product crates directory",
                crates.display()
            ));
            continue;
        }
        let mut manifests = fs::read_dir(crates)?
            .map(|entry| entry.map(|entry| entry.path().join("Cargo.toml")))
            .collect::<Result<Vec<_>, _>>()?;
        manifests.retain(|path| path.is_file());
        manifests.sort();
        for manifest in manifests {
            check_crate(&root, &manifest, &mut errors);
        }
    }
    Ok(errors)
}

fn check_crate(root: &Path, manifest: &Path, errors: &mut Vec<String>) {
    let package = fs::read_to_string(manifest)
        .map_err(|error| error.to_string())
        .and_then(|source| {
            toml::from_str::<toml::Value>(&source).map_err(|error| error.to_string())
        })
        .and_then(|value| {
            value
                .get("package")
                .and_then(toml::Value::as_table)
                .cloned()
                .ok_or_else(|| "missing package table".to_owned())
        });
    let package = match package {
        Ok(package) => package,
        Err(error) => {
            errors.push(format!(
                "{}: cannot read package metadata: {error}",
                manifest.display()
            ));
            return;
        }
    };
    if !package
        .get("description")
        .and_then(toml::Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        errors.push(format!(
            "{}: missing non-empty package description",
            manifest.display()
        ));
    }
    if package
        .get("readme")
        .is_some_and(|value| value.as_str() != Some("README.md"))
    {
        errors.push(format!(
            "{}: package readme must be README.md",
            manifest.display()
        ));
    }
    let directory = manifest.parent().expect("manifest has crate directory");
    let readme = directory.join("README.md");
    match fs::read_to_string(&readme) {
        Ok(source) => check_links(root, directory, &readme, &source, errors),
        Err(error) => errors.push(format!(
            "{}: missing or unreadable crate README: {error}",
            readme.display()
        )),
    }
}

fn check_links(
    root: &Path,
    directory: &Path,
    readme: &Path,
    source: &str,
    errors: &mut Vec<String>,
) {
    let mut fence = None;
    for (index, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        let marker = if trimmed.starts_with("```") {
            Some('`')
        } else if trimmed.starts_with("~~~") {
            Some('~')
        } else {
            None
        };
        if let Some(marker) = marker {
            fence = match fence {
                Some(active) if active == marker => None,
                Some(active) => Some(active),
                None => Some(marker),
            };
            continue;
        }
        if fence.is_some() {
            continue;
        }
        for (offset, _) in line.match_indices("](") {
            let suffix = &line[offset + 2..];
            let destination = if let Some(angle) = suffix.strip_prefix('<') {
                angle.split_once('>').map(|(destination, _)| destination)
            } else {
                Some(
                    suffix
                        .split(|character: char| character.is_whitespace() || character == ')')
                        .next()
                        .unwrap_or(""),
                )
            };
            if let Some(destination) = destination {
                check_destination(root, directory, readme, index + 1, destination, errors);
            }
        }
    }
}

fn check_destination(
    root: &Path,
    directory: &Path,
    readme: &Path,
    line: usize,
    destination: &str,
    errors: &mut Vec<String>,
) {
    if destination.starts_with("//") || url::Url::parse(destination).is_ok() {
        return;
    }
    let path = destination.split(['#', '?']).next().unwrap_or("");
    if path.is_empty() {
        return;
    }
    let decoded = match decode_path(path) {
        Ok(decoded) => decoded,
        Err(error) => {
            errors.push(format!(
                "{}:{line}: invalid local link {destination}: {error}",
                readme.display()
            ));
            return;
        }
    };
    let path = if decoded.starts_with('/') {
        root.join(decoded.trim_start_matches('/'))
    } else {
        directory.join(decoded)
    };
    match path.canonicalize() {
        Ok(target) if target.starts_with(root) => {}
        Ok(_) => errors.push(format!(
            "{}:{line}: local link leaves repository: {destination}",
            readme.display()
        )),
        Err(_) => errors.push(format!(
            "{}:{line}: broken local link: {destination}",
            readme.display()
        )),
    }
}

fn decode_path(path: &str) -> Result<String, std::string::FromUtf8Error> {
    let mut result = Vec::new();
    let bytes = path.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let high = char::from(bytes[index + 1]).to_digit(16);
            let low = char::from(bytes[index + 2]).to_digit(16);
            if let (Some(high), Some(low)) = (high, low) {
                result.push((high * 16 + low) as u8);
                index += 3;
                continue;
            }
        }
        result.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(result)
}

#[cfg(test)]
#[path = "product_documentation_tests.rs"]
mod tests;
