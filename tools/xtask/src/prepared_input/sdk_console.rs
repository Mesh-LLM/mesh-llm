use super::html_modules::asset_references;
use super::{Checked, Rejected, positional, text_io};
use std::fs;
use std::path::Path;

mod references;

pub(super) fn manifest(args: &[String]) -> Checked<String> {
    let [directory] = positional(args, "DIRECTORY")?;
    let root = normalized_directory(directory);
    let mut entries = Vec::new();
    collect(&root, "", &mut entries)?;
    entries.sort();
    let path = root.join("manifest.txt");
    fs::write(&path, entries.join("\n") + "\n")
        .map_err(|error| Rejected(text_io::os_error(&path, &error)))?;
    Ok(String::new())
}

fn collect(directory: &Path, prefix: &str, entries: &mut Vec<String>) -> Checked<()> {
    let children = match fs::read_dir(directory) {
        Ok(children) => children,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(Rejected(text_io::os_error(directory, &error))),
    };
    for child in children {
        let child = child.map_err(|error| Rejected(text_io::os_error(directory, &error)))?;
        let path = child.path();
        let name = format!("{prefix}{}", child.file_name().to_string_lossy());
        let kind = child
            .file_type()
            .map_err(|error| Rejected(text_io::os_error(&path, &error)))?;
        if kind.is_dir() {
            collect(&path, &format!("{name}/"), entries)?;
        } else if path.is_file() && name != "manifest.txt" && !name.starts_with('.') {
            entries.push(name);
        }
    }
    Ok(())
}

pub(super) fn verify(args: &[String]) -> Checked<String> {
    let [directory] = positional(args, "DIRECTORY")?;
    let root = normalized_directory(directory);
    for name in ["index.html", "manifest.txt"] {
        let path = root.join(name);
        if !path.is_file() {
            return Err(format!("missing console {name}: {}", path.display()).into());
        }
    }
    let html = text_io::read_text(&root.join("index.html"))?;
    let refs = asset_references(&html)?;
    let local: Vec<String> = refs
        .iter()
        .filter_map(|value| references::local_path(value))
        .collect();
    let mut missing: Vec<&str> = local
        .iter()
        .filter(|relative| !root.join(relative).is_file())
        .map(String::as_str)
        .collect();
    missing.sort();
    if !missing.is_empty() {
        return Err(format!(
            "console index references missing assets: {}",
            missing.join(", ")
        )
        .into());
    }
    verify_manifest(&root)?;
    if !has_suffix(&root, ".js")? {
        return Err(
            "console assets must include at least one JavaScript asset under assets/".into(),
        );
    }
    if local.iter().any(|path| path.ends_with(".css")) && !has_suffix(&root, ".css")? {
        return Err("console index references CSS, but no CSS asset exists under assets/".into());
    }
    Ok(format!("verified console assets: {}\n", root.display()))
}

fn normalized_directory(directory: &str) -> std::path::PathBuf {
    let path: std::path::PathBuf = Path::new(directory)
        .components()
        .filter(|component| *component != std::path::Component::CurDir)
        .collect();
    if path.as_os_str().is_empty() {
        std::path::PathBuf::from(".")
    } else {
        path
    }
}

fn verify_manifest(root: &Path) -> Checked<()> {
    let text = text_io::read_text(&root.join("manifest.txt"))?;
    let entries: Vec<&str> = text
        .split([
            '\n', '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}',
            '\u{2029}',
        ])
        .map(|line| {
            line.trim_matches(|ch: char| ch.is_whitespace() || matches!(ch, '\u{1c}'..='\u{1f}'))
        })
        .filter(|line| !line.is_empty())
        .collect();
    if !entries.contains(&"index.html") {
        return Err("console manifest.txt must include index.html".into());
    }
    for relative in entries {
        if relative.starts_with('/')
            || relative.contains('\\')
            || relative
                .split('/')
                .any(|part| part.is_empty() || part == "..")
        {
            return Err(format!("unsafe console manifest path: {relative}").into());
        }
        if !root.join(relative).is_file() {
            return Err(format!("console manifest references missing asset: {relative}").into());
        }
    }
    Ok(())
}

fn has_suffix(root: &Path, suffix: &str) -> Checked<bool> {
    let directory = root.join("assets");
    let children = match fs::read_dir(&directory) {
        Ok(children) => children,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(false);
        }
        Err(error) => return Err(Rejected(text_io::os_error(&directory, &error))),
    };
    for child in children {
        let child = child.map_err(|error| Rejected(text_io::os_error(&directory, &error)))?;
        let name = child.file_name();
        let name = name.to_string_lossy();
        if name.len() > suffix.len() && name.ends_with(suffix) {
            return Ok(true);
        }
    }
    Ok(false)
}
