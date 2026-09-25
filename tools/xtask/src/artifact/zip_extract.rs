//! `artifact extract-zip <archive> <destination>`: extract a ZIP archive
//! without permitting path or symlink escapes. Diagnostics, streams and
//! statuses match `scripts/safe-extract-zip.py`; an exception the legacy
//! script leaves uncaught is reported as its final traceback line.

use super::zip_directory::{self, Archive, Failure, Info};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: scripts/safe-extract-zip.py ARCHIVE.zip DESTINATION\n";
const S_IFMT: u32 = 0o170_000;
const S_IFDIR: u32 = 0o040_000;
const S_IFREG: u32 = 0o100_000;
const S_IFLNK: u32 = 0o120_000;

pub(super) fn run(args: &[String]) -> CheckReport {
    let [archive, destination] = args else {
        return CheckReport::failure(String::new(), USAGE.to_owned());
    };
    match extract(Path::new(archive), Path::new(destination)) {
        Ok(()) => CheckReport::success(String::new()),
        Err(Failure::Unsafe(message)) => {
            CheckReport::failure(String::new(), format!("unsafe ZIP archive: {message}\n"))
        }
        Err(Failure::Raised(line)) => CheckReport::failure(String::new(), format!("{line}\n")),
    }
}

fn fail<T>(message: String) -> Result<T, Failure> {
    Err(Failure::Unsafe(message))
}

/// An uncaught `OSError` subclass, as its traceback's last line.
fn os_failure(error: &std::io::Error, path: &Path) -> Failure {
    Failure::Raised(os_error_line(error, &python_path_display(path)))
}

fn os_error_line(error: &std::io::Error, shown: &str) -> String {
    let class = match error.raw_os_error() {
        Some(2) => "FileNotFoundError",
        Some(1 | 13) => "PermissionError",
        Some(17) => "FileExistsError",
        Some(20) => "NotADirectoryError",
        Some(21) => "IsADirectoryError",
        _ => "OSError",
    };
    format!("{class}: {}", os_error_text(error, shown))
}

enum Kind {
    Directory,
    File,
    Symlink(String),
}

struct Entry<'a> {
    info: &'a Info,
    parts: Vec<String>,
    kind: Kind,
    mode: u32,
}

fn has_unportable_text(text: &str) -> bool {
    let bytes = text.as_bytes();
    text.is_empty()
        || text.contains(['\0', '\r', '\n', '\t', '\\'])
        || text.starts_with('/')
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
}

/// `PurePosixPath(text).parts` for a relative path.
fn posix_parts(text: &str) -> Vec<String> {
    text.split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(str::to_owned)
        .collect()
}

fn portable_parts(name: &str) -> Result<Vec<String>, Failure> {
    if has_unportable_text(name) {
        return fail(format!(
            "entry is not a portable relative path: {}",
            repr(name)
        ));
    }
    let parts = posix_parts(name);
    if parts.is_empty() || parts.iter().any(|part| part == "..") {
        return fail(format!("entry escapes the extraction root: {}", repr(name)));
    }
    Ok(parts)
}

fn resolve_link(parts: &[String], target: &str) -> Result<(), Failure> {
    if has_unportable_text(target) {
        return fail(format!("symlink target is not portable: {}", repr(target)));
    }
    let mut resolved = parts[..parts.len() - 1].to_vec();
    for part in posix_parts(target) {
        if part != ".." {
            resolved.push(part);
        } else if resolved.pop().is_none() {
            let shown = repr(target);
            return fail(format!(
                "symlink target escapes the extraction root: {shown}"
            ));
        }
    }
    if resolved.is_empty() {
        let shown = repr(target);
        return fail(format!(
            "symlink target resolves to the extraction root: {shown}"
        ));
    }
    Ok(())
}

fn classify<'a>(archive: &Archive, info: &'a Info) -> Result<Entry<'a>, Failure> {
    let name = if info.is_dir() {
        info.filename.trim_end_matches('/')
    } else {
        info.filename.as_str()
    };
    let parts = portable_parts(name)?;
    let mode = info.external >> 16;
    let kind = match mode & S_IFMT {
        _ if info.is_dir() => Kind::Directory,
        S_IFDIR => Kind::Directory,
        S_IFLNK => {
            let Ok(target) = String::from_utf8(archive.decode(info, archive.open_member(info)?)?)
            else {
                return fail(format!(
                    "symlink target is not UTF-8: {}",
                    repr(&info.filename)
                ));
            };
            resolve_link(&parts, &target)?;
            Kind::Symlink(target)
        }
        0 | S_IFREG => Kind::File,
        _ => {
            return fail(format!(
                "unsupported entry type for {}",
                repr(&info.filename)
            ));
        }
    };
    Ok(Entry {
        info,
        parts,
        kind,
        mode,
    })
}

fn inspect(archive: &Archive) -> Result<Vec<Entry<'_>>, Failure> {
    let entries = archive
        .infos
        .iter()
        .map(|info| classify(archive, info))
        .collect::<Result<Vec<_>, _>>()?;
    let symlinks: HashSet<&[String]> = entries
        .iter()
        .filter(|entry| matches!(entry.kind, Kind::Symlink(_)))
        .map(|entry| entry.parts.as_slice())
        .collect();
    let mut seen = HashSet::new();
    for entry in &entries {
        let name = repr(&entry.info.filename);
        if !seen.insert(entry.parts.as_slice()) {
            return fail(format!("duplicate entry path: {name}"));
        }
        if (1..entry.parts.len()).any(|index| symlinks.contains(&entry.parts[..index])) {
            return fail(format!(
                "entry is nested beneath an archive symlink: {name}"
            ));
        }
    }
    Ok(entries)
}

fn prepare(destination: &Path) -> Result<(), Failure> {
    let shown = python_path_display(destination);
    if fs::symlink_metadata(destination).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return fail(format!("destination cannot be a symlink: {shown}"));
    }
    mkdir_parents(destination)?;
    let mut listing = fs::read_dir(destination).map_err(|error| os_failure(&error, destination))?;
    if listing.next().is_some() {
        return fail(format!("destination must be empty: {shown}"));
    }
    Ok(())
}

fn mkdir_parents(path: &Path) -> Result<(), Failure> {
    fs::create_dir_all(path).map_err(|error| os_failure(&error, path))
}

fn extract(archive_path: &Path, destination: &Path) -> Result<(), Failure> {
    let probe = if archive_path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        archive_path
    };
    if !fs::metadata(probe).is_ok_and(|meta| meta.is_file()) {
        let shown = python_path_display(archive_path);
        return fail(format!("archive does not exist: {shown}"));
    }
    let root = if destination.as_os_str().is_empty() {
        Path::new(".")
    } else {
        destination
    };
    prepare(root)?;
    let data = fs::read(archive_path).map_err(|error| os_failure(&error, archive_path))?;
    let archive = zip_directory::open(data)?;
    let entries = inspect(&archive)?;
    for entry in entries
        .iter()
        .filter(|entry| matches!(entry.kind, Kind::Directory))
    {
        mkdir_parents(&joined(root, &entry.parts))?;
    }
    for entry in entries
        .iter()
        .filter(|entry| matches!(entry.kind, Kind::File))
    {
        write_file(&archive, entry, &joined(root, &entry.parts))?;
    }
    for entry in &entries {
        if let Kind::Symlink(target) = &entry.kind {
            let output = joined(root, &entry.parts);
            mkdir_parents(output.parent().unwrap_or(root))?;
            symlink(target, &output).map_err(|error| symlink_failure(&error, target, &output))?;
        }
    }
    Ok(())
}

fn joined(root: &Path, parts: &[String]) -> PathBuf {
    parts
        .iter()
        .fold(root.to_path_buf(), |path, part| path.join(part))
}

/// `open(..., "xb")` plus `shutil.copyfileobj`: the output file is created
/// before the payload is verified, so a corrupt member leaves it behind.
fn write_file(archive: &Archive, entry: &Entry<'_>, output: &Path) -> Result<(), Failure> {
    mkdir_parents(output.parent().unwrap_or(output))?;
    let range = archive.open_member(entry.info)?;
    let mut handle = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|error| os_failure(&error, output))?;
    let bytes = archive.decode(entry.info, range)?;
    handle
        .write_all(&bytes)
        .map_err(|error| os_failure(&error, output))?;
    let permissions = entry.mode & 0o777;
    if permissions != 0 {
        fs::set_permissions(output, fs::Permissions::from_mode(permissions))
            .map_err(|error| os_failure(&error, output))?;
    }
    Ok(())
}

/// `os.symlink` renders both paths: `[Errno N] reason: 'target' -> 'link'`.
fn symlink_failure(error: &std::io::Error, target: &str, output: &Path) -> Failure {
    let line = os_error_line(error, target);
    Failure::Raised(format!("{line} -> {}", repr(&python_path_display(output))))
}
