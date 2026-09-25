//! `artifact extract-tar <archive> <destination>`: extract a tar archive
//! (plain or gzip) without permitting writes outside the destination.
//! Diagnostics, streams and statuses are byte-compatible with
//! `scripts/safe-extract-tar.py`.

use super::argv::Program;
use super::tar_header::Member;
use super::tar_read::{self, Archive};
use crate::ci_plan::catalog::os_error_text;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};

const PROGRAM: Program = Program {
    name: "safe-extract-tar.py",
    positionals: &["archive", "destination"],
};

pub(super) fn run(args: &[String]) -> CheckReport {
    match PROGRAM.parse(args) {
        Ok(values) => match safe_extract(values[0], values[1]) {
            Ok(()) => CheckReport::success(String::new()),
            Err(message) => CheckReport::failure(
                String::new(),
                format!("unsafe or invalid tar archive: {message}\n"),
            ),
        },
        Err(report) => report,
    }
}

type Parts = Vec<String>;

/// `normalized_parts`: reject NUL, backslash, absolute, drive and `..` names.
fn normalized_parts(raw: &str, label: &str, allow_root: bool) -> Result<Parts, String> {
    if raw.is_empty() || raw.contains(['\0', '\\']) {
        return Err(format!("unsafe {label}: {}", repr(raw)));
    }
    let bytes = raw.as_bytes();
    if raw.starts_with('/')
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
    {
        return Err(format!("absolute {label} is not allowed: {raw}"));
    }
    let parts: Parts = raw
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(str::to_owned)
        .collect();
    if parts.is_empty() {
        if allow_root {
            return Ok(parts);
        }
        return Err(format!("empty {label} is not allowed: {}", repr(raw)));
    }
    if parts.iter().any(|part| part == "..") {
        return Err(format!("traversing {label} is not allowed: {raw}"));
    }
    Ok(parts)
}

/// `validate_members`: every member's normalized parts, in archive order.
fn validate(members: &[Member]) -> Result<Vec<(&Member, Parts)>, String> {
    let mut validated = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for member in members {
        let parts = normalized_parts(&member.name, "archive member path", member.is_dir())?;
        if !seen.insert(parts.clone()) {
            return Err(format!("duplicate archive member path: {}", member.name));
        }
        if parts.is_empty() {
            continue;
        }
        if !(member.is_dir() || member.is_reg() || member.is_sym() || member.is_lnk()) {
            return Err(format!(
                "unsupported archive member type for {}: {}",
                member.name,
                bytes_repr(member.kind)
            ));
        }
        if member.is_sym() || member.is_lnk() {
            let label = format!("link target for {}", member.name);
            // Parts never contain `..`, so the joined target cannot escape;
            // only the legacy empty-target check remains reachable.
            normalized_parts(&member.link, &label, false)?;
        }
        validated.push((member, parts));
    }
    Ok(validated)
}

/// Python's `repr(bytes([kind]))`.
fn bytes_repr(kind: u8) -> String {
    match kind {
        b'\'' => "b\"'\"".to_owned(),
        b'\\' => "b'\\\\'".to_owned(),
        b'\t' => "b'\\t'".to_owned(),
        b'\n' => "b'\\n'".to_owned(),
        b'\r' => "b'\\r'".to_owned(),
        0x20..=0x7e => format!("b'{}'", char::from(kind)),
        _ => format!("b'\\x{kind:02x}'"),
    }
}

/// `Path.mkdir(parents=True, exist_ok=True)`.
fn mkdir_parents(path: &Path, shown: &str) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|error| os_error_text(&error, shown))
}

fn apply_mode(path: &Path, mode: i64) -> Result<(), String> {
    let bits = u32::try_from(mode & 0o777).unwrap_or_default();
    fs::set_permissions(path, fs::Permissions::from_mode(bits))
        .map_err(|error| os_error_text(&error, &path.to_string_lossy()))
}

fn occupied(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok() || path.exists()
}

fn safe_extract(archive_path: &str, destination: &str) -> Result<(), String> {
    let dest = Path::new(destination);
    mkdir_parents(dest, destination)?;
    if fs::symlink_metadata(dest).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return Err(format!(
            "extraction destination cannot be a symlink: {destination}"
        ));
    }
    let root = fs::canonicalize(dest).map_err(|error| os_error_text(&error, destination))?;
    let shown_root = root.to_string_lossy().into_owned();
    let mut entries = fs::read_dir(&root).map_err(|error| os_error_text(&error, &shown_root))?;
    if entries.next().is_some() {
        return Err(format!(
            "extraction destination must be empty: {shown_root}"
        ));
    }
    let raw = fs::read(archive_path).map_err(|error| os_error_text(&error, archive_path))?;
    let archive = tar_read::open(raw)?;
    let validated = validate(&archive.members)?;
    extract(&archive, &validated, &root)
}

fn joined(root: &Path, parts: &Parts) -> PathBuf {
    parts
        .iter()
        .fold(root.to_path_buf(), |path, part| path.join(part))
}

fn extract(archive: &Archive, validated: &[(&Member, Parts)], root: &Path) -> Result<(), String> {
    let directories: Vec<_> = validated.iter().filter(|(m, _)| m.is_dir()).collect();
    for (_, parts) in &directories {
        let path = joined(root, parts);
        mkdir_parents(&path, &path.to_string_lossy())?;
    }
    for (member, parts) in validated.iter().filter(|(m, _)| m.is_reg()) {
        let output = prepare_output(root, parts, member, "archive member")?;
        let payload = archive
            .data
            .get(member.data_start..member.data_start + member.size)
            .ok_or_else(|| "unexpected end of data".to_owned())?;
        let shown = output.to_string_lossy().into_owned();
        let mut handle = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|error| os_error_text(&error, &shown))?;
        handle
            .write_all(payload)
            .map_err(|error| os_error_text(&error, &shown))?;
        apply_mode(&output, member.mode)?;
    }
    for (member, parts) in validated.iter().filter(|(m, _)| m.is_sym() || m.is_lnk()) {
        let output = prepare_output(root, parts, member, "archive link")?;
        let shown = output.to_string_lossy().into_owned();
        if member.is_sym() {
            symlink(&member.link, &output).map_err(|error| os_error_text(&error, &member.link))?;
            continue;
        }
        let label = format!("hard-link target for {}", member.name);
        let target = joined(root, &normalized_parts(&member.link, &label, false)?);
        let regular = fs::symlink_metadata(&target).is_ok_and(|meta| meta.is_file());
        if !regular {
            return Err(format!(
                "hard-link target is not a regular extracted file: {}",
                member.link
            ));
        }
        fs::hard_link(&target, &output).map_err(|error| os_error_text(&error, &shown))?;
    }
    for (member, parts) in directories.iter().rev() {
        apply_mode(&joined(root, parts), member.mode)?;
    }
    Ok(())
}

/// Create the parent of `parts` and refuse to overwrite anything there.
fn prepare_output(
    root: &Path,
    parts: &Parts,
    member: &Member,
    noun: &str,
) -> Result<PathBuf, String> {
    let output = joined(root, parts);
    if let Some(parent) = output.parent() {
        mkdir_parents(parent, &parent.to_string_lossy())?;
    }
    if occupied(&output) {
        return Err(format!(
            "{noun} would overwrite an existing path: {}",
            member.name
        ));
    }
    Ok(output)
}
