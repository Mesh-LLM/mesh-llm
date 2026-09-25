//! Value semantics of `scripts/manage-build-cache.py`: `parse_size`,
//! `parse_age` (Python `int()`), `human_size`, and non-strict
//! `Path.resolve()`.

use crate::repository::python_text::{is_space, strip};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

pub(crate) const DEFAULT_MAX_AGE_DAYS: i128 = 14;

/// A parsed `--max-age`, kept as Python's unbounded `int` would print it.
#[derive(Clone, Copy)]
pub(crate) struct Age(pub(crate) i128);

impl Age {
    pub(crate) fn default_days() -> Self {
        Self(DEFAULT_MAX_AGE_DAYS)
    }
}

fn unit_bytes(unit: &str) -> Option<u128> {
    Some(match unit {
        "" | "b" => 1,
        "kb" => 1000,
        "kib" => 1024,
        "mb" => 1000_u128.pow(2),
        "mib" => 1024_u128.pow(2),
        "gb" => 1000_u128.pow(3),
        "gib" => 1024_u128.pow(3),
        "tb" => 1000_u128.pow(4),
        "tib" => 1024_u128.pow(4),
        _ => return None,
    })
}

/// `^(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?$` (case-insensitive) on the stripped text.
fn size_parts(text: &str) -> Option<(&str, String)> {
    let digits_end = text
        .char_indices()
        .find(|(_, ch)| !ch.is_ascii_digit())
        .map_or(text.len(), |(index, _)| index);
    if digits_end == 0 {
        return None;
    }
    let mut number_end = digits_end;
    if let Some(fraction) = text[digits_end..].strip_prefix('.') {
        let length = fraction
            .char_indices()
            .find(|(_, ch)| !ch.is_ascii_digit())
            .map_or(fraction.len(), |(index, _)| index);
        if length == 0 {
            return None;
        }
        number_end = digits_end + 1 + length;
    }
    let unit = text[number_end..]
        .trim_start_matches(is_space)
        .to_lowercase();
    let mut chars = unit.chars().peekable();
    chars.next_if(|ch| "kmgt".contains(*ch));
    chars.next_if(|ch| *ch == 'i');
    let valid = unit.is_empty() || (chars.next() == Some('b') && chars.next().is_none());
    valid.then(|| (&text[..number_end], unit))
}

/// Returns the byte count, or argparse's `ArgumentTypeError` message.
pub(crate) fn parse_size(raw: &str) -> Result<i128, String> {
    let value = raw.strip_prefix("max_size=").unwrap_or(raw);
    let invalid = || format!("invalid size: {value}");
    let (number, unit) = size_parts(strip(value)).ok_or_else(invalid)?;
    // `[kmgt]?i?b` also admits `ib`, which the legacy table lacks (a
    // KeyError traceback there); it is reported as an invalid size here.
    let multiplier = unit_bytes(&unit).ok_or_else(invalid)?;
    let amount: f64 = number.parse().map_err(|_| invalid())?;
    let bytes = (amount * multiplier as f64) as i128;
    Ok(bytes)
}

/// Python `int(text)` for decimal text: surrounding whitespace, a sign, and
/// single underscores between digits.
pub(crate) fn parse_age(raw: &str) -> Result<Age, String> {
    let value = raw.strip_prefix("max_age=").unwrap_or(raw);
    let invalid = || format!("invalid age in days: {value}");
    let text = strip(value);
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let well_formed = !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits.chars().all(|ch| ch.is_ascii_digit() || ch == '_');
    if !well_formed {
        return Err(invalid());
    }
    let magnitude: i128 = digits.replace('_', "").parse().map_err(|_| invalid())?;
    Ok(Age(if negative { -magnitude } else { magnitude }))
}

/// `human_size`: `f"{amount:.1f} {unit}"` over binary units.
pub(crate) fn human_size(value: i128) -> String {
    let mut amount = value as f64;
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"] {
        if amount < 1024.0 || unit == "TiB" {
            return format!("{amount:.1} {unit}");
        }
        amount /= 1024.0;
    }
    String::new()
}

/// Non-strict `Path.resolve()` (`os.path.realpath`): symlinks in existing
/// prefixes are followed, missing components are kept lexically.
pub(crate) fn resolve(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut pending: Vec<OsString> = components(&absolute);
    pending.reverse();
    let mut resolved = PathBuf::from("/");
    let mut hops = 0;
    while let Some(part) = pending.pop() {
        if part == ".." {
            resolved.pop();
            continue;
        }
        let next = resolved.join(&part);
        match std::fs::read_link(&next) {
            Ok(target) if hops < 40 => {
                hops += 1;
                if target.is_absolute() {
                    resolved = PathBuf::from("/");
                }
                let mut parts = components(&target);
                parts.reverse();
                pending.extend(parts);
            }
            _ => resolved = next,
        }
    }
    resolved
}

fn components(path: &Path) -> Vec<OsString> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name.to_os_string()),
            Component::ParentDir => Some(OsString::from("..")),
            _ => None,
        })
        .collect()
}
