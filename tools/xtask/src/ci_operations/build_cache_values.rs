//! Cache-size parsing, human-readable sizes, and non-strict path resolution.

use crate::repository::text::{is_space, strip};
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

pub(crate) const DEFAULT_MAX_AGE_DAYS: i128 = 14;

/// A parsed maximum age in whole days.
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

/// Returns the byte count or a deterministic size-validation message.
pub(crate) fn parse_size(raw: &str) -> Result<i128, String> {
    let invalid = || format!("invalid size: {raw}");
    let (number, unit) = size_parts(strip(raw)).ok_or_else(invalid)?;
    let multiplier = unit_bytes(&unit).ok_or_else(invalid)?;
    let amount: f64 = number.parse().map_err(|_| invalid())?;
    let bytes = (amount * multiplier as f64) as i128;
    Ok(bytes)
}

/// Format a cache size with one decimal place and binary units.
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
