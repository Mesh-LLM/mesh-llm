//! GLIBC floor arithmetic for `verify-host-dependencies.py`: `parse_version`
//! (Python `int()` of each half), `readelf -V` version-needs scanning, and
//! the comments-aware floor file. Versions compare as Python int tuples.

use crate::ci_operations::ci_metrics_int::python_int_text;
use crate::ci_plan::catalog::os_error_text;
use crate::repository::python_text::{repr, splitlines, strip};
use std::cmp::Ordering;
use std::path::Path;

const VERSION_NEEDS_HEADING: &str = "Version needs section";

/// One Python int as canonical decimal text (`-` sign, no leading zeros).
#[derive(Clone, PartialEq, Eq)]
struct Int(String);

impl Int {
    fn parse(text: &str) -> Result<Self, String> {
        python_int_text(text)
            .map(Self)
            .ok_or_else(|| format!("invalid literal for int() with base 10: {}", repr(text)))
    }

    fn key(&self) -> (bool, usize, &str) {
        match self.0.strip_prefix('-') {
            Some(digits) => (false, usize::MAX - digits.len(), digits),
            None => (true, self.0.len(), &self.0),
        }
    }
}

impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        let (left, right) = (self.key(), other.key());
        let ordering = left.cmp(&right);
        match (left.0, right.0) {
            // Among negatives a larger magnitude is smaller.
            (false, false) if left.1 == right.1 => right.2.cmp(left.2),
            _ => ordering,
        }
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A `(major, minor)` int tuple.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Version(Int, Int);

impl Version {
    /// `format_version`.
    pub(super) fn render(&self) -> String {
        format!("{}.{}", self.0.0, self.1.0)
    }
}

/// `parse_version(value)`: `ValueError` text on failure.
pub(super) fn parse_version(value: &str) -> Result<Version, String> {
    let stripped = strip(value);
    let (major, minor) = stripped.split_once('.').unwrap_or((stripped, ""));
    Ok(Version(Int::parse(major)?, Int::parse(minor)?))
}

/// `parse_elf_glibc_floor(output)`.
pub(super) fn elf_floor(output: &str) -> Option<Version> {
    let (_, needs) = output.split_once(VERSION_NEEDS_HEADING)?;
    let mut floor: Option<Version> = None;
    let mut rest = needs;
    while let Some(offset) = rest.find("GLIBC_") {
        rest = &rest[offset + "GLIBC_".len()..];
        let major_len = rest.bytes().take_while(u8::is_ascii_digit).count();
        let after = &rest[major_len..];
        let minor_len = after.strip_prefix('.').map_or(0, |minor| {
            minor.bytes().take_while(u8::is_ascii_digit).count()
        });
        if major_len == 0 || minor_len == 0 {
            continue;
        }
        let parsed = parse_version(&rest[..major_len + 1 + minor_len]).ok();
        rest = &rest[major_len + 1 + minor_len..];
        floor = floor.max(parsed);
    }
    floor
}

/// `read_declared_glibc_floor(path)`: the first line that is neither blank
/// nor a comment. `shown` is `str(path)`.
pub(super) fn declared_floor(path: &Path, shown: &str) -> Result<Version, String> {
    let raw = std::fs::read(path).map_err(|error| os_error_text(&error, shown))?;
    let text = std::str::from_utf8(&raw)
        .map_err(|_| format!("'utf-8' codec can't decode the file {}", repr(shown)))?
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    splitlines(&text)
        .into_iter()
        .map(strip)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map_or_else(
            || Err(format!("no glibc floor declared in {shown}")),
            parse_version,
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(text: &str) -> Version {
        parse_version(text).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn migration_native_policy_versions_compare_as_int_tuples() {
        assert!(version("2.39") > version("2.4"));
        assert!(version("-1.0") < version("0.0"));
        assert!(version("-10.0") < version("-9.0"));
        assert_eq!(version(" 02.0_1 ").render(), "2.1");
        assert_eq!(
            parse_version("2").err().as_deref(),
            Some("invalid literal for int() with base 10: ''")
        );
        assert_eq!(
            parse_version("x.1").err().as_deref(),
            Some("invalid literal for int() with base 10: 'x'")
        );
    }

    #[test]
    fn migration_native_policy_floor_reads_only_version_needs() {
        let output = "Version definition section\n  Name: GLIBC_2.99\n\
                      Version needs section '.gnu.version_r'\n  Name: GLIBC_2.17 GLIBC_2.34x GLIBC_.1 GLIBC_2.";
        assert_eq!(
            elf_floor(output).map(|floor| floor.render()).as_deref(),
            Some("2.34")
        );
        assert!(elf_floor("GLIBC_2.1").is_none());
        assert!(elf_floor("Version needs section\n").is_none());
    }
}
