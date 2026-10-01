//! Textual source-version extraction used by the native runtime packager.

use crate::repository::python_text::{is_space, strip};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AbiComponent {
    Major,
    Minor,
    Patch,
}

impl std::fmt::Display for AbiComponent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Major => "MAJOR",
            Self::Minor => "MINOR",
            Self::Patch => "PATCH",
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum VersionError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid UTF-8 source: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("invalid Skippy runtime release version")]
    RuntimeInvalid,
    #[error("workspace package version not found")]
    WorkspaceMissing,
    #[error("missing ABI component '{0}'")]
    AbiMissing(AbiComponent),
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceVersion<'a>(&'a str);

impl std::fmt::Display for WorkspaceVersion<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AbiVersion<'a> {
    major: &'a str,
    minor: &'a str,
    patch: &'a str,
}

impl std::fmt::Display for AbiVersion<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Extract the first eligible version prefix, without interpreting TOML.
/// The complete byte slice must be UTF-8. Legacy text-file iteration may return
/// before decoding an invalid suffix; this boundary is not approved legacy parity.
pub(crate) fn workspace_version(bytes: &[u8]) -> Result<WorkspaceVersion<'_>, VersionError> {
    let text = std::str::from_utf8(bytes)?;
    let mut eligible = false;
    for line in text.split(['\r', '\n']).map(strip) {
        if line == "[workspace.package]" {
            eligible = true;
            continue;
        }
        if line.starts_with('[') {
            eligible = false;
        }
        if eligible && let Some(value) = version_prefix(line) {
            return Ok(WorkspaceVersion(value));
        }
    }
    Err(VersionError::WorkspaceMissing)
}

fn version_prefix(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("version")?.trim_start_matches(is_space);
    let rest = rest.strip_prefix('=')?.trim_start_matches(is_space);
    let rest = rest.strip_prefix('"')?;
    let (value, _) = rest.split_once('"')?;
    (!value.is_empty()).then_some(value)
}

/// Extract the last exact declaration prefix for each textual ABI component.
pub(crate) fn abi_version(bytes: &[u8]) -> Result<AbiVersion<'_>, VersionError> {
    let text = std::str::from_utf8(bytes)?;
    let mut major = None;
    let mut minor = None;
    let mut patch = None;
    for line in text.split(['\r', '\n']).map(strip) {
        let Some(rest) = line.strip_prefix("pub const ABI_VERSION_") else {
            continue;
        };
        for (name, value) in [
            ("MAJOR", &mut major),
            ("MINOR", &mut minor),
            ("PATCH", &mut patch),
        ] {
            if let Some(digits) = rest.strip_prefix(name).and_then(abi_digits) {
                *value = Some(digits);
            }
        }
    }
    Ok(AbiVersion {
        major: major.ok_or(VersionError::AbiMissing(AbiComponent::Major))?,
        minor: minor.ok_or(VersionError::AbiMissing(AbiComponent::Minor))?,
        patch: patch.ok_or(VersionError::AbiMissing(AbiComponent::Patch))?,
    })
}

fn abi_digits(rest: &str) -> Option<&str> {
    let rest = rest.strip_prefix(": u32 = ")?;
    let end = rest
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(rest.len());
    let (digits, suffix) = rest.split_at(end);
    (!digits.is_empty() && suffix.starts_with(';')).then_some(digits)
}

pub(crate) fn read_source(path: &Path) -> Result<Vec<u8>, VersionError> {
    std::fs::read(path).map_err(|source| VersionError::Read {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
#[path = "package_version_tests.rs"]
mod tests;

/// Read the independent runtime release identity, preserving its text-file contract.
pub(crate) fn runtime_version(bytes: &[u8]) -> Result<String, VersionError> {
    let text = std::str::from_utf8(bytes)?.trim();
    let (base, suffix) = text
        .split_once('-')
        .map_or((text, None), |(base, suffix)| (base, Some(suffix)));
    let components: Vec<_> = base.split('.').collect();
    let base_valid = components.len() == 3
        && components
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()));
    let suffix_valid = suffix.is_none_or(|suffix| {
        !suffix.is_empty()
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
    });
    if !base_valid || !suffix_valid {
        return Err(VersionError::RuntimeInvalid);
    }
    Ok(text.to_owned())
}

#[cfg(test)]
mod runtime_release_tests {
    use super::runtime_version;
    #[test]
    fn runtime_identity_accepts_independent_release_and_rejects_invalid_source() {
        assert_eq!(runtime_version(b"  0.75.1-rc.2\n").unwrap(), "0.75.1-rc.2");
        for source in [
            b"1.2".as_slice(),
            b"1.2.3-",
            b"1.2.3+build",
            b"1.2.3\nextra",
            b"1.2.3\xff",
        ] {
            assert!(runtime_version(source).is_err());
        }
    }
}
