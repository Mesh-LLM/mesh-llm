use super::{PrivacyError, policy};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) struct PendingNativeLint {
    pub(crate) template: PathBuf,
    pub(crate) embedded: Vec<PathBuf>,
}

#[derive(Debug)]
pub(crate) struct GenericVerification {
    pub(crate) pending_native_lint: PendingNativeLint,
}

pub(crate) fn verify_files(
    template: &Path,
    xcframework: Option<&Path>,
) -> Result<GenericVerification, PrivacyError> {
    let bytes = fs::read(template).map_err(|source| PrivacyError::Read {
        path: template.to_path_buf(),
        source,
    })?;
    policy::validate(&bytes)?;
    let embedded = match xcframework {
        None => Vec::new(),
        Some(root) => compare_embedded(root, &bytes)?,
    };
    Ok(GenericVerification {
        pending_native_lint: PendingNativeLint {
            template: template.to_path_buf(),
            embedded,
        },
    })
}

fn compare_embedded(root: &Path, template: &[u8]) -> Result<Vec<PathBuf>, PrivacyError> {
    if !root.is_dir() {
        return Err(PrivacyError::MissingFramework(root.to_path_buf()));
    }
    let mut embedded = Vec::new();
    discover(root, &mut embedded)?;
    if embedded.is_empty() {
        return Err(PrivacyError::NotEmbedded(root.to_path_buf()));
    }
    for path in &embedded {
        match fs::read(path) {
            Ok(bytes) if bytes == template => {}
            Ok(_) | Err(_) => return Err(PrivacyError::EmbeddedDifference(path.clone())),
        }
    }
    Ok(embedded)
}

fn discover(path: &Path, found: &mut Vec<PathBuf>) -> Result<(), PrivacyError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| PrivacyError::Discovery {
        path: path.to_path_buf(),
        source,
    })?;
    if path
        .file_name()
        .is_some_and(|name| name == "PrivacyInfo.xcprivacy")
    {
        found.push(path.to_path_buf());
    }
    if metadata.is_dir() {
        let entries = fs::read_dir(path).map_err(|source| PrivacyError::Discovery {
            path: path.to_path_buf(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| PrivacyError::Discovery {
                path: path.to_path_buf(),
                source,
            })?;
            discover(&entry.path(), found)?;
        }
    }
    Ok(())
}

pub(super) struct ValidatedTemplate<'a> {
    path: &'a Path,
    bytes: Vec<u8>,
}

impl<'a> ValidatedTemplate<'a> {
    pub(super) fn read(path: &'a Path) -> Result<Self, PrivacyError> {
        let bytes = fs::read(path).map_err(|source| PrivacyError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        policy::validate(&bytes)?;
        Ok(Self { path, bytes })
    }

    pub(super) fn path(&self) -> &'a Path {
        self.path
    }

    pub(super) fn compare<'b>(&self, path: &'b Path) -> Result<&'b Path, PrivacyError> {
        match fs::read(path) {
            Ok(bytes) if bytes == self.bytes => Ok(path),
            Ok(_) | Err(_) => Err(PrivacyError::EmbeddedDifference(path.to_path_buf())),
        }
    }
}

pub(super) fn discover_embedded(root: &Path) -> Result<Vec<PathBuf>, PrivacyError> {
    if !root.is_dir() {
        return Err(PrivacyError::MissingFramework(root.to_path_buf()));
    }
    let mut embedded = Vec::new();
    discover(root, &mut embedded)?;
    if embedded.is_empty() {
        return Err(PrivacyError::NotEmbedded(root.to_path_buf()));
    }
    Ok(embedded)
}
