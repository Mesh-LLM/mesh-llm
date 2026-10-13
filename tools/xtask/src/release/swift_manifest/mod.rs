//! Frozen inline-Python manifest operations, without native checksum execution.

mod command;
#[cfg(test)]
mod command_tests;
mod field;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod verification_tests;

pub(crate) use command::run;
use field::Field;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ManifestError {
    #[error("missing {0}")]
    Missing(Field),
    #[error("Package.swift still contains the Swift release URL placeholder")]
    UrlPlaceholder,
    #[error("Package.swift still contains the Swift release checksum placeholder")]
    ChecksumPlaceholder,
    #[error("Swift release URL mismatch: {actual} != {expected}")]
    UrlMismatch { actual: String, expected: String },
    #[error("Swift release checksum mismatch: {actual} != {expected}")]
    ChecksumMismatch { actual: String, expected: String },
    #[error("Package.swift is not strict UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("Swift manifest I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Opaque output previously obtained from the explicit Swift checksum adapter.
pub(crate) struct SwiftChecksum<'a>(pub(crate) &'a str);

/// Legacy tags are opaque strings, not parsed versions or validated URLs.
pub(crate) struct ReleaseTag<'a>(pub(crate) &'a str);

pub(crate) struct ReleaseValues<'a> {
    pub(crate) tag: ReleaseTag<'a>,
    pub(crate) checksum: SwiftChecksum<'a>,
}

impl ReleaseValues<'_> {
    pub(crate) fn url(&self) -> String {
        format!(
            "https://github.com/Mesh-LLM/mesh-llm/releases/download/{}/MeshLLMFFI.xcframework.zip",
            self.tag.0
        )
    }
}

fn text(bytes: &[u8]) -> Result<String, ManifestError> {
    Ok(std::str::from_utf8(bytes)?
        .replace("\r\n", "\n")
        .replace('\r', "\n"))
}

pub(crate) fn update(bytes: &[u8], values: &ReleaseValues<'_>) -> Result<String, ManifestError> {
    let manifest = text(bytes)?;
    let manifest = Field::Url.replace_first(&manifest, &values.url())?;
    Field::Checksum.replace_first(&manifest, values.checksum.0)
}

pub(crate) fn verify(bytes: &[u8], values: &ReleaseValues<'_>) -> Result<(), ManifestError> {
    let manifest = text(bytes)?;
    let actual_url = Field::Url.value(&manifest)?;
    let actual_checksum = Field::Checksum.value(&manifest)?;
    if actual_url.contains("__MESH_SWIFT_RELEASE_TAG__") {
        return Err(ManifestError::UrlPlaceholder);
    }
    if actual_checksum.contains("__MESH_SWIFT_RELEASE_CHECKSUM__") {
        return Err(ManifestError::ChecksumPlaceholder);
    }
    let expected_url = values.url();
    if actual_url != expected_url {
        return Err(ManifestError::UrlMismatch {
            actual: actual_url.to_owned(),
            expected: expected_url,
        });
    }
    if actual_checksum != values.checksum.0 {
        return Err(ManifestError::ChecksumMismatch {
            actual: actual_checksum.to_owned(),
            expected: values.checksum.0.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn update_file(path: &Path, values: &ReleaseValues<'_>) -> Result<(), ManifestError> {
    let bytes = std::fs::read(path)?;
    let updated = update(&bytes, values)?;
    std::fs::write(path, updated)?;
    Ok(())
}

pub(crate) fn verify_file(path: &Path, values: &ReleaseValues<'_>) -> Result<(), ManifestError> {
    verify(&std::fs::read(path)?, values)
}
