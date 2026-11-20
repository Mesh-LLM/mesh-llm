//! Startup source policy. Catalog and cache I/O belong to the concrete model owner.
use std::path::{Path, PathBuf};

/// Exact catalog coordinates selected for startup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteModelRef {
    pub name: String,
    pub repo: String,
    pub revision: Option<String>,
    pub file: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StartupModelSource {
    Passthrough(PathBuf),
    Existing(PathBuf),
    Catalog(RemoteModelRef),
    Installed(PathBuf),
    Download {
        reference: String,
        bare_name_fallback: bool,
    },
    MissingBareName,
}

impl StartupModelSource {
    /// Explicit package references take precedence even over an existing path.
    pub fn explicit(input: &Path, exists: impl FnOnce() -> bool) -> Option<Self> {
        if input.to_string_lossy().starts_with("hf://") {
            Some(Self::Passthrough(input.to_path_buf()))
        } else if exists() {
            Some(Self::Existing(input.to_path_buf()))
        } else {
            None
        }
    }

    /// Called only after catalog and installed-cache lookup have missed.
    pub fn missing(input: &str, canonical: Option<String>) -> Self {
        if input.contains('/') {
            return Self::Download {
                reference: input.to_string(),
                bare_name_fallback: false,
            };
        }
        match canonical {
            Some(reference) if reference != input => Self::Download {
                reference,
                bare_name_fallback: true,
            },
            _ => Self::MissingBareName,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_sources_precede_discovery() {
        for exists in [false, true] {
            let input = Path::new("hf://owner/package@revision");
            assert_eq!(
                StartupModelSource::explicit(input, || exists),
                Some(StartupModelSource::Passthrough(input.into()))
            );
        }
        let input = Path::new("local.gguf");
        assert_eq!(
            StartupModelSource::explicit(input, || true),
            Some(StartupModelSource::Existing(input.into()))
        );
        assert_eq!(StartupModelSource::explicit(input, || false), None);
    }

    #[test]
    fn discovery_misses_keep_bare_name_and_remote_semantics() {
        for canonical in [None, Some("name".into())] {
            assert_eq!(
                StartupModelSource::missing("name", canonical),
                StartupModelSource::MissingBareName
            );
        }
        assert_eq!(
            StartupModelSource::missing("name", Some("org/name".into())),
            StartupModelSource::Download {
                reference: "org/name".into(),
                bare_name_fallback: true
            }
        );
        assert_eq!(
            StartupModelSource::missing("org/name", None),
            StartupModelSource::Download {
                reference: "org/name".into(),
                bare_name_fallback: false
            }
        );
    }
}
