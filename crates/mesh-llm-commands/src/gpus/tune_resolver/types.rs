use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConfigModelMatch {
    pub row_index: usize,
    pub configured_model: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DuplicateTuneTarget {
    pub input: String,
    pub canonical_model_ref: String,
    pub first_input: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LocalTargetSource {
    FilesystemPath { synthetic_model_ref: String },
    HuggingFaceCache { canonical_ref: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TuneTargetSelection {
    Configured,
    Explicit { configured: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedTuneTarget {
    pub requested_input: String,
    pub canonical_model_ref: String,
    pub resolved_path: PathBuf,
    pub local_source: LocalTargetSource,
    pub config_matches: Vec<ConfigModelMatch>,
    pub selection: TuneTargetSelection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TuneTargetResolution {
    pub resolved: Vec<ResolvedTuneTarget>,
    pub duplicates: Vec<DuplicateTuneTarget>,
    pub errors: Vec<TuneTargetResolveError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TuneTargetResolveError {
    pub input: String,
    pub context: TuneTargetContext,
    pub reason: TuneTargetResolveReason,
}

impl fmt::Display for TuneTargetResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let subject = match &self.context {
            TuneTargetContext::ConfiguredRow { row_index } => {
                format!("configured model row {}", row_index + 1)
            }
            TuneTargetContext::ExplicitInput => "requested target".to_string(),
        };
        write!(f, "{subject} `{}`: {}", self.input, self.reason)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TuneTargetContext {
    ConfiguredRow { row_index: usize },
    ExplicitInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TuneTargetResolveReason {
    EmptyInput,
    RemoteRefRequiresDownload,
    NotFoundLocally,
}

impl fmt::Display for TuneTargetResolveReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => f.write_str("target is empty"),
            Self::RemoteRefRequiresDownload => {
                f.write_str("remote-only refs are unsupported here; benchmark tune is local-only and will not download")
            }
            Self::NotFoundLocally => {
                f.write_str("target is not an existing local path or installed cache ref")
            }
        }
    }
}
