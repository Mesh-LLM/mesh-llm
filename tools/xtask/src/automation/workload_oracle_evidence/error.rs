use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub(crate) enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json(String),
    Worker(std::io::Error),
    UnknownClass(String),
    SmokeLane(String),
    ComparatorLog,
    Object,
    Identity(&'static str),
    Projector(&'static str),
    Executable,
    Comparison,
    TtsResult,
    Metrics,
    PositiveInteger(&'static str),
    MetricRange {
        field: &'static str,
        minimum: &'static str,
        maximum: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Json(message) => formatter.write_str(message),
            Self::Worker(source) => write!(formatter, "workload evidence worker: {source}"),
            Self::UnknownClass(name) => write!(formatter, "unknown workload class: {name}"),
            Self::SmokeLane(lane) => write!(formatter, "smoke lane must end with '-smoke': {lane}"),
            Self::ComparatorLog => formatter
                .write_str("oracle comparator did not emit an explicit class-specific pass"),
            Self::Object => formatter.write_str("oracle evidence must be an object"),
            Self::Identity(field) => {
                write!(formatter, "oracle evidence {field} does not match this run")
            }
            Self::Projector(class) => write!(
                formatter,
                "{class} oracle evidence requires a projector path"
            ),
            Self::Executable => formatter.write_str("wrong oracle executable for workload class"),
            Self::Comparison => {
                formatter.write_str("oracle evidence lacks an explicit comparator pass")
            }
            Self::TtsResult => formatter
                .write_str("TTS comparator result is missing or does not match the pinned patch"),
            Self::Metrics => formatter.write_str("TTS oracle evidence lacks PCM metrics"),
            Self::PositiveInteger(field) => write!(
                formatter,
                "TTS PCM metric {field} must be a positive integer"
            ),
            Self::MetricRange {
                field,
                minimum,
                maximum,
            } => write!(
                formatter,
                "TTS PCM metric {field} must be finite and within [{minimum}, {maximum}]"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Worker(source) => Some(source),
            Self::Json(_)
            | Self::UnknownClass(_)
            | Self::SmokeLane(_)
            | Self::ComparatorLog
            | Self::Object
            | Self::Identity(_)
            | Self::Projector(_)
            | Self::Executable
            | Self::Comparison
            | Self::TtsResult
            | Self::Metrics
            | Self::PositiveInteger(_)
            | Self::MetricRange { .. } => None,
        }
    }
}
