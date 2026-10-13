use std::fmt;

#[derive(Debug)]
pub(super) enum Rejected {
    Input(serde_json::Error),
    Read(std::io::Error),
    InputSize,
    Event,
    Controller,
    Repository,
    ProducerEvent,
    Branch,
    Conclusion,
    Workflow,
    RunIdentity,
    RowProducer,
    Catalog(String),
    MissingRow,
    RowIdentity,
    IncompleteArtifacts,
    DuplicateArtifactId,
    MissingArtifact,
    AmbiguousArtifact,
    ExpiredArtifact,
    ArtifactIdentity,
    ArtifactDigest,
}

impl fmt::Display for Rejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(error) => write!(formatter, "invalid SDK evidence JSON: {error}"),
            Self::Read(error) => write!(formatter, "cannot read SDK evidence: {error}"),
            Self::InputSize => formatter.write_str("SDK evidence exceeds 32 MiB"),
            Self::Event => {
                formatter.write_str("only completed workflow_run or workflow_dispatch is allowed")
            }
            Self::Controller => formatter
                .write_str("SDK controller must belong to Mesh-LLM/mesh-llm on refs/heads/main"),
            Self::Repository => {
                formatter.write_str("SDK producer repository or head repository is not canonical")
            }
            Self::ProducerEvent => {
                formatter.write_str("SDK producer must be a main push, including manual selection")
            }
            Self::Branch => formatter.write_str("SDK producer branch must be main"),
            Self::Conclusion => formatter.write_str("SDK producer must be completed successfully"),
            Self::Workflow => {
                formatter.write_str("SDK producer workflow name and path are not authorized")
            }
            Self::RunIdentity => {
                formatter.write_str("SDK event and API producer identities disagree")
            }
            Self::RowProducer => {
                formatter.write_str("SDK product row belongs to a different producer workflow")
            }
            Self::Catalog(error) => write!(formatter, "invalid SDK row catalog: {error}"),
            Self::MissingRow => {
                formatter.write_str("SDK product row is missing or duplicated in the catalog")
            }
            Self::RowIdentity => formatter.write_str(
                "SDK platform, architecture, backend or target differs from its closed row",
            ),
            Self::IncompleteArtifacts => {
                formatter.write_str("SDK artifact inventory is incomplete")
            }
            Self::DuplicateArtifactId => {
                formatter.write_str("SDK artifact inventory repeats an immutable artifact ID")
            }
            Self::MissingArtifact => {
                formatter.write_str("SDK product artifact is missing; rebuilding is forbidden")
            }
            Self::AmbiguousArtifact => {
                formatter.write_str("SDK product artifact name is ambiguous")
            }
            Self::ExpiredArtifact => formatter.write_str("SDK product artifact has expired"),
            Self::ArtifactIdentity => formatter.write_str(
                "SDK artifact run, repository, branch or source SHA does not match its producer",
            ),
            Self::ArtifactDigest => {
                formatter.write_str("SDK artifact lacks a canonical sha256 digest")
            }
        }
    }
}

impl std::error::Error for Rejected {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            Self::Read(error) => Some(error),
            Self::InputSize
            | Self::Event
            | Self::Controller
            | Self::Repository
            | Self::ProducerEvent
            | Self::Branch
            | Self::Conclusion
            | Self::Workflow
            | Self::RunIdentity
            | Self::RowProducer
            | Self::Catalog(_)
            | Self::MissingRow
            | Self::RowIdentity
            | Self::IncompleteArtifacts
            | Self::DuplicateArtifactId
            | Self::MissingArtifact
            | Self::AmbiguousArtifact
            | Self::ExpiredArtifact
            | Self::ArtifactIdentity
            | Self::ArtifactDigest => None,
        }
    }
}

pub(super) type Checked<T> = Result<T, Rejected>;
