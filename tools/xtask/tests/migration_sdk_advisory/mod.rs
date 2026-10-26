use super::admission::{Controller, Trigger, admit};
use super::error::Rejected;
use super::rows::Catalog;
use serde::Deserialize;
use serde_json::Value;

mod admission;
mod artifacts;
mod catalog;
mod command;
mod coverage;

const LINUX_RUN: &[u8] = include_bytes!("fixtures/main-linux-run.json");
const LINUX_EVENT: &[u8] = include_bytes!("fixtures/main-linux-event.json");
const LINUX_ARTIFACTS: &[u8] = include_bytes!("fixtures/main-linux-artifacts.json");
const MANUAL_EVENT: &[u8] = include_bytes!("fixtures/manual-linux-cuda.json");
const MACOS_RUN: &[u8] = include_bytes!("fixtures/main-macos-run.json");
const MACOS_EVENT: &[u8] = include_bytes!("fixtures/main-macos-event.json");
const MACOS_ARTIFACTS: &[u8] = include_bytes!("fixtures/main-macos-artifacts.json");
const OWNERSHIP: &[u8] = include_bytes!("../../../../ci/ownership.yml");
const SLICES: &[u8] = include_bytes!("../../../../ci/slices.yml");
const CONTROLLER: Controller<'static> = Controller {
    repository: "Mesh-LLM/mesh-llm",
    reference: "refs/heads/main",
};

fn catalog() -> Catalog {
    Catalog::parse(OWNERSHIP, SLICES).expect("valid existing planner catalog")
}

fn value(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("valid fixture")
}

fn bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).expect("serialize fixture")
}

#[derive(Deserialize)]
struct Mutation {
    name: String,
    pointer: String,
    value: Value,
    reason: Denial,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Denial {
    Input,
    Repository,
    ProducerEvent,
    Branch,
    Conclusion,
    Workflow,
    RunIdentity,
    IncompleteArtifacts,
    DuplicateArtifactId,
    MissingArtifact,
    AmbiguousArtifact,
    ExpiredArtifact,
    ArtifactIdentity,
    ArtifactDigest,
}

impl Denial {
    fn matches(&self, error: &Rejected) -> bool {
        match self {
            Self::Input => matches!(error, Rejected::Input(_)),
            Self::Repository => matches!(error, Rejected::Repository),
            Self::ProducerEvent => matches!(error, Rejected::ProducerEvent),
            Self::Branch => matches!(error, Rejected::Branch),
            Self::Conclusion => matches!(error, Rejected::Conclusion),
            Self::Workflow => matches!(error, Rejected::Workflow),
            Self::RunIdentity => matches!(error, Rejected::RunIdentity),
            Self::IncompleteArtifacts => matches!(error, Rejected::IncompleteArtifacts),
            Self::DuplicateArtifactId => matches!(error, Rejected::DuplicateArtifactId),
            Self::MissingArtifact => matches!(error, Rejected::MissingArtifact),
            Self::AmbiguousArtifact => matches!(error, Rejected::AmbiguousArtifact),
            Self::ExpiredArtifact => matches!(error, Rejected::ExpiredArtifact),
            Self::ArtifactIdentity => matches!(error, Rejected::ArtifactIdentity),
            Self::ArtifactDigest => matches!(error, Rejected::ArtifactDigest),
        }
    }
}

fn mutated(original: &[u8], mutation: &Mutation) -> Vec<u8> {
    let mut document = value(original);
    *document
        .pointer_mut(&mutation.pointer)
        .expect("fixture pointer exists") = mutation.value.clone();
    bytes(&document)
}
