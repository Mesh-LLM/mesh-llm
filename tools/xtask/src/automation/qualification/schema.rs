use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Platform {
    Linux,
    Macos,
    Windows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Scenario {
    ProductReadiness,
    ProtocolPair,
    CorruptRuntime,
    ReadinessTimeout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Backend {
    Cpu,
    Cuda,
    Metal,
    Rocm,
    Vulkan,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Artifact {
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Product {
    pub(crate) backend: Backend,
    pub(crate) hardware_evidence: Artifact,
    pub(crate) host_manifest: Artifact,
    pub(crate) runtime_manifest: Artifact,
    pub(crate) product_manifest: Artifact,
    pub(crate) files: Vec<Artifact>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Model {
    pub(crate) artifact_id: String,
    pub(crate) revision: String,
    pub(crate) files: Vec<Artifact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ProbeKind {
    Path,
    Absolute,
    Shebang,
    Versioned,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Probe {
    pub(crate) kind: ProbeKind,
    pub(crate) candidates: Vec<String>,
    pub(crate) found: Vec<PathBuf>,
    pub(crate) evidence: Artifact,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InterpreterProof {
    pub(crate) path: String,
    pub(crate) attempts: u64,
    pub(crate) probes: Vec<Probe>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Execution {
    pub(crate) argv: Vec<String>,
    pub(crate) exit_code: i32,
    pub(crate) case_count: u64,
    pub(crate) cleanup_complete: bool,
    pub(crate) evidence: Artifact,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProtocolCase {
    pub(crate) backend: Backend,
    pub(crate) model_id: String,
    pub(crate) execution: Execution,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScenarioReceipt {
    pub(crate) scenario: Scenario,
    pub(crate) execution: Execution,
    pub(crate) expected_failure_observed: bool,
    pub(crate) failure_kind: Option<FailureKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureKind {
    DigestMismatch,
    ReadinessTimeout,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Receipt {
    pub(crate) schema_version: u32,
    pub(crate) platform: Platform,
    pub(crate) source_sha: String,
    pub(crate) source_snapshot: Artifact,
    pub(crate) contracts: Artifact,
    pub(crate) interpreters: InterpreterProof,
    pub(crate) roots: std::collections::BTreeMap<String, Execution>,
    pub(crate) products: Vec<Product>,
    pub(crate) models: Vec<Model>,
    pub(crate) protocol_cases: Vec<ProtocolCase>,
    pub(crate) scenarios: Vec<ScenarioReceipt>,
}
