use std::path::PathBuf;

use anyhow::{Result, ensure};
use model_artifact::safetensors::TensorHeader;
use serde::{Deserialize, Serialize};

pub(crate) const MANIFEST_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsStageRequest {
    pub repo: String,
    /// Hugging Face commit SHA. The endpoint must honor commit-addressed immutability.
    pub revision: String,
    pub layer_start: u32,
    pub layer_end: u32,
    #[serde(default)]
    pub include_prefixes: Vec<String>,
}

impl SafetensorsStageRequest {
    pub(crate) fn normalized(mut self) -> Result<Self> {
        ensure!(
            !self.repo.trim().is_empty(),
            "Hugging Face repo is required"
        );
        ensure!(
            self.repo.split('/').count() == 2 && self.repo.split('/').all(|part| !part.is_empty()),
            "Hugging Face repo must be owner/name"
        );
        ensure!(
            self.revision.len() == 40 && self.revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "SafeTensors stage revision must be an immutable 40-character commit SHA"
        );
        ensure!(
            self.layer_start < self.layer_end,
            "SafeTensors stage layer range must be non-empty"
        );
        self.include_prefixes = self
            .include_prefixes
            .into_iter()
            .map(|prefix| prefix.trim().to_string())
            .filter(|prefix| !prefix.is_empty())
            .collect();
        self.include_prefixes.sort();
        self.include_prefixes.dedup();
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsStagePlan {
    /// Topology-wide identity shared by every layer range of this checkpoint.
    pub checkpoint_sha256: String,
    pub repo: String,
    pub revision: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub include_prefixes: Vec<String>,
    pub total_model_tensor_bytes: Option<u64>,
    pub config_bytes: u64,
    pub index_bytes: u64,
    pub selected_tensor_count: usize,
    pub selected_tensor_bytes: u64,
    pub largest_selected_tensor_bytes: u64,
    pub source_shard_count: usize,
    pub source_shard_bytes: u64,
    pub range_request_count: usize,
    pub range_payload_bytes: u64,
    pub header_probe_bytes: u64,
    pub planned_download_bytes: u64,
    pub source_shard_bytes_avoided: u64,
    pub full_model_tensor_bytes_avoided: Option<u64>,
    pub shards: Vec<SafetensorsShardPlan>,
}

/// Metadata-only description used to plan a distributed MLX topology without
/// downloading checkpoint tensor payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsCheckpointDescriptor {
    pub checkpoint_sha256: String,
    pub repo: String,
    pub revision: String,
    pub model_type: String,
    pub layer_count: u32,
    pub hidden_size: u32,
    pub native_context_length: u32,
    pub dense_tensor_bytes: u64,
    pub estimated_affine4_weight_bytes: u64,
    /// Conservative affine-4 runtime bytes attributed to each transformer
    /// layer. Boundary tensors are charged to the stages that load them.
    #[serde(default)]
    pub estimated_affine4_layer_bytes: Vec<u64>,
    /// K + V cache bytes per token across all layers at BF16 precision.
    pub kv_bytes_per_token_bf16: u64,
}

#[derive(Clone, Debug)]
pub struct PreparedSafetensorsCheckpoint {
    pub path: PathBuf,
    pub descriptor: SafetensorsCheckpointDescriptor,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsShardPlan {
    pub file: String,
    pub file_bytes: u64,
    pub header_probe_bytes: u64,
    pub selected_tensor_count: usize,
    pub selected_tensor_bytes: u64,
    pub largest_selected_tensor_bytes: u64,
    pub ranges: Vec<ByteRange>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end_exclusive: u64,
}

impl ByteRange {
    pub fn len(&self) -> u64 {
        self.end_exclusive.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end_exclusive
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsStageManifest {
    pub schema_version: u32,
    pub cache_key: String,
    pub checkpoint_sha256: String,
    pub source_endpoint: String,
    pub request: SafetensorsStageRequest,
    pub selected_tensor_count: usize,
    pub selected_tensor_bytes: u64,
    pub output_file_bytes: u64,
    pub output_sha256: String,
    pub config_sha256: String,
    pub config_etag: Option<String>,
    pub index_sha256: Option<String>,
    pub index_etag: Option<String>,
    pub source_shards: Vec<SafetensorsSourceShard>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsSourceShard {
    pub file: String,
    pub file_bytes: u64,
    pub etag: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SafetensorsStageArtifact {
    pub path: PathBuf,
    pub manifest: SafetensorsStageManifest,
    pub plan: SafetensorsStagePlan,
    pub cache_hit: bool,
}

/// One selected tensor materialized as an ephemeral, valid SafeTensors file.
///
/// The file exists only for the duration of the visitor callback that receives
/// this value. Callers must consume it before returning from that callback.
#[derive(Debug)]
pub struct SafetensorsStageTensorFile {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub source_file: String,
    pub source_range: ByteRange,
    pub path: PathBuf,
    pub file_bytes: u64,
}

/// Summary of a sequential selected-tensor visit.
#[derive(Clone, Debug)]
pub struct SafetensorsStageTensorVisitReport {
    /// The artifact-oriented range plan used to select and verify tensors.
    /// Its `range_request_count` describes coalesced materialization spans;
    /// `source_range_request_count` is the visitor's actual request count.
    pub plan: SafetensorsStagePlan,
    pub visited_tensor_count: usize,
    pub visited_tensor_bytes: u64,
    pub source_range_request_count: usize,
    /// Largest ephemeral source file produced by `model-hf` during this visit.
    /// This excludes consumer output files, filesystem overhead, and RSS.
    pub temporary_file_peak_bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct SelectedTensor {
    pub name: String,
    pub source_file: String,
    pub source_range: ByteRange,
    pub header: TensorHeader,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedStage {
    pub checkpoint_sha256: String,
    pub plan: SafetensorsStagePlan,
    pub tensors: Vec<SelectedTensor>,
    pub config: Vec<u8>,
    pub config_sha256: String,
    pub config_etag: Option<String>,
    pub index_sha256: Option<String>,
    pub index_etag: Option<String>,
    pub source_shards: Vec<SafetensorsSourceShard>,
}
