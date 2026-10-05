//! Repository handles, metadata types, and list/create/delete/move APIs.
//!
//! Start from [`HFClient`] — [`HFClient::model`], [`HFClient::dataset`],
//! [`HFClient::space`], and [`HFClient::kernel`] return a typed
//! [`HFRepository<T>`](HFRepository) for the corresponding repo kind. All read/write
//! file and revision APIs hang off that value; methods that differ per repo kind (such
//! as [`info`](HFRepository::info)) are resolved at compile time via the [`RepoType`]
//! type parameter.
//!
//! Submodule pages group related builders and types:
//!
//! - [`commits`] — history, refs, compare/diff, branches, tags
//! - [`listing`] — list files/tree, path metadata
//! - [`download`] — single-file and snapshot download builders
//! - [`upload`] — uploads, deletes, and [`CommitOperation`] batches
//! - [`diff`] — parsed raw diff lines ([`HFFileDiff`])
//! - [`files`] — shared types such as [`CommitOperation`] and [`RepoTreeEntry`]
//!
//! Most items are also re-exported at [`crate::repository`] for a flat `hf_hub::repository::…`
//! path in addition to the submodule paths rustdoc lists.

pub mod commits;
pub mod diff;
pub mod download;
pub mod files;
pub mod listing;
pub mod repo_type;
pub mod upload;

use std::collections::HashMap;
use std::str::FromStr;

use bon::bon;
pub use commits::{CommitAuthor, DiffEntry, GitCommitInfo, GitRefInfo, GitRefs};
pub use diff::{GitStatus, HFDiffParseError, HFFileDiff};
pub use files::{
    AddSource, BlobLfsInfo, BlobSecurityInfo, CommitInfo, CommitOperation, FileMetadataInfo,
    LastCommitInfo, RepoTreeEntry, SourceByteStream, StreamFactory, StreamSource,
};
#[cfg(not(target_family = "wasm"))]
pub(crate) use files::{extract_file_size, extract_xet_hash};
use futures::Stream;
pub use repo_type::{
    RepoType, RepoTypeAny, RepoTypeDataset, RepoTypeKernel, RepoTypeModel, RepoTypeSpace,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize, Serializer};
use url::Url;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::{constants, retry};

/// Access-gating mode for a repository.
///
/// Controls whether users must request access and how requests are approved.
/// Serializes as `false` when [`GatedApprovalMode::Disabled`], or as the lowercase mode string otherwise.
#[derive(Debug, Clone)]
pub enum GatedApprovalMode {
    /// Access is open; no request is required.
    Disabled,
    /// Access requests are approved automatically once the user accepts the terms.
    Auto,
    /// Access requests must be reviewed and approved by a repo owner.
    Manual,
}

/// Notification cadence for gated-access requests on a repository.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GatedNotificationsMode {
    /// Bundle notifications and deliver them periodically.
    Bulk,
    /// Notify on every access request as it arrives.
    RealTime,
}

/// Notification preferences for gated-access requests on a repository.
///
/// Groups the cadence (`mode`) with the optional override `email`. Pass to
/// [`HFRepository::update_settings`] via the `gated_notifications` parameter so
/// the two fields move together — leaving the email out keeps the existing
/// recipient.
#[derive(Debug, Clone)]
pub struct GatedNotifications {
    /// Cadence at which gated-access notifications are sent.
    pub mode: GatedNotificationsMode,
    /// Override the email address that receives gated-access notifications.
    /// When `None`, the existing recipient configured on the repository is left in place.
    pub email: Option<String>,
}

impl GatedNotifications {
    /// Construct a notification configuration with just the cadence.
    pub fn new(mode: GatedNotificationsMode) -> Self {
        Self { mode, email: None }
    }

    /// Set or replace the override email recipient for gated-access notifications.
    pub fn with_email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }
}

/// A single file entry in a repository's flat "siblings" listing, as returned by the repo info endpoint.
///
/// Most fields are populated only when the repo info request asks for file metadata (the `files_metadata`
/// option in the Python client / `?blobs=true` on the API). When listing repos, only `rfilename` is set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoSibling {
    /// File path relative to the repo root.
    pub rfilename: String,
    /// File size in bytes. Populated only when file metadata was requested.
    pub size: Option<u64>,
    /// LFS metadata for the file. Populated only when the file is stored with Git LFS and file metadata was
    /// requested.
    pub lfs: Option<BlobLfsInfo>,
}

/// SafeTensors footprint for a model: per-dtype parameter counts and total.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SafeTensorsInfo {
    /// Parameter counts keyed by dtype (e.g., `"F32"`, `"BF16"`, `"I8"`).
    pub parameters: HashMap<String, u64>,
    /// Total number of parameters across all dtypes.
    pub total: u64,
}

/// Transformers-specific metadata for a model (auto class, processor, etc.).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TransformersInfo {
    /// Name of the Transformers auto class for this model (e.g., `"AutoModelForCausalLM"`).
    pub auto_model: String,
    /// Custom Python class declared by the model, if any.
    #[serde(default)]
    pub custom_class: Option<String>,
    /// Pipeline tag declared in `transformersInfo` (may differ from the top-level `pipeline_tag`).
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    /// Processor name declared by the model, if any.
    #[serde(default)]
    pub processor: Option<String>,
}

/// Inference-providers mapping for a model.
///
/// Mirrors `huggingface_hub.InferenceProviderMapping`. Each entry describes how a single provider
/// serves the model.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceProviderMapping {
    /// Provider name (e.g., `"hf-inference"`, `"together"`).
    pub provider: String,
    /// ID of the model on the provider's side.
    pub provider_id: String,
    /// Status of the mapping: `"error"`, `"live"`, or `"staging"`.
    pub status: String,
    /// Task served by this provider (e.g., `"text-generation"`).
    pub task: String,
    /// Adapter name, if the mapping uses an adapter.
    #[serde(default)]
    pub adapter: Option<String>,
    /// Path to adapter weights, if applicable.
    #[serde(default)]
    pub adapter_weights_path: Option<String>,
    /// Mapping kind: `"single-model"` or `"tag-filter"`.
    #[serde(default, rename = "type")]
    pub r#type: Option<String>,
}

// The Hub returns `inferenceProviderMapping` as an object keyed by provider name on
// `GET /api/models/{id}` and as an array on `GET /api/models`. See:
//   https://huggingface.co/docs/inference-providers/hub-api
//   https://huggingface.co/docs/inference-providers/register-as-a-provider
// The Python `huggingface_hub` library handles both shapes the same way and notes:
// "a dict on model_info and a list on list_models. Let's harmonize to list."
fn deserialize_inference_provider_mapping<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<InferenceProviderMapping>>, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    use serde::de::Error;
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value else { return Ok(None) };
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(serde_json::from_value(item).map_err(D::Error::custom)?);
            }
            Ok(Some(out))
        }
        serde_json::Value::Object(map) => {
            let mut out = Vec::with_capacity(map.len());
            for (provider, mut value) in map {
                if let serde_json::Value::Object(ref mut obj) = value {
                    obj.insert("provider".to_string(), serde_json::Value::String(provider));
                }
                out.push(serde_json::from_value(value).map_err(D::Error::custom)?);
            }
            Ok(Some(out))
        }
        _ => Err(D::Error::custom(
            "expected list or object for inferenceProviderMapping",
        )),
    }
}

/// One evaluation-result entry from the `.eval_results/*.yaml` format.
///
/// See <https://huggingface.co/docs/hub/eval-results>.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalResultEntry {
    /// Benchmark dataset and task identifiers.
    pub dataset: EvalResultDataset,
    /// The metric value (numeric, string, or other JSON shape depending on the benchmark).
    pub value: serde_json::Value,
    /// Signature proving the evaluation is auditable and reproducible.
    #[serde(default, rename = "verifyToken")]
    pub verify_token: Option<String>,
    /// ISO-8601 datetime when the evaluation was run; defaults to git commit time when omitted.
    #[serde(default)]
    pub date: Option<String>,
    /// Source attribution for the evaluation (Space, dataset, user, org), if any.
    #[serde(default)]
    pub source: Option<EvalResultSource>,
    /// Free-text notes about the evaluation setup.
    #[serde(default)]
    pub notes: Option<String>,
}

/// Benchmark dataset and task identifiers for an [`EvalResultEntry`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalResultDataset {
    /// Benchmark dataset ID (e.g., `"cais/hle"`).
    pub id: String,
    /// Task identifier within the benchmark (e.g., `"gpqa_diamond"`).
    pub task_id: String,
    /// Git SHA of the benchmark dataset, if pinned.
    #[serde(default)]
    pub revision: Option<String>,
}

/// Source attribution for an [`EvalResultEntry`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalResultSource {
    /// URL pointing to the evaluation source (Space, dataset, etc.).
    #[serde(default)]
    pub url: Option<String>,
    /// Display name for the source.
    #[serde(default)]
    pub name: Option<String>,
    /// HF user attributed for the evaluation.
    #[serde(default)]
    pub user: Option<String>,
    /// HF org attributed for the evaluation.
    #[serde(default)]
    pub org: Option<String>,
}

/// Metadata for a model repository on the Hub.
///
/// Returned by [`HFClient::list_models`] and by
/// [`HFRepository::info`] when the repo is a model.
/// Most fields are optional because they depend on the `expand` parameter and the repo's state.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// Repo ID, in the form `owner/name`.
    pub id: String,
    /// Internal Hub identifier (the API's `_id` field). Most callers should use `id` instead.
    #[serde(rename = "_id")]
    pub internal_id: Option<String>,
    /// Owner of the repo (the part before `/` in `id`).
    pub author: Option<String>,
    /// Base models this model is derived from.
    #[serde(default)]
    pub base_models: Option<Vec<String>>,
    /// Parsed YAML metadata from the model card (`README.md` front matter). Modeled as raw JSON
    /// because the schema varies by library.
    pub card_data: Option<serde_json::Value>,
    /// Number of children (derived) models.
    pub children_model_count: Option<u64>,
    /// Model configuration (e.g., parsed `config.json` for Transformers models).
    pub config: Option<serde_json::Value>,
    /// ISO-8601 timestamp when the repo was created. The earliest possible value is
    /// `2022-03-02T23:29:04.000Z` (when the Hub started recording creation dates).
    pub created_at: Option<String>,
    /// Whether the repo is disabled.
    pub disabled: Option<bool>,
    /// Number of downloads over the last 30 days.
    pub downloads: Option<u64>,
    /// Cumulative download count since repo creation.
    pub downloads_all_time: Option<u64>,
    /// Evaluation results parsed from the model's `.eval_results/*.yaml` files.
    #[serde(default, rename = "evalResults")]
    pub eval_results: Option<Vec<EvalResultEntry>>,
    /// Gated-access state. Either the boolean `false` (open) or the string `"auto"`/`"manual"` indicating
    /// the approval mode for access requests. Modeled as raw JSON because the field is union-typed.
    pub gated: Option<serde_json::Value>,
    /// GGUF-specific metadata, when the repo contains GGUF files.
    pub gguf: Option<serde_json::Value>,
    /// Inference-providers status. Currently, `Some("warm")` when the model is served by at least
    /// one provider, `None` otherwise.
    pub inference: Option<String>,
    /// Per-provider inference mappings, ordered by the user's provider preference.
    ///
    /// The Hub returns this either as a list (modern shape) or as an object keyed by provider name
    /// (legacy shape); both are accepted.
    #[serde(default, deserialize_with = "deserialize_inference_provider_mapping")]
    pub inference_provider_mapping: Option<Vec<InferenceProviderMapping>>,
    /// ISO-8601 timestamp of the most recent commit to the repo.
    pub last_modified: Option<String>,
    /// Library this model is associated with (e.g., `"transformers"`, `"diffusers"`).
    #[serde(rename = "library_name")]
    pub library_name: Option<String>,
    /// Number of likes on the repo.
    pub likes: Option<u64>,
    /// Mask token used by the model (for fill-mask tasks).
    #[serde(rename = "mask_token")]
    pub mask_token: Option<String>,
    /// Model-index data describing benchmark results in the `model-index` format.
    #[serde(rename = "model-index")]
    pub model_index: Option<serde_json::Value>,
    /// Primary task tag (e.g., `"text-generation"`, `"image-classification"`).
    #[serde(rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    /// Whether the repo is private.
    pub private: Option<bool>,
    /// Resource-group information, when the repo belongs to one.
    pub resource_group: Option<serde_json::Value>,
    /// Per-dtype parameter counts produced from the model's safetensors files, if any.
    pub safetensors: Option<SafeTensorsInfo>,
    /// Security-scan summary for the repo.
    pub security_repo_status: Option<serde_json::Value>,
    /// Git commit SHA at the revision the response describes.
    pub sha: Option<String>,
    /// Files in the repo. Only populated when file metadata is requested; otherwise only `rfilename`
    /// is set on each entry. See [`RepoSibling`].
    pub siblings: Option<Vec<RepoSibling>>,
    /// IDs of Spaces that use this model.
    pub spaces: Option<Vec<String>>,
    /// Hub tags. Includes both author-provided tags from the model card and tags computed by the Hub
    /// (e.g., supported libraries, arXiv references).
    pub tags: Option<Vec<String>>,
    /// Transformers-specific metadata declared by the model.
    pub transformers_info: Option<TransformersInfo>,
    /// Trending score used to rank the repo on the Hub's trending lists.
    pub trending_score: Option<f64>,
    /// Total size of the repo on disk, in bytes.
    pub used_storage: Option<u64>,
    /// Inference-widget configuration declared in the model card.
    pub widget_data: Option<serde_json::Value>,
}

/// Metadata for a dataset repository on the Hub.
///
/// Returned by [`HFClient::list_datasets`] and by
/// [`HFRepository::info`] when the repo is a dataset.
/// Most fields are optional because they depend on the `expand` parameter and the repo's state.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetInfo {
    /// Repo ID, in the form `owner/name`.
    pub id: String,
    /// Internal Hub identifier (the API's `_id` field). Most callers should use `id` instead.
    #[serde(rename = "_id")]
    pub internal_id: Option<String>,
    /// Owner of the repo (the part before `/` in `id`).
    pub author: Option<String>,
    /// Git commit SHA at the revision the response describes.
    pub sha: Option<String>,
    /// Whether the repo is private.
    pub private: Option<bool>,
    /// Gated-access state. Either the boolean `false` (open) or the string `"auto"`/`"manual"` indicating
    /// the approval mode for access requests. Modeled as raw JSON because the field is union-typed.
    pub gated: Option<serde_json::Value>,
    /// Whether the repo is disabled.
    pub disabled: Option<bool>,
    /// Number of downloads over the last 30 days.
    pub downloads: Option<u64>,
    /// Cumulative download count since repo creation.
    pub downloads_all_time: Option<u64>,
    /// Number of likes on the repo.
    pub likes: Option<u64>,
    /// Hub tags declared on the dataset.
    pub tags: Option<Vec<String>>,
    /// ISO-8601 timestamp when the repo was created. The earliest possible value is
    /// `2022-03-02T23:29:04.000Z` (when the Hub started recording creation dates).
    pub created_at: Option<String>,
    /// ISO-8601 timestamp of the most recent commit to the repo.
    pub last_modified: Option<String>,
    /// Files in the repo. Only populated when file metadata is requested; otherwise only `rfilename`
    /// is set on each entry. See [`RepoSibling`].
    pub siblings: Option<Vec<RepoSibling>>,
    /// Parsed YAML metadata from the dataset card (`README.md` front matter).
    pub card_data: Option<serde_json::Value>,
    /// Citation information for the dataset.
    pub citation: Option<String>,
    /// Papers-with-code identifier, when the dataset is registered there.
    #[serde(rename = "paperswithcode_id")]
    pub paperswithcode_id: Option<String>,
    /// Resource-group information, when the repo belongs to one.
    pub resource_group: Option<serde_json::Value>,
    /// Trending score used to rank the repo on the Hub's trending lists.
    pub trending_score: Option<f64>,
    /// Free-text description of the dataset.
    pub description: Option<String>,
    /// Total size of the repo on disk, in bytes.
    pub used_storage: Option<u64>,
}

/// Metadata for a Space repository on the Hub.
///
/// Returned by [`HFClient::list_spaces`] and by
/// [`HFRepository::info`] when the repo is a Space.
/// Most fields are optional because they depend on the `expand` parameter and the Space's state.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpaceInfo {
    /// Repo ID, in the form `owner/name`.
    pub id: String,
    /// Internal Hub identifier (the API's `_id` field). Most callers should use `id` instead.
    #[serde(rename = "_id")]
    pub internal_id: Option<String>,
    /// Owner of the repo (the part before `/` in `id`).
    pub author: Option<String>,
    /// Git commit SHA at the revision the response describes.
    pub sha: Option<String>,
    /// Whether the repo is private.
    pub private: Option<bool>,
    /// Gated-access state. Either the boolean `false` (open) or the string `"auto"`/`"manual"` indicating
    /// the approval mode for access requests. Modeled as raw JSON because the field is union-typed.
    pub gated: Option<serde_json::Value>,
    /// Whether the Space is disabled.
    pub disabled: Option<bool>,
    /// Number of likes on the Space.
    pub likes: Option<u64>,
    /// Hub tags declared on the Space.
    pub tags: Option<Vec<String>>,
    /// ISO-8601 timestamp when the repo was created. The earliest possible value is
    /// `2022-03-02T23:29:04.000Z` (when the Hub started recording creation dates).
    pub created_at: Option<String>,
    /// ISO-8601 timestamp of the most recent commit to the repo.
    pub last_modified: Option<String>,
    /// Files in the repo. Only populated when file metadata is requested; otherwise only `rfilename`
    /// is set on each entry. See [`RepoSibling`].
    pub siblings: Option<Vec<RepoSibling>>,
    /// Parsed YAML metadata from the Space card (`README.md` front matter).
    pub card_data: Option<serde_json::Value>,
    /// SDK powering the Space (e.g., `"gradio"`, `"streamlit"`, `"docker"`, `"static"`).
    pub sdk: Option<String>,
    /// Trending score used to rank the Space on the Hub's trending lists.
    pub trending_score: Option<f64>,
    /// Hostname serving the Space.
    pub host: Option<String>,
    /// Subdomain serving the Space.
    pub subdomain: Option<String>,
    /// Runtime state of the Space (stage, hardware, sleep time, volumes, etc.).
    ///
    /// Populated when the repo info request expands the `runtime` field. The same shape is also
    /// returned by [`HFRepository::<RepoTypeSpace>::runtime`](HFRepository::runtime).
    pub runtime: Option<crate::spaces::SpaceRuntime>,
    /// Datasets used by the Space, declared in the Space card.
    pub datasets: Option<Vec<String>>,
    /// Models used by the Space, declared in the Space card.
    pub models: Option<Vec<String>>,
    /// Resource-group information, when the repo belongs to one.
    pub resource_group: Option<serde_json::Value>,
    /// Total size of the repo on disk, in bytes.
    pub used_storage: Option<u64>,
}

/// Metadata for a kernel repository on the Hub.
///
/// Returned by [`HFRepository::info`] when the repo is a kernel. The Hub's
/// `/api/kernels/{repo_id}` endpoint returns a slim shape compared with
/// model/dataset/space repos — fields like `tags`, `cardData`, and `siblings`
/// are not exposed by this endpoint and are intentionally absent here. Most
/// fields are optional because they depend on the repo's state.
///
/// Kernels are also retrievable via `/api/models/{repo_id}` (kernels carry
/// `library_name: "kernels"`) if you need the full model-style metadata; in
/// that case go through [`HFClient::model`] and the [`ModelInfo`] response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelInfo {
    /// Repo ID, in the form `owner/name`.
    pub id: String,
    /// Internal Hub identifier (the API's `_id` field). Most callers should use `id` instead.
    #[serde(rename = "_id")]
    pub internal_id: Option<String>,
    /// Owner of the repo (the part before `/` in `id`).
    pub author: Option<String>,
    /// Git commit SHA at the revision the response describes.
    pub sha: Option<String>,
    /// Whether the repo is private.
    pub private: Option<bool>,
    /// Gated-access state. Either the boolean `false` (open) or the string `"auto"`/`"manual"` indicating
    /// the approval mode for access requests. Modeled as raw JSON because the field is union-typed.
    pub gated: Option<serde_json::Value>,
    /// Total downloads of this kernel.
    pub downloads: Option<u64>,
    /// Number of likes on the kernel.
    pub likes: Option<u64>,
    /// ISO-8601 timestamp of the most recent commit to the repo.
    pub last_modified: Option<String>,
    /// Whether the publisher is trusted by the Hub. Kernels from trusted publishers receive a
    /// distinct badge in the UI.
    pub trusted_publisher: Option<bool>,
    /// Driver families the prebuilt kernel artifacts support, e.g., `"cuda"`, `"xpu"`, `"cpu"`.
    /// May be absent when the kernel has not declared its supported drivers.
    pub supported_driver_families: Option<Vec<String>>,
}

/// URL of a repository, returned by create/move endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct RepoUrl {
    /// Absolute URL of the repository on the Hub.
    pub url: String,
}

/// A handle for a single repository on the Hugging Face Hub, parameterized by the
/// repo kind via the type-level marker `T`.
///
/// `HFRepository<T>` is created via the typed factories on [`HFClient`] —
/// [`HFClient::model`], [`HFClient::dataset`], [`HFClient::space`], or
/// [`HFClient::kernel`] — and binds together the client, owner, and repo name. The
/// repo kind lives in the type system rather than in a runtime field, so methods
/// that differ per repo kind (such as [`info`](HFRepository::info)) are dispatched
/// at compile time and mismatches are impossible by construction.
///
/// Cheap to clone — the inner [`HFClient`] is `Arc`-backed.
///
/// # Example
///
/// ```rust,no_run
/// # use hf_hub::HFClient;
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::builder().build()?;
/// let repo = client.model("openai-community", "gpt2");
/// let info = repo.info().send().await?;
/// # Ok(()) }
/// ```
pub struct HFRepository<T: RepoType> {
    pub(crate) hf_client: HFClient,
    pub(super) owner: String,
    pub(super) name: String,
    pub(super) repo_type: T,
}

impl<T: RepoType> Clone for HFRepository<T> {
    fn clone(&self) -> Self {
        Self {
            hf_client: self.hf_client.clone(),
            owner: self.owner.clone(),
            name: self.name.clone(),
            repo_type: self.repo_type,
        }
    }
}

impl Serialize for GatedApprovalMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            GatedApprovalMode::Disabled => serializer.serialize_bool(false),
            GatedApprovalMode::Auto => serializer.serialize_str("auto"),
            GatedApprovalMode::Manual => serializer.serialize_str("manual"),
        }
    }
}

impl FromStr for GatedApprovalMode {
    type Err = HFError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "false" | "disabled" => Ok(GatedApprovalMode::Disabled),
            "auto" => Ok(GatedApprovalMode::Auto),
            "manual" => Ok(GatedApprovalMode::Manual),
            _ => Err(HFError::InvalidParameter(format!(
                "unknown gated approval mode: {s:?}. Expected 'auto', 'manual', or 'false'"
            ))),
        }
    }
}

impl FromStr for GatedNotificationsMode {
    type Err = HFError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bulk" => Ok(GatedNotificationsMode::Bulk),
            "real-time" | "realtime" => Ok(GatedNotificationsMode::RealTime),
            _ => Err(HFError::InvalidParameter(format!(
                "unknown gated notifications mode: {s:?}. Expected 'bulk' or 'real-time'"
            ))),
        }
    }
}

impl<T: RepoType> std::fmt::Debug for HFRepository<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFRepository")
            .field("owner", &self.owner)
            .field("name", &self.name)
            .field("repo_type", &self.repo_type.singular())
            .finish()
    }
}

impl HFClient {
    /// Create an [`HFRepository`] handle for any repo kind via a turbofished generic.
    ///
    /// Equivalent to the typed shortcuts ([`model`](Self::model), [`dataset`](Self::dataset),
    /// [`space`](Self::space), [`kernel`](Self::kernel)) but useful when the kind is
    /// expressed by an external type parameter or a `match` arm — for example, dispatching
    /// from a CLI flag:
    ///
    /// ```rust,no_run
    /// # use hf_hub::{HFClient, RepoTypeDataset};
    /// # let client = HFClient::builder().build().unwrap();
    /// let repo = client.repository(RepoTypeDataset, "rajpurkar", "squad");
    /// # let _ = repo;
    /// ```
    pub fn repository<T: RepoType>(
        &self,
        repo_type: T,
        owner: impl Into<String>,
        name: impl Into<String>,
    ) -> HFRepository<T> {
        HFRepository::new(self.clone(), repo_type, owner, name)
    }

    /// Create an [`HFRepository`] handle for a model repository.
    pub fn model(
        &self,
        owner: impl Into<String>,
        name: impl Into<String>,
    ) -> HFRepository<RepoTypeModel> {
        self.repository(RepoTypeModel, owner, name)
    }

    /// Create an [`HFRepository`] handle for a dataset repository.
    pub fn dataset(
        &self,
        owner: impl Into<String>,
        name: impl Into<String>,
    ) -> HFRepository<RepoTypeDataset> {
        self.repository(RepoTypeDataset, owner, name)
    }

    /// Create an [`HFRepository`] handle for a Space repository.
    pub fn space(
        &self,
        owner: impl Into<String>,
        name: impl Into<String>,
    ) -> HFRepository<RepoTypeSpace> {
        self.repository(RepoTypeSpace, owner, name)
    }

    /// Create an [`HFRepository`] handle for a kernel repository.
    pub fn kernel(
        &self,
        owner: impl Into<String>,
        name: impl Into<String>,
    ) -> HFRepository<RepoTypeKernel> {
        self.repository(RepoTypeKernel, owner, name)
    }
}

#[bon]
impl HFClient {
    /// List models on the Hub. Endpoint: `GET /api/models`.
    ///
    /// Returns a stream of [`ModelInfo`] entries. Pagination is automatic.
    ///
    /// # Parameters
    ///
    /// - `search`: free-text query forwarded as the `?search=` parameter. The Hub matches it substring-style against
    ///   the model `id` and (when present) the model card description — it is **not** a tag filter.
    /// - `author`: namespace owner to filter on, forwarded as `?author=`. Pass a Hub user or organization name (e.g.,
    ///   `"google"`, `"meta-llama"`) — bare names, not paths.
    /// - `filter`: a single Hub **tag** value forwarded as `?filter=`. Tags use the Hub's namespaced format, e.g.,
    ///   `"pytorch"`, `"text-generation"`, `"license:apache-2.0"`, `"language:en"`, `"dataset:wikipedia"`,
    ///   `"region:us"`. To combine tags, narrow the results client-side (only one `filter` value is sent).
    /// - `sort`: API field name to sort by, forwarded as `?sort=`. Common values are `"downloads"`, `"likes"`,
    ///   `"createdAt"`, `"lastModified"`, and `"trendingScore"`. Use the camelCase Hub field names (not Rust struct
    ///   field names).
    /// - `pipeline_tag`: pipeline-tag filter (e.g., `"text-classification"`, `"automatic-speech-recognition"`),
    ///   forwarded as `?pipeline_tag=`. Same vocabulary as the `pipeline_tag` field on a model card.
    /// - `full`: fetch the full model information including all fields.
    /// - `card_data`: include the model card metadata in the response.
    /// - `fetch_config`: include the model configuration in the response.
    /// - `limit`: cap on the total number of items yielded by the stream. When less than 1000, also used as the server
    ///   page size.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_models(
        &self,
        /// Free-text query forwarded as the `?search=` parameter. The Hub matches it substring-style against
        /// the model `id` and (when present) the model card description — it is **not** a tag filter.
        #[builder(into)]
        search: Option<String>,
        /// Namespace owner to filter on, forwarded as `?author=`. Pass a Hub user or organization name (e.g.,
        /// `"google"`, `"meta-llama"`) — bare names, not paths.
        #[builder(into)]
        author: Option<String>,
        /// A single Hub **tag** value forwarded as `?filter=`. Tags use the Hub's namespaced format, e.g.,
        /// `"pytorch"`, `"text-generation"`, `"license:apache-2.0"`, `"language:en"`, `"dataset:wikipedia"`,
        /// `"region:us"`. To combine tags, narrow the results client-side (only one `filter` value is sent).
        #[builder(into)]
        filter: Option<String>,
        /// API field name to sort by, forwarded as `?sort=`. Common values are `"downloads"`, `"likes"`,
        /// `"createdAt"`, `"lastModified"`, and `"trendingScore"`. Use the camelCase Hub field names (not Rust struct
        /// field names).
        #[builder(into)]
        sort: Option<String>,
        /// Pipeline-tag filter (e.g., `"text-classification"`, `"automatic-speech-recognition"`),
        /// forwarded as `?pipeline_tag=`. Same vocabulary as the `pipeline_tag` field on a model card.
        #[builder(into)]
        pipeline_tag: Option<String>,
        /// Fetch the full model information including all fields.
        full: Option<bool>,
        /// Include the model card metadata in the response.
        card_data: Option<bool>,
        /// Include the model configuration in the response.
        fetch_config: Option<bool>,
        /// Cap on the total number of items yielded by the stream. When less than 1000, also used as the server
        /// page size.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<ModelInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/models", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = search {
            query.push(("search".into(), s.clone()));
        }
        if let Some(ref a) = author {
            query.push(("author".into(), a.clone()));
        }
        if let Some(ref f) = filter {
            query.push(("filter".into(), f.clone()));
        }
        if let Some(ref s) = sort {
            query.push(("sort".into(), s.clone()));
        }
        if let Some(max) = limit {
            // The Hub API usually returns up to 1000 items per page by default,
            // so only set an explicit limit for smaller requests.
            if max < 1000 {
                query.push(("limit".into(), max.to_string()));
            }
        }
        if let Some(ref pt) = pipeline_tag {
            query.push(("pipeline_tag".into(), pt.clone()));
        }
        if full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        if card_data == Some(true) {
            query.push(("cardData".into(), "true".into()));
        }
        if fetch_config == Some(true) {
            query.push(("config".into(), "true".into()));
        }
        Ok(self.paginate(url, query, limit))
    }

    /// List datasets on the Hub. Endpoint: `GET /api/datasets`.
    ///
    /// Returns a stream of [`DatasetInfo`] entries. Pagination is automatic.
    ///
    /// # Parameters
    ///
    /// - `search`: free-text query forwarded as `?search=`. The Hub matches it substring-style against the dataset `id`
    ///   and card description — not a tag filter.
    /// - `author`: namespace owner forwarded as `?author=`. Pass a bare Hub user or organization name (e.g.,
    ///   `"HuggingFaceH4"`, `"allenai"`).
    /// - `filter`: a single Hub **tag** value forwarded as `?filter=`. Tags use the Hub's namespaced format, e.g.,
    ///   `"task_categories:text-classification"`, `"language:en"`, `"size_categories:10K<n<100K"`, `"license:mit"`. To
    ///   combine tags, narrow client-side — only one `filter` value is sent.
    /// - `sort`: API field name to sort by, forwarded as `?sort=`. Common values are `"downloads"`, `"likes"`,
    ///   `"createdAt"`, `"lastModified"`, and `"trendingScore"` (Hub camelCase field names).
    /// - `full`: fetch the full dataset information including all fields.
    /// - `limit`: cap on the total number of items yielded by the stream. When less than 1000, also used as the server
    ///   page size.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_datasets(
        &self,
        /// Free-text query forwarded as `?search=`. The Hub matches it substring-style against the dataset `id`
        /// and card description — not a tag filter.
        #[builder(into)]
        search: Option<String>,
        /// Namespace owner forwarded as `?author=`. Pass a bare Hub user or organization name (e.g.,
        /// `"HuggingFaceH4"`, `"allenai"`).
        #[builder(into)]
        author: Option<String>,
        /// A single Hub **tag** value forwarded as `?filter=`. Tags use the Hub's namespaced format, e.g.,
        /// `"task_categories:text-classification"`, `"language:en"`, `"size_categories:10K<n<100K"`, `"license:mit"`.
        /// To combine tags, narrow client-side — only one `filter` value is sent.
        #[builder(into)]
        filter: Option<String>,
        /// API field name to sort by, forwarded as `?sort=`. Common values are `"downloads"`, `"likes"`,
        /// `"createdAt"`, `"lastModified"`, and `"trendingScore"` (Hub camelCase field names).
        #[builder(into)]
        sort: Option<String>,
        /// Fetch the full dataset information including all fields.
        full: Option<bool>,
        /// Cap on the total number of items yielded by the stream. When less than 1000, also used as the server
        /// page size.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<DatasetInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/datasets", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = search {
            query.push(("search".into(), s.clone()));
        }
        if let Some(ref a) = author {
            query.push(("author".into(), a.clone()));
        }
        if let Some(ref f) = filter {
            query.push(("filter".into(), f.clone()));
        }
        if let Some(ref s) = sort {
            query.push(("sort".into(), s.clone()));
        }
        if let Some(max) = limit
            && max < 1000
        {
            query.push(("limit".into(), max.to_string()));
        }
        if full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        Ok(self.paginate(url, query, limit))
    }

    /// List Spaces on the Hub. Endpoint: `GET /api/spaces`.
    ///
    /// Returns a stream of [`SpaceInfo`] entries. Pagination is automatic.
    ///
    /// # Parameters
    ///
    /// - `search`: free-text query forwarded as `?search=`. The Hub matches it substring-style against the Space `id`
    ///   and card description — not a tag filter.
    /// - `author`: namespace owner forwarded as `?author=`. Pass a bare Hub user or organization name (e.g.,
    ///   `"openai"`, `"stabilityai"`).
    /// - `filter`: a single Hub **tag** value forwarded as `?filter=`. Tags use the Hub's namespaced format, e.g.,
    ///   `"sdk:gradio"`, `"sdk:streamlit"`, `"sdk:docker"`, `"language:en"`, `"license:mit"`. To combine tags, narrow
    ///   client-side — only one `filter` value is sent.
    /// - `sort`: API field name to sort by, forwarded as `?sort=`. Common values are `"likes"`, `"createdAt"`,
    ///   `"lastModified"`, and `"trendingScore"` (Hub camelCase field names).
    /// - `full`: fetch the full Space information including all fields.
    /// - `limit`: cap on the total number of items yielded by the stream. When less than 1000, also used as the server
    ///   page size.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_spaces(
        &self,
        /// Free-text query forwarded as `?search=`. The Hub matches it substring-style against the Space `id`
        /// and card description — not a tag filter.
        #[builder(into)]
        search: Option<String>,
        /// Namespace owner forwarded as `?author=`. Pass a bare Hub user or organization name (e.g., `"openai"`,
        /// `"stabilityai"`).
        #[builder(into)]
        author: Option<String>,
        /// A single Hub **tag** value forwarded as `?filter=`. Tags use the Hub's namespaced format, e.g.,
        /// `"sdk:gradio"`, `"sdk:streamlit"`, `"sdk:docker"`, `"language:en"`, `"license:mit"`. To combine tags,
        /// narrow client-side — only one `filter` value is sent.
        #[builder(into)]
        filter: Option<String>,
        /// API field name to sort by, forwarded as `?sort=`. Common values are `"likes"`, `"createdAt"`,
        /// `"lastModified"`, and `"trendingScore"` (Hub camelCase field names).
        #[builder(into)]
        sort: Option<String>,
        /// Fetch the full Space information including all fields.
        full: Option<bool>,
        /// Cap on the total number of items yielded by the stream. When less than 1000, also used as the server
        /// page size.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<SpaceInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/spaces", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = search {
            query.push(("search".into(), s.clone()));
        }
        if let Some(ref a) = author {
            query.push(("author".into(), a.clone()));
        }
        if let Some(ref f) = filter {
            query.push(("filter".into(), f.clone()));
        }
        if let Some(ref s) = sort {
            query.push(("sort".into(), s.clone()));
        }
        if let Some(max) = limit
            && max < 1000
        {
            query.push(("limit".into(), max.to_string()));
        }
        if full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        Ok(self.paginate(url, query, limit))
    }

    /// Create a new repository. Endpoint: `POST /api/repos/create`.
    ///
    /// The repo kind is picked at the call site by passing one of the four marker
    /// structs to [`repo_type`](HFClientCreateRepositoryBuilder::repo_type) — e.g.,
    /// `client.create_repository().repo_type(RepoTypeDataset)...` to create a dataset.
    /// The body always includes the `type` field, matching the Hub's per-kind
    /// defaults.
    ///
    /// # Parameters
    ///
    /// - `repo_id` (required): repository ID in `"owner/name"` or `"name"` format.
    /// - `repo_type` (required): the repo kind ([`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`], or
    ///   [`RepoTypeKernel`]).
    /// - `private`: whether the repository should be private.
    /// - `exist_ok` (default `false`): if `true`, do not error when the repository already exists.
    /// - `space_sdk`: SDK for a Space (e.g., `"gradio"`, `"streamlit"`, `"docker"`). Required when `repo_type` is
    ///   [`RepoTypeSpace`]; ignored for other repo kinds.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn create_repository(
        &self,
        /// Repository ID in `"owner/name"` or `"name"` format.
        repo_id: &str,
        /// The repo kind ([`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`],
        /// or [`RepoTypeKernel`]).
        repo_type: impl RepoType,
        /// Whether the repository should be private.
        private: Option<bool>,
        /// If `true`, do not error when the repository already exists.
        #[builder(default)]
        exist_ok: bool,
        /// SDK for a Space (e.g., `"gradio"`, `"streamlit"`, `"docker"`). Required when `repo_type` is
        /// [`RepoTypeSpace`]; ignored for other repo kinds.
        #[builder(into)]
        space_sdk: Option<String>,
    ) -> HFResult<RepoUrl> {
        let url = format!("{}/api/repos/create", self.endpoint());

        let (namespace, name) = split_repo_id(repo_id);

        let mut body = serde_json::json!({
            "name": name,
            "private": private.unwrap_or(false),
            "type": repo_type.singular(),
        });

        if let Some(ns) = namespace {
            body["organization"] = serde_json::Value::String(ns.to_string());
        }
        if let Some(sdk) = space_sdk {
            body["sdk"] = serde_json::Value::String(sdk);
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        if response.status().as_u16() == 409 && exist_ok {
            return Ok(RepoUrl {
                url: format!("{}/{}{}", self.endpoint(), repo_type.url_prefix(), repo_id),
            });
        }

        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Delete a repository. Endpoint: `DELETE /api/repos/delete`.
    ///
    /// The repo kind is picked at the call site by passing one of the four marker
    /// structs to [`repo_type`](HFClientDeleteRepositoryBuilder::repo_type) — e.g.,
    /// `client.delete_repository().repo_type(RepoTypeSpace)...`. The body always includes
    /// the `type` field.
    ///
    /// # Parameters
    ///
    /// - `repo_id` (required): repository ID in `"owner/name"` or `"name"` format.
    /// - `repo_type` (required): the repo kind ([`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`], or
    ///   [`RepoTypeKernel`]).
    /// - `missing_ok` (default `false`): if `true`, do not error when the repository does not exist.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_repository(
        &self,
        /// Repository ID in `"owner/name"` or `"name"` format.
        repo_id: &str,
        /// The repo kind ([`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`],
        /// or [`RepoTypeKernel`]).
        repo_type: impl RepoType,
        /// If `true`, do not error when the repository does not exist.
        #[builder(default)]
        missing_ok: bool,
    ) -> HFResult<()> {
        let url = format!("{}/api/repos/delete", self.endpoint());

        let (namespace, name) = split_repo_id(repo_id);

        let mut body = serde_json::json!({
            "name": name,
            "type": repo_type.singular(),
        });
        if let Some(ns) = namespace {
            body["organization"] = serde_json::Value::String(ns.to_string());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client()
                .delete(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        if response.status().as_u16() == 404 && missing_ok {
            return Ok(());
        }

        self.check_response(response, Some(repo_id), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Move (rename) a repository. Endpoint: `POST /api/repos/move`.
    ///
    /// The repo kind is picked at the call site by passing one of the four marker
    /// structs to [`repo_type`](HFClientMoveRepositoryBuilder::repo_type) — e.g.,
    /// `client.move_repository().repo_type(RepoTypeModel)...`. The body always includes
    /// the `type` field.
    ///
    /// # Parameters
    ///
    /// - `from_id` (required): current repository ID in `"owner/name"` format.
    /// - `to_id` (required): new repository ID in `"owner/name"` format.
    /// - `repo_type` (required): the repo kind ([`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`], or
    ///   [`RepoTypeKernel`]).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn move_repository(
        &self,
        /// Current repository ID in `"owner/name"` format.
        from_id: &str,
        /// New repository ID in `"owner/name"` format.
        to_id: &str,
        /// The repo kind ([`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`],
        /// or [`RepoTypeKernel`]).
        repo_type: impl RepoType,
    ) -> HFResult<RepoUrl> {
        let url = format!("{}/api/repos/move", self.endpoint());
        let body = serde_json::json!({
            "fromRepo": from_id,
            "toRepo": to_id,
            "type": repo_type.singular(),
        });

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        self.check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(RepoUrl {
            url: format!("{}/{}{}", self.endpoint(), repo_type.url_prefix(), to_id),
        })
    }
}

impl<T: RepoType> HFRepository<T> {
    /// Construct a new repository handle. Prefer the factory methods on [`HFClient`] instead.
    pub fn new(
        client: HFClient,
        repo_type: T,
        owner: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            hf_client: client,
            owner: owner.into(),
            name: name.into(),
            repo_type,
        }
    }

    /// Return a reference to the underlying [`HFClient`].
    pub fn client(&self) -> &HFClient {
        &self.hf_client
    }

    /// The repository owner (user or organization name).
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The repository name (without owner prefix).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full `"owner/name"` identifier used in Hub API calls.
    ///
    /// If no owner is set, returns just the name (for repos using short-form IDs like `"gpt2"`).
    pub fn repo_path(&self) -> String {
        if self.owner.is_empty() {
            self.name.clone()
        } else {
            format!("{}/{}", self.owner, self.name)
        }
    }

    /// The marker for this handle's repo kind.
    ///
    /// Returns a reference to the stored marker value. Call
    /// [`singular`](RepoType::singular), [`plural`](RepoType::plural), or
    /// [`url_prefix`](RepoType::url_prefix) on it to get the corresponding string.
    /// For `T = RepoTypeAny`, the returned value reflects the runtime variant the
    /// handle was constructed with.
    pub fn repo_type(&self) -> &T {
        &self.repo_type
    }

    /// Fetch repo info for this repository's kind, deserializing into `I`.
    /// Endpoint: `GET /api/{plural}/{repo_id}[/revision/{revision}]`.
    pub(crate) async fn fetch_repo_info<I: DeserializeOwned>(
        &self,
        revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<I> {
        let mut url = self
            .hf_client
            .api_url(self.repo_type.plural(), &self.repo_path());
        if let Some(ref revision) = revision {
            url = format!("{url}/revision/{}", crate::client::encode_ref(revision));
        }
        let headers = self.hf_client.auth_headers();
        let expand_params: Option<Vec<(&str, &str)>> = expand
            .as_ref()
            .map(|e| e.iter().map(|v| ("expand", v.as_str())).collect());
        let response = retry::retry(self.hf_client.retry_config(), || {
            let mut req = self
                .hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone());
            if let Some(ref params) = expand_params {
                req = req.query(params);
            }
            req.send()
        })
        .await?;
        let repo_path = self.repo_path();
        let not_found_ctx = match revision {
            Some(rev) => crate::error::NotFoundContext::Revision { revision: rev },
            None => crate::error::NotFoundContext::Repo,
        };
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), not_found_ctx)
            .await?;
        Ok(response.json().await?)
    }
}

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Return `true` if the repository exists.
    ///
    /// Returns `Ok(false)` only when the Hub responds with 404. If the repo exists but the current
    /// credentials don't have access (private/gated), this returns an error
    /// ([`HFError::AuthRequired`] or [`HFError::Forbidden`]), not `Ok(false)`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn exists(&self) -> HFResult<bool> {
        let url = self
            .hf_client
            .api_url(self.repo_type.plural(), &self.repo_path());
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Generic,
            )
            .await?;
        Ok(true)
    }

    /// Return `true` if the given revision (branch, tag, or commit SHA) exists.
    ///
    /// Returns `Ok(false)` only when the Hub responds with 404. If the repo exists but the current
    /// credentials don't have access (private/gated), this returns an error
    /// ([`HFError::AuthRequired`] or [`HFError::Forbidden`]), not `Ok(false)`.
    ///
    /// # Parameters
    ///
    /// - `revision` (required): Git revision to check for existence.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn revision_exists(
        &self,
        /// Git revision to check for existence.
        revision: &str,
    ) -> HFResult<bool> {
        let url = format!(
            "{}/revision/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            crate::client::encode_ref(revision)
        );
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Generic,
            )
            .await?;
        Ok(true)
    }

    /// Return `true` if the given file exists in the repository at the specified revision.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to check within the repository.
    /// - `revision`: Git revision to check. Defaults to the main branch.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn file_exists(
        &self,
        /// Path of the file to check within the repository.
        filename: &str,
        /// Git revision to check. Defaults to the main branch.
        revision: Option<&str>,
    ) -> HFResult<bool> {
        let revision = revision.unwrap_or(constants::DEFAULT_REVISION);
        let url = self.hf_client.download_url(
            self.repo_type.url_prefix(),
            &self.repo_path(),
            revision,
            filename,
        )?;
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .head(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            if self.revision_exists().revision(revision).send().await? {
                return Ok(false);
            }
            return Err(HFError::RevisionNotFound {
                repo_id: self.repo_path(),
                revision: revision.to_string(),
                context: None,
            });
        }
        self.hf_client
            .check_response(
                response,
                Some(&self.repo_path()),
                crate::error::NotFoundContext::Generic,
            )
            .await?;
        Ok(true)
    }

    /// Update repository settings such as visibility, gating policy, description,
    /// discussion settings, and gated notification preferences.
    ///
    /// Endpoint: `PUT /api/{plural}/{repo_id}/settings`.
    ///
    /// # Parameters
    ///
    /// - `private`: whether the repository should be private.
    /// - `gated`: access-gating mode for the repository (e.g., `auto`, `manual`, disabled).
    /// - `description`: repository description shown on the Hub page.
    /// - `discussions_disabled`: whether discussions are disabled on this repository.
    /// - `gated_notifications`: notification cadence (and optional email override) for gated-access requests. The
    ///   cadence is required when this is set; pass [`GatedNotifications::with_email`] to override the recipient too.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn update_settings(
        &self,
        /// Whether the repository should be private.
        private: Option<bool>,
        /// Access-gating mode for the repository (e.g., `auto`, `manual`, disabled).
        gated: Option<GatedApprovalMode>,
        /// Repository description shown on the Hub page.
        #[builder(into)]
        description: Option<String>,
        /// Whether discussions are disabled on this repository.
        discussions_disabled: Option<bool>,
        /// Notification cadence (and optional email override) for gated-access requests.
        gated_notifications: Option<GatedNotifications>,
    ) -> HFResult<()> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct UpdateSettingsBody<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            private: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            gated: Option<&'a GatedApprovalMode>,
            #[serde(skip_serializing_if = "Option::is_none")]
            description: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            discussions_disabled: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            gated_notifications_email: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            gated_notifications_mode: Option<&'a GatedNotificationsMode>,
        }

        let body = UpdateSettingsBody {
            private,
            gated: gated.as_ref(),
            description: description.as_deref(),
            discussions_disabled,
            gated_notifications_email: gated_notifications
                .as_ref()
                .and_then(|g| g.email.as_deref()),
            gated_notifications_mode: gated_notifications.as_ref().map(|g| &g.mode),
        };

        let url = format!(
            "{}/settings",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path())
        );
        let headers = self.hf_client.auth_headers();

        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .put(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }
}

/// Documentation shared by the per-repo-kind `info` builders.
macro_rules! info_method_doc {
    () => {
        "Fetch repository metadata for this repo kind, returning the concrete info struct.\n\n\
        # Parameters\n\n\
        - `revision`: Git revision (branch, tag, or commit SHA). Defaults to the main branch.\n\
        - `expand`: list of properties to expand in the response (e.g., `\"trendingScore\"`, `\"cardData\"`).\n  \
          When set, only the listed properties (plus `_id` and `id`) are returned. Kernel info ignores this."
    };
}

#[bon]
impl HFRepository<RepoTypeModel> {
    /// Fetch [`ModelInfo`] for this model repository.
    #[doc = info_method_doc!()]
    #[builder(
        finish_fn = send,
        builder_type = HFModelInfoBuilder,
        state_mod(name = hf_model_info_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub async fn info(
        &self,
        /// Git revision (branch, tag, or commit SHA). Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// List of properties to expand in the response (e.g., `"trendingScore"`, `"cardData"`).
        expand: Option<Vec<String>>,
    ) -> HFResult<ModelInfo> {
        self.fetch_repo_info(revision, expand).await
    }
}

#[bon]
impl HFRepository<RepoTypeDataset> {
    /// Fetch [`DatasetInfo`] for this dataset repository.
    #[doc = info_method_doc!()]
    #[builder(
        finish_fn = send,
        builder_type = HFDatasetInfoBuilder,
        state_mod(name = hf_dataset_info_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub async fn info(
        &self,
        /// Git revision (branch, tag, or commit SHA). Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// List of properties to expand in the response (e.g., `"trendingScore"`, `"cardData"`).
        expand: Option<Vec<String>>,
    ) -> HFResult<DatasetInfo> {
        self.fetch_repo_info(revision, expand).await
    }
}

#[bon]
impl HFRepository<RepoTypeSpace> {
    /// Fetch [`SpaceInfo`] for this Space repository.
    #[doc = info_method_doc!()]
    #[builder(
        finish_fn = send,
        builder_type = HFSpaceInfoBuilder,
        state_mod(name = hf_space_info_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub async fn info(
        &self,
        /// Git revision (branch, tag, or commit SHA). Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// List of properties to expand in the response (e.g., `"trendingScore"`, `"cardData"`).
        expand: Option<Vec<String>>,
    ) -> HFResult<SpaceInfo> {
        self.fetch_repo_info(revision, expand).await
    }
}

#[bon]
impl HFRepository<RepoTypeKernel> {
    /// Fetch [`KernelInfo`] for this kernel repository.
    ///
    /// Note: the Hub's `/api/kernels/{repo_id}` endpoint returns a slim shape (no `tags`,
    /// `cardData`, or `siblings`) and silently ignores `expand`. To get the full
    /// model-style metadata for a kernel, build a model handle for the same repo id
    /// (`client.model(owner, name)`) and call `info()` on it.
    #[doc = info_method_doc!()]
    #[builder(
        finish_fn = send,
        builder_type = HFKernelInfoBuilder,
        state_mod(name = hf_kernel_info_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub async fn info(
        &self,
        /// Git revision (branch, tag, or commit SHA). Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// List of properties to expand in the response. Kernel info ignores this parameter.
        expand: Option<Vec<String>>,
    ) -> HFResult<KernelInfo> {
        self.fetch_repo_info(revision, expand).await
    }
}

/// Split "namespace/name" into (Some("namespace"), "name") or (None, "name")
fn split_repo_id(repo_id: &str) -> (Option<&str>, &str) {
    match repo_id.split_once('/') {
        Some((ns, name)) => (Some(ns), name),
        None => (None, repo_id),
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::list_models`]. Returns the collected stream as a
    /// `Vec<ModelInfo>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_models(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        #[builder(into)] pipeline_tag: Option<String>,
        full: Option<bool>,
        card_data: Option<bool>,
        fetch_config: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<Vec<ModelInfo>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_models()
                .maybe_search(search)
                .maybe_author(author)
                .maybe_filter(filter)
                .maybe_sort(sort)
                .maybe_pipeline_tag(pipeline_tag)
                .maybe_full(full)
                .maybe_card_data(card_data)
                .maybe_fetch_config(fetch_config)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_datasets`]. Returns the collected stream as a
    /// `Vec<DatasetInfo>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_datasets(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        full: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<Vec<DatasetInfo>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_datasets()
                .maybe_search(search)
                .maybe_author(author)
                .maybe_filter(filter)
                .maybe_sort(sort)
                .maybe_full(full)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_spaces`]. Returns the collected stream as a
    /// `Vec<SpaceInfo>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_spaces(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        full: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<Vec<SpaceInfo>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_spaces()
                .maybe_search(search)
                .maybe_author(author)
                .maybe_filter(filter)
                .maybe_sort(sort)
                .maybe_full(full)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::create_repository`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn create_repository(
        &self,
        repo_id: &str,
        repo_type: impl RepoType,
        private: Option<bool>,
        #[builder(default)] exist_ok: bool,
        #[builder(into)] space_sdk: Option<String>,
    ) -> HFResult<RepoUrl> {
        self.runtime.block_on(
            self.inner
                .create_repository()
                .repo_id(repo_id)
                .repo_type(repo_type)
                .maybe_private(private)
                .exist_ok(exist_ok)
                .maybe_space_sdk(space_sdk)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::delete_repository`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_repository(
        &self,
        repo_id: &str,
        repo_type: impl RepoType,
        #[builder(default)] missing_ok: bool,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .delete_repository()
                .repo_id(repo_id)
                .repo_type(repo_type)
                .missing_ok(missing_ok)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::move_repository`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn move_repository(
        &self,
        from_id: &str,
        to_id: &str,
        repo_type: impl RepoType,
    ) -> HFResult<RepoUrl> {
        self.runtime.block_on(
            self.inner
                .move_repository()
                .from_id(from_id)
                .to_id(to_id)
                .repo_type(repo_type)
                .send(),
        )
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::exists`].
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn exists(&self) -> HFResult<bool> {
        self.runtime.block_on(self.inner.exists().send())
    }

    /// Blocking counterpart of [`HFRepository::revision_exists`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn revision_exists(&self, revision: &str) -> HFResult<bool> {
        self.runtime
            .block_on(self.inner.revision_exists().revision(revision).send())
    }

    /// Blocking counterpart of [`HFRepository::file_exists`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn file_exists(&self, filename: &str, revision: Option<&str>) -> HFResult<bool> {
        self.runtime.block_on(
            self.inner
                .file_exists()
                .filename(filename)
                .maybe_revision(revision)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::update_settings`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn update_settings(
        &self,
        private: Option<bool>,
        gated: Option<GatedApprovalMode>,
        #[builder(into)] description: Option<String>,
        discussions_disabled: Option<bool>,
        gated_notifications: Option<GatedNotifications>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .update_settings()
                .maybe_private(private)
                .maybe_gated(gated)
                .maybe_description(description)
                .maybe_discussions_disabled(discussions_disabled)
                .maybe_gated_notifications(gated_notifications)
                .send(),
        )
    }
}

/// Sync `info()` builders for each repo kind.
///
/// Each block mirrors the async per-kind `info()` impl in shape and forwards through the
/// runtime. Builder type names are unique to keep the bon-generated state types from
/// colliding across the four impl blocks.

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync<RepoTypeModel> {
    /// Blocking counterpart of [`HFRepository::<RepoTypeModel>::info`].
    #[builder(
        finish_fn = send,
        builder_type = HFModelInfoSyncBuilder,
        state_mod(name = hf_model_info_sync_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub fn info(
        &self,
        #[builder(into)] revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<ModelInfo> {
        self.runtime.block_on(
            self.inner
                .info()
                .maybe_revision(revision)
                .maybe_expand(expand)
                .send(),
        )
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync<RepoTypeDataset> {
    /// Blocking counterpart of [`HFRepository::<RepoTypeDataset>::info`].
    #[builder(
        finish_fn = send,
        builder_type = HFDatasetInfoSyncBuilder,
        state_mod(name = hf_dataset_info_sync_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub fn info(
        &self,
        #[builder(into)] revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<DatasetInfo> {
        self.runtime.block_on(
            self.inner
                .info()
                .maybe_revision(revision)
                .maybe_expand(expand)
                .send(),
        )
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync<RepoTypeSpace> {
    /// Blocking counterpart of [`HFRepository::<RepoTypeSpace>::info`].
    #[builder(
        finish_fn = send,
        builder_type = HFSpaceInfoSyncBuilder,
        state_mod(name = hf_space_info_sync_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub fn info(
        &self,
        #[builder(into)] revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<SpaceInfo> {
        self.runtime.block_on(
            self.inner
                .info()
                .maybe_revision(revision)
                .maybe_expand(expand)
                .send(),
        )
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync<RepoTypeKernel> {
    /// Blocking counterpart of [`HFRepository::<RepoTypeKernel>::info`].
    #[builder(
        finish_fn = send,
        builder_type = HFKernelInfoSyncBuilder,
        state_mod(name = hf_kernel_info_sync_builder, vis = "pub(crate)"),
        derive(Debug, Clone),
    )]
    pub fn info(
        &self,
        #[builder(into)] revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<KernelInfo> {
        self.runtime.block_on(
            self.inner
                .info()
                .maybe_revision(revision)
                .maybe_expand(expand)
                .send(),
        )
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::{
        DatasetInfo, EvalResultEntry, HFRepository, InferenceProviderMapping, KernelInfo,
        ModelInfo, RepoType, RepoTypeDataset, RepoTypeKernel, RepoTypeModel, RepoTypeSpace,
        SafeTensorsInfo, SpaceInfo, TransformersInfo, split_repo_id,
    };
    use crate::client::HFClient;

    #[test]
    fn test_repo_path_and_accessors() {
        let client = HFClient::builder().build().unwrap();
        let repo: HFRepository<RepoTypeModel> =
            HFRepository::new(client, RepoTypeModel, "openai-community", "gpt2");

        assert_eq!(repo.owner(), "openai-community");
        assert_eq!(repo.name(), "gpt2");
        assert_eq!(repo.repo_path(), "openai-community/gpt2");
        assert_eq!(repo.repo_type().singular(), "model");
    }

    #[test]
    fn test_marker_struct_singular_and_plural() {
        assert_eq!(RepoTypeModel.singular(), "model");
        assert_eq!(RepoTypeDataset.singular(), "dataset");
        assert_eq!(RepoTypeSpace.singular(), "space");
        assert_eq!(RepoTypeKernel.singular(), "kernel");
        assert_eq!(RepoTypeModel.plural(), "models");
        assert_eq!(RepoTypeDataset.plural(), "datasets");
        assert_eq!(RepoTypeSpace.plural(), "spaces");
        assert_eq!(RepoTypeKernel.plural(), "kernels");
    }

    #[test]
    fn test_marker_struct_url_prefix() {
        assert_eq!(RepoTypeModel.url_prefix(), "");
        assert_eq!(RepoTypeDataset.url_prefix(), "datasets/");
        assert_eq!(RepoTypeSpace.url_prefix(), "spaces/");
        assert_eq!(RepoTypeKernel.url_prefix(), "kernels/");
    }

    #[test]
    fn test_marker_struct_display() {
        assert_eq!(format!("{}", RepoTypeModel), "model");
        assert_eq!(format!("{}", RepoTypeDataset), "dataset");
        assert_eq!(format!("{}", RepoTypeSpace), "space");
        assert_eq!(format!("{}", RepoTypeKernel), "kernel");
    }

    #[test]
    fn test_split_repo_id() {
        assert_eq!(split_repo_id("user/repo"), (Some("user"), "repo"));
        assert_eq!(split_repo_id("repo"), (None, "repo"));
        assert_eq!(split_repo_id("org/sub/repo"), (Some("org"), "sub/repo"));
    }

    #[tokio::test]
    async fn test_list_models_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let stream = client.list_models().limit(0_usize).send().unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_datasets_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let stream = client.list_datasets().limit(0_usize).send().unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_spaces_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let stream = client.list_spaces().limit(0_usize).send().unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn test_safetensors_info_deserialize() {
        let json = r#"{"parameters":{"F32":124000000,"BF16":1000000},"total":125000000}"#;
        let info: SafeTensorsInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.total, 125_000_000);
        assert_eq!(info.parameters.get("F32"), Some(&124_000_000));
    }

    #[test]
    fn test_transformers_info_deserialize() {
        let json = r#"{"auto_model":"AutoModelForCausalLM","pipeline_tag":"text-generation"}"#;
        let info: TransformersInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.auto_model, "AutoModelForCausalLM");
        assert_eq!(info.pipeline_tag.as_deref(), Some("text-generation"));
        assert!(info.processor.is_none());
    }

    #[test]
    fn test_eval_result_entry_minimal() {
        let json = r#"{"dataset":{"id":"cais/hle","task_id":"default"},"value":20.9}"#;
        let entry: EvalResultEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.dataset.id, "cais/hle");
        assert_eq!(entry.dataset.task_id, "default");
        assert_eq!(entry.value.as_f64(), Some(20.9));
        assert!(entry.source.is_none());
    }

    #[test]
    fn test_eval_result_entry_with_source() {
        let json = r#"{"dataset":{"id":"d/x","task_id":"t","revision":"abc"},"value":0.5,"source":{"url":"u","name":"n","org":"o"},"verifyToken":"vt","notes":"n"}"#;
        let entry: EvalResultEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.dataset.id, "d/x");
        assert_eq!(entry.dataset.revision.as_deref(), Some("abc"));
        let source = entry.source.as_ref().unwrap();
        assert_eq!(source.url.as_deref(), Some("u"));
        assert_eq!(source.org.as_deref(), Some("o"));
        assert_eq!(entry.verify_token.as_deref(), Some("vt"));
    }

    #[test]
    fn test_inference_provider_mapping_list_form() {
        let json = r#"{
            "id":"o/m",
            "inferenceProviderMapping":[
                {"provider":"hf-inference","providerId":"o/m","status":"live","task":"text-generation"}
            ]
        }"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();
        let mappings = info.inference_provider_mapping.unwrap();
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].provider, "hf-inference");
        assert_eq!(mappings[0].provider_id, "o/m");
        assert_eq!(mappings[0].status, "live");
    }

    #[test]
    fn test_inference_provider_mapping_dict_form() {
        let json = r#"{
            "id":"o/m",
            "inferenceProviderMapping":{
                "together":{"providerId":"o/m","status":"live","task":"text-generation"}
            }
        }"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();
        let mappings = info.inference_provider_mapping.unwrap();
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].provider, "together");
        assert_eq!(mappings[0].task, "text-generation");
    }

    #[test]
    fn test_inference_provider_mapping_helper_directly() {
        let info_helper: InferenceProviderMapping = serde_json::from_str(
            r#"{"provider":"x","providerId":"y","status":"live","task":"t","adapterWeightsPath":"w"}"#,
        )
        .unwrap();
        assert_eq!(info_helper.adapter_weights_path.as_deref(), Some("w"));
    }

    #[test]
    fn test_model_info_ignores_unknown_and_legacy_fields() {
        let json = r#"{"id":"o/m","modelId":"o/m","brandNewField":42}"#;
        let info: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.id, "o/m");
    }

    /// Real `/api/kernels/{repo_id}` response shape (slim — no `tags`,
    /// `cardData`, or `siblings`). The `authorData` and `_id` fields are
    /// ignored on deserialize but must not break the parse.
    #[test]
    fn test_kernel_info_deserializes_real_response() {
        let json = r#"{
            "_id":"69d02879cbdc347de53cced2",
            "author":"kernels-community",
            "authorData":{"name":"kernels-community","type":"org"},
            "trustedPublisher":false,
            "downloads":7199,
            "gated":false,
            "id":"kernels-community/flash-attn2",
            "isLikedByUser":false,
            "lastModified":"2026-04-20T20:31:57.000Z",
            "likes":6,
            "private":false,
            "repoType":"kernel",
            "sha":"e16b327d7c5b015cac48944d4058f688e4d0c62f",
            "supportedDriverFamilies":["cuda","xpu","cpu"]
        }"#;
        let info: KernelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.id, "kernels-community/flash-attn2");
        assert_eq!(info.author.as_deref(), Some("kernels-community"));
        assert_eq!(
            info.sha.as_deref(),
            Some("e16b327d7c5b015cac48944d4058f688e4d0c62f")
        );
        assert_eq!(info.downloads, Some(7199));
        assert_eq!(info.likes, Some(6));
        assert_eq!(info.trusted_publisher, Some(false));
        assert_eq!(
            info.supported_driver_families.as_deref(),
            Some(&["cuda".into(), "xpu".into(), "cpu".into()][..])
        );
        // `gated: false` is kept as JSON (consistent with ModelInfo/SpaceInfo).
        assert_eq!(info.gated.as_ref().and_then(|v| v.as_bool()), Some(false));
    }

    /// `supportedDriverFamilies` is absent on some kernels — must remain optional.
    #[test]
    fn test_kernel_info_missing_supported_driver_families() {
        let json =
            r#"{"id":"o/k","sha":"abc","downloads":0,"likes":0,"private":false,"gated":false}"#;
        let info: KernelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.id, "o/k");
        assert!(info.supported_driver_families.is_none());
    }

    #[test]
    fn test_dataset_info_new_fields() {
        let json = r#"{
            "id":"u/d",
            "citation":"Doe et al. 2024",
            "paperswithcode_id":"pwc-id",
            "resourceGroup":{"id":"rg-1","name":"Team A"}
        }"#;
        let info: DatasetInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.citation.as_deref(), Some("Doe et al. 2024"));
        assert_eq!(info.paperswithcode_id.as_deref(), Some("pwc-id"));
        assert!(info.resource_group.is_some());
    }

    #[test]
    fn test_space_info_new_fields() {
        let json = r#"{
            "id":"u/s",
            "models":["org/model-a","org/model-b"],
            "datasets":["org/dataset"],
            "resourceGroup":{"id":"rg-2"}
        }"#;
        let info: SpaceInfo = serde_json::from_str(json).unwrap();
        assert_eq!(
            info.models.as_deref(),
            Some(&["org/model-a".to_string(), "org/model-b".to_string()][..])
        );
        assert_eq!(
            info.datasets.as_deref(),
            Some(&["org/dataset".to_string()][..])
        );
        assert!(info.resource_group.is_some());
    }
}
