use anyhow::{Context, Result};
pub use mesh_llm_types::models::capabilities::{CapabilityLevel, ModelCapabilities};
use mesh_llm_types::models::capabilities::{merge_config_signals, merge_name_signals};
use model_artifact::{ModelFormat, ResolvedModelArtifact, resolve_model_artifact_ref};
use model_hf::HfModelRepository;
use model_ref::{format_model_ref, normalize_gguf_distribution_id, quant_selector_from_gguf_file};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstalledModel {
    pub model_ref: String,
    pub path: PathBuf,
    pub size_bytes: Option<u64>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSearchQuery {
    pub query: String,
    pub limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDetails {
    pub id: String,
    pub name: String,
    pub source: ModelSource,
    pub kind: ModelKind,
    pub model_ref: String,
    pub download_ref: String,
    pub path: Option<PathBuf>,
    pub size_bytes: Option<u64>,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub installed: bool,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelSource {
    Catalog,
    HuggingFace,
    Local,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    Gguf,
    Safetensors,
    LayerPackage,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadedModel {
    pub model_ref: String,
    pub paths: Vec<PathBuf>,
    pub primary_path: Option<PathBuf>,
    pub details: Option<ModelDetails>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeleteModelOptions {
    pub force: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeleteModelResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CleanupPolicy {
    pub remove_all: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CleanupResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
    pub skipped_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrunePolicy {
    pub remove_all: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PruneResult {
    pub deleted_paths: Vec<PathBuf>,
    pub reclaimed_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct CatalogAsset {
    file: String,
    url: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CatalogModel {
    name: String,
    file: String,
    url: String,
    size: String,
    description: String,
    draft: Option<String>,
    #[serde(default)]
    extra_files: Vec<CatalogAsset>,
    mmproj: Option<CatalogAsset>,
}

pub fn default_huggingface_cache_dir() -> PathBuf {
    if let Some(path) = env_path("HF_HUB_CACHE").or_else(|| env_path("HUGGINGFACE_HUB_CACHE")) {
        return path;
    }
    if let Some(path) = env_path("HF_HOME") {
        return path.join("hub");
    }
    if let Some(path) = env_path("XDG_CACHE_HOME") {
        return path.join("huggingface").join("hub");
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

pub fn scan_installed_models(cache_dir: impl AsRef<Path>) -> Vec<InstalledModel> {
    let cache_dir = cache_dir.as_ref();
    let mut models = Vec::new();
    if cache_dir.exists() {
        scan_dir(cache_dir, cache_dir, &mut models);
    }
    models.sort_by(|left, right| {
        left.model_ref
            .cmp(&right.model_ref)
            .then_with(|| left.path.cmp(&right.path))
    });
    models.dedup_by(|left, right| left.model_ref == right.model_ref && left.path == right.path);
    models
}

fn env_path(name: &str) -> Option<PathBuf> {
    let value = std::env::var_os(name)?;
    let path = PathBuf::from(value);
    (!path.as_os_str().is_empty()).then_some(path)
}

fn scan_dir(root: &Path, dir: &Path, models: &mut Vec<InstalledModel>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            scan_dir(root, &path, models);
        } else if file_type.is_file() {
            maybe_push_model(root, path, models);
        }
    }
}

fn maybe_push_model(root: &Path, path: PathBuf, models: &mut Vec<InstalledModel>) {
    let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
        return;
    };
    if file_name.contains("mmproj") || !is_model_artifact(file_name) {
        return;
    }
    let Some(model_ref) = model_ref_for_path(root, &path) else {
        return;
    };
    let size_bytes = std::fs::metadata(&path).map(|metadata| metadata.len()).ok();
    let capabilities = infer_local_capabilities(&model_ref, &path);
    models.push(InstalledModel {
        model_ref,
        path,
        size_bytes,
        capabilities,
    });
}

pub fn recommended_models() -> Vec<ModelSummary> {
    catalog_models()
        .into_iter()
        .map(|model| ModelSummary {
            id: model.name.clone(),
            name: model.name.clone(),
            size_label: Some(model.size.clone()),
            description: Some(model.description.clone()),
            capabilities: infer_catalog_capabilities(&model),
        })
        .collect()
}

pub fn search_models(query: ModelSearchQuery, cache_dir: impl AsRef<Path>) -> Vec<ModelSummary> {
    let needle = query.query.trim().to_ascii_lowercase();
    let limit = if query.limit == 0 { 20 } else { query.limit };
    let mut results = recommended_models()
        .into_iter()
        .chain(
            scan_installed_models(cache_dir)
                .into_iter()
                .map(ModelSummary::from),
        )
        .filter(|model| needle.is_empty() || model_matches(model, &needle))
        .collect::<Vec<_>>();

    results.sort_by(|left, right| {
        search_rank(left, &needle)
            .cmp(&search_rank(right, &needle))
            .then_with(|| left.name.cmp(&right.name))
    });
    results.dedup_by(|left, right| left.id == right.id);
    results.truncate(limit);
    results
}

pub async fn show_model(
    model_ref: impl AsRef<str>,
    cache_dir: impl AsRef<Path>,
) -> Result<ModelDetails> {
    let input = model_ref.as_ref().trim();
    if let Some(installed) = scan_installed_models(cache_dir.as_ref())
        .into_iter()
        .find(|model| model.model_ref == input)
    {
        return Ok(ModelDetails {
            id: installed.model_ref.clone(),
            name: installed.model_ref.clone(),
            source: ModelSource::Local,
            kind: kind_for_path(&installed.path),
            model_ref: installed.model_ref.clone(),
            download_ref: installed.model_ref,
            path: Some(installed.path),
            size_bytes: installed.size_bytes,
            size_label: None,
            description: None,
            draft: None,
            installed: true,
            capabilities: installed.capabilities,
        });
    }

    if let Some(model) = find_catalog_model(input) {
        let (download_ref, kind) = catalog_download_ref_and_kind(&model);
        let capabilities = infer_catalog_capabilities(&model);
        return Ok(ModelDetails {
            id: model.name.clone(),
            name: model.name.clone(),
            source: ModelSource::Catalog,
            kind,
            model_ref: model.name,
            download_ref,
            path: None,
            size_bytes: None,
            size_label: Some(model.size),
            description: Some(model.description),
            draft: model.draft,
            installed: false,
            capabilities,
        });
    }

    let repo = HfModelRepository::builder()
        .cache_dir(cache_dir.as_ref())
        .build()
        .context("build Hugging Face model repository")?;
    let artifact = resolve_model_artifact_ref(input, &repo).await?;
    Ok(details_for_artifact(&artifact, None, false))
}

pub async fn download_model(
    model_ref: impl AsRef<str>,
    cache_dir: impl AsRef<Path>,
) -> Result<DownloadedModel> {
    let input = model_ref.as_ref().trim();
    let details = show_model(input, cache_dir.as_ref()).await.ok();
    if let Some(details) = details.as_ref().filter(|details| details.installed) {
        let paths = details.path.iter().cloned().collect::<Vec<_>>();
        return Ok(DownloadedModel {
            model_ref: details.model_ref.clone(),
            primary_path: paths.first().cloned(),
            paths,
            details: Some(details.clone()),
        });
    }
    let download_ref = details
        .as_ref()
        .map(|details| details.download_ref.as_str())
        .unwrap_or(input);
    let repo = HfModelRepository::builder()
        .cache_dir(cache_dir.as_ref())
        .build()
        .context("build Hugging Face model repository")?;
    let artifact = resolve_model_artifact_ref(download_ref, &repo).await?;
    let paths = repo.download_artifact_files(&artifact).await?;
    let primary_path = paths.first().cloned();
    Ok(DownloadedModel {
        model_ref: artifact.model_id.clone(),
        paths,
        primary_path,
        details: Some(details_for_artifact(&artifact, details, true)),
    })
}

pub async fn delete_model(
    model_ref: impl AsRef<str>,
    cache_dir: impl AsRef<Path>,
    _options: DeleteModelOptions,
) -> Result<DeleteModelResult> {
    let cache_dir = cache_dir.as_ref();
    let input = model_ref.as_ref();
    let installed = scan_installed_models(cache_dir);
    let matches = installed
        .into_iter()
        .filter(|model| model.model_ref == input)
        .collect::<Vec<_>>();
    if matches.is_empty() {
        anyhow::bail!("installed model not found: {input}");
    }
    delete_paths(
        cache_dir,
        matches.into_iter().map(|model| model.path).collect(),
    )
}

pub fn cleanup_models(cache_dir: impl AsRef<Path>, policy: CleanupPolicy) -> Result<CleanupResult> {
    let cache_dir = cache_dir.as_ref();
    let installed = scan_installed_models(cache_dir);
    if !policy.remove_all {
        return Ok(CleanupResult {
            skipped_paths: installed.into_iter().map(|model| model.path).collect(),
            ..CleanupResult::default()
        });
    }

    let result = delete_paths(
        cache_dir,
        installed.into_iter().map(|model| model.path).collect(),
    )?;
    Ok(CleanupResult {
        deleted_paths: result.deleted_paths,
        reclaimed_bytes: result.reclaimed_bytes,
        skipped_paths: Vec::new(),
    })
}

pub fn prune_derived_cache(
    runtime_dir: impl AsRef<Path>,
    policy: PrunePolicy,
) -> Result<PruneResult> {
    let runtime_dir = runtime_dir.as_ref();
    if !policy.remove_all {
        return Ok(PruneResult::default());
    }

    let candidates = [
        runtime_dir.join("materialized"),
        runtime_dir.join("skippy-runtime").join("materialized"),
    ];
    let mut paths = Vec::new();
    for candidate in candidates {
        collect_files(&candidate, &mut paths);
    }
    let result = delete_paths(runtime_dir, paths)?;
    Ok(PruneResult {
        deleted_paths: result.deleted_paths,
        reclaimed_bytes: result.reclaimed_bytes,
    })
}

fn delete_paths(root: &Path, paths: Vec<PathBuf>) -> Result<DeleteModelResult> {
    let root = normalize_existing_or_parent(root)?;
    let mut reclaimed_bytes = 0;
    let mut deleted_paths = Vec::new();
    let mut unique_paths = BTreeSet::new();

    for path in paths {
        let path = normalize_existing_or_parent(&path)?;
        if !path.starts_with(&root) {
            anyhow::bail!(
                "refusing to delete path outside configured root: {}",
                path.display()
            );
        }
        unique_paths.insert(path);
    }

    for path in unique_paths {
        if !path.is_file() {
            continue;
        }
        if let Ok(metadata) = std::fs::metadata(&path) {
            reclaimed_bytes += metadata.len();
        }
        std::fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        prune_empty_ancestors(&path, &root);
        deleted_paths.push(path);
    }

    Ok(DeleteModelResult {
        deleted_paths,
        reclaimed_bytes,
    })
}

fn normalize_existing_or_parent(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
    }
    let Some(parent) = path.parent() else {
        return Ok(path.to_path_buf());
    };
    let parent = parent
        .canonicalize()
        .unwrap_or_else(|_| parent.to_path_buf());
    Ok(parent.join(
        path.file_name()
            .map(|value| value.to_owned())
            .unwrap_or_default(),
    ))
}

fn collect_files(dir: &Path, paths: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_files(&path, paths);
        } else if file_type.is_file() {
            paths.push(path);
        }
    }
}

fn prune_empty_ancestors(path: &Path, stop_at: &Path) {
    let mut current = path.parent();
    while let Some(dir) = current {
        if dir == stop_at || !dir.starts_with(stop_at) {
            break;
        }
        match std::fs::remove_dir(dir) {
            Ok(()) => current = dir.parent(),
            Err(_) => break,
        }
    }
}

fn details_for_artifact(
    artifact: &ResolvedModelArtifact,
    base: Option<ModelDetails>,
    installed: bool,
) -> ModelDetails {
    let capabilities = base
        .as_ref()
        .map(|details| details.capabilities)
        .unwrap_or_else(|| {
            infer_remote_capabilities(&artifact.source_repo, &artifact.primary_file)
        });
    ModelDetails {
        id: artifact.model_id.clone(),
        name: artifact.model_id.clone(),
        source: base
            .as_ref()
            .map(|details| details.source.clone())
            .unwrap_or(ModelSource::HuggingFace),
        kind: kind_for_artifact_format(artifact.format),
        model_ref: artifact.model_id.clone(),
        download_ref: artifact.model_id.clone(),
        path: None,
        size_bytes: artifact
            .files
            .iter()
            .filter_map(|file| file.size_bytes)
            .reduce(|left, right| left.saturating_add(right)),
        size_label: base.as_ref().and_then(|details| details.size_label.clone()),
        description: base
            .as_ref()
            .and_then(|details| details.description.clone()),
        draft: base.as_ref().and_then(|details| details.draft.clone()),
        installed,
        capabilities,
    }
}

fn is_model_artifact(file_name: &str) -> bool {
    file_name.ends_with(".gguf")
        || file_name == "model.safetensors"
        || file_name == "model.safetensors.index.json"
        || is_split_safetensors_shard(file_name)
}

fn is_split_safetensors_shard(file_name: &str) -> bool {
    let Some(rest) = file_name.strip_prefix("model-") else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((part, total)) = rest.split_once("-of-") else {
        return false;
    };
    part.len() == 5
        && total.len() == 5
        && part.bytes().all(|byte| byte.is_ascii_digit())
        && total.bytes().all(|byte| byte.is_ascii_digit())
}

fn model_ref_for_path(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let mut components = relative.components();
    let repo_folder = components.next()?.as_os_str().to_str()?;
    let repo_id = repo_folder
        .strip_prefix("models--")
        .map(|value| value.replace("--", "/"))?;
    if components.next()?.as_os_str() != "snapshots" {
        return None;
    }
    let _revision = components.next()?.as_os_str().to_str()?;
    let relative_file = components
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()?
        .join("/");

    if repo_id.ends_with("-layers") && is_layer_package_file(&relative_file) {
        return Some(format_model_ref(&repo_id, None, None));
    }

    let selector = quant_selector_from_gguf_file(&relative_file)
        .or_else(|| normalize_gguf_distribution_id(&relative_file));
    Some(format_model_ref(&repo_id, None, selector.as_deref()))
}

fn is_layer_package_file(relative_file: &str) -> bool {
    relative_file.ends_with(".gguf")
        && (relative_file.starts_with("shared/") || relative_file.starts_with("layers/"))
}

fn catalog_models() -> Vec<CatalogModel> {
    serde_json::from_str(include_str!("../../mesh-client/src/models/catalog.json"))
        .expect("parse bundled mesh client model catalog")
}

fn find_catalog_model(query: &str) -> Option<CatalogModel> {
    let query_lower = query.to_ascii_lowercase();
    catalog_models()
        .into_iter()
        .find(|model| model.name.eq_ignore_ascii_case(query))
        .or_else(|| {
            catalog_models()
                .into_iter()
                .find(|model| model.name.to_ascii_lowercase().contains(&query_lower))
        })
}

fn catalog_download_ref_and_kind(model: &CatalogModel) -> (String, ModelKind) {
    if let Some((repo, _revision, file)) = parse_hf_resolve_url_parts(&model.url) {
        let selector =
            quant_selector_from_gguf_file(file).or_else(|| normalize_gguf_distribution_id(file));
        return (
            format_model_ref(repo, None, selector.as_deref()),
            kind_for_file(file),
        );
    }
    (model.name.clone(), kind_for_file(&model.file))
}

fn parse_hf_resolve_url_parts(url: &str) -> Option<(&str, Option<&str>, &str)> {
    let tail = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = tail.split_once("/resolve/")?;
    let (revision, file) = rest.split_once('/')?;
    Some((repo, Some(revision), file))
}

fn kind_for_path(path: &Path) -> ModelKind {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(kind_for_file)
        .unwrap_or(ModelKind::Unknown)
}

fn kind_for_file(file: &str) -> ModelKind {
    if file.ends_with(".gguf") {
        ModelKind::Gguf
    } else if file.ends_with(".safetensors") || file == "model.safetensors.index.json" {
        ModelKind::Safetensors
    } else {
        ModelKind::Unknown
    }
}

fn kind_for_artifact_format(format: ModelFormat) -> ModelKind {
    match format {
        ModelFormat::Gguf => ModelKind::Gguf,
        ModelFormat::Safetensors => ModelKind::Safetensors,
    }
}

fn infer_catalog_capabilities(model: &CatalogModel) -> ModelCapabilities {
    let mut caps = ModelCapabilities::default();
    if let Some(mmproj) = &model.mmproj {
        caps.vision = CapabilityLevel::Supported;
        caps.multimodal = true;
        caps = merge_name_signals(caps, &[mmproj.file.as_str(), mmproj.url.as_str()]);
    }
    let extra_file_signals = model
        .extra_files
        .iter()
        .flat_map(|asset| [asset.file.as_str(), asset.url.as_str()])
        .collect::<Vec<_>>();
    caps = merge_name_signals(
        caps,
        &[
            model.name.as_str(),
            model.file.as_str(),
            model.description.as_str(),
        ],
    );
    caps = merge_name_signals(caps, &extra_file_signals);
    caps.normalize()
}

fn infer_remote_capabilities(repo: &str, file: &str) -> ModelCapabilities {
    merge_name_signals(ModelCapabilities::default(), &[repo, file]).normalize()
}

fn infer_local_capabilities(model_ref: &str, path: &Path) -> ModelCapabilities {
    let mut caps = merge_name_signals(
        ModelCapabilities::default(),
        &[
            model_ref,
            path.file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default(),
        ],
    );
    for config in read_local_metadata_jsons(path) {
        caps = merge_config_signals(caps, &config);
    }
    caps.normalize()
}

fn read_local_metadata_jsons(path: &Path) -> Vec<Value> {
    let mut values = Vec::new();
    for dir in path.ancestors().skip(1).take(6) {
        for name in ["config.json", "tokenizer_config.json", "chat_template.json"] {
            let candidate = dir.join(name);
            let Ok(text) = std::fs::read_to_string(candidate) else {
                continue;
            };
            if let Ok(value) = serde_json::from_str(&text) {
                values.push(value);
            }
        }
    }
    values
}

impl From<InstalledModel> for ModelSummary {
    fn from(value: InstalledModel) -> Self {
        Self {
            id: value.model_ref.clone(),
            name: value.model_ref,
            size_label: value.size_bytes.map(format_size_label),
            description: value
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| format!("Installed model artifact: {name}")),
            capabilities: value.capabilities,
        }
    }
}

fn model_matches(model: &ModelSummary, needle: &str) -> bool {
    let fields = [
        model.id.as_str(),
        model.name.as_str(),
        model.size_label.as_deref().unwrap_or_default(),
        model.description.as_deref().unwrap_or_default(),
    ];
    fields
        .iter()
        .any(|field| field.to_ascii_lowercase().contains(needle))
}

fn search_rank(model: &ModelSummary, needle: &str) -> u8 {
    if needle.is_empty() {
        return 0;
    }
    let id = model.id.to_ascii_lowercase();
    let name = model.name.to_ascii_lowercase();
    if id == needle || name == needle {
        0
    } else if id.starts_with(needle) || name.starts_with(needle) {
        1
    } else if id.contains(needle) || name.contains(needle) {
        2
    } else {
        3
    }
}

fn format_size_label(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} bytes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_installed_models_finds_hf_snapshot_gguf() {
        let temp = unique_temp_dir("mesh-llm-node-installed-gguf");
        let model = temp
            .join("models--org--repo-GGUF")
            .join("snapshots")
            .join("abc")
            .join("Repo-Q4_K_M.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"gguf").unwrap();

        let installed = scan_installed_models(&temp);
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].model_ref, "org/repo-GGUF:Q4_K_M");
        assert_eq!(installed[0].path, model);
        assert_eq!(installed[0].size_bytes, Some(4));
        assert_eq!(installed[0].capabilities.reasoning, CapabilityLevel::None);

        let _ = std::fs::remove_dir_all(temp);
    }

    #[test]
    fn scan_installed_models_collapses_layer_package_refs() {
        let temp = unique_temp_dir("mesh-llm-node-installed-layers");
        let shared = temp
            .join("models--meshllm--Qwen-layers")
            .join("snapshots")
            .join("abc")
            .join("shared")
            .join("tok.gguf");
        let layer = temp
            .join("models--meshllm--Qwen-layers")
            .join("snapshots")
            .join("abc")
            .join("layers")
            .join("000.gguf");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::create_dir_all(layer.parent().unwrap()).unwrap();
        std::fs::write(&shared, b"shared").unwrap();
        std::fs::write(&layer, b"layer").unwrap();

        let installed = scan_installed_models(&temp);
        assert!(
            installed
                .iter()
                .all(|model| model.model_ref == "meshllm/Qwen-layers")
        );
        assert_eq!(installed.len(), 2);

        let _ = std::fs::remove_dir_all(temp);
    }

    #[test]
    fn recommended_models_include_capabilities() {
        let recommended = recommended_models();
        assert!(!recommended.is_empty());
        assert!(
            recommended
                .iter()
                .any(|model| model.id == "Qwen3-4B-Q4_K_M")
        );
    }

    #[test]
    fn search_models_finds_catalog_and_installed_models_with_capabilities() {
        let temp = unique_temp_dir("mesh-llm-node-search");
        let model = temp
            .join("models--org--Qwen2-VL-GGUF")
            .join("snapshots")
            .join("abc")
            .join("Qwen2-VL-Q4_K_M.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"gguf").unwrap();

        let catalog = search_models(
            ModelSearchQuery {
                query: "qwen3".to_string(),
                limit: 5,
            },
            &temp,
        );
        assert!(catalog.iter().any(|model| {
            model.id.to_ascii_lowercase().contains("qwen3")
                && model.capabilities.reasoning == CapabilityLevel::Supported
        }));

        let installed = search_models(
            ModelSearchQuery {
                query: "vl".to_string(),
                limit: 5,
            },
            &temp,
        );
        assert!(installed.iter().any(|model| {
            model.id == "org/Qwen2-VL-GGUF:Q4_K_M"
                && model.capabilities.vision == CapabilityLevel::Supported
        }));

        let _ = std::fs::remove_dir_all(temp);
    }

    #[tokio::test]
    async fn show_model_returns_installed_details_with_capabilities() {
        let temp = unique_temp_dir("mesh-llm-node-show-installed");
        let model = temp
            .join("models--org--Qwen2-VL-GGUF")
            .join("snapshots")
            .join("abc")
            .join("Qwen2-VL-Q4_K_M.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"gguf").unwrap();

        let details = show_model("org/Qwen2-VL-GGUF:Q4_K_M", &temp)
            .await
            .expect("show installed model");
        assert_eq!(details.source, ModelSource::Local);
        assert_eq!(details.kind, ModelKind::Gguf);
        assert!(details.installed);
        assert_eq!(details.path.as_deref(), Some(model.as_path()));
        assert!(details.capabilities.multimodal);

        let _ = std::fs::remove_dir_all(temp);
    }

    #[tokio::test]
    async fn download_model_returns_existing_installed_model_without_network() {
        let temp = unique_temp_dir("mesh-llm-node-download-installed");
        let model = temp
            .join("models--org--repo-GGUF")
            .join("snapshots")
            .join("abc")
            .join("Repo-Q4_K_M.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"gguf").unwrap();

        let downloaded = download_model("org/repo-GGUF:Q4_K_M", &temp)
            .await
            .expect("download installed model");
        assert_eq!(downloaded.model_ref, "org/repo-GGUF:Q4_K_M");
        assert_eq!(downloaded.primary_path.as_deref(), Some(model.as_path()));
        assert_eq!(downloaded.paths, vec![model]);
        assert!(
            downloaded
                .details
                .as_ref()
                .is_some_and(|details| details.installed)
        );

        let _ = std::fs::remove_dir_all(temp);
    }

    #[tokio::test]
    async fn delete_model_removes_matching_installed_artifact() {
        let temp = unique_temp_dir("mesh-llm-node-delete-model");
        let model = temp
            .join("models--org--repo-GGUF")
            .join("snapshots")
            .join("abc")
            .join("Repo-Q4_K_M.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"gguf").unwrap();
        let expected_model = model.canonicalize().unwrap();

        let result = delete_model("org/repo-GGUF:Q4_K_M", &temp, DeleteModelOptions::default())
            .await
            .expect("delete model");
        assert_eq!(result.deleted_paths, vec![expected_model]);
        assert_eq!(result.reclaimed_bytes, 4);
        assert!(!model.exists());

        let _ = std::fs::remove_dir_all(temp);
    }

    #[test]
    fn cleanup_models_requires_opt_in_and_can_remove_all() {
        let temp = unique_temp_dir("mesh-llm-node-cleanup-models");
        let model = temp
            .join("models--org--repo-GGUF")
            .join("snapshots")
            .join("abc")
            .join("Repo-Q4_K_M.gguf");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"gguf").unwrap();
        let expected_model = model.canonicalize().unwrap();

        let skipped = cleanup_models(&temp, CleanupPolicy::default()).expect("cleanup preview");
        assert!(skipped.deleted_paths.is_empty());
        assert_eq!(skipped.skipped_paths, vec![model.clone()]);
        assert!(model.exists());

        let deleted =
            cleanup_models(&temp, CleanupPolicy { remove_all: true }).expect("cleanup remove all");
        assert_eq!(deleted.deleted_paths, vec![expected_model]);
        assert_eq!(deleted.reclaimed_bytes, 4);
        assert!(!model.exists());

        let _ = std::fs::remove_dir_all(temp);
    }

    #[test]
    fn prune_derived_cache_removes_materialized_files_when_enabled() {
        let temp = unique_temp_dir("mesh-llm-node-prune-derived");
        let materialized = temp.join("materialized").join("stage.gguf");
        std::fs::create_dir_all(materialized.parent().unwrap()).unwrap();
        std::fs::write(&materialized, b"stage").unwrap();
        let expected_materialized = materialized.canonicalize().unwrap();

        let skipped = prune_derived_cache(&temp, PrunePolicy::default()).expect("prune preview");
        assert!(skipped.deleted_paths.is_empty());
        assert!(materialized.exists());

        let pruned =
            prune_derived_cache(&temp, PrunePolicy { remove_all: true }).expect("prune remove all");
        assert_eq!(pruned.deleted_paths, vec![expected_materialized]);
        assert_eq!(pruned.reclaimed_bytes, 5);
        assert!(!materialized.exists());

        let _ = std::fs::remove_dir_all(temp);
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
