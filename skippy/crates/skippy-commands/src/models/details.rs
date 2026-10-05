//! Exact model details and fit-aware GGUF variant resolution.
use super::{capabilities, storage::format_size_bytes};
use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde::Deserialize;
use skippy_model_artifact::capabilities::ModelCapabilities;
use skippy_model_artifact::{ModelArtifactFile, select_primary_artifact_file};
use skippy_model_hf::{
    remote_catalog,
    store::local::{HuggingFaceModelIdentity, find_model_path},
};
#[cfg(test)]
use std::sync::{Arc, LazyLock, Mutex};
use std::{collections::HashSet, path::Path};
#[derive(Clone, Debug)]
pub(super) struct ModelDetails {
    pub display_name: String,
    pub exact_ref: String,
    pub source: &'static str,
    pub kind: &'static str,
    pub download_url: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub capabilities: ModelCapabilities,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ShowVariantsProgress {
    Inspecting { completed: usize, total: usize },
}
pub(super) fn find_remote_catalog_model_exact(
    query: &str,
) -> Option<remote_catalog::RemoteCatalogModel> {
    remote_catalog::find_model_exact(query)
}
pub(super) fn find_loaded_remote_catalog_model_exact(
    query: &str,
) -> Option<remote_catalog::RemoteCatalogModel> {
    remote_catalog::find_loaded_model_exact(query)
}
pub(super) fn remote_catalog_model_ref(model: &remote_catalog::RemoteCatalogModel) -> String {
    model.exact_ref()
}
pub(super) fn remote_catalog_model_draft_ref(
    model: &remote_catalog::RemoteCatalogModel,
) -> Option<String> {
    model.draft.as_deref().map(|draft| {
        remote_catalog::find_model_exact(draft)
            .map(|model| model.exact_ref())
            .unwrap_or_else(|| draft.to_string())
    })
}

#[derive(Clone, Debug)]
enum ExactModelRef {
    Catalog(Box<remote_catalog::RemoteCatalogModel>),
    HuggingFace {
        repo: String,
        revision: Option<String>,
        file: String,
    },
}
pub(super) fn merge_capabilities(
    left: ModelCapabilities,
    right: ModelCapabilities,
) -> ModelCapabilities {
    ModelCapabilities {
        multimodal: left.multimodal || right.multimodal,
        vision: left.vision.max(right.vision),
        audio: left.audio.max(right.audio),
        reasoning: left.reasoning.max(right.reasoning),
        tool_use: left.tool_use.max(right.tool_use),
        moe: false,
        system_one: left.system_one.max(right.system_one),
    }
}
pub async fn show_exact_model(input: &str) -> Result<ModelDetails> {
    let input = canonicalize_model_ref_input(input).await?;
    match parse_exact_model_ref(&input)? {
        ExactModelRef::Catalog(model) => {
            let exact_ref = remote_catalog_model_ref(&model);
            Ok(ModelDetails {
                display_name: exact_ref.clone(),
                exact_ref,
                source: "catalog",
                kind: remote_catalog_model_kind(&model),
                download_url: model.resolve_url(),
                size_label: model.size.clone(),
                description: model.description.clone(),
                draft: remote_catalog_model_draft_ref(&model),
                capabilities: capabilities::infer_remote_catalog_capabilities(&model),
            })
        }
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            let file = resolve_huggingface_file(&repo, revision.as_deref(), &file).await?;
            let exact_ref = format_huggingface_display_ref(&repo, revision.as_deref(), &file);
            let catalog =
                matching_remote_catalog_model_for_huggingface(&repo, revision.as_deref(), &file);
            let download_url = huggingface_resolve_url(&repo, revision.as_deref(), &file);
            let size_label = match catalog {
                Some(ref model) => model.size.clone(),
                None => remote_size_label(&download_url).await,
            };
            let capabilities = match catalog {
                Some(ref model) => {
                    let base = capabilities::infer_remote_catalog_capabilities(model);
                    let remote = capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &file,
                        None,
                    )
                    .await;
                    merge_capabilities(base, remote)
                }
                None => {
                    capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &file,
                        None,
                    )
                    .await
                }
            };
            Ok(ModelDetails {
                display_name: Path::new(&file)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(&file)
                    .to_string(),
                exact_ref,
                source: "huggingface",
                kind: artifact_kind_for_file(&file),
                download_url,
                size_label,
                description: catalog.as_ref().and_then(|model| model.description.clone()),
                draft: catalog.as_ref().and_then(remote_catalog_model_draft_ref),
                capabilities,
            })
        }
    }
}

pub async fn show_model_variants_with_progress<F>(
    input: &str,
    mut progress: F,
) -> Result<Option<Vec<ModelDetails>>>
where
    F: FnMut(ShowVariantsProgress),
{
    let input = canonicalize_model_ref_input(input).await?;
    let parsed = parse_huggingface_repo_ref(&input).or_else(|| parse_huggingface_repo_url(&input));
    let Some((repo, revision, _selector)) = parsed else {
        return Ok(None);
    };
    let revision_ref = revision.as_deref().unwrap_or("main");
    let sibling_entries = fetch_repo_sibling_entries(&repo, revision_ref).await?;
    let available_bytes = crate::models::output::fit_budget_bytes();
    let variants = collect_show_gguf_variants_from_siblings(&sibling_entries, available_bytes);
    if variants.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let quant_variants: Vec<_> = variants
        .into_iter()
        .filter(|(file, _)| quant_selector_from_gguf_file(file).is_some())
        .collect();
    let total = quant_variants.len();
    progress(ShowVariantsProgress::Inspecting {
        completed: 0,
        total,
    });

    let mut seen_refs = HashSet::new();
    let mut out = Vec::new();
    for (idx, (file, size_bytes)) in quant_variants.into_iter().enumerate() {
        let exact_ref = format_huggingface_display_ref(&repo, revision.as_deref(), &file);
        if !seen_refs.insert(exact_ref.clone()) {
            progress(ShowVariantsProgress::Inspecting {
                completed: idx + 1,
                total,
            });
            continue;
        }
        let size_label = match size_bytes {
            Some(bytes) => Some(format_size_bytes(bytes)),
            None => {
                remote_size_label(&huggingface_resolve_url(&repo, revision.as_deref(), &file)).await
            }
        };
        out.push(ModelDetails {
            display_name: Path::new(&file)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or(&file)
                .to_string(),
            exact_ref,
            source: "huggingface",
            kind: artifact_kind_for_file(&file),
            download_url: huggingface_resolve_url(&repo, revision.as_deref(), &file),
            size_label,
            description: None,
            draft: None,
            capabilities: ModelCapabilities::default(),
        });
        progress(ShowVariantsProgress::Inspecting {
            completed: idx + 1,
            total,
        });
    }

    Ok(Some(out))
}

pub(super) fn quant_selector_from_gguf_file(file: &str) -> Option<String> {
    skippy_model_ref::quant_selector_from_gguf_file(file)
}

fn format_huggingface_display_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    skippy_model_resolver::format_huggingface_display_ref(repo, revision, file)
}

fn artifact_kind_for_file(file: &str) -> &'static str {
    if file.ends_with(".safetensors") || file.ends_with(".safetensors.index.json") {
        "🍎 MLX"
    } else {
        "🦙 GGUF"
    }
}

fn remote_catalog_model_kind(model: &remote_catalog::RemoteCatalogModel) -> &'static str {
    artifact_kind_for_file(model.source_file())
}

pub fn installed_model_capabilities(model_name: &str) -> ModelCapabilities {
    let path = find_model_path(model_name);
    capabilities::infer_local_model_capabilities(model_name, &path)
}

pub fn loaded_remote_catalog_display_name(model_name: &str) -> Option<String> {
    find_loaded_remote_catalog_model_exact(model_name).map(|model| model.name.clone())
}

pub fn installed_model_display_name(model_name: &str) -> String {
    loaded_remote_catalog_display_name(model_name).unwrap_or_else(|| model_name.to_string())
}

pub fn installed_model_huggingface_ref(identity: &HuggingFaceModelIdentity) -> String {
    format_huggingface_display_ref(&identity.repo_id, None, &identity.file)
}

pub(super) fn matching_remote_catalog_model_for_huggingface(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<remote_catalog::RemoteCatalogModel> {
    remote_catalog::matching_model_for_huggingface(repo, revision, file)
}

pub(super) fn parse_huggingface_ref(input: &str) -> Option<(String, Option<String>, String)> {
    skippy_model_resolver::parse_huggingface_file_ref(input)
}

fn parse_huggingface_repo_ref(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    skippy_model_resolver::parse_huggingface_repo_ref(input)
}

fn parse_huggingface_repo_url(input: &str) -> Option<(String, Option<String>, Option<String>)> {
    skippy_model_resolver::parse_huggingface_repo_url(input)
}

fn parse_exact_model_ref(input: &str) -> Result<ExactModelRef> {
    if let Some((repo, revision, file)) = parse_huggingface_ref(input) {
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        });
    }
    if let Some((repo, revision, selector)) = parse_huggingface_repo_ref(input) {
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file: selector.unwrap_or_default(),
        });
    }
    if let Some((repo, revision, selector)) = parse_huggingface_repo_url(input) {
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file: selector.unwrap_or_default(),
        });
    }
    if let Some(model) = find_remote_catalog_model_exact(input) {
        return Ok(ExactModelRef::Catalog(Box::new(model)));
    }
    bail!(
        "Expected an exact model ref. Use a catalog id or a Hugging Face ref like org/repo, org/repo@rev:QUANT, org/repo/file.gguf, org/repo/file-stem for split GGUFs, org/repo/model.safetensors, or org/repo/model-00001-of-00048.safetensors."
    )
}

fn split_bare_name_selector(input: &str) -> (&str, Option<&str>) {
    match input.split_once(':') {
        Some((name, selector))
            if !name.is_empty()
                && !selector.is_empty()
                && !name.contains('/')
                && !name.contains('@')
                && !name.contains("://") =>
        {
            (name, Some(selector))
        }
        _ => (input, None),
    }
}

fn normalize_repo_leaf_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn select_strong_repo_hit(query: &str, repo_ids: &[String]) -> Option<String> {
    let query_norm = normalize_repo_leaf_name(query);
    if query_norm.is_empty() {
        return None;
    }
    let mut exact = Vec::new();
    for repo_id in repo_ids {
        let leaf = repo_id.rsplit('/').next().unwrap_or(repo_id);
        if normalize_repo_leaf_name(leaf) == query_norm {
            exact.push(repo_id.clone());
        }
    }
    if let Some(first) = exact.into_iter().next() {
        return Some(first);
    }
    None
}

async fn discover_hf_repo_for_bare_name(name: &str) -> Result<Option<String>> {
    let api = skippy_model_hf::build_hf_async_api()?;
    let stream = api
        .list_models()
        .search(name.to_string())
        .filter("gguf".to_string())
        .limit(20_usize)
        .send()
        .with_context(|| format!("Search Hugging Face for '{name}'"))?;
    tokio::pin!(stream);
    let mut repo_ids = Vec::new();
    while let Some(repo) = stream.next().await {
        repo_ids.push(repo?.id);
    }
    Ok(select_strong_repo_hit(name, &repo_ids))
}

async fn canonicalize_model_ref_input(input: &str) -> Result<String> {
    if parse_exact_model_ref(input).is_ok() {
        return Ok(input.to_string());
    }
    if input.contains('/') || input.starts_with("http://") || input.starts_with("https://") {
        return Ok(input.to_string());
    }

    let (name, selector) = split_bare_name_selector(input);
    if let Some(repo) = discover_hf_repo_for_bare_name(name).await? {
        if let Some(selector) = selector {
            return Ok(format!("{repo}:{selector}"));
        }
        return Ok(repo);
    }
    Ok(input.to_string())
}

fn is_split_mlx_first_shard(file: &str) -> bool {
    skippy_model_resolver::is_split_mlx_first_shard(file)
}

fn select_default_hf_file_from_siblings(siblings: &[String]) -> Option<String> {
    resolve_hf_file_from_siblings("", siblings)
}

fn resolve_hf_file_from_siblings(requested: &str, siblings: &[String]) -> Option<String> {
    if requested.ends_with(".gguf")
        || requested.ends_with(".safetensors")
        || requested.ends_with(".safetensors.index.json")
    {
        return Some(requested.to_string());
    }

    let files = siblings
        .iter()
        .cloned()
        .map(ModelArtifactFile::new)
        .collect::<Vec<_>>();
    let selector = (!requested.is_empty()).then_some(requested);
    select_primary_artifact_file(selector, &files)
        .ok()
        .map(|file| file.path)
}

pub(super) fn is_known_gguf_sidecar(file: &str) -> bool {
    let basename = file.rsplit('/').next().unwrap_or(file);
    basename.to_ascii_lowercase().starts_with("mmproj")
}

fn split_gguf_shard_info(file: &str) -> Option<(&str, &str, &str)> {
    skippy_model_ref::split_gguf_shard_info(file)
        .map(|shard| (shard.prefix, shard.part, shard.total))
}

fn is_split_gguf_first_shard(file: &str) -> bool {
    split_gguf_shard_info(file)
        .map(|(_, part, _)| part == "00001")
        .unwrap_or(false)
}

pub(super) fn gguf_variant_size_bytes_from_siblings(
    file: &str,
    siblings: &[(String, Option<u64>)],
) -> Option<u64> {
    let files = siblings
        .iter()
        .map(|(path, size_bytes)| ModelArtifactFile {
            path: path.clone(),
            size_bytes: *size_bytes,
            sha256: None,
        })
        .collect::<Vec<_>>();
    skippy_model_artifact::selection::gguf_variant_size_bytes(file, &files)
}

fn collect_show_gguf_variants_from_siblings(
    siblings: &[(String, Option<u64>)],
    available_bytes: u64,
) -> Vec<(String, Option<u64>)> {
    let mut gguf_candidates: Vec<(String, Option<u64>)> = siblings
        .iter()
        .filter_map(|(file, _size)| {
            let lower = file.to_lowercase();
            if !lower.ends_with(".gguf") {
                return None;
            }
            if is_known_gguf_sidecar(file) {
                return None;
            }
            if split_gguf_shard_info(file).is_some() && !is_split_gguf_first_shard(file) {
                return None;
            }
            Some((
                file.clone(),
                gguf_variant_size_bytes_from_siblings(file, siblings),
            ))
        })
        .collect();

    if available_bytes == 0 {
        gguf_candidates.sort_by(|left, right| {
            file_preference_score(&left.0)
                .cmp(&file_preference_score(&right.0))
                .then_with(|| left.0.cmp(&right.0))
        });
        return gguf_candidates;
    }

    gguf_candidates.sort_by(|left, right| {
        compare_gguf_candidates_by_fit(&left.0, left.1, &right.0, right.1, available_bytes)
    });
    gguf_candidates
}

fn compare_gguf_candidates_by_fit(
    left_file: &str,
    left_size: Option<u64>,
    right_file: &str,
    right_size: Option<u64>,
    available_bytes: u64,
) -> std::cmp::Ordering {
    skippy_model_artifact::selection::compare_gguf_candidates_by_fit(
        left_file,
        left_size,
        right_file,
        right_size,
        available_bytes,
    )
}

async fn remote_size_bytes(url: &str) -> Option<u64> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("skippy/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .ok()?;
    let response = client
        .head(url)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?;
    response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
}

#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    size: Option<u64>,
}

async fn fetch_hf_tree_entries(
    repo: &str,
    revision: Option<&str>,
    path: Option<&str>,
) -> Option<Vec<HfTreeEntry>> {
    let revision = revision.unwrap_or("main");
    let base = format!("https://huggingface.co/api/models/{repo}/tree/{revision}");
    let url = match path {
        Some(path) if !path.is_empty() => format!("{base}/{path}?recursive=1&expand=1"),
        _ => format!("{base}?recursive=1&expand=1"),
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("skippy/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .ok()?;
    client
        .get(url)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json::<Vec<HfTreeEntry>>()
        .await
        .ok()
}

fn sibling_entries_with_tree_sizes(
    siblings: &[(String, Option<u64>)],
    tree_entries: Vec<HfTreeEntry>,
) -> Vec<(String, Option<u64>)> {
    let tree_sizes = tree_entries
        .into_iter()
        .filter(|entry| entry.entry_type == "file")
        .filter_map(|entry| Some((entry.path, entry.size?)))
        .collect::<std::collections::HashMap<_, _>>();

    siblings
        .iter()
        .map(|(file, size)| (file.clone(), size.or_else(|| tree_sizes.get(file).copied())))
        .collect()
}

async fn select_default_hf_file_fit_aware(
    repo: &str,
    revision: Option<&str>,
    siblings: &[(String, Option<u64>)],
) -> Option<String> {
    let mut gguf_candidates: Vec<(String, Option<u64>)> = Vec::new();
    for (file, api_size) in siblings {
        let lower = file.to_lowercase();
        if !lower.ends_with(".gguf") {
            continue;
        }
        if is_known_gguf_sidecar(file) {
            continue;
        }
        if lower.contains("-000") && !lower.contains("-00001-of-") {
            continue;
        }
        gguf_candidates.push((file.clone(), *api_size));
    }
    if gguf_candidates.is_empty() {
        return None;
    }

    let available_bytes = crate::models::output::fit_budget_bytes();
    if available_bytes == 0 {
        gguf_candidates.sort_by(|left, right| {
            file_preference_score(&left.0)
                .cmp(&file_preference_score(&right.0))
                .then_with(|| left.0.cmp(&right.0))
        });
        return gguf_candidates.first().map(|(f, _)| f.clone());
    }

    // Prefer API-provided sizes; only fall back to HEAD for files missing a size.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("skippy/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .ok();
    let mut scored: Vec<(String, Option<u64>)> = Vec::with_capacity(gguf_candidates.len());
    for (file, api_size) in gguf_candidates {
        let size = if api_size.is_some() {
            api_size
        } else if let Some(ref c) = client {
            let url = huggingface_resolve_url(repo, revision, &file);
            c.head(&url)
                .send()
                .await
                .ok()
                .and_then(|r| r.error_for_status().ok())
                .and_then(|r| {
                    r.headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                })
        } else {
            None
        };
        scored.push((file, size));
    }
    pick_gguf_for_budget(scored, available_bytes)
}

/// The preferred GGUF among sized candidates for a local fit budget: the
/// largest file that fits, before anything that does not.
fn pick_gguf_for_budget(
    mut scored: Vec<(String, Option<u64>)>,
    available_bytes: u64,
) -> Option<String> {
    scored.sort_by(|left, right| {
        compare_gguf_candidates_by_fit(&left.0, left.1, &right.0, right.1, available_bytes)
    });
    scored.first().map(|(file, _)| file.clone())
}

fn repo_prefers_gguf_only(repo: &str) -> bool {
    skippy_model_artifact::selection::repo_prefers_gguf_only(repo)
}

#[cfg(test)]
type RepoSiblingEntriesOverrideFn =
    Arc<dyn Fn(&str, &str) -> Option<Vec<(String, Option<u64>)>> + Send + Sync>;

#[cfg(test)]
static REPO_SIBLING_ENTRIES_OVERRIDE: LazyLock<Mutex<Option<RepoSiblingEntriesOverrideFn>>> =
    LazyLock::new(|| Mutex::new(None));

#[cfg(test)]
struct RepoSiblingEntriesOverrideGuard;

#[cfg(test)]
impl RepoSiblingEntriesOverrideGuard {
    fn set(func: RepoSiblingEntriesOverrideFn) -> Self {
        let mut slot = REPO_SIBLING_ENTRIES_OVERRIDE.lock().unwrap();
        *slot = Some(func);
        Self
    }
}

#[cfg(test)]
impl Drop for RepoSiblingEntriesOverrideGuard {
    fn drop(&mut self) {
        let mut slot = REPO_SIBLING_ENTRIES_OVERRIDE.lock().unwrap();
        *slot = None;
    }
}

async fn fetch_repo_sibling_entries(
    repo: &str,
    revision: &str,
) -> Result<Vec<(String, Option<u64>)>> {
    #[cfg(test)]
    {
        let func = REPO_SIBLING_ENTRIES_OVERRIDE.lock().unwrap().clone();
        if let Some(func) = func
            && let Some(entries) = func(repo, revision)
        {
            return Ok(entries);
        }
    }

    let api = skippy_model_hf::build_hf_async_api()?;
    let (owner, name) = repo.split_once('/').unwrap_or(("", repo));
    let detail = api
        .model(owner, name)
        .info()
        .revision(revision.to_string())
        .send()
        .await
        .with_context(|| format!("Fetch Hugging Face repo {repo}@{revision}"))?;
    let siblings = detail
        .siblings
        .unwrap_or_default()
        .iter()
        .map(|sibling| (sibling.rfilename.clone(), sibling.size))
        .collect::<Vec<_>>();

    if siblings.iter().all(|(_, size)| size.is_some()) {
        return Ok(siblings);
    }

    let tree_entries = fetch_hf_tree_entries(repo, Some(revision), None).await;
    Ok(tree_entries
        .map(|entries| sibling_entries_with_tree_sizes(&siblings, entries))
        .unwrap_or(siblings))
}

pub(crate) async fn resolve_huggingface_file_from_sibling_entries(
    repo: &str,
    revision: Option<&str>,
    file: &str,
    sibling_entries: &[(String, Option<u64>)],
) -> Result<String> {
    if file.ends_with(".gguf")
        || file.ends_with(".safetensors")
        || file.ends_with(".safetensors.index.json")
    {
        return Ok(file.to_string());
    }

    let revision = revision.unwrap_or("main");
    let siblings: Vec<String> = sibling_entries.iter().map(|(f, _)| f.clone()).collect();
    let has_mlx_weights = siblings
        .iter()
        .any(|entry| entry == "model.safetensors" || is_split_mlx_first_shard(entry));

    if file.is_empty() {
        let gguf_only = repo_prefers_gguf_only(repo);
        if gguf_only {
            if let Some(resolved) =
                select_default_hf_file_fit_aware(repo, Some(revision), sibling_entries).await
            {
                return Ok(resolved);
            }
            bail!("No GGUF model files found in {repo}@{revision}.");
        }

        if let Some(resolved) = select_default_hf_file_from_siblings(&siblings) {
            return Ok(resolved);
        }

        if let Some(resolved) =
            select_default_hf_file_fit_aware(repo, Some(revision), sibling_entries).await
        {
            return Ok(resolved);
        }
    }
    if file == "model" && has_mlx_weights {
        bail!(
            "MLX shorthand '/model' is not supported. Use '{repo}' or a full file ref like '{repo}/model.safetensors'."
        );
    }

    if let Some(resolved) = resolve_hf_file_from_siblings(file, &siblings) {
        return Ok(resolved);
    }

    bail!(
        "No model file matching stem '{file}' in {repo}@{revision}. Use a full ref like org/repo/file.gguf or org/repo/model.safetensors."
    )
}

async fn resolve_huggingface_file(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Result<String> {
    let revision = revision.unwrap_or("main");
    let sibling_entries = fetch_repo_sibling_entries(repo, revision).await?;
    resolve_huggingface_file_from_sibling_entries(repo, Some(revision), file, &sibling_entries)
        .await
}

pub(super) fn huggingface_resolve_url(repo: &str, revision: Option<&str>, file: &str) -> String {
    skippy_model_resolver::huggingface_resolve_url(repo, revision, file)
}

pub(super) fn file_preference_score(file: &str) -> usize {
    skippy_model_artifact::selection::file_preference_score(file)
}

async fn remote_size_label(url: &str) -> Option<String> {
    let size = remote_size_bytes(url).await?;
    Some(format_size_bytes(size))
}

pub(super) async fn remote_hf_size_label_with_api(
    _api: &hf_hub::HFClient,
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<String> {
    if split_gguf_shard_info(file).is_some() {
        let tree_path = Path::new(file)
            .parent()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty());
        if let Some(tree_entries) = fetch_hf_tree_entries(repo, revision, tree_path).await {
            let siblings = tree_entries
                .into_iter()
                .filter(|entry| entry.entry_type == "file")
                .map(|entry| (entry.path, entry.size))
                .collect::<Vec<_>>();
            if let Some(size) = gguf_variant_size_bytes_from_siblings(file, &siblings) {
                return Some(format_size_bytes(size));
            }
        }
    }

    let url = huggingface_resolve_url(repo, revision, file);
    remote_size_label(&url).await
}

#[cfg(test)]
mod tests;
