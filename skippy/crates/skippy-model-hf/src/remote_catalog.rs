//! Fetches and caches the meshllm/catalog HuggingFace dataset for layer package discovery.
//!
//! The catalog lives at <https://huggingface.co/datasets/meshllm/catalog> with entries like:
//! ```text
//! entries/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF.json
//! ```
use std::{
    collections::HashSet,
    fs,
    path::{Component, Path, PathBuf},
    sync::{
        Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime},
};

use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result, bail};
use hf_hub::HFClientSync;
use skippy_model_resolver::{
    CatalogProvider, CatalogSidecarAsset, CatalogSidecarRef,
    CatalogVariant as ResolverCatalogVariant, HfCatalogProvider, ModelArtifactCandidate,
    ModelResolver,
};

fn build_hf_api() -> Result<HFClientSync> {
    let repository = crate::HfModelRepository::from_env()?;
    HFClientSync::from_inner(repository.api).context("build Hugging Face sync API client")
}

// ---------------------------------------------------------------------------
// Schema types
// ---------------------------------------------------------------------------

pub use skippy_model_resolver::CatalogEntry;
#[cfg(test)]
pub use skippy_model_resolver::{
    CatalogPackage, CatalogSidecarAsset as CatalogSidecarAssetRef,
    CatalogSidecarRef as CatalogSidecar, CatalogSource, CatalogVariant,
    CuratedMeta as CatalogCurated,
};

// ---------------------------------------------------------------------------
// Static catalog cache
// ---------------------------------------------------------------------------

static CATALOG_ENTRIES: RwLock<Option<Vec<CatalogEntry>>> = RwLock::new(None);
static CATALOG_ENSURE_LOCK: Mutex<()> = Mutex::new(());

/// Tracks the most recent failed catalog refresh so we don't re-attempt a slow
/// network refresh on every request when the cache is already loaded but the
/// staleness marker can't be refreshed (e.g. a download error). Without this,
/// a persistently failing refresh turns every `/api/models` call into a fresh
/// multi-second download attempt.
static CATALOG_REFRESH_BACKOFF_UNTIL: Mutex<Option<Instant>> = Mutex::new(None);

/// How long to suppress repeated refresh attempts after a failure when a stale
/// cached catalog is already available.
const CATALOG_REFRESH_BACKOFF: Duration = Duration::from_secs(5 * 60);

static CATALOG_ENTRIES_OVERRIDE_ACTIVE: AtomicBool = AtomicBool::new(false);

type HfModelFileProbe = Arc<dyn Fn(&str, &str, &str) -> bool + Send + Sync>;

static HF_MODEL_FILE_PROBE_OVERRIDE: LazyLock<Mutex<Option<HfModelFileProbe>>> =
    LazyLock::new(|| Mutex::new(None));

#[doc(hidden)]
pub struct CatalogEntriesOverrideGuard {
    previous_entries: Option<Vec<CatalogEntry>>,
    previous_override_active: bool,
}

#[doc(hidden)]
pub struct HfModelFileProbeOverrideGuard {
    previous_probe: Option<HfModelFileProbe>,
}

#[doc(hidden)]
pub fn set_catalog_entries_for_test(entries: Vec<CatalogEntry>) -> CatalogEntriesOverrideGuard {
    let previous_override_active = CATALOG_ENTRIES_OVERRIDE_ACTIVE.swap(true, Ordering::SeqCst);
    let mut lock = CATALOG_ENTRIES.write().unwrap();
    let previous = lock.replace(entries);
    CatalogEntriesOverrideGuard {
        previous_entries: previous,
        previous_override_active,
    }
}

impl Drop for CatalogEntriesOverrideGuard {
    fn drop(&mut self) {
        *CATALOG_ENTRIES.write().unwrap() = self.previous_entries.take();
        CATALOG_ENTRIES_OVERRIDE_ACTIVE.store(self.previous_override_active, Ordering::SeqCst);
    }
}

#[doc(hidden)]
pub fn set_hf_model_file_probe_for_test<F>(probe: F) -> HfModelFileProbeOverrideGuard
where
    F: Fn(&str, &str, &str) -> bool + Send + Sync + 'static,
{
    let mut slot = HF_MODEL_FILE_PROBE_OVERRIDE.lock().unwrap();
    let previous_probe = slot.replace(Arc::new(probe));
    HfModelFileProbeOverrideGuard { previous_probe }
}

impl Drop for HfModelFileProbeOverrideGuard {
    fn drop(&mut self) {
        *HF_MODEL_FILE_PROBE_OVERRIDE.lock().unwrap() = self.previous_probe.take();
    }
}

/// Returns the directory where the catalog dataset is cached locally.
pub fn catalog_cache_dir() -> PathBuf {
    std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .map(|path| path.join("meshllm-catalog"))
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|path| path.join(".cache/meshllm/catalog"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("meshllm/catalog"))
}

/// Returns true if the catalog cache is older than 24 hours or doesn't exist.
pub fn is_catalog_stale() -> bool {
    let cache_dir = catalog_cache_dir();
    let entries_dir = cache_dir.join("entries");
    if !entries_dir.is_dir() {
        return true;
    }
    let refresh_marker = entries_dir.join(".last_refresh");
    let Ok(metadata) = fs::metadata(&refresh_marker) else {
        return true;
    };
    let Ok(modified) = metadata.modified() else {
        return true;
    };
    let Ok(elapsed) = SystemTime::now().duration_since(modified) else {
        return true;
    };
    elapsed > Duration::from_secs(24 * 60 * 60)
}

/// Downloads/refreshes the catalog dataset from HuggingFace and loads entries into memory.
///
/// Lists all files in the `meshllm/catalog` dataset via the HF API, then downloads
/// every `entries/**/*.json` file. No hardcoded file list — new models added to the
/// catalog are discovered automatically.
pub fn refresh_catalog() -> Result<()> {
    crate::blocking::run_hf_sync(refresh_catalog_sync)
}

fn refresh_catalog_sync() -> Result<()> {
    let api = build_hf_api()?;
    let dataset = api.dataset("meshllm", "catalog");

    // List all files in the dataset repo
    let info = dataset
        .info()
        .revision("main".to_string())
        .send()
        .context("fetch meshllm/catalog dataset info")?;

    let siblings = info.siblings.as_ref();
    let Some(siblings) = siblings else {
        bail!("meshllm/catalog dataset info has no file listing");
    };

    let cache_dir = catalog_cache_dir();
    let entry_files = siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|f| f.starts_with("entries/") && f.ends_with(".json"))
        .map(|entry_file| {
            catalog_entry_cache_path(&cache_dir, entry_file)?;
            Ok(entry_file.to_string())
        })
        .collect::<Result<Vec<_>>>()?;

    if entry_files.is_empty() {
        bail!("meshllm/catalog has no entry files");
    }

    let entries_dir = cache_dir.join("entries");
    fs::create_dir_all(&entries_dir)
        .with_context(|| format!("create catalog cache dir {}", entries_dir.display()))?;

    // Download each entry file
    for entry_file in &entry_files {
        let downloaded = dataset
            .download_file()
            .filename(entry_file.clone())
            .revision("main".to_string())
            .send()
            .with_context(|| format!("download catalog entry {entry_file}"))?;

        // Copy to our cache dir structure if needed
        let dest = catalog_entry_cache_path(&cache_dir, entry_file)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        if downloaded != dest {
            fs::copy(&downloaded, &dest)
                .with_context(|| format!("copy catalog entry to cache: {entry_file}"))?;
        }
    }

    prune_stale_catalog_entry_files(&cache_dir, &entry_files)?;

    // Touch marker to update mtime for staleness check. Directory mtimes do
    // not reliably change when existing entry files are overwritten.
    let _ = fs::File::create(entries_dir.join(".last_refresh"));

    load_catalog_from_disk()
}

/// Loads catalog entries from the on-disk cache without downloading.
/// Useful if the cache is already fresh.
pub fn load_catalog_from_disk() -> Result<()> {
    let cache_dir = catalog_cache_dir();
    let entries_dir = cache_dir.join("entries");
    if !entries_dir.is_dir() {
        bail!(
            "catalog entries directory does not exist: {}",
            entries_dir.display()
        );
    }

    let entries = parse_entries_recursive(&entries_dir)?;
    for entry in &entries {
        remote_models_from_entry(entry)?;
    }
    let mut lock = CATALOG_ENTRIES
        .write()
        .map_err(|_| anyhow::anyhow!("catalog lock poisoned"))?;
    *lock = Some(entries);
    Ok(())
}

/// Ensures the catalog is loaded — refreshes if stale, otherwise loads from disk.
pub fn ensure_catalog() -> Result<()> {
    if CATALOG_ENTRIES_OVERRIDE_ACTIVE.load(Ordering::SeqCst) {
        let lock = CATALOG_ENTRIES
            .read()
            .map_err(|_| anyhow::anyhow!("catalog lock poisoned"))?;
        if lock.is_some() {
            return Ok(());
        }
    }

    {
        let lock = CATALOG_ENTRIES
            .read()
            .map_err(|_| anyhow::anyhow!("catalog lock poisoned"))?;
        if lock.is_some() && !is_catalog_stale() {
            return Ok(());
        }
    }

    let _ensure = CATALOG_ENSURE_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("catalog ensure lock poisoned"))?;

    {
        let lock = CATALOG_ENTRIES
            .read()
            .map_err(|_| anyhow::anyhow!("catalog lock poisoned"))?;
        if lock.is_some() && !is_catalog_stale() {
            return Ok(());
        }
    }

    if is_catalog_stale() {
        // If a recent refresh failed and we already have a (stale) catalog
        // loaded, don't hammer the network on every request — serve the loaded
        // catalog until the backoff window elapses.
        if catalog_entries().is_some() && refresh_in_backoff() {
            return Ok(());
        }

        match refresh_catalog() {
            Ok(()) => {
                clear_refresh_backoff();
                Ok(())
            }
            Err(refresh_err) => {
                if catalog_entries().is_some() {
                    set_refresh_backoff();
                    tracing::warn!(
                        "failed to refresh stale meshllm/catalog; using already-loaded stale catalog \
                         (suppressing retries for {}s): {refresh_err:#}",
                        CATALOG_REFRESH_BACKOFF.as_secs()
                    );
                    return Ok(());
                }

                load_catalog_from_disk().with_context(|| {
                    format!(
                        "failed to refresh meshllm/catalog ({refresh_err:#}) and failed to load stale cache"
                    )
                })
            }
        }
    } else {
        load_catalog_from_disk()
    }
}

/// Returns true if a recent refresh failure means we should skip another
/// refresh attempt for now.
fn refresh_in_backoff() -> bool {
    let guard = CATALOG_REFRESH_BACKOFF_UNTIL.lock();
    match guard {
        Ok(until) => until.map(|t| Instant::now() < t).unwrap_or(false),
        Err(_) => false,
    }
}

/// Records that a refresh just failed, suppressing retries for the backoff
/// window.
fn set_refresh_backoff() {
    if let Ok(mut guard) = CATALOG_REFRESH_BACKOFF_UNTIL.lock() {
        *guard = Some(Instant::now() + CATALOG_REFRESH_BACKOFF);
    }
}

/// Clears any active refresh backoff after a successful refresh.
fn clear_refresh_backoff() {
    if let Ok(mut guard) = CATALOG_REFRESH_BACKOFF_UNTIL.lock() {
        *guard = None;
    }
}

/// Searches the cached catalog for a layer-package matching `model_query`.
///
/// The query is matched (case-insensitive contains) against:
/// - variant name (the key in the variants map)
/// - curated name
/// - source_repo
/// - exact layer-package repo
///
/// Returns the first matching layer-package repo as an `hf://` reference.
/// Catalog entries, variants, and package repos are traversed in sorted order
/// so overlapping contains-matches resolve deterministically.
pub fn find_layer_package(model_query: &str) -> Option<String> {
    let resolver = resolver_from_loaded_entries()?;
    resolver
        .resolve(model_query)
        .ok()?
        .into_iter()
        .find_map(|candidate| match candidate {
            ModelArtifactCandidate::RemoteLayerPackage(package) => {
                Some(format!("hf://{}", package.package_repo))
            }
            _ => None,
        })
}

/// Probes a Hugging Face model repo directly and treats it as a layer package
/// only when the package manifest exists. Repo naming is intentionally ignored.
pub fn find_huggingface_layer_package(model_query: &str) -> Option<String> {
    let (repo, revision) = parse_exact_huggingface_repo(model_query)?;
    let revision_ref = revision.as_deref().unwrap_or("main");
    match hf_model_repo_has_file(&repo, revision_ref, "model-package.json") {
        Ok(true) => Some(format_hf_package_ref(&repo, revision.as_deref())),
        Ok(false) => None,
        Err(err) => {
            tracing::debug!(
                "Hugging Face layer package probe failed for {repo}@{revision_ref}: {err:#}"
            );
            None
        }
    }
}

/// Resolve the catalog package preferred by `models download` in either CLI.
pub fn resolve_layer_package_download(model_query: &str) -> Option<String> {
    let _ = ensure_catalog();
    find_layer_package(model_query).or_else(|| find_huggingface_layer_package(model_query))
}

fn parse_exact_huggingface_repo(input: &str) -> Option<(String, Option<String>)> {
    let (repo, revision, selector) = skippy_model_resolver::parse_huggingface_repo_ref(input)
        .or_else(|| skippy_model_resolver::parse_huggingface_repo_url(input))?;
    selector.is_none().then_some((repo, revision))
}

fn format_hf_package_ref(repo: &str, revision: Option<&str>) -> String {
    match revision {
        Some(revision) => format!("hf://{repo}@{revision}"),
        None => format!("hf://{repo}"),
    }
}

fn hf_model_repo_has_file(repo: &str, revision: &str, file: &str) -> Result<bool> {
    {
        let probe = HF_MODEL_FILE_PROBE_OVERRIDE.lock().unwrap().clone();
        if let Some(probe) = probe {
            return Ok(probe(repo, revision, file));
        }
    }

    let repo = repo.to_string();
    let revision = revision.to_string();
    let file = file.to_string();
    crate::blocking::run_hf_sync(move || {
        let api = build_hf_api()?;
        let (owner, name) = repo.split_once('/').unwrap_or(("", repo.as_str()));
        let info = api
            .model(owner, name)
            .info()
            .revision(revision.clone())
            .send()
            .with_context(|| format!("fetch Hugging Face model repo {repo}@{revision}"))?;
        Ok(info
            .siblings
            .unwrap_or_default()
            .iter()
            .any(|sibling| sibling.rfilename == file))
    })
}

/// A resolved model download reference from the remote catalog.
pub struct RemoteModelRef {
    pub name: String,
    pub repo: String,
    pub revision: Option<String>,
    pub file: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteCatalogAsset {
    pub file: String,
    pub repo: String,
    pub revision: Option<String>,
    pub source_file: String,
}

impl RemoteCatalogAsset {
    /// Download this exact source filename, preserving its repository revision.
    pub fn download_ref(&self) -> String {
        skippy_model_ref::format_model_ref(
            &self.repo,
            self.revision.as_deref(),
            Some(&self.source_file),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteCatalogModel {
    pub name: String,
    pub file: String,
    pub repo: String,
    pub revision: Option<String>,
    pub source_file: String,
    pub size: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub extra_files: Vec<RemoteCatalogAsset>,
    pub mmproj: Option<RemoteCatalogAsset>,
}

impl RemoteCatalogModel {
    pub fn source_repo(&self) -> &str {
        &self.repo
    }

    pub fn source_file(&self) -> &str {
        &self.source_file
    }

    pub fn resolve_url(&self) -> String {
        skippy_model_resolver::huggingface_resolve_url(
            &self.repo,
            self.revision.as_deref(),
            &self.source_file,
        )
    }

    pub fn exact_ref(&self) -> String {
        skippy_model_resolver::format_huggingface_display_ref(
            &self.repo,
            self.revision.as_deref(),
            &self.source_file,
        )
    }

    pub fn source_asset(&self) -> RemoteCatalogAsset {
        RemoteCatalogAsset {
            file: self.file.clone(),
            repo: self.repo.clone(),
            revision: self.revision.clone(),
            source_file: self.source_file.clone(),
        }
    }
}

/// Searches the remote catalog for a model matching `query` and returns
/// download coordinates (repo, revision, file) if found.
///
/// This enables models not in the baked-in catalog to be resolved and
/// downloaded from HuggingFace when they exist in the remote catalog.
pub fn resolve_model_download(query: &str) -> Option<RemoteModelRef> {
    if ensure_catalog().is_err() {
        return None;
    }
    let resolver = resolver_from_loaded_entries()?;
    resolver
        .resolve(query)
        .ok()?
        .into_iter()
        .find_map(|candidate| match candidate {
            ModelArtifactCandidate::RemoteGguf(remote) => {
                let file = remote.source.file?;
                let name = remote
                    .curated
                    .as_ref()
                    .map(|curated| curated.name.clone())
                    .unwrap_or_else(|| query.to_string());
                Some(RemoteModelRef {
                    name,
                    repo: remote.source.repo,
                    revision: remote.source.revision,
                    file,
                })
            }
            _ => None,
        })
}

pub fn find_model_exact(query: &str) -> Option<RemoteCatalogModel> {
    if ensure_catalog().is_err() {
        return None;
    }
    find_loaded_model_exact(query)
}

pub fn find_loaded_model_exact(query: &str) -> Option<RemoteCatalogModel> {
    let q = query.to_lowercase();
    loaded_models().ok()?.into_iter().find(|model| {
        model.name.to_lowercase() == q
            || model.exact_ref().to_lowercase() == q
            || model.file.to_lowercase() == q
            || model.file.trim_end_matches(".gguf").to_lowercase() == q
    })
}

pub fn matching_model_for_huggingface(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<RemoteCatalogModel> {
    if ensure_catalog().is_err() {
        return None;
    }
    matching_loaded_model_for_huggingface(repo, revision, file)
}

pub fn matching_primary_for_huggingface(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<RemoteCatalogModel> {
    let model = matching_model_for_huggingface(repo, revision, file)?;
    let asset = model.source_asset();
    asset_matches_hf(&asset, repo, revision, file).then_some(model)
}

#[cfg(test)]
pub fn matching_primary_for_url(url: &str) -> Option<RemoteCatalogModel> {
    let (repo, revision, file) = skippy_model_resolver::parse_hf_resolve_url(url)?;
    matching_primary_for_huggingface(&repo, revision.as_deref(), &file)
}

pub fn loaded_models() -> Result<Vec<RemoteCatalogModel>> {
    let entries = catalog_entries().context("remote catalog is not loaded")?;
    let mut models = Vec::new();
    for entry in &entries {
        models.extend(remote_models_from_entry(entry)?);
    }
    Ok(models)
}

/// Returns all loaded catalog entries (if any).
pub fn catalog_entries() -> Option<Vec<CatalogEntry>> {
    let lock = CATALOG_ENTRIES.read().ok()?;
    lock.clone()
}

fn resolver_from_loaded_entries() -> Option<ModelResolver<HfCatalogProvider>> {
    let entries = catalog_entries()?;
    Some(ModelResolver::new(
        HfCatalogProvider::from_entries(entries),
        Vec::new(),
    ))
}

fn matching_loaded_model_for_huggingface(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<RemoteCatalogModel> {
    loaded_models()
        .ok()?
        .into_iter()
        .find(|model| {
            std::iter::once(model.source_asset())
                .chain(model.extra_files.clone())
                .chain(model.mmproj.clone())
                .any(|asset| asset_matches_hf(&asset, repo, revision, file))
        })
        .or_else(|| {
            if revision.is_some() {
                None
            } else {
                matching_loaded_model_by_basename(file)
            }
        })
}

fn matching_loaded_model_by_basename(repo_file: &str) -> Option<RemoteCatalogModel> {
    let basename = repo_file
        .rsplit('/')
        .next()
        .unwrap_or(repo_file)
        .to_lowercase();
    loaded_models().ok()?.into_iter().find(|model| {
        model.file.to_lowercase() == basename
            || model.file.trim_end_matches(".gguf").to_lowercase()
                == basename.trim_end_matches(".gguf")
    })
}

fn asset_matches_hf(
    asset: &RemoteCatalogAsset,
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> bool {
    if !asset.repo.eq_ignore_ascii_case(repo) || !asset.source_file.eq_ignore_ascii_case(file) {
        return false;
    }
    match revision {
        Some(revision) => asset
            .revision
            .as_deref()
            .map(|value| value.eq_ignore_ascii_case(revision))
            .unwrap_or(false),
        None => true,
    }
}

fn remote_models_from_entry(entry: &CatalogEntry) -> Result<Vec<RemoteCatalogModel>> {
    let mut variants = entry.variants.iter().collect::<Vec<_>>();
    variants.sort_by(|left, right| left.0.cmp(right.0));
    variants
        .into_iter()
        .map(|(variant_name, variant)| remote_model_from_variant(variant_name, variant))
        .collect()
}

fn remote_model_from_variant(
    variant_name: &str,
    variant: &ResolverCatalogVariant,
) -> Result<RemoteCatalogModel> {
    let source_file = variant
        .source
        .file
        .clone()
        .unwrap_or_else(|| format!("{variant_name}.gguf"));
    Ok(RemoteCatalogModel {
        name: variant.curated.name.clone(),
        file: source_file
            .rsplit('/')
            .next()
            .unwrap_or(source_file.as_str())
            .to_string(),
        repo: variant.source.repo.clone(),
        revision: variant.source.revision.clone(),
        source_file,
        size: variant.curated.size.clone(),
        description: variant.curated.description.clone(),
        draft: variant.curated.draft.clone(),
        extra_files: parse_extra_file_assets(&variant.curated.extra_files)?,
        mmproj: variant
            .curated
            .mmproj
            .as_ref()
            .map(parse_sidecar_ref)
            .transpose()?
            .flatten(),
    })
}

fn parse_extra_file_assets(values: &[serde_json::Value]) -> Result<Vec<RemoteCatalogAsset>> {
    values
        .iter()
        .map(|value| {
            let object: &serde_json::Map<String, serde_json::Value> = value
                .as_object()
                .context("catalog extra_files entry is not an object")?;
            let file = object
                .get("file")
                .and_then(|value| value.as_str())
                .context("catalog extra_files entry missing string file")?
                .to_string();
            let repo = object
                .get("repo")
                .and_then(|value| value.as_str())
                .context("catalog extra_files entry missing string repo")?
                .to_string();
            let revision = object
                .get("revision")
                .and_then(|value| value.as_str())
                .map(str::to_string);
            let source_file = object
                .get("source_file")
                .or_else(|| object.get("file_path"))
                .and_then(|value| value.as_str())
                .unwrap_or(file.as_str())
                .to_string();
            Ok(RemoteCatalogAsset {
                file,
                repo,
                revision,
                source_file,
            })
        })
        .collect()
}

fn parse_sidecar_ref(value: &CatalogSidecarRef) -> Result<Option<RemoteCatalogAsset>> {
    match value {
        CatalogSidecarRef::Ref(value) => parse_sidecar_string_ref(value),
        CatalogSidecarRef::Asset(asset) => parse_sidecar_asset_ref(asset),
    }
}

fn parse_sidecar_string_ref(value: &str) -> Result<Option<RemoteCatalogAsset>> {
    let (repo, revision, source_file) = skippy_model_resolver::parse_huggingface_file_ref(value)
        .or_else(|| skippy_model_resolver::parse_hf_resolve_url(value))
        .with_context(|| format!("catalog sidecar ref is not a Hugging Face file ref: {value}"))?;
    let file = source_file
        .rsplit('/')
        .next()
        .unwrap_or(source_file.as_str())
        .to_string();
    Ok(Some(RemoteCatalogAsset {
        file,
        repo,
        revision,
        source_file,
    }))
}

fn parse_sidecar_asset_ref(asset: &CatalogSidecarAsset) -> Result<Option<RemoteCatalogAsset>> {
    let file = asset
        .file
        .rsplit('/')
        .next()
        .unwrap_or(asset.file.as_str())
        .to_string();
    Ok(Some(RemoteCatalogAsset {
        file,
        repo: asset.repo.clone(),
        revision: asset.revision.clone(),
        source_file: asset
            .source_file
            .clone()
            .unwrap_or_else(|| asset.file.clone()),
    }))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_entries_recursive(dir: &std::path::Path) -> Result<Vec<CatalogEntry>> {
    Ok(HfCatalogProvider::from_entries_dir(dir)?.entries().to_vec())
}

fn catalog_entry_cache_path(cache_dir: &Path, entry_file: &str) -> Result<PathBuf> {
    let path = Path::new(entry_file);
    let mut components = path.components();

    match components.next() {
        Some(Component::Normal(component)) if component == "entries" => {}
        _ => bail!("invalid catalog entry path outside entries/: {entry_file}"),
    }

    let mut dest = cache_dir.join("entries");
    let mut saw_child = false;
    for component in components {
        match component {
            Component::Normal(part) => {
                saw_child = true;
                dest = dest.join(part);
            }
            _ => bail!("invalid catalog entry path component in {entry_file}"),
        }
    }

    if !saw_child || dest.extension().is_none_or(|ext| ext != "json") {
        bail!("invalid catalog entry path: {entry_file}");
    }

    Ok(dest)
}

fn prune_stale_catalog_entry_files(cache_dir: &Path, entry_files: &[String]) -> Result<()> {
    let entries_dir = cache_dir.join("entries");
    if !entries_dir.is_dir() {
        return Ok(());
    }

    let expected_paths: HashSet<PathBuf> = entry_files
        .iter()
        .map(|path| catalog_entry_cache_path(cache_dir, path))
        .collect::<Result<_>>()?;
    prune_stale_json_files(&entries_dir, &expected_paths)
}

fn prune_stale_json_files(dir: &Path, expected_paths: &HashSet<PathBuf>) -> Result<()> {
    let read_dir =
        fs::read_dir(dir).with_context(|| format!("read catalog cache dir {}", dir.display()))?;
    let mut dir_entries = read_dir
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("read cached catalog entries under {}", dir.display()))?;
    dir_entries.sort_by_key(|dir_entry| dir_entry.path());

    for dir_entry in dir_entries {
        let path = dir_entry.path();
        if path.is_dir() {
            prune_stale_json_files(&path, expected_paths)?;
            continue;
        }
        if path.extension().is_some_and(|ext| ext == "json") && !expected_paths.contains(&path) {
            fs::remove_file(&path)
                .with_context(|| format!("remove stale catalog entry {}", path.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
