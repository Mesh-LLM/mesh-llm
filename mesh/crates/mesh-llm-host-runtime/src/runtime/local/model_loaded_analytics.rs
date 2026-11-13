//! `model_loaded`, and the identity serving events use for each loaded model.
//!
//! A model is named only when the runtime holds a public identity for it: the
//! catalog entry it was loaded from. That is the privacy rule this module
//! enforces, and it is enforced on the *source* of the name, not on the name's
//! shape. A model with no catalog identity is `redacted` and described by its
//! architecture and a coarse size instead.

use std::path::Path;

use crate::inference_analytics::model_identity::{self, ModelIdentity};
use crate::models;

/// Where the model being reported came from.
///
/// This decides where a public identity may be looked for at all, so it is a
/// type rather than a string: the privacy rule belongs to the source, not to
/// a comparison at one call site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::runtime) enum ModelLoadSource {
    /// A GGUF file loaded directly: a `--gguf` path, or a catalog or Hugging
    /// Face model resolved to its file in the Hugging Face cache.
    DirectGguf,
    /// A catalog or repository model reference.
    LayerPackage,
    /// A whisper.cpp model file, by path or repository reference.
    WhisperFile,
    /// A Hugging Face layer package served as stage 0 of a split.
    SplitLayerPackage,
    /// Stage 0 of a split whose model came from anywhere but Hugging Face.
    SplitLocalSource,
}

impl ModelLoadSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::DirectGguf => "direct_gguf",
            Self::LayerPackage => "layer_package",
            Self::WhisperFile => "whisper_file",
            Self::SplitLayerPackage => "split_layer_package",
            Self::SplitLocalSource => "split_local_source",
        }
    }
}

/// Everything `model_loaded` may consult about one load.
pub(in crate::runtime) struct LoadedModel<'a> {
    /// The name skippy and the router serve it under. Never reported as such.
    pub(in crate::runtime) served_name: &'a str,
    pub(in crate::runtime) source: ModelLoadSource,
    pub(in crate::runtime) metadata: Option<&'a models::gguf::GgufCompactMeta>,
    /// The weights file, for a direct load.
    pub(in crate::runtime) path: Option<&'a Path>,
    /// The package reference, for a package or split load.
    pub(in crate::runtime) package_ref: Option<&'a str>,
    /// The default DFlash pairing attached to this model, if any.
    pub(in crate::runtime) dflash: Option<&'a mesh_llm_skippy_adapter::config::DefaultDFlash>,
    /// The device the load selected (`CUDA0`, `CPU`), if it selected one.
    pub(in crate::runtime) device: Option<&'a str>,
}

/// Catalog entries a load matched, found by the means its source allows.
#[derive(Debug, Default)]
struct CatalogMatches {
    /// The served name is itself a catalog name or exact reference.
    by_name: Option<String>,
    /// The file is a catalog entry's file in the Hugging Face cache.
    by_file: Option<String>,
    /// The package repository is a catalog entry's published package.
    by_package: Option<String>,
}

impl CatalogMatches {
    fn lookup(model: &LoadedModel<'_>) -> Self {
        match model.source {
            ModelLoadSource::DirectGguf => Self {
                by_file: model.path.and_then(catalog_name_for_cached_file),
                ..Self::default()
            },
            ModelLoadSource::LayerPackage | ModelLoadSource::SplitLayerPackage => Self {
                by_name: models::find_loaded_remote_catalog_model_exact(model.served_name)
                    .map(|entry| entry.name),
                by_package: model.package_ref.and_then(catalog_name_for_package_ref),
                ..Self::default()
            },
            ModelLoadSource::WhisperFile | ModelLoadSource::SplitLocalSource => Self::default(),
        }
    }
}

/// The catalog name a load may be reported under, if any.
///
/// A direct load is named only through its file's place in the Hugging Face
/// cache. Its served name is a file stem, and a stem the user chose (say
/// `acme-merger-finetune`) passes the label grammar cleanly, so the served
/// name is never consulted. A package is named by its served name only when
/// that is exactly a catalog name, since a configured alias such as
/// `server.alias` is operator configuration, not a catalog name.
fn reportable_catalog_name(source: ModelLoadSource, matches: CatalogMatches) -> Option<String> {
    match source {
        ModelLoadSource::DirectGguf => matches.by_file,
        ModelLoadSource::LayerPackage | ModelLoadSource::SplitLayerPackage => {
            matches.by_name.or(matches.by_package)
        }
        ModelLoadSource::WhisperFile | ModelLoadSource::SplitLocalSource => None,
    }
}

/// The catalog entry whose primary file is `path` in the Hugging Face cache.
///
/// Only an entry already loaded is consulted: model loading must not wait on
/// a catalog fetch for analytics. A file from a repository the catalog does
/// not list, private or not, matches nothing.
fn catalog_name_for_cached_file(path: &Path) -> Option<String> {
    let identity = models::huggingface_identity_for_path(path)?;
    let catalog = models::remote_catalog::loaded_models().ok()?;
    catalog_name_for_file(&catalog, &identity.repo_id, &identity.file)
}

fn catalog_name_for_file(
    catalog: &[models::remote_catalog::RemoteCatalogModel],
    repo: &str,
    file: &str,
) -> Option<String> {
    catalog
        .iter()
        .find(|entry| entry.repo.eq_ignore_ascii_case(repo) && entry.source_file == file)
        .map(|entry| entry.name.clone())
}

fn catalog_name_for_package_ref(package_ref: &str) -> Option<String> {
    let repo = package_ref.strip_prefix("hf://")?;
    let repo = repo.split_once('@').map_or(repo, |(repo, _)| repo);
    let entries = models::remote_catalog::catalog_entries()?;
    catalog_name_for_package_repo(&entries, repo)
}

fn catalog_name_for_package_repo(
    entries: &[skippy_model_resolver::CatalogEntry],
    repo: &str,
) -> Option<String> {
    entries
        .iter()
        .flat_map(|entry| entry.variants.values())
        .find(|variant| {
            variant
                .packages
                .iter()
                .any(|package| package.repo.eq_ignore_ascii_case(repo))
        })
        .map(|variant| variant.curated.name.clone())
}

/// The identity serving events report for a load.
fn model_identity(model: &LoadedModel<'_>, catalog_name: Option<&str>) -> ModelIdentity {
    let metadata = model.metadata;
    let pairing = model.dflash.map(|dflash| &dflash.pairing);
    ModelIdentity {
        model: catalog_name.map_or_else(
            mesh_llm_analytics::Label::redacted,
            mesh_llm_analytics::Label::sanitize_or_redact,
        ),
        architecture: metadata
            .and_then(|meta| mesh_llm_analytics::Label::sanitize(&meta.architecture)),
        parameters: metadata
            .and_then(|meta| meta.parameter_size.as_deref())
            .and_then(model_identity::parameter_bucket),
        moe: metadata.map(|meta| meta.expert_count > 0),
        family: pairing
            .and_then(|pairing| pairing.target.family.as_deref())
            .and_then(mesh_llm_analytics::Label::sanitize),
        dflash_draft: pairing
            .and_then(|pairing| mesh_llm_analytics::Label::sanitize(&pairing.draft.repo)),
        backend: model
            .device
            .and_then(model_identity::backend_of_selected_device),
    }
}

/// Report `model_loaded` and register the model's identity for the serving
/// events that follow.
pub(in crate::runtime) fn report_loaded_model_analytics(model: LoadedModel<'_>) {
    let catalog_name = reportable_catalog_name(model.source, CatalogMatches::lookup(&model));
    let identity = model_identity(&model, catalog_name.as_deref());
    model_identity::register(model.served_name, identity.clone());
    let mut properties = identity
        .apply(mesh_llm_analytics::Properties::new())
        .with("source", model.source.as_str());
    if let Some(backend) = system_one_backend(model.metadata) {
        properties = properties.with("system_one_backend", backend);
    }
    mesh_llm_analytics::capture(mesh_llm_analytics::Event::ModelLoaded, properties);
}

/// Report `model_loaded` for a load with no file path, package, or draft to
/// consult.
pub(in crate::runtime) fn report_model_loaded_analytics(
    model: &str,
    source: ModelLoadSource,
    metadata: Option<&models::gguf::GgufCompactMeta>,
) {
    report_loaded_model_analytics(LoadedModel {
        served_name: model,
        source,
        metadata,
        path: None,
        package_ref: None,
        dflash: None,
        device: None,
    });
}

/// Report `model_loaded` for stage 0 of a split.
///
/// The name is reportable only when the package is a Hugging Face reference.
/// A split can also run from a local `--gguf`, whose name may be a file stem
/// or an alias matching one.
pub(in crate::runtime) fn report_split_model_loaded_analytics(model: &str, package_ref: &str) {
    report_loaded_model_analytics(LoadedModel {
        served_name: model,
        source: split_model_load_source(package_ref),
        metadata: None,
        path: None,
        package_ref: Some(package_ref),
        dflash: None,
        device: None,
    });
}

fn split_model_load_source(package_ref: &str) -> ModelLoadSource {
    if package_ref.starts_with("hf://") {
        ModelLoadSource::SplitLayerPackage
    } else {
        ModelLoadSource::SplitLocalSource
    }
}

/// Which System One (OpenJEV) backend a loaded model is, if any.
///
/// Read from the GGUF architecture, never from the model name, so it stays a
/// closed vocabulary even for a direct `--gguf` load whose name is redacted.
fn system_one_backend(metadata: Option<&models::gguf::GgufCompactMeta>) -> Option<&'static str> {
    match metadata?.architecture.as_str() {
        crate::runtime::local_laya::LAYA_ARCHITECTURE => Some("laya"),
        "diffusion-gemma" => Some("openjev"),
        _ => None,
    }
}

#[cfg(test)]
#[path = "model_loaded_analytics/tests.rs"]
mod tests;
