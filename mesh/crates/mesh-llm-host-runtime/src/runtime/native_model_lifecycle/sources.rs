//! Source identity and startup-equivalent validation for managed cold models.
use super::*;
use crate::plugin::{MeshConfig, ModelConfigEntry};
use crate::runtime::{RuntimeOptions, startup_models};
use std::path::Path;

/// Use the eager spec builder for configured files, tuning, projector and device selectors.
pub(in crate::runtime) fn startup_spec(
    config: &MeshConfig,
    options: &RuntimeOptions,
    model: &ModelConfigEntry,
) -> Result<startup_models::StartupModelSpec> {
    let mut config = config.clone();
    let mut model = model.clone();
    model.lifecycle.load = ModelLoadPolicy::Startup;
    config.models = vec![model];
    let mut options = options.clone();
    options.model.clear();
    options.gguf.clear();
    startup_models::build_startup_model_specs(&options, &config)?
        .into_iter()
        .next()
        .context("configured native model has no launch spec")
}

/// Reject unsupported split policy before skipping any eager startup models.
pub(in crate::runtime) fn validate_split(
    config: &MeshConfig,
    options: &RuntimeOptions,
) -> Result<()> {
    if options.client || startup_models::cli_has_explicit_models(options) {
        return Ok(());
    }
    anyhow::ensure!(
        !(options.split || options.split_topology_lock.is_some())
            || !config
                .models
                .iter()
                .any(|model| is_on_demand_managed(config, model)),
        "on-demand native lifecycle does not support split serving; use lifecycle.load = startup for --split or --split-topology-lock"
    );
    Ok(())
}

/// Validate configured files before starting prefetch or accepting requests.
pub(super) fn validate_managed_sources(config: &MeshConfig) -> Result<()> {
    for model in config
        .models
        .iter()
        .filter(|model| is_on_demand_managed(config, model))
    {
        startup_spec(config, &RuntimeOptions::default(), model)?;
    }
    Ok(())
}

/// Resolve explicit and cached public names before the first native load.
pub(super) fn model_names(config: &MeshConfig, model: &ModelConfigEntry) -> Vec<String> {
    let mut names = vec![model.model.clone()];
    if let Some(alias) = startup_models::configured_server_alias(model, config.defaults.as_ref()) {
        names.push(alias);
    }
    let spec = model
        .hardware
        .as_ref()
        .and_then(|hardware| hardware.model_path.as_deref())
        .or_else(|| {
            config
                .defaults
                .as_ref()
                .and_then(|defaults| defaults.hardware.as_ref())
                .and_then(|hardware| hardware.model_path.as_deref())
        })
        .unwrap_or(&model.model);
    let direct = Path::new(spec);
    let cached = if direct.exists() {
        direct.to_path_buf()
    } else {
        crate::models::find_model_path(&model.model)
    };
    if cached.exists() {
        names.push(crate::models::model_ref_for_path(&cached));
    }
    names.sort();
    names.dedup();
    names
}

/// Materialize package assets as well as ordinary GGUF/projector downloads.
pub(super) async fn prefetch(source: &Path) -> Result<()> {
    let path = startup_models::resolve_model(source).await?;
    if crate::inference::skippy::is_layer_package_ref(&path.to_string_lossy()) {
        let package = path.to_string_lossy().into_owned();
        tokio::task::spawn_blocking(move || {
            crate::inference::skippy::resolve_package_v2_full_model_to_local(&package)
        })
        .await
        .context("join native package prefetch")??;
    }
    Ok(())
}

/// Warm model and configured projector disk assets without creating native sessions.
pub(super) fn spawn_prefetch(
    config: &MeshConfig,
    residencies: &BTreeMap<String, Arc<Residency>>,
) -> Result<()> {
    for model in config
        .models
        .iter()
        .filter(|model| residencies.contains_key(&model.model))
    {
        let residency = &residencies[&model.model];
        let strict_local = startup_models::skippy_local_source_required(
            model.skippy.as_ref(),
            config
                .defaults
                .as_ref()
                .and_then(|defaults| defaults.skippy.as_ref()),
        );
        if !residency.policy.prefetch || strict_local {
            continue;
        }
        let spec = startup_spec(config, &RuntimeOptions::default(), model)?;
        let name = model.model.clone();
        tokio::spawn(async move {
            let result = async {
                prefetch(&spec.model_ref).await?;
                if let Some(projector) = spec.mmproj_ref {
                    prefetch(&projector).await?;
                }
                Ok::<_, anyhow::Error>(())
            }
            .await;
            if let Err(error) = result {
                tracing::warn!(model = name, %error, "native asset prefetch failed; requests may retry");
            }
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_specs_preserve_projector_and_cli_device_precedence() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "cold"
[models.lifecycle]
load = "on_demand"
[models.hardware]
device = "uuid:configured"
[models.multimodal]
mmproj = "org/projector/projector.gguf"
"#,
        )
        .unwrap();
        let options = RuntimeOptions {
            device: Some("CPU".into()),
            ..Default::default()
        };
        let spec = startup_spec(&config, &options, &config.models[0]).unwrap();
        assert_eq!(spec.gpu_id.as_deref(), Some("CPU"));
        assert!(spec.cli_device_override);
        assert!(spec.resolve_pinned_gpu);
        assert_eq!(
            spec.mmproj_ref.as_deref(),
            Some(Path::new("org/projector/projector.gguf"))
        );
        let persisted =
            startup_spec(&config, &RuntimeOptions::default(), &config.models[0]).unwrap();
        assert_eq!(persisted.gpu_id.as_deref(), Some("uuid:configured"));
    }

    #[test]
    fn cold_local_paths_reject_relative_and_symlink_sources_before_resolution() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.gguf");
        std::fs::write(&file, b"GGUF").unwrap();
        let mut config = MeshConfig::default();
        let mut model = ModelConfigEntry {
            model: "cold".into(),
            lifecycle: ModelLifecycle {
                load: ModelLoadPolicy::OnDemand,
                ..Default::default()
            },
            ..Default::default()
        };
        model.hardware = Some(crate::plugin::HardwareConfig {
            model_path: Some("relative.gguf".into()),
            ..Default::default()
        });
        config.models.push(model);
        assert!(validate_managed_sources(&config).is_err());
        config.models[0].hardware.as_mut().unwrap().model_path = Some(file.display().to_string());
        validate_managed_sources(&config).unwrap();
        #[cfg(unix)]
        {
            let link = dir.path().join("link.gguf");
            std::os::unix::fs::symlink(&file, &link).unwrap();
            config.models[0].hardware.as_mut().unwrap().model_path =
                Some(link.display().to_string());
            assert!(validate_managed_sources(&config).is_err());
        }
    }

    #[test]
    fn cold_alias_and_cached_public_id_are_bound_before_loading_and_split_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.gguf");
        std::fs::write(&file, b"GGUF").unwrap();
        let mut config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "cold"
[models.lifecycle]
load = "on_demand"
[models.advanced.server]
alias = "public-alias"
"#,
        )
        .unwrap();
        config.models[0].hardware = Some(crate::plugin::HardwareConfig {
            model_path: Some(file.display().to_string()),
            ..Default::default()
        });
        let names = model_names(&config, &config.models[0]);
        assert!(names.contains(&"cold".to_owned()));
        assert!(names.contains(&"public-alias".to_owned()));
        assert!(names.contains(&crate::models::model_ref_for_path(&file)));
        let options = RuntimeOptions {
            split: true,
            ..Default::default()
        };
        assert!(validate_split(&config, &options).is_err());
        assert!(startup_models::build_startup_model_specs(&options, &config).is_err());
        let explicit = RuntimeOptions {
            model: vec!["explicit-model".into()],
            ..options
        };
        validate_split(&config, &explicit).unwrap();
        assert_eq!(
            startup_models::build_startup_model_specs(&explicit, &config)
                .unwrap()
                .len(),
            1
        );
    }
}
