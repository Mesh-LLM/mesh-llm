//! Bind cache construction to proof of the runtime's loaded model inputs.
use super::*;

impl KvStageIntegration {
    /// Construct durable reuse only from proof of the runtime's actual inputs.
    /// Raw StageConfig constructors retain in-memory caching without disk reuse.
    pub fn from_runtime_with_l3_manager(
        config: &StageConfig,
        runtime: &Arc<Mutex<crate::runtime_state::RuntimeState>>,
        manager: Option<L3CacheManager>,
        observer: Option<Arc<dyn KvLifecycleObserver>>,
    ) -> Result<Option<Self>> {
        let (kind, memory, content) = {
            let runtime = crate::runtime_state::panic_recovery::lock_runtime(runtime);
            let capability = runtime.model.capability();
            (
                capability.as_ref().map(|value| value.state_kind),
                capability.as_ref().map(|value| value.cache_capabilities),
                runtime.verified_model_content.clone(),
            )
        };
        // Verification stays outside the runtime lock. Unix can reuse the
        // load-time digest through a strong file fingerprint; other platforms
        // recheck bytes when equivalent identity/change-time evidence is absent.
        let content = content.filter(|proof| proof.still_matches(config));
        Self::from_loaded_model_with_l3(config, kind, memory, content, || Ok(manager), observer)
    }

    #[cfg(test)]
    pub(crate) fn from_test_model_with_l3_manager(
        config: &StageConfig,
        model_state_kind: Option<ModelStateKind>,
        memory_cache: Option<skippy_runtime::MemoryCacheCapabilities>,
        manager: Option<L3CacheManager>,
        observer: Option<Arc<dyn KvLifecycleObserver>>,
    ) -> Result<Option<Self>> {
        // Unit fixtures have no native model. Supply actual deterministic file
        // content through the same verifier rather than authorizing a declared
        // digest or creating a production escape hatch.
        let root = tempfile::tempdir()?;
        let path = root.path().join("model.gguf");
        std::fs::write(&path, config.model_id.as_bytes())?;
        let mut proof_config = config.clone();
        proof_config.model_path = Some(path.to_string_lossy().into_owned());
        proof_config.model_part_paths.clear();
        proof_config.projector_path = None;
        proof_config.checkpoint_imatrix = None;
        let content =
            crate::runtime_state::model_content::VerifiedModelContent::observe(&proof_config)?
                .map(Arc::new);
        Self::from_loaded_model_with_l3(
            config,
            model_state_kind,
            memory_cache,
            content,
            || Ok(manager),
            observer,
        )
    }
}
