use skippy_runtime::ModelStateKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ModelKvCapability {
    KnownDense,
    KnownRecurrent,
    Unknown(String),
}

/// Translate the authoritative descriptor captured from the loaded native
/// model into the cache payload families understood by the server.
///
/// Model names, repository paths, and tensor-name heuristics are deliberately
/// excluded: llama.cpp has already resolved the actual architecture by this
/// point, including hybrid/recurrent state that is not visible in a package
/// name.
pub(super) fn loaded_model_kv_capability(state_kind: Option<ModelStateKind>) -> ModelKvCapability {
    match state_kind {
        Some(ModelStateKind::Dense) => ModelKvCapability::KnownDense,
        Some(ModelStateKind::Recurrent | ModelStateKind::Hybrid) => {
            ModelKvCapability::KnownRecurrent
        }
        Some(ModelStateKind::Diffusion) => ModelKvCapability::Unknown(
            "loaded diffusion model has no causal KV state to cache".to_string(),
        ),
        None => ModelKvCapability::Unknown(
            "loaded model capability descriptor is unavailable".to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_model_state_is_the_only_kv_authority() {
        for (state_kind, expected) in [
            (ModelStateKind::Dense, ModelKvCapability::KnownDense),
            (ModelStateKind::Recurrent, ModelKvCapability::KnownRecurrent),
            (ModelStateKind::Hybrid, ModelKvCapability::KnownRecurrent),
        ] {
            assert_eq!(loaded_model_kv_capability(Some(state_kind)), expected);
        }
    }

    #[test]
    fn unsupported_or_missing_loaded_descriptor_fails_closed() {
        assert!(matches!(
            loaded_model_kv_capability(Some(ModelStateKind::Diffusion)),
            ModelKvCapability::Unknown(_)
        ));
        assert!(matches!(
            loaded_model_kv_capability(None),
            ModelKvCapability::Unknown(_)
        ));
    }
}
