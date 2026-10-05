// Copyright 2024 mesh-llm contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Daemon startup sequence for runtime initialization.
//!
//! This module handles the ordered startup of mesh components, mode resolution,
//! and failure policies during daemon initialization.

use crate::runtime::{RuntimeSurface, options::RuntimeOptions};
use mesh_llm_config::RuntimeMode;

/// Resolve effective runtime mode with priority: client > config > default serve
pub(super) fn resolve_effective_mode(
    options: &RuntimeOptions,
    configured_mode: RuntimeMode,
) -> RuntimeMode {
    if options.client {
        RuntimeMode::Client
    } else {
        configured_mode
    }
}

/// Check for conflicting flags and return error if found
pub(super) fn check_mode_conflicts(
    options: &RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    configured_mode: RuntimeMode,
) -> Result<(), String> {
    let has_explicit_model =
        !options.model.is_empty() || !options.gguf.is_empty() || options.mmproj.is_some();
    if options.client && has_explicit_model {
        return Err("client mode cannot be combined with --model, --gguf, or --mmproj".to_string());
    }
    if configured_mode == RuntimeMode::Client
        && (explicit_surface == Some(RuntimeSurface::Serve) || has_explicit_model)
    {
        return Err(
            "persisted runtime.mode is 'client', which cannot be overridden by serve or model \
             flags; change [runtime].mode to 'serve' or 'on_demand', or remove the conflicting \
             serve/model arguments"
                .to_string(),
        );
    }

    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::field_reassign_with_default,
    reason = "tests vary individual RuntimeOptions fields to keep each conflict scenario explicit"
)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_resolve_effective_mode_priority() {
        let options = RuntimeOptions::default();
        assert_eq!(
            resolve_effective_mode(&options, RuntimeMode::OnDemand),
            RuntimeMode::OnDemand
        );

        let mut client_options = RuntimeOptions::default();
        client_options.client = true;
        assert_eq!(
            resolve_effective_mode(&client_options, RuntimeMode::OnDemand),
            RuntimeMode::Client
        );
    }

    #[test]
    fn test_check_mode_conflicts_client_with_model() {
        let mut options = RuntimeOptions::default();
        options.client = true;
        options.model.push(PathBuf::from("test.gguf"));

        let result =
            check_mode_conflicts(&options, Some(RuntimeSurface::Client), RuntimeMode::Serve);

        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("client mode cannot be combined"));
    }

    #[test]
    fn persisted_client_rejects_explicit_serve_and_model_flags() {
        let options = RuntimeOptions::default();
        let error =
            check_mode_conflicts(&options, Some(RuntimeSurface::Serve), RuntimeMode::Client)
                .expect_err("explicit serve must not override persisted client mode");
        assert!(error.contains("change [runtime].mode"));

        let mut model_options = RuntimeOptions::default();
        model_options.model.push(PathBuf::from("test.gguf"));
        assert!(
            check_mode_conflicts(&model_options, None, RuntimeMode::Client).is_err(),
            "model flags must not override persisted client mode"
        );
    }
}
