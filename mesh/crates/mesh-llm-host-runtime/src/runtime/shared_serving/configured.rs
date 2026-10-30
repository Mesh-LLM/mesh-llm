//! Adapt local launch configuration to the shared model host behind `serve`.

use anyhow::{Context, Result, ensure};
use mesh_llm_config::memberships::MembershipsConfig;

use super::{ServingMembershipConfig, SharedServingConfig, run_shared_serving};
use crate::runtime::RuntimeOptions;

/// Validate all local launch inputs before native initialization or state writes.
pub(crate) fn load(options: &RuntimeOptions) -> Result<(SharedServingConfig, u16)> {
    let path = options
        .memberships_config
        .as_deref()
        .context("missing memberships config")?;
    ensure!(
        options.gguf.len() == 1,
        "configured memberships require exactly one local GGUF"
    );
    let manifest = MembershipsConfig::load(path)?;
    let config = SharedServingConfig {
        model_path: options.gguf[0].clone(),
        model_id: manifest.model_id,
        state_dir: manifest.state_dir,
        context_size: options.ctx_size.unwrap_or(manifest.context_size),
        gpu_layers: manifest.gpu_layers,
        concurrency: options.parallel.unwrap_or(manifest.concurrency),
        queue_capacity: manifest.queue_capacity,
        membership_queue_capacity: manifest.membership_queue_capacity,
        lan_only: manifest.lan_only,
        bind_ip: manifest.bind_ip,
        memberships: manifest
            .memberships
            .into_iter()
            .map(|member| ServingMembershipConfig {
                name: member.name,
                token_file: member.join_token_file,
                trust_store: member.trust_store,
                owner_key: member.owner_key,
            })
            .collect(),
    };
    config.validate()?;
    Ok((config, manifest.management_port))
}

pub(crate) async fn run(options: &RuntimeOptions) -> Result<()> {
    let (config, management_port) = load(options)?;
    run_shared_serving(config, management_port).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_membership_token_fails_before_state_creation() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.gguf");
        let manifest = directory.path().join("memberships.toml");
        std::fs::write(&model, b"GGUF").unwrap();
        std::fs::write(directory.path().join("work.token"), "SECRET_INVALID_TOKEN").unwrap();
        std::fs::write(&manifest, "version=1\nmodel_id='shared'\nstate_dir='state'\n[[memberships]]\nname='work'\njoin_token_file='work.token'").unwrap();
        let options = RuntimeOptions {
            gguf: vec![model],
            memberships_config: Some(manifest),
            ..RuntimeOptions::default()
        };
        let error = load(&options).unwrap_err().to_string();
        assert!(error.contains("invalid membership join token"));
        assert!(!error.contains("SECRET_INVALID_TOKEN"));
        assert!(!directory.path().join("state").exists());
    }
}
