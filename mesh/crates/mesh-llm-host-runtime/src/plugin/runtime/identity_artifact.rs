//! Bind software identity to the configured executable, never an unused package.

use std::path::{Path, PathBuf};

use anyhow::Result;

use super::super::config::ExternalPluginSpec;

pub(super) fn installed_executable(spec: &ExternalPluginSpec) -> Option<PathBuf> {
    let installed = spec.installed_metadata.as_ref()?.executable_path();
    // Command::new receives this immutable spec.command literally. Do not
    // resolve wrappers, PATH aliases, or executable names supplied as arguments.
    (Path::new(&spec.command) == installed).then_some(installed)
}

pub(super) async fn capture_startup_digest(spec: &ExternalPluginSpec) -> Result<Option<String>> {
    if !spec
        .openai_exchange_grant
        .as_ref()
        .is_some_and(|grant| grant.read_identity_bundle || grant.delegate_signing_key)
    {
        return Ok(None);
    }
    let Some(executable) = installed_executable(spec) else {
        return Ok(None);
    };
    super::super::identity_registration::artifact_sha256(&executable)
        .await
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn installed_spec(root: &Path) -> ExternalPluginSpec {
        let metadata = mesh_llm_plugin_manager::InstalledPluginMetadata {
            name: "observer".into(),
            source_repository: "local:fixture".into(),
            installed_version: "1.0.0".into(),
            target_triple: "fixture".into(),
            downloaded_asset_name: "observer.tar.gz".into(),
            install_path: root.into(),
            enabled: true,
            default_managed: false,
            manifest: None,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        };
        ExternalPluginSpec {
            name: metadata.name.clone(),
            command: metadata.executable_path().to_str().unwrap().into(),
            args: vec!["--identity".into()],
            url: None,
            env: BTreeMap::new(),
            startup: Default::default(),
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            installed_metadata: Some(metadata),
            openai_exchange_grant: Some(Box::new(mesh_llm_config::OpenAiExchangeGrant {
                read_identity_bundle: true,
                ..Default::default()
            })),
        }
    }

    #[tokio::test]
    async fn startup_identity_captures_only_the_executable_actually_configured() {
        let root = tempfile::tempdir().unwrap();
        let mut spec = installed_spec(root.path());
        let executable = installed_executable(&spec).unwrap();
        tokio::fs::write(&executable, b"installed executable")
            .await
            .unwrap();
        assert_eq!(
            capture_startup_digest(&spec).await.unwrap(),
            Some(hex::encode(Sha256::digest(b"installed executable")))
        );
        let plugin = super::super::tests::plugin_for_spec(spec.clone());
        let command = plugin.configured_child_command("fixture", "local");
        assert_eq!(command.as_std().get_program(), executable.as_os_str());
        assert!(command.as_std().get_args().any(|arg| arg == "--identity"));

        // A wrapper with the package executable among its arguments must not
        // receive that package's identity, even if the package later disappears.
        spec.command = root.path().join("wrapper").to_str().unwrap().into();
        spec.args.push(executable.to_str().unwrap().into());
        tokio::fs::remove_file(&executable).await.unwrap();
        assert!(installed_executable(&spec).is_none());
        assert_eq!(capture_startup_digest(&spec).await.unwrap(), None);
        spec.command.clear();
        assert_eq!(capture_startup_digest(&spec).await.unwrap(), None);
    }

    #[tokio::test]
    async fn overridden_command_cannot_borrow_identity_before_manifest_or_artifact_work() {
        let root = tempfile::tempdir().unwrap();
        let mut spec = installed_spec(root.path());
        spec.command = "different-plugin".into();
        let mut manager = super::super::super::PluginManager::for_test_summaries(Vec::new());
        Arc::get_mut(&mut manager.inner).unwrap().plugins.insert(
            spec.name.clone(),
            super::super::tests::plugin_for_spec(spec),
        );
        let error = manager
            .identity_grants_and_artifact("observer")
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("configured command to match the installed executable")
        );
    }
}
