use crate::crypto::{
    LoadedEmbeddedReleaseAttestation, ReleaseSignerTrustStore,
    load_embedded_release_attestation_for_binary,
};
use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub(crate) struct LoadedReleaseAttestation {
    pub(crate) binary_path: std::path::PathBuf,
    pub(crate) summary: crate::ReleaseAttestationSummary,
    pub(crate) attestation: Option<crate::ReleaseBuildAttestation>,
}

/// Verify the running executable's embedded release proof using the default trust store.
pub(crate) fn load_for_current_binary() -> Result<LoadedReleaseAttestation> {
    let binary_path =
        std::env::current_exe().context("failed to determine mesh-llm binary path")?;
    load_for_binary_path(&binary_path, &ReleaseSignerTrustStore::default())
}

/// Install the verified binary proof before the node accepts or joins peers.
/// Missing or unreadable proofs leave admission to the mesh's genesis policy.
pub(super) async fn attach_to_node(
    node: &crate::mesh::Node,
    loaded: Result<LoadedReleaseAttestation>,
) {
    let loaded = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            tracing::warn!(
                error = %format!("{error:#}"),
                "failed to load local embedded release attestation; continuing without advertising one"
            );
            return;
        }
    };
    log_loaded_attestation(&loaded);
    node.set_release_attestation_report(loaded.summary, loaded.attestation)
        .await;
}

/// Log whether a binary's proof is missing, invalid or successfully verified.
fn log_loaded_attestation(loaded: &LoadedReleaseAttestation) {
    match loaded.summary.status {
        crate::ReleaseAttestationStatus::Missing => tracing::info!(
            path = %loaded.binary_path.display(),
            "no embedded release attestation found for local binary"
        ),
        crate::ReleaseAttestationStatus::Invalid => tracing::warn!(
            path = %loaded.binary_path.display(),
            error = %loaded.summary.error.as_deref().unwrap_or("unknown release attestation error"),
            "local binary has an invalid embedded release attestation; continuing without advertising one"
        ),
        crate::ReleaseAttestationStatus::Valid => log_verified_attestation(loaded),
    }
}

/// Log verified signer and proof identity, warning if the expected payload is absent.
fn log_verified_attestation(loaded: &LoadedReleaseAttestation) {
    let Some(attestation) = &loaded.attestation else {
        tracing::warn!(
            path = %loaded.binary_path.display(),
            "embedded release attestation verified but no release attestation payload was produced"
        );
        return;
    };
    if loaded.summary.verified {
        let attestation_hash = attestation.canonical_hash_hex().ok();
        tracing::info!(
            path = %loaded.binary_path.display(),
            signer_key_id = %attestation.signer_key_id,
            attestation_hash = attestation_hash.as_deref().unwrap_or("unknown"),
            "loaded local embedded release attestation"
        );
    }
}

#[cfg(test)]
pub(crate) fn assert_release_attestation_reports_missing_for_unstamped_binary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let binary_path = dir.path().join("mesh-llm");
    std::fs::write(&binary_path, b"plain-binary").expect("write binary");

    let loaded = load_for_binary_path(&binary_path, &ReleaseSignerTrustStore::default())
        .expect("load release attestation");

    assert_eq!(
        loaded.summary.status,
        crate::ReleaseAttestationStatus::Missing
    );
    assert!(loaded.attestation.is_none());
}

/// Load and verify a binary's embedded proof against the supplied signer trust store.
/// Preserve missing/invalid proof status and contextualize errors reading the binary.
pub(super) fn load_for_binary_path(
    binary_path: &std::path::Path,
    trust_store: &ReleaseSignerTrustStore,
) -> Result<LoadedReleaseAttestation> {
    let LoadedEmbeddedReleaseAttestation {
        binary_path,
        summary,
        attestation,
    } = load_embedded_release_attestation_for_binary(binary_path, trust_store)
        .map_err(anyhow::Error::from)
        .with_context(|| {
            format!(
                "failed to verify embedded release attestation for {}",
                binary_path.display()
            )
        })?;
    Ok(LoadedReleaseAttestation {
        binary_path,
        summary,
        attestation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReleaseAttestationStatus;
    use crate::crypto::release_attestation::tests::{
        stamped_binary_bytes, test_release_signing_key,
    };

    #[test]
    fn load_for_binary_path_reports_missing_when_no_footer_exists() {
        assert_release_attestation_reports_missing_for_unstamped_binary();
    }

    #[test]
    fn load_for_binary_path_reads_embedded_attestation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let binary_path = dir.path().join("mesh-llm");
        let signing_key = test_release_signing_key(8);
        std::fs::write(&binary_path, stamped_binary_bytes(&signing_key)).expect("write binary");

        let loaded = load_for_binary_path(&binary_path, &ReleaseSignerTrustStore::default())
            .expect("load release attestation");

        assert_eq!(loaded.summary.status, ReleaseAttestationStatus::Valid);
        loaded
            .attestation
            .expect("embedded attestation")
            .verify()
            .expect("embedded attestation should verify as canonical protocol attestation");
    }

    #[tokio::test]
    /// Preserve the root I/O failure in diagnostics when the binary proof cannot be read.
    async fn attach_to_node_logs_binary_read_error_chain() {
        use tracing::instrument::WithSubscriber;

        let directory = tempfile::tempdir().expect("tempdir");
        let binary_path = directory.path().join("missing-binary");
        let error = load_for_binary_path(&binary_path, &ReleaseSignerTrustStore::default())
            .expect_err("missing binary must fail to load");
        let source_error = error.root_cause().to_string();
        let log = tempfile::NamedTempFile::new().expect("log file");
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_ansi(false)
            .with_writer(log.reopen().expect("log writer"))
            .finish();
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
            .await
            .expect("test node");

        attach_to_node(&node, Err(error))
            .with_subscriber(subscriber)
            .await;
        node.close_endpoint().await;

        let record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(log.path()).expect("read log"))
                .expect("single structured warning");
        assert_eq!(record["level"], "WARN");
        assert_eq!(
            record["fields"]["message"],
            "failed to load local embedded release attestation; continuing without advertising one"
        );
        let logged_error = record["fields"]["error"].as_str().expect("error field");
        assert!(logged_error.contains("failed to verify embedded release attestation"));
        assert!(logged_error.contains(&source_error), "{logged_error}");
    }
}
