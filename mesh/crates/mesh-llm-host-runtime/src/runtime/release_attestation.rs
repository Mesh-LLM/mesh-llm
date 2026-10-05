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

pub(crate) fn load_for_current_binary() -> Result<LoadedReleaseAttestation> {
    let binary_path =
        std::env::current_exe().context("failed to determine mesh-llm binary path")?;
    load_for_binary_path(&binary_path, &ReleaseSignerTrustStore::default())
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

fn load_for_binary_path(
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
}
