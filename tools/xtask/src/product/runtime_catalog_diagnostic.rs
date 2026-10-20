//! Prepare public-only diagnostic inputs from a real native runtime archive.

use crate::repository::check_report::CheckReport;
use std::path::Path;

pub(super) fn run(args: &[String]) -> CheckReport {
    match prepare(args) {
        Ok(()) => CheckReport::default(),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}

fn prepare(args: &[String]) -> Result<(), String> {
    if let [mode, output] = args
        && mode == "disposable-sign"
    {
        return disposable_sign(Path::new(output));
    }
    let [archive, release, output] = args else {
        return Err(
            "usage: product runtime-catalog-diagnostic <archive> <runtime-release> <output-dir>"
                .into(),
        );
    };
    let output = Path::new(output);
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let archive_out = output.join("runtime.tar.gz");
    std::fs::copy(archive, &archive_out).map_err(|e| e.to_string())?;
    let catalog = output.join("native-runtimes.json");
    let temp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let report = super::release_manifest::run(&[
        catalog.to_string_lossy().into_owned(),
        "Mesh-LLM/mesh-llm".into(),
        "diagnostic-not-a-release".into(),
        release.clone(),
        temp.path().to_string_lossy().into_owned(),
        archive_out.to_string_lossy().into_owned(),
    ]);
    if report.code != 0 {
        return Err(report.stderr);
    }
    let bytes = std::fs::read(&catalog).map_err(|e| e.to_string())?;
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    manifest["artifacts"][0]["url"] = "http://127.0.0.1:18764/runtime.tar.gz".into();
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    std::fs::write(&catalog, &bytes).map_err(|e| e.to_string())?;
    let digest = super::digest::file_sha256(&catalog).map_err(|e| e.error.to_string())?;
    std::fs::write(
        output.join("native-runtimes.json.sha256"),
        format!("{digest}  native-runtimes.json\n"),
    )
    .map_err(|e| e.to_string())
}

fn disposable_sign(output: &Path) -> Result<(), String> {
    let password = "disposable diagnostic password, not a credential";
    let pair = minisign::KeyPair::generate_encrypted_keypair(Some(password.into()))
        .map_err(|e| e.to_string())?;
    let secret = pair
        .sk
        .to_box(None)
        .map_err(|e| e.to_string())?
        .into_string();
    let public = output.join("disposable-test-key.pub");
    std::fs::write(
        &public,
        pair.pk.to_box().map_err(|e| e.to_string())?.into_string(),
    )
    .map_err(|e| e.to_string())?;
    super::runtime_catalog_signing::sign_catalog(
        &output.join("native-runtimes.json"),
        &output.join("native-runtimes.json.minisig"),
        &public,
        &secret,
        password,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_archive_and_invalid_arguments() {
        assert!(prepare(&[]).is_err());
        let temp = tempfile::tempdir().unwrap();
        assert!(
            prepare(&[
                "missing.tar.gz".into(),
                "1.0.0".into(),
                temp.path().display().to_string()
            ])
            .is_err()
        );
    }

    #[test]
    fn disposable_signing_retains_only_public_material() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("native-runtimes.json"), b"catalog").unwrap();
        disposable_sign(temp.path()).unwrap();
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 3);
        assert!(temp.path().join("native-runtimes.json.minisig").is_file());
    }
}
