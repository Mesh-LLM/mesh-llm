use super::package::CaseFixture;
use crate::canary_receipts::{Digest, ErrorKind};
use std::{error::Error, fs};

#[test]
fn canary_package_admission_accepts_source_derived_package_when_hashes_match()
-> Result<(), Box<dyn Error>> {
    let given = CaseFixture::complete()?;
    let when = given.reverify_package();
    assert!(when.is_ok());
    Ok(())
}

#[test]
fn canary_package_admission_rejects_artifact_when_bytes_change() -> Result<(), Box<dyn Error>> {
    let given = CaseFixture::complete()?;
    fs::write(given.package().join("binaries.tar"), b"tampered")?;
    let when = given.reverify_package();
    assert_eq!(
        when.err().map(|error| error.kind),
        Some(ErrorKind::PackageArtifact)
    );
    Ok(())
}

#[test]
fn canary_package_admission_rejects_artifact_when_required_file_is_missing()
-> Result<(), Box<dyn Error>> {
    let given = CaseFixture::complete()?;
    fs::remove_file(given.package().join("workload-oracles.tar"))?;
    let when = given.reverify_package();
    assert_eq!(when.err().map(|error| error.kind), Some(ErrorKind::Io));
    Ok(())
}

#[test]
fn canary_package_admission_rejects_package_when_identity_digest_differs()
-> Result<(), Box<dyn Error>> {
    let given = CaseFixture::complete()?;
    let expected: Digest = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        .to_owned()
        .try_into()?;
    let when = given.verify_package_with_identity(expected);
    assert_eq!(
        when.err().map(|error| error.kind),
        Some(ErrorKind::PackageIdentity)
    );
    Ok(())
}

#[test]
fn canary_package_admission_rejects_null_mesh_source_when_none_was_selected()
-> Result<(), Box<dyn Error>> {
    let given = CaseFixture::complete()?;
    let identity_path = given.package().join("identity.json");
    let mut identity: serde_json::Value = serde_json::from_slice(&fs::read(&identity_path)?)?;
    identity["mesh_source"] = serde_json::Value::Null;
    let identity_bytes = serde_json::to_vec(&identity)?;
    fs::write(identity_path, &identity_bytes)?;
    let expected_identity = Digest::of_bytes(&identity_bytes);

    let when = given.verify_package_with_identity(expected_identity);

    assert_eq!(
        when.err().map(|error| error.kind),
        Some(ErrorKind::PackageIdentity)
    );
    Ok(())
}
