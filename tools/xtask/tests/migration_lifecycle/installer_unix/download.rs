//! Archive selection and integrity of local release fixtures.
use super::fixture::{FALLBACK, Fixture, PREFERRED, stderr, stdout};
use sha2::{Digest, Sha256};
use std::fs;

#[test]
fn installer_prefers_platform_archive_and_uses_bundle_only_when_missing() {
    for preferred in [true, false] {
        let fixture = Fixture::new();
        fixture.asset(FALLBACK, b"fallback archive\n");
        if preferred {
            fixture.asset(PREFERRED, b"platform archive\n");
        }
        let report = fixture.download();
        assert!(report.success(), "{report:?}");
        let name = if preferred { PREFERRED } else { FALLBACK };
        assert!(stdout(&report).contains(&format!("asset={name}\n")));
        assert!(stdout(&report).contains(&format!(
            "archive={}\n",
            fixture.root.join("download").join(name).display()
        )));
        assert_eq!(
            fs::read(fixture.root.join("download").join(name)).unwrap(),
            fs::read(fixture.root.join("assets").join(name)).unwrap()
        );
        assert_eq!(
            stdout(&report).contains("Using runtime-enabled mesh bundle fallback"),
            !preferred
        );
        if preferred {
            assert!(!fixture.root.join("download").join(FALLBACK).exists());
        }
    }
}

#[test]
fn installer_missing_release_shape_fails_without_success_identity() {
    let fixture = Fixture::new();
    let report = fixture.download();
    assert!(!report.success());
    assert!(stderr(&report).contains("could not download release archive"));
    assert!(!stdout(&report).contains("asset="));
    assert_eq!(
        fs::read_dir(fixture.root.join("download")).unwrap().count(),
        0
    );
}

#[test]
fn installer_rejects_bad_preferred_checksum_without_falling_back() {
    for sidecar in [None, Some("malformed"), Some("mismatched")] {
        let fixture = Fixture::new();
        fixture.asset(PREFERRED, b"platform archive\n");
        fixture.asset(FALLBACK, b"valid fallback archive\n");
        let path = fixture
            .root
            .join("assets")
            .join(format!("{PREFERRED}.sha256"));
        match sidecar {
            None => fs::remove_file(path).unwrap(),
            Some("malformed") => fs::write(path, "invalid digest\n").unwrap(),
            Some(_) => fs::write(path, format!("{}  {PREFERRED}\n", "0".repeat(64))).unwrap(),
        }
        let report = fixture.download();
        assert!(!report.success(), "{report:?}");
        assert!(!stdout(&report).contains("asset="));
        assert!(!fixture.root.join("download").join(FALLBACK).exists());
        assert!(stderr(&report).contains("checksum"));
    }
}

#[test]
fn installer_release_url_normalizes_one_trailing_slash_without_network_access() {
    let fixture = Fixture::new();
    let report = fixture.run(
        "RELEASE_URL_BASE=https://example.invalid/assets/\nrelease_url mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
    );
    assert!(report.success(), "{report:?}");
    assert_eq!(
        stdout(&report),
        "https://example.invalid/assets/mesh-llm-aarch64-unknown-linux-gnu.tar.gz\n"
    );
}

#[test]
fn installer_sidecar_accepts_uppercase_hash_and_rejects_missing_digest() {
    let fixture = Fixture::new();
    fixture.stub("awk", "#!/bin/bash\nexit 2\n");
    let sidecar = fixture.root.join("download/sidecar");
    let digest = hex::encode(Sha256::digest(b"fixture\n"));
    fs::write(
        &sidecar,
        format!("{}  archive.tar.gz\n", digest.to_uppercase()),
    )
    .unwrap();
    let command = "checksum_from_sidecar \"$FIXTURE_DOWNLOAD_DIR/sidecar\"";
    let report = fixture.run(command);
    assert!(report.success(), "{report:?}");
    assert_eq!(stdout(&report), format!("{digest}\n"));
    fs::write(sidecar, "invalid checksum\n").unwrap();
    let report = fixture.run(command);
    assert!(!report.success());
    assert!(stderr(&report).contains("did not contain a SHA-256 digest"));
}
