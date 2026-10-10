use super::{TestResult, root, run};
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
const SOURCE_COMMIT: &str = "a12b535d7c4e1f09b8d3427a66c5e0f19d8a7b34";

struct IsolatedVerifier(PathBuf);

impl IsolatedVerifier {
    fn new() -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "xtask-copied-verifier-{}-{}-{nanos}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        fs::copy(
            env!("CARGO_BIN_EXE_xtask"),
            path.join("release-attestation-verifier"),
        )?;
        Ok(Self(path))
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(Command::new(self.0.join("release-attestation-verifier"))
            .current_dir(&self.0)
            .args(args)
            .output()?)
    }

    fn compare(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        let direct = run(&self.0, args)?;
        let stamped_bytes = if args.contains(&"stamp") && direct.status.success() {
            let path = args
                .windows(2)
                .find(|pair| pair[0] == "--binary")
                .ok_or("stamp missing --binary")?[1];
            Some((path, fs::read(path)?))
        } else {
            None
        };
        let copied = self.run(args)?;
        assert_eq!(direct.status, copied.status);
        assert_eq!(direct.stdout, copied.stdout);
        assert_eq!(direct.stderr, copied.stderr);
        if let Some((path, bytes)) = stamped_bytes {
            assert_eq!(bytes, fs::read(path)?);
        }
        Ok(copied)
    }

    fn keys(&self) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
        let private = self.0.join("signing key.json");
        let public = self.0.join("public key.json");
        let result = self.run(&[
            "release-attestation",
            "generate-keypair",
            "--private-key-out",
            private.to_str().ok_or("non-UTF8 private key")?,
            "--public-key-out",
            public.to_str().ok_or("non-UTF8 public key")?,
        ])?;
        assert_eq!(
            result.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        Ok((private, public))
    }
}

impl Drop for IsolatedVerifier {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn migration_cli_copied_verifier_inspects_missing_footer_without_checkout() -> TestResult {
    // Given: a copied verifier and an unstamped host outside the checkout.
    let isolated = IsolatedVerifier::new()?;
    let host = isolated.0.join("host.exe");
    fs::write(&host, b"unstamped host")?;
    // When: both verifiers inspect the host from the isolated directory.
    let result = isolated.compare(&[
        "release-attestation",
        "inspect",
        "--binary",
        host.to_str().ok_or("non-UTF8 host")?,
        "--json",
    ])?;
    // Then: the complete consumed JSON and error channel agree.
    assert_eq!(result.status.code(), Some(0));
    assert!(result.stderr.is_empty());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&result.stdout)?,
        serde_json::json!({"status":"missing","version":null,"signer_key_id":null,"artifact_digest":null,"error":null})
    );
    Ok(())
}

#[test]
fn migration_cli_copied_verifier_rejects_missing_stamp_input_without_checkout() -> TestResult {
    // Given: valid signing keys but no host input in an isolated directory.
    let isolated = IsolatedVerifier::new()?;
    let (key, _) = isolated.keys()?;
    let absent = isolated.0.join("absent host.exe");
    // When: stamp is given an explicit version and a missing host.
    let result = isolated.compare(&[
        "release-attestation",
        "stamp",
        "--binary",
        absent.to_str().ok_or("non-UTF8 host")?,
        "--signing-key-file",
        key.to_str().ok_or("non-UTF8 key")?,
        "--node-version",
        "1.0.0",
    ])?;
    // Then: both fail on the missing host rather than trying to discover a checkout.
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
    assert!(!String::from_utf8_lossy(&result.stderr).contains("repo root"));
    Ok(())
}

#[test]
fn migration_cli_copied_verifier_stamps_and_inspects_unicode_host() -> TestResult {
    // Given: the action's copied verifier, host path with spaces, and signing keys.
    let isolated = IsolatedVerifier::new()?;
    let host = isolated.0.join("höst input with spaces.exe");
    let (key, public) = isolated.keys()?;
    fs::write(&host, b"standalone host fixture")?;
    let host = host.to_str().ok_or("non-UTF8 host")?;
    let key = key.to_str().ok_or("non-UTF8 key")?;
    let public = public.to_str().ok_or("non-UTF8 public key")?;
    // When: both verifiers stamp with the Windows action's source-commit requirement.
    let stamped = isolated.compare(&[
        "release-attestation",
        "stamp",
        "--binary",
        host,
        "--signing-key-file",
        key,
        "--require-source-commit",
        "--commit",
        SOURCE_COMMIT,
        "--node-version",
        "1.2.3",
    ])?;
    // Then: stamp JSON agrees byte-for-byte and the footer inspects as valid.
    assert_eq!(stamped.status.code(), Some(0));
    assert!(stamped.stderr.is_empty());
    let stamp: serde_json::Value = serde_json::from_slice(&stamped.stdout)?;
    assert_eq!(stamp["node_version"], "1.2.3");
    assert_eq!(stamp["binary"], host);
    let inspected = isolated.compare(&[
        "release-attestation",
        "inspect",
        "--binary",
        host,
        "--public-key-file",
        public,
        "--json",
    ])?;
    assert_eq!(inspected.status.code(), Some(0));
    assert!(inspected.stderr.is_empty());
    let summary: serde_json::Value = serde_json::from_slice(&inspected.stdout)?;
    assert_eq!(summary["status"], "valid");
    assert_eq!(summary["version"], 1);
    assert_eq!(summary["artifact_digest"], stamp["artifact_digest"]);
    Ok(())
}

#[test]
fn migration_cli_copied_verifier_explicit_root_supplies_default_version() -> TestResult {
    // Given: a copied verifier, keys and an unstamped host outside the checkout.
    let isolated = IsolatedVerifier::new()?;
    let host = isolated.0.join("host.exe");
    let (key, public) = isolated.keys()?;
    fs::write(&host, b"host bytes")?;
    let host = host.to_str().ok_or("non-UTF8 host")?;
    let key = key.to_str().ok_or("non-UTF8 key")?;
    let public = public.to_str().ok_or("non-UTF8 public key")?;
    let checkout = root();
    // When: stamp resolves the default version from an explicit repository root.
    let stamped = isolated.compare(&[
        "--repo-root",
        checkout.to_str().ok_or("non-UTF8 checkout")?,
        "release-attestation",
        "stamp",
        "--binary",
        host,
        "--signing-key-file",
        key,
        "--require-source-commit",
        "--commit",
        SOURCE_COMMIT,
    ])?;
    // Then: the recorded version equals the checkout's workspace version.
    assert_eq!(stamped.status.code(), Some(0));
    let result: serde_json::Value = serde_json::from_slice(&stamped.stdout)?;
    let manifest = fs::read_to_string(checkout.join("Cargo.toml"))?;
    let version = manifest
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("version = \"")
                .and_then(|value| value.strip_suffix('"'))
        })
        .ok_or("workspace version missing")?;
    assert_eq!(result["node_version"], version);
    let inspected = isolated.compare(&[
        "release-attestation",
        "inspect",
        "--binary",
        host,
        "--public-key-file",
        public,
        "--json",
    ])?;
    assert_eq!(inspected.status.code(), Some(0));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&inspected.stdout)?["status"],
        "valid"
    );
    Ok(())
}
