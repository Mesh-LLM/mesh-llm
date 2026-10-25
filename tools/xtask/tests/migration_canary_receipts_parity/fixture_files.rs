use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) struct TestRoot(pub(super) PathBuf);

impl TestRoot {
    pub(super) fn new() -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "canary-receipts-parity-{}-{nanos}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove parity test-owned fixture only");
    }
}

pub(super) fn write_artifacts(package: &Path) -> Result<(), Box<dyn Error>> {
    let artifacts: [(&str, &[u8]); 5] = [
        ("binaries.tar", b"synthetic binaries artifact\n"),
        ("workload-oracles.tar", b"synthetic workload artifact\n"),
        ("llama-source.bundle", b"synthetic source bundle\n"),
        ("llama-source.json", b"synthetic source provenance\n"),
        ("upstream-summary.md", b"synthetic source summary\n"),
    ];
    for (name, bytes) in artifacts {
        fs::write(package.join(name), bytes)?;
    }
    Ok(())
}

pub(super) fn artifact_digest(package: &Path, name: &str) -> Result<String, Box<dyn Error>> {
    Ok(hash_bytes(&fs::read(package.join(name))?))
}

pub(super) fn source_identity(package: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let identity = json!({
        "schema": 3,
        "candidate": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "base": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "branch": "llama-canary/repair-123-2-aaaaaaaaaa",
        "pass_id": "repair-1",
        "run_id": "123",
        "run_attempt": "2",
        "platform": "macos-arm64-metal",
        "controller": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "mesh_source": "",
        "plan_sha256": artifact_digest(package, "plan.json")?,
        "manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "binaries_sha256": artifact_digest(package, "binaries.tar")?,
        "workload_oracles_sha256": artifact_digest(package, "workload-oracles.tar")?,
        "llama_bundle_sha256": artifact_digest(package, "llama-source.bundle")?,
        "llama_provenance_sha256": artifact_digest(package, "llama-source.json")?,
        "summary_sha256": artifact_digest(package, "upstream-summary.md")?,
        "bundle_sha256": null
    });
    let mut bytes = serde_json::to_vec_pretty(&identity)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn write_worker(
    evidence: &Path,
    directory: &str,
    attempt: &str,
    outcome: &str,
    results: &[u8],
    identity_sha256: &str,
) -> Result<(), Box<dyn Error>> {
    let worker = evidence.join(directory);
    fs::create_dir(&worker)?;
    fs::write(worker.join("results.jsonl"), results)?;
    let receipt = json!({
        "candidate": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "family": if directory == "new-dense" { "dense" } else { directory },
        "identity_sha256": identity_sha256,
        "outcome": outcome,
        "pass_id": "repair-1",
        "results_sha256": hash_bytes(results),
        "run_attempt": attempt,
        "run_id": "123",
        "runner": "fixture"
    });
    fs::write(worker.join("receipt.json"), serde_json::to_vec(&receipt)?)?;
    Ok(())
}

pub(super) fn case_paths(package: &Path, evidence: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut files = fs::read_dir(package)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    for entry in fs::read_dir(evidence)? {
        let worker = entry?.path();
        files.extend([worker.join("receipt.json"), worker.join("results.jsonl")]);
    }
    files.sort();
    Ok(files)
}

pub(super) fn hash_paths(paths: Vec<PathBuf>) -> Result<Vec<(PathBuf, String)>, Box<dyn Error>> {
    paths
        .into_iter()
        .map(|path| Ok((path.clone(), hash_bytes(&fs::read(path)?))))
        .collect()
}

pub(super) fn hash_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
