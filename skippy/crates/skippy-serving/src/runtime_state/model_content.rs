//! Content proof captured once around model open, outside runtime locks.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use skippy_protocol::StageConfig;
use std::{
    fs::{File, Metadata},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::{Instant, SystemTime},
};

type ObservedFiles = Vec<(String, PathBuf, FileStamp)>;

#[derive(Clone)]
pub(crate) struct VerifiedModelContent {
    digest: String,
    files: ObservedFiles,
}

#[derive(Clone, PartialEq, Eq)]
struct FileStamp {
    bytes: u64,
    modified: Option<SystemTime>,
    created: Option<SystemTime>,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}

impl FileStamp {
    fn read(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Self {
            bytes: metadata.len(),
            modified: metadata.modified().ok(),
            created: metadata.created().ok(),
            #[cfg(unix)]
            identity: (
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ),
        }
    }
}

impl VerifiedModelContent {
    /// Unix captures strong file stamps before native open, then hashes on a
    /// scoped worker while the caller loads the model. Other platforms retain
    /// the sequential read so weak timestamps cannot authorize a raced load.
    pub(crate) fn verify_during_load<T>(
        config: &StageConfig,
        load: impl FnOnce() -> Result<T>,
    ) -> Result<(T, Option<Self>)> {
        #[cfg(unix)]
        let (model, content) = match Self::observe_files(config).ok().flatten() {
            Some(files) => verify_alongside_load(load, || Self::from_observed(files))?,
            None => (load()?, None),
        };
        #[cfg(not(unix))]
        let (model, content) = {
            let content = Self::observe(config).ok().flatten();
            (load()?, content)
        };
        Ok((model, content.filter(|proof| proof.still_matches(config))))
    }

    #[cfg(any(not(unix), test))]
    pub(crate) fn observe(config: &StageConfig) -> Result<Option<Self>> {
        Self::observe_files(config)?
            .map(Self::from_observed)
            .transpose()
    }

    fn observe_files(config: &StageConfig) -> Result<Option<ObservedFiles>> {
        if !crate::kv_integration::KvStageIntegration::cache_enabled_for_stage(config) {
            return Ok(None);
        }
        let files = source_files(config)?;
        if files.is_empty() {
            return Ok(None);
        }
        capture_files(files).map(Some)
    }

    #[cfg(any(not(unix), test))]
    fn from_files(files: Vec<(String, PathBuf)>) -> Result<Self> {
        Self::from_observed(capture_files(files)?)
    }

    fn from_observed(files: ObservedFiles) -> Result<Self> {
        let started = Instant::now();
        let mut verified_bytes = 0_u64;
        let mut digest = Sha256::new();
        digest.update(b"skippy-loaded-model-content-v1\0");
        let mut observed = Vec::with_capacity(files.len());
        let mut buffer = vec![0; 1024 * 1024];
        for (name, path, stamp) in files {
            let file = File::open(&path)
                .with_context(|| format!("open model content {}", path.display()))?;
            let before = file.metadata()?;
            ensure!(
                before.is_file(),
                "model content proof requires regular files"
            );
            ensure!(
                FileStamp::read(&before) == stamp,
                "model content changed before verification"
            );
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update(stamp.bytes.to_le_bytes());
            let mut reader = BufReader::new(file);
            loop {
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
                verified_bytes = verified_bytes.saturating_add(count as u64);
            }
            ensure!(
                FileStamp::read(&reader.get_ref().metadata()?) == stamp,
                "model content changed during verification"
            );
            observed.push((name, path, stamp));
        }
        let digest = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        tracing::info!(
            target: "skippy::cache",
            verified_bytes,
            file_count = observed.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "hashed model content"
        );
        Ok(Self {
            digest,
            files: observed,
        })
    }

    /// Reject a proof if model input selection or bytes changed while opening.
    pub(crate) fn still_matches(&self, config: &StageConfig) -> bool {
        let Ok(files) = source_files(config) else {
            return false;
        };
        let unchanged = files.len() == self.files.len()
            && files.iter().zip(&self.files).all(
                |((name, path), (expected_name, expected_path, stamp))| {
                    name == expected_name
                        && path == expected_path
                        && std::fs::metadata(path).is_ok_and(|metadata| {
                            metadata.is_file() && FileStamp::read(&metadata) == *stamp
                        })
                },
            );
        if !unchanged {
            return false;
        }
        #[cfg(unix)]
        {
            true
        }
        #[cfg(not(unix))]
        {
            // Size and last-write timestamps are not a strong file identity on
            // these platforms. Recheck actual content after model open and at
            // backend construction rather than trusting preserved timestamps.
            Self::from_files(files).is_ok_and(|current| current.digest == self.digest)
        }
    }

    pub(crate) fn bind(&self, identity: &str) -> String {
        let mut digest = blake3::Hasher::new();
        digest.update(b"skippy-cache-loaded-content-v1\0");
        digest.update(self.digest.as_bytes());
        digest.update(identity.as_bytes());
        format!("blake3:{}", digest.finalize().to_hex())
    }
}

fn capture_files(files: Vec<(String, PathBuf)>) -> Result<ObservedFiles> {
    files
        .into_iter()
        .map(|(name, path)| {
            let metadata = path.metadata()?;
            ensure!(
                metadata.is_file(),
                "model content proof requires regular files"
            );
            Ok((name, path, FileStamp::read(&metadata)))
        })
        .collect()
}

#[cfg(unix)]
fn verify_alongside_load<T>(
    load: impl FnOnce() -> Result<T>,
    verify: impl FnOnce() -> Result<VerifiedModelContent> + Send,
) -> Result<(T, Option<VerifiedModelContent>)> {
    std::thread::scope(|scope| {
        let verification = std::thread::Builder::new()
            .name("skippy-model-content".into())
            .spawn_scoped(scope, verify)
            .ok();
        let model = load();
        let content = verification
            .and_then(|verification| verification.join().ok())
            .and_then(Result::ok);
        model.map(|model| (model, content))
    })
}

fn source_files(config: &StageConfig) -> Result<Vec<(String, PathBuf)>> {
    let mut files = if config.model_part_paths.is_empty() {
        let Some(path) = config.model_path.as_deref().map(Path::new) else {
            return Ok(Vec::new());
        };
        if skippy_runtime::is_safetensors_checkpoint(path) {
            let root = if path.is_dir() {
                path
            } else {
                path.parent().context("checkpoint has no parent")?
            };
            checkpoint_files(root)?
        } else {
            skippy_runtime::gguf_shard_paths(path)?
                .into_iter()
                .enumerate()
                .map(|(index, path)| (format!("part-{index}"), path))
                .collect()
        }
    } else {
        config
            .model_part_paths
            .iter()
            .enumerate()
            .map(|(index, path)| (format!("part-{index}"), PathBuf::from(path)))
            .collect()
    };
    for (role, path) in [
        ("projector", config.projector_path.as_deref()),
        ("imatrix", config.checkpoint_imatrix.as_deref()),
    ] {
        if let Some(path) = path {
            files.push((role.into(), PathBuf::from(path)));
        }
    }
    files
        .into_iter()
        .map(|(role, path)| Ok((role, path.canonicalize()?)))
        .collect()
}

fn checkpoint_files(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut files = Vec::new();
    let mut pending = vec![(root.to_path_buf(), 0)];
    while let Some((directory, depth)) = pending.pop() {
        ensure!(depth < 16, "checkpoint content tree is too deep");
        for entry in std::fs::read_dir(directory)? {
            let path = entry?.path();
            let metadata = path.metadata()?;
            if metadata.is_dir() {
                pending.push((path, depth + 1));
            } else {
                ensure!(
                    metadata.is_file(),
                    "checkpoint content contains a special file"
                );
                files.push((
                    path.strip_prefix(root)?
                        .to_string_lossy()
                        .replace('\\', "/"),
                    path.canonicalize()?,
                ));
                ensure!(
                    files.len() <= 100_000,
                    "checkpoint content has too many files"
                );
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn model_open_and_content_verification_progress_concurrently() {
        use std::{sync::mpsc, time::Duration};
        let (started, observed_start) = mpsc::channel();
        let (opened, observed_open) = mpsc::channel();
        let (model, content) = verify_alongside_load(
            || {
                observed_start.recv_timeout(Duration::from_secs(5))?;
                opened.send(())?;
                Ok("loaded")
            },
            move || {
                started.send(())?;
                observed_open.recv_timeout(Duration::from_secs(5))?;
                Ok(VerifiedModelContent {
                    digest: "proof".into(),
                    files: Vec::new(),
                })
            },
        )
        .unwrap();
        assert_eq!(model, "loaded");
        assert_eq!(content.unwrap().digest, "proof");
    }

    #[test]
    fn content_changes_during_model_open_disable_persistent_reuse() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        std::fs::write(&path, b"old weights").unwrap();
        let config = StageConfig {
            model_path: Some(path.to_string_lossy().into_owned()),
            kv_cache: Some(
                serde_json::from_value(serde_json::json!({"mode":"lookup-record"})).unwrap(),
            ),
            ..StageConfig::default()
        };
        let (_, unchanged) = VerifiedModelContent::verify_during_load(&config, || Ok(())).unwrap();
        assert!(unchanged.is_some());
        let (model, proof) = VerifiedModelContent::verify_during_load(&config, || {
            std::fs::write(&path, b"new weights")?;
            Ok("loaded")
        })
        .unwrap();
        assert_eq!(model, "loaded");
        assert!(proof.is_none());
    }

    #[test]
    fn unchanged_declared_digest_cannot_hide_changed_weight_bytes() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        std::fs::write(&model, b"first weights").unwrap();
        let mut config = StageConfig {
            model_path: Some(model.to_string_lossy().into_owned()),
            source_model_sha256: Some("ab".repeat(32)),
            ..StageConfig::default()
        };
        let first = VerifiedModelContent::from_files(source_files(&config).unwrap()).unwrap();
        assert!(first.still_matches(&config));
        let modified = model.metadata().unwrap().modified().unwrap();
        std::fs::write(&model, b"other weights").unwrap();
        File::options()
            .write(true)
            .open(&model)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert!(!first.still_matches(&config));
        let second = VerifiedModelContent::from_files(source_files(&config).unwrap()).unwrap();
        assert_ne!(first.bind("same namespace"), second.bind("same namespace"));
        config.model_path = Some(
            root.path()
                .join("different.gguf")
                .to_string_lossy()
                .into_owned(),
        );
        assert!(!second.still_matches(&config));
    }

    #[test]
    fn content_binding_is_stable_across_relocation_and_includes_every_part() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let files = |root: &Path| {
            ["one", "two"]
                .into_iter()
                .enumerate()
                .map(|(index, name)| {
                    let path = root.join(name);
                    std::fs::write(&path, name).unwrap();
                    (format!("part-{index}"), path)
                })
                .collect()
        };
        let one = VerifiedModelContent::from_files(files(first.path())).unwrap();
        let relocated = VerifiedModelContent::from_files(files(second.path())).unwrap();
        assert_eq!(one.bind("namespace"), relocated.bind("namespace"));
        std::fs::write(second.path().join("two"), b"changed").unwrap();
        let changed = VerifiedModelContent::from_files(vec![
            ("part-0".into(), second.path().join("one")),
            ("part-1".into(), second.path().join("two")),
        ])
        .unwrap();
        assert_ne!(one.bind("namespace"), changed.bind("namespace"));
    }

    #[test]
    fn environment_enabled_cache_captures_model_proof() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "runtime_state::model_content::tests::environment_enabled_cache_captures_model_proof_fixture", "--test-threads=1"])
            .env("SKIPPY_CONTENT_PROOF_ENV_FIXTURE", "1")
            .env_remove("SKIPPY_KV_CACHE")
            .env("SKIPPY_PREFIX_CACHE", "on")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn environment_enabled_cache_captures_model_proof_fixture() {
        if std::env::var_os("SKIPPY_CONTENT_PROOF_ENV_FIXTURE").is_none() {
            return;
        }
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        std::fs::write(&model, b"actual model bytes").unwrap();
        let config = StageConfig {
            model_path: Some(model.to_string_lossy().into_owned()),
            kv_cache: None,
            ..StageConfig::default()
        };
        assert!(VerifiedModelContent::observe(&config).unwrap().is_some());
    }
}
