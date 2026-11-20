//! Content proof captured once around model open, outside runtime locks.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use skippy_protocol::StageConfig;
use std::{
    fs::{File, Metadata},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
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
    /// Volume serial number, 128-bit file ID and `ChangeTime`.
    #[cfg(windows)]
    identity: (u64, u128, i64),
}

impl FileStamp {
    /// Reads the stamp of an open file. Windows only exposes a file's identity
    /// and change time through a handle, hence the file and not just its
    /// metadata.
    fn read(file: &File, metadata: &Metadata) -> std::io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(not(windows))]
        let _ = file;
        Ok(Self {
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
            #[cfg(windows)]
            identity: windows_file_identity(file)?,
        })
    }

    fn of_path(path: &Path) -> std::io::Result<(Metadata, Self)> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let stamp = Self::read(&file, &metadata)?;
        Ok((metadata, stamp))
    }
}

/// std exposes no device, inode or change time on Windows, so read them from
/// the open handle: the volume serial number and 128-bit file ID (`FileIdInfo`,
/// which also covers ReFS) identify the file, and `ChangeTime` advances on any
/// content or metadata change, which is what lets a same-size rewrite with a
/// restored last-write time be told apart. Unlike Unix ctime, a process with
/// write-attributes access can set `ChangeTime`; like the rest of this proof
/// it gates reuse of a verified hash and is not a defence against a local
/// writer. `skippy-api`'s source registry reads the same identity.
#[cfg(windows)]
fn windows_file_identity(file: &File) -> std::io::Result<(u64, u128, i64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_BASIC_INFO, FILE_ID_INFO, FileBasicInfo, FileIdInfo, GetFileInformationByHandleEx,
    };

    let handle = file.as_raw_handle();
    let mut id = FILE_ID_INFO::default();
    let mut basic = FILE_BASIC_INFO::default();
    // SAFETY: `handle` belongs to `file`, which outlives both calls, and each
    // buffer is the structure its information class documents, passed with
    // that structure's exact size.
    let queried = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            (&raw mut id).cast(),
            size_of::<FILE_ID_INFO>() as u32,
        ) != 0
            && GetFileInformationByHandleEx(
                handle,
                FileBasicInfo,
                (&raw mut basic).cast(),
                size_of::<FILE_BASIC_INFO>() as u32,
            ) != 0
    };
    if !queried {
        return Err(std::io::Error::last_os_error());
    }
    Ok((
        id.VolumeSerialNumber,
        u128::from_le_bytes(id.FileId.Identifier),
        basic.ChangeTime,
    ))
}

impl VerifiedModelContent {
    /// Unix and Windows capture strong file stamps before native open, then
    /// hash on a scoped worker while the caller loads the model. Other
    /// platforms retain the sequential read so weak timestamps cannot
    /// authorize a raced load.
    pub(crate) fn verify_during_load<T>(
        config: &StageConfig,
        load: impl FnOnce() -> Result<T>,
    ) -> Result<(T, Option<Self>)> {
        #[cfg(any(unix, windows))]
        let (model, content) = match Self::observe_files(config).ok().flatten() {
            Some(files) => {
                verify_alongside_load(load, |cancelled| Self::from_observed(files, cancelled))?
            }
            None => (load()?, None),
        };
        #[cfg(not(any(unix, windows)))]
        let (model, content) = {
            let content = Self::observe(config).ok().flatten();
            (load()?, content)
        };
        Ok((model, content.filter(|proof| proof.still_matches(config))))
    }

    #[cfg(any(not(any(unix, windows)), test))]
    pub(crate) fn observe(config: &StageConfig) -> Result<Option<Self>> {
        Self::observe_files(config)?
            .map(|files| Self::from_observed(files, &AtomicBool::new(false)))
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

    #[cfg(any(not(any(unix, windows)), test))]
    fn from_files(files: Vec<(String, PathBuf)>) -> Result<Self> {
        Self::from_observed(capture_files(files)?, &AtomicBool::new(false))
    }

    fn from_observed(files: ObservedFiles, cancelled: &AtomicBool) -> Result<Self> {
        let started = Instant::now();
        let mut verified_bytes = 0_u64;
        let mut digest = Sha256::new();
        digest.update(b"skippy-loaded-model-content-v1\0");
        let mut observed = Vec::with_capacity(files.len());
        let mut buffer = vec![0; 1024 * 1024];
        for (name, path, stamp) in files {
            ensure!(
                !cancelled.load(Ordering::Acquire),
                "model content verification cancelled"
            );
            let file = File::open(&path)
                .with_context(|| format!("open model content {}", path.display()))?;
            let before = file.metadata()?;
            ensure!(
                before.is_file(),
                "model content proof requires regular files"
            );
            ensure!(
                FileStamp::read(&file, &before)? == stamp,
                "model content changed before verification"
            );
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update(stamp.bytes.to_le_bytes());
            let mut reader = BufReader::new(file);
            verified_bytes = verified_bytes.saturating_add(hash_content(
                &mut reader,
                &mut digest,
                &mut buffer,
                cancelled,
            )?);
            let after = reader.get_ref().metadata()?;
            ensure!(
                FileStamp::read(reader.get_ref(), &after)? == stamp,
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
                        && FileStamp::of_path(path).is_ok_and(|(metadata, current)| {
                            metadata.is_file() && current == *stamp
                        })
                },
            );
        if !unchanged {
            return false;
        }
        #[cfg(any(unix, windows))]
        {
            true
        }
        #[cfg(not(any(unix, windows)))]
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

/// Hash bounded chunks, abandoning further input reads after a failed model open.
fn hash_content(
    reader: &mut impl Read,
    digest: &mut Sha256,
    buffer: &mut [u8],
    cancelled: &AtomicBool,
) -> Result<u64> {
    let mut bytes = 0_u64;
    loop {
        ensure!(
            !cancelled.load(Ordering::Acquire),
            "model content verification cancelled"
        );
        let count = reader.read(buffer)?;
        if count == 0 {
            return Ok(bytes);
        }
        digest.update(&buffer[..count]);
        bytes = bytes.saturating_add(count as u64);
    }
}

fn capture_files(files: Vec<(String, PathBuf)>) -> Result<ObservedFiles> {
    files
        .into_iter()
        .map(|(name, path)| {
            let (metadata, stamp) = FileStamp::of_path(&path)?;
            ensure!(
                metadata.is_file(),
                "model content proof requires regular files"
            );
            Ok((name, path, stamp))
        })
        .collect()
}

#[cfg(any(unix, windows))]
fn verify_alongside_load<T>(
    load: impl FnOnce() -> Result<T>,
    verify: impl FnOnce(&AtomicBool) -> Result<VerifiedModelContent> + Send,
) -> Result<(T, Option<VerifiedModelContent>)> {
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let verification = std::thread::Builder::new()
            .name("skippy-model-content".into())
            .spawn_scoped(scope, || verify(&cancelled))
            .ok();
        let model = load();
        if model.is_err() {
            cancelled.store(true, Ordering::Release);
        }
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

    /// These tests rewrite a file in place with same-length bytes, so on Unix
    /// only ctime, and on Windows only `ChangeTime`, can reveal the change.
    /// Filesystems stamp both from a coarse clock; wait until that clock has
    /// moved past `path`'s stamp so the rewrite cannot land in the same tick as
    /// the observed one.
    fn wait_for_ctime_to_advance_past(path: &Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let ctime = |metadata: Metadata| (metadata.ctime(), metadata.ctime_nsec());
            let observed = ctime(std::fs::metadata(path).unwrap());
            let probe = path.with_extension("ctime-probe");
            let deadline = Instant::now() + std::time::Duration::from_secs(60);
            loop {
                std::fs::write(&probe, b"").unwrap();
                if ctime(std::fs::metadata(&probe).unwrap()) > observed {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "filesystem ctime did not advance"
                );
                std::thread::yield_now();
            }
            std::fs::remove_file(probe).unwrap();
        }
        #[cfg(windows)]
        {
            let change_time = |path: &Path| FileStamp::of_path(path).unwrap().1.identity.2;
            let observed = change_time(path);
            let probe = path.with_extension("ctime-probe");
            let deadline = Instant::now() + std::time::Duration::from_secs(60);
            loop {
                std::fs::write(&probe, b"").unwrap();
                if change_time(&probe) > observed {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "filesystem ChangeTime did not advance"
                );
                std::thread::yield_now();
            }
            std::fs::remove_file(probe).unwrap();
        }
        #[cfg(not(any(unix, windows)))]
        let _ = path;
    }

    /// The strong stamp is what lets Unix and Windows skip rehashing: a
    /// same-size rewrite that puts the last-write time back must still be
    /// caught (#3158).
    #[cfg(any(unix, windows))]
    #[test]
    fn same_size_rewrite_with_restored_mtime_invalidates_the_proof() {
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
        let proof = VerifiedModelContent::observe(&config).unwrap().unwrap();
        assert!(proof.still_matches(&config));

        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        wait_for_ctime_to_advance_past(&path);
        std::fs::write(&path, b"new weights").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );

        assert!(!proof.still_matches(&config));
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn failed_model_open_cancels_verification_before_joining() {
        use std::{sync::mpsc, time::Duration};
        let (started, observed_start) = mpsc::channel();
        let (cancellation, observed_cancellation) = mpsc::channel();
        let error = verify_alongside_load(
            || -> Result<()> {
                observed_start.recv_timeout(Duration::from_secs(5))?;
                anyhow::bail!("native model load failed")
            },
            move |cancelled| {
                started.send(())?;
                let deadline = Instant::now() + Duration::from_secs(5);
                while !cancelled.load(Ordering::Acquire) && Instant::now() < deadline {
                    std::thread::yield_now();
                }
                cancellation.send(cancelled.load(Ordering::Acquire))?;
                anyhow::bail!("verification abandoned")
            },
        )
        .err()
        .expect("failed load must stay an error");
        assert!(observed_cancellation.recv().unwrap());
        assert_eq!(error.to_string(), "native model load failed");
    }

    #[test]
    fn cancelled_verification_stops_before_the_next_chunk() {
        struct CancelAfterRead<'a> {
            cancelled: &'a AtomicBool,
            reads: usize,
        }
        impl Read for CancelAfterRead<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                self.reads += 1;
                if self.reads > 1 {
                    return Ok(0);
                }
                buffer.fill(7);
                self.cancelled.store(true, Ordering::Release);
                Ok(buffer.len())
            }
        }
        let cancelled = AtomicBool::new(false);
        let mut reader = CancelAfterRead {
            cancelled: &cancelled,
            reads: 0,
        };
        let error =
            hash_content(&mut reader, &mut Sha256::new(), &mut [0; 16], &cancelled).unwrap_err();
        assert_eq!(reader.reads, 1);
        assert!(error.to_string().contains("verification cancelled"));
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn verification_failure_does_not_fail_a_successful_model_open() {
        let (model, proof) =
            verify_alongside_load(|| Ok("loaded"), |_| anyhow::bail!("content unavailable"))
                .unwrap();
        assert_eq!(model, "loaded");
        assert!(proof.is_none());
    }

    #[cfg(any(unix, windows))]
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
            move |cancelled| {
                started.send(())?;
                observed_open.recv_timeout(Duration::from_secs(5))?;
                assert!(!cancelled.load(Ordering::Acquire));
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
            wait_for_ctime_to_advance_past(&path);
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
        wait_for_ctime_to_advance_past(&model);
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
