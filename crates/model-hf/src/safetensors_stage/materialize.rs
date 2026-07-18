use std::{
    collections::BTreeMap,
    env,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result, anyhow, ensure};
use model_artifact::safetensors::{TensorHeader, parse_header};
use sha2::{Digest, Sha256};

use super::{
    http::RemoteSource,
    layout,
    locking::CacheKeyLock,
    types::{
        MANIFEST_SCHEMA_VERSION, PreparedSafetensorsCheckpoint, PreparedStage,
        SafetensorsCheckpointDescriptor, SafetensorsSourceShard, SafetensorsStageArtifact,
        SafetensorsStageManifest, SafetensorsStagePlan, SafetensorsStageRequest, SelectedTensor,
    },
};

const MODEL_FILE: &str = "model.safetensors";
const CONFIG_FILE: &str = "config.json";
const PLAN_FILE: &str = "stage-plan.json";
const MANIFEST_FILE: &str = "stage-manifest.json";
pub const CHECKPOINT_DESCRIPTOR_FILE: &str = "checkpoint-descriptor.json";
const MAX_LOCAL_HEADER_BYTES: u64 = 256 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn read_checkpoint_descriptor(path: &Path) -> Result<SafetensorsCheckpointDescriptor> {
    read_json(&path.join(CHECKPOINT_DESCRIPTOR_FILE)).with_context(|| {
        format!(
            "read prepared SafeTensors checkpoint descriptor from {}",
            path.display()
        )
    })
}

pub struct SafetensorsStageMaterializer {
    pub(crate) remote: RemoteSource,
    pub(crate) cache_root: PathBuf,
}

impl SafetensorsStageMaterializer {
    pub fn from_environment() -> Result<Self> {
        let endpoint = env::var("HF_ENDPOINT").ok();
        Self::new(
            crate::store::mesh_llm_cache_dir().join("mlx-safetensors-stages"),
            endpoint.as_deref(),
            hf_token(),
        )
    }

    pub fn new(cache_root: PathBuf, endpoint: Option<&str>, token: Option<String>) -> Result<Self> {
        Ok(Self {
            remote: RemoteSource::new(endpoint, token)?,
            cache_root,
        })
    }

    pub fn plan(&self, request: SafetensorsStageRequest) -> Result<SafetensorsStagePlan> {
        let request = request.normalized()?;
        Ok(layout::prepare(&self.remote, &request)?.plan)
    }

    /// Fetches config/index/header metadata only. Tensor payload ranges are
    /// planned but not downloaded.
    pub fn describe_checkpoint(
        &self,
        repo: &str,
        revision: &str,
    ) -> Result<SafetensorsCheckpointDescriptor> {
        layout::describe(&self.remote, repo, revision)
    }

    /// Caches only the tokenizer/config/chat-template files needed by the
    /// distributed stage-0 frontend. No model tensor payload is downloaded.
    pub fn prepare_checkpoint(
        &self,
        repo: &str,
        revision: &str,
    ) -> Result<PreparedSafetensorsCheckpoint> {
        let descriptor = self.describe_checkpoint(repo, revision)?;
        let cache_key = format!(
            "checkpoint-{:x}",
            Sha256::digest(format!("{repo}@{revision}").as_bytes())
        );
        let sidecar_root = self.cache_root.join("checkpoint-sidecars");
        fs::create_dir_all(&sidecar_root).with_context(|| {
            format!(
                "create SafeTensors checkpoint sidecar cache {}",
                sidecar_root.display()
            )
        })?;
        let _cache_lock = CacheKeyLock::acquire(&self.cache_root, &cache_key)?;
        let destination = sidecar_root.join(&cache_key);
        if checkpoint_sidecars_match(&destination, &descriptor) {
            return Ok(PreparedSafetensorsCheckpoint {
                path: destination,
                descriptor,
            });
        }
        remove_stale_partials(&sidecar_root, &cache_key)?;
        let temporary = temporary_path(&sidecar_root, &cache_key);
        fs::create_dir(&temporary).with_context(|| {
            format!(
                "create temporary checkpoint sidecar cache {}",
                temporary.display()
            )
        })?;
        let write_result = self.write_checkpoint_sidecars(repo, revision, &temporary, &descriptor);
        if let Err(error) = write_result {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
        let quarantine = destination
            .exists()
            .then(|| temporary_path(&sidecar_root, &format!("{cache_key}.stale")));
        if let Some(quarantine) = quarantine.as_ref() {
            fs::rename(&destination, quarantine).with_context(|| {
                format!(
                    "quarantine incomplete checkpoint sidecar cache {}",
                    destination.display()
                )
            })?;
        }
        if let Err(error) = fs::rename(&temporary, &destination) {
            if let Some(quarantine) = quarantine.as_ref() {
                let _ = fs::rename(quarantine, &destination);
            }
            return Err(error).context("publish checkpoint sidecar cache");
        }
        if let Some(quarantine) = quarantine {
            let _ = fs::remove_dir_all(quarantine);
        }
        sync_directory(&sidecar_root)?;
        Ok(PreparedSafetensorsCheckpoint {
            path: destination,
            descriptor,
        })
    }

    fn write_checkpoint_sidecars(
        &self,
        repo: &str,
        revision: &str,
        destination: &Path,
        descriptor: &SafetensorsCheckpointDescriptor,
    ) -> Result<()> {
        self.cache_required_sidecar(repo, revision, "config.json", destination)?;
        self.cache_required_sidecar(repo, revision, "tokenizer.json", destination)?;
        for file in [
            "tokenizer_config.json",
            "chat_template.jinja",
            "chat_template.json",
            "generation_config.json",
            "special_tokens_map.json",
        ] {
            self.cache_optional_sidecar(repo, revision, file, destination)?;
        }
        write_json(destination.join(CHECKPOINT_DESCRIPTOR_FILE), descriptor)?;
        sync_directory(destination)
    }

    fn cache_required_sidecar(
        &self,
        repo: &str,
        revision: &str,
        file: &str,
        destination: &Path,
    ) -> Result<()> {
        let remote = self
            .remote
            .small_file(
                self.remote.url(repo, revision, file)?,
                MAX_LOCAL_HEADER_BYTES,
            )
            .with_context(|| format!("download required checkpoint sidecar {file}"))?;
        write_synced(destination.join(file), &remote.bytes)
    }

    fn cache_optional_sidecar(
        &self,
        repo: &str,
        revision: &str,
        file: &str,
        destination: &Path,
    ) -> Result<()> {
        let Some(remote) = self.remote.optional_small_file(
            self.remote.url(repo, revision, file)?,
            MAX_LOCAL_HEADER_BYTES,
        )?
        else {
            return Ok(());
        };
        write_synced(destination.join(file), &remote.bytes)
    }

    pub fn materialize(
        &self,
        request: SafetensorsStageRequest,
    ) -> Result<SafetensorsStageArtifact> {
        let request = request.normalized()?;
        let cache_key = cache_key(self.remote.endpoint(), &request)?;
        let destination = self.cache_root.join(&cache_key);
        if let Ok(Some(artifact)) = self.load_cached(&destination, &cache_key, &request) {
            return Ok(artifact);
        }

        let _cache_lock = CacheKeyLock::acquire(&self.cache_root, &cache_key)?;
        if let Ok(Some(artifact)) = self.load_cached(&destination, &cache_key, &request) {
            return Ok(artifact);
        }
        remove_stale_partials(&self.cache_root, &cache_key)?;
        let prepared = layout::prepare(&self.remote, &request)?;
        let temporary = self.temporary_path(&cache_key);
        fs::create_dir(&temporary)
            .with_context(|| format!("create temporary stage cache {}", temporary.display()))?;
        if let Err(error) = self.write_stage(&temporary, &cache_key, &request, &prepared) {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error);
        }
        let cache_hit = self.publish_cache(&temporary, &destination, &cache_key, &request)?;
        let mut artifact = self
            .load_cached(&destination, &cache_key, &request)?
            .ok_or_else(|| anyhow!("published SafeTensors stage cache is missing"))?;
        artifact.cache_hit = cache_hit;
        Ok(artifact)
    }

    fn write_stage(
        &self,
        directory: &Path,
        cache_key: &str,
        request: &SafetensorsStageRequest,
        prepared: &PreparedStage,
    ) -> Result<()> {
        let output_path = directory.join(MODEL_FILE);
        let (output_file_bytes, output_sha256) = self
            .write_model(&output_path, prepared)
            .context("materialize SafeTensors stage")?;
        write_synced(directory.join(CONFIG_FILE), &prepared.config)?;
        write_json(directory.join(PLAN_FILE), &prepared.plan)?;
        let manifest = SafetensorsStageManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            cache_key: cache_key.to_string(),
            checkpoint_sha256: prepared.checkpoint_sha256.clone(),
            source_endpoint: self.remote.endpoint().to_string(),
            request: request.clone(),
            selected_tensor_count: prepared.plan.selected_tensor_count,
            selected_tensor_bytes: prepared.plan.selected_tensor_bytes,
            output_file_bytes,
            output_sha256,
            config_sha256: prepared.config_sha256.clone(),
            config_etag: prepared.config_etag.clone(),
            index_sha256: prepared.index_sha256.clone(),
            index_etag: prepared.index_etag.clone(),
            source_shards: prepared.source_shards.clone(),
        };
        write_json(directory.join(MANIFEST_FILE), &manifest)?;
        sync_directory(directory)?;
        Ok(())
    }

    fn write_model(&self, path: &Path, prepared: &PreparedStage) -> Result<(u64, String)> {
        let mut tensors = prepared.tensors.iter().collect::<Vec<_>>();
        tensors.sort_by(|left, right| {
            (&left.source_file, left.source_range.start)
                .cmp(&(&right.source_file, right.source_range.start))
        });
        let (header, payload_bytes) = output_header(&tensors)?;
        let mut header_bytes = serde_json::to_vec(&header)?;
        while header_bytes.len() % 8 != 0 {
            header_bytes.push(b' ');
        }
        let header_len = u64::try_from(header_bytes.len()).context("output header is too large")?;

        let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut writer = HashingWriter::new(BufWriter::new(file));
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header_bytes)?;
        let source_shards = prepared
            .source_shards
            .iter()
            .map(|shard| (shard.file.as_str(), shard))
            .collect::<BTreeMap<_, _>>();
        let mut downloaded = 0_u64;
        for span in materialization_spans(&tensors) {
            let identity = source_shards
                .get(span.source_file.as_str())
                .with_context(|| format!("missing source identity for {}", span.source_file))?;
            let url = self.remote.url(
                &prepared.plan.repo,
                &prepared.plan.revision,
                &span.source_file,
            )?;
            let expected_etag = identity
                .etag
                .as_deref()
                .context("planned SafeTensors shard has no ETag")?;
            let response = self.remote.exact_range_if_range(
                url,
                span.start..span.end_exclusive,
                expected_etag,
            )?;
            ensure!(
                response.total_file_bytes == identity.file_bytes,
                "SafeTensors shard {} changed size during materialization",
                span.source_file
            );
            ensure_source_identity(identity, response.etag())?;
            downloaded = downloaded
                .checked_add(response.copy_to(&mut writer)?)
                .context("materialized tensor byte count overflow")?;
        }
        ensure!(
            downloaded == payload_bytes && downloaded == prepared.plan.selected_tensor_bytes,
            "materialized {downloaded} tensor bytes but planned {}",
            prepared.plan.selected_tensor_bytes
        );
        writer.flush()?;
        writer.inner.get_ref().sync_all()?;
        let output_file_bytes = writer.bytes_written;
        let expected_output_file_bytes = 8_u64
            .checked_add(header_len)
            .and_then(|bytes| bytes.checked_add(payload_bytes))
            .context("materialized SafeTensors output length overflow")?;
        ensure!(
            output_file_bytes == expected_output_file_bytes,
            "materialized SafeTensors output length mismatch"
        );
        let output_sha256 = writer.finish_hash();
        Ok((output_file_bytes, output_sha256))
    }

    fn load_cached(
        &self,
        directory: &Path,
        cache_key: &str,
        request: &SafetensorsStageRequest,
    ) -> Result<Option<SafetensorsStageArtifact>> {
        if !directory.exists() {
            return Ok(None);
        }
        ensure!(
            directory.is_dir(),
            "SafeTensors stage cache entry is not a directory"
        );
        let manifest: SafetensorsStageManifest = read_json(&directory.join(MANIFEST_FILE))?;
        let plan: SafetensorsStagePlan = read_json(&directory.join(PLAN_FILE))?;
        ensure!(
            manifest.schema_version == MANIFEST_SCHEMA_VERSION,
            "SafeTensors stage cache schema is stale"
        );
        ensure!(
            manifest.cache_key == cache_key,
            "SafeTensors cache key mismatch"
        );
        ensure!(
            manifest.request == *request,
            "SafeTensors cache request mismatch"
        );
        ensure!(
            manifest.source_endpoint == self.remote.endpoint(),
            "SafeTensors cache source endpoint mismatch"
        );
        ensure!(
            plan.selected_tensor_count == manifest.selected_tensor_count
                && plan.selected_tensor_bytes == manifest.selected_tensor_bytes,
            "SafeTensors cached plan and manifest disagree"
        );
        ensure!(
            plan.checkpoint_sha256 == manifest.checkpoint_sha256,
            "SafeTensors checkpoint identity mismatch"
        );
        let model_path = directory.join(MODEL_FILE);
        let config_path = directory.join(CONFIG_FILE);
        ensure_regular_file(&model_path)?;
        ensure_regular_file(&config_path)?;
        ensure!(
            fs::metadata(&model_path)?.len() == manifest.output_file_bytes,
            "cached SafeTensors output length mismatch"
        );
        ensure!(
            sha256_file(&model_path)? == manifest.output_sha256,
            "cached SafeTensors output hash mismatch"
        );
        ensure!(
            sha256_file(&config_path)? == manifest.config_sha256,
            "cached SafeTensors config hash mismatch"
        );
        validate_local_safetensors(&model_path)?;
        Ok(Some(SafetensorsStageArtifact {
            path: directory.to_path_buf(),
            manifest,
            plan,
            cache_hit: true,
        }))
    }

    fn publish_cache(
        &self,
        temporary: &Path,
        destination: &Path,
        cache_key: &str,
        request: &SafetensorsStageRequest,
    ) -> Result<bool> {
        if destination.exists() {
            if matches!(
                self.load_cached(destination, cache_key, request),
                Ok(Some(_))
            ) {
                fs::remove_dir_all(temporary)?;
                return Ok(true);
            }
            let quarantine = self.temporary_path(&format!("{cache_key}.corrupt"));
            fs::rename(destination, &quarantine).with_context(|| {
                format!("quarantine corrupt stage cache {}", destination.display())
            })?;
            if let Err(error) = fs::rename(temporary, destination) {
                let _ = fs::rename(&quarantine, destination);
                return Err(error).context("publish repaired SafeTensors stage cache");
            }
            let _ = fs::remove_dir_all(quarantine);
        } else {
            fs::rename(temporary, destination).context("publish SafeTensors stage cache")?;
        }
        sync_directory(&self.cache_root)?;
        Ok(false)
    }

    fn temporary_path(&self, cache_key: &str) -> PathBuf {
        temporary_path(&self.cache_root, cache_key)
    }
}

fn checkpoint_sidecars_match(
    destination: &Path,
    expected: &SafetensorsCheckpointDescriptor,
) -> bool {
    destination.join(CONFIG_FILE).is_file()
        && destination.join("tokenizer.json").is_file()
        && read_checkpoint_descriptor(destination).is_ok_and(|actual| actual == *expected)
}

fn temporary_path(cache_root: &Path, cache_key: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    cache_root.join(format!(
        ".{cache_key}.{}.{}.partial",
        std::process::id(),
        sequence
    ))
}

struct MaterializationSpan {
    source_file: String,
    start: u64,
    end_exclusive: u64,
}

fn output_header(tensors: &[&SelectedTensor]) -> Result<(BTreeMap<String, TensorHeader>, u64)> {
    let mut output_offset = 0_u64;
    let mut output_header = BTreeMap::new();
    for tensor in tensors {
        let end = output_offset
            .checked_add(tensor.source_range.len())
            .context("partial SafeTensors output offset overflow")?;
        let mut header = tensor.header.clone();
        header.data_offsets = [output_offset, end];
        output_header.insert(tensor.name.clone(), header);
        output_offset = end;
    }
    Ok((output_header, output_offset))
}

fn materialization_spans(tensors: &[&SelectedTensor]) -> Vec<MaterializationSpan> {
    let mut spans: Vec<MaterializationSpan> = Vec::new();
    for tensor in tensors {
        if let Some(previous) = spans.last_mut()
            && previous.source_file == tensor.source_file
            && previous.end_exclusive == tensor.source_range.start
        {
            previous.end_exclusive = tensor.source_range.end_exclusive;
        } else {
            spans.push(MaterializationSpan {
                source_file: tensor.source_file.clone(),
                start: tensor.source_range.start,
                end_exclusive: tensor.source_range.end_exclusive,
            });
        }
    }
    spans
}

pub(super) fn ensure_source_identity(
    identity: &SafetensorsSourceShard,
    actual_etag: Option<&str>,
) -> Result<()> {
    let expected = identity
        .etag
        .as_deref()
        .context("planned SafeTensors shard has no ETag")?;
    let actual = actual_etag.context("SafeTensors payload response omitted ETag")?;
    ensure!(
        expected == actual,
        "SafeTensors shard {} changed identity during materialization",
        identity.file
    );
    Ok(())
}

#[cfg(unix)]
fn remove_stale_partials(cache_root: &Path, cache_key: &str) -> Result<()> {
    let prefix = format!(".{cache_key}.");
    for entry in fs::read_dir(cache_root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with(&prefix) && name.ends_with(".partial") && entry.file_type()?.is_dir() {
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn remove_stale_partials(_cache_root: &Path, _cache_key: &str) -> Result<()> {
    Ok(())
}

fn cache_key(endpoint: &str, request: &SafetensorsStageRequest) -> Result<String> {
    let identity = serde_json::to_vec(&(MANIFEST_SCHEMA_VERSION, endpoint, request))?;
    Ok(format!("{:x}", Sha256::digest(identity)))
}

fn hf_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|name| env::var(name).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn write_json(path: PathBuf, value: &impl serde::Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_synced(path, &bytes)
}

fn write_synced(path: PathBuf, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    ensure_regular_file(path)?;
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

fn ensure_regular_file(path: &Path) -> Result<()> {
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_file(),
        "cache path is not a regular file: {}",
        path.display()
    );
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    std::io::copy(&mut reader, &mut HashWriter(&mut hasher))?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_local_safetensors(path: &Path) -> Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let file_bytes = reader.get_ref().metadata()?.len();
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    ensure!(
        header_len <= MAX_LOCAL_HEADER_BYTES,
        "cached SafeTensors header is too large"
    );
    let data_start = 8_u64
        .checked_add(header_len)
        .context("cached SafeTensors header offset overflow")?;
    ensure!(
        data_start <= file_bytes,
        "cached SafeTensors header is truncated"
    );
    let mut header = vec![0_u8; usize::try_from(header_len)?];
    reader.read_exact(&mut header)?;
    parse_header(&header, file_bytes - data_start)?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
    bytes_written: u64,
}

impl<W> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            bytes_written: 0,
        }
    }

    fn finish_hash(&self) -> String {
        format!("{:x}", self.hasher.clone().finalize())
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hasher.update(&bytes[..written]);
        self.bytes_written = self.bytes_written.saturating_add(written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

struct HashWriter<'a>(&'a mut Sha256);

impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Seek, SeekFrom},
        net::{TcpListener, TcpStream},
        sync::{
            Arc, Barrier, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
    };

    use model_artifact::safetensors::TensorHeader;

    use super::*;
    use crate::safetensors_stage::types::ByteRange;

    fn selected(name: &str, file: &str, start: u64, end: u64) -> SelectedTensor {
        SelectedTensor {
            name: name.to_string(),
            source_file: file.to_string(),
            source_range: ByteRange {
                start,
                end_exclusive: end,
            },
            header: TensorHeader {
                dtype: "U8".to_string(),
                shape: vec![end - start],
                data_offsets: [start, end],
            },
        }
    }

    #[test]
    fn materialization_spans_join_only_same_shard_contiguous_tensors() {
        let tensors = [
            selected("a", "one", 10, 20),
            selected("b", "one", 20, 30),
            selected("c", "one", 40, 50),
            selected("d", "two", 50, 60),
        ];
        let refs = tensors.iter().collect::<Vec<_>>();

        let spans = materialization_spans(&refs);

        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].start, 10);
        assert_eq!(spans[0].end_exclusive, 30);
        assert_eq!(spans[2].source_file, "two");
    }

    #[test]
    fn cache_identity_changes_with_layer_range() {
        let request = SafetensorsStageRequest {
            repo: "org/model".to_string(),
            revision: "a".repeat(40),
            layer_start: 0,
            layer_end: 10,
            include_prefixes: Vec::new(),
        };
        let mut other = request.clone();
        other.layer_start = 10;
        other.layer_end = 20;

        assert_ne!(
            cache_key("https://huggingface.co/", &request).unwrap(),
            cache_key("https://huggingface.co/", &other).unwrap()
        );
    }

    #[test]
    fn materializes_exact_ranges_reuses_cache_and_repairs_corruption() {
        let checkpoint = Arc::new(test_checkpoint());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_checkpoint_server(Arc::clone(&checkpoint), Arc::clone(&requests));
        let cache = tempfile::tempdir().unwrap();
        let materializer =
            SafetensorsStageMaterializer::new(cache.path().join("cache"), Some(&endpoint), None)
                .unwrap();
        let request = test_request();

        let first = materializer.materialize(request.clone()).unwrap();

        assert!(!first.cache_hit);
        assert_eq!(first.plan.selected_tensor_count, 2);
        assert_eq!(first.plan.selected_tensor_bytes, 4);
        assert!(first.manifest.output_file_bytes < checkpoint.len() as u64);
        let request_count = requests.lock().unwrap().len();
        assert!(requests.lock().unwrap().iter().any(|request| {
            request
                .lines()
                .any(|line| line.eq_ignore_ascii_case("range: bytes=0-7"))
        }));
        assert!(requests.lock().unwrap().iter().any(|request| {
            request
                .lines()
                .any(|line| line.eq_ignore_ascii_case("if-range: \"model-id\""))
        }));

        let cached = materializer.materialize(request.clone()).unwrap();

        assert!(cached.cache_hit);
        assert_eq!(requests.lock().unwrap().len(), request_count);

        let model_path = cached.path.join(MODEL_FILE);
        let mut model = fs::OpenOptions::new().write(true).open(model_path).unwrap();
        model.seek(SeekFrom::Start(16)).unwrap();
        model.write_all(b"broken").unwrap();
        model.sync_all().unwrap();

        let repaired = materializer.materialize(request).unwrap();

        assert!(!repaired.cache_hit);
        assert!(requests.lock().unwrap().len() > request_count);
        validate_local_safetensors(&repaired.path.join(MODEL_FILE)).unwrap();
    }

    #[test]
    fn visits_selected_tensors_as_one_ephemeral_file_at_a_time() {
        let checkpoint = Arc::new(test_checkpoint());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_checkpoint_server(checkpoint, Arc::clone(&requests));
        let cache = tempfile::tempdir().unwrap();
        let cache_root = cache.path().join("cache");
        let materializer =
            SafetensorsStageMaterializer::new(cache_root.clone(), Some(&endpoint), None).unwrap();
        let mut visited = Vec::new();

        let visit = materializer.prepare_tensor_visit(test_request()).unwrap();
        assert_eq!(visit.checkpoint_sha256(), visit.plan().checkpoint_sha256);
        assert_eq!(
            format!("{:x}", Sha256::digest(visit.config())),
            visit.config_sha256()
        );
        let metadata_request_count = requests.lock().unwrap().len();
        let report = visit
            .visit_tensor_files(|tensor| {
                assert!(tensor.path.is_file());
                assert_eq!(
                    fs::read_dir(tensor.path.parent().unwrap())?.count(),
                    1,
                    "more than one ephemeral source tensor was retained"
                );
                validate_local_safetensors(&tensor.path)?;
                let mut reader = BufReader::new(File::open(&tensor.path)?);
                let mut header_len = [0_u8; 8];
                reader.read_exact(&mut header_len)?;
                let header_len = u64::from_le_bytes(header_len);
                let mut header = vec![0_u8; usize::try_from(header_len)?];
                reader.read_exact(&mut header)?;
                let parsed = parse_header(&header, tensor.source_range.len())?;
                assert_eq!(parsed.keys().collect::<Vec<_>>(), vec![&tensor.name]);
                visited.push(tensor.name.clone());
                Ok(())
            })
            .unwrap();

        assert_eq!(visited.len(), report.plan.selected_tensor_count);
        assert_eq!(report.visited_tensor_count, visited.len());
        assert_eq!(report.source_range_request_count, visited.len());
        assert_eq!(
            report.visited_tensor_bytes,
            report.plan.selected_tensor_bytes
        );
        assert!(report.temporary_file_peak_bytes > 0);
        let payload_request_count = requests.lock().unwrap().len() - metadata_request_count;
        assert_eq!(payload_request_count, report.source_range_request_count);
        assert!(fs::read_dir(cache_root).unwrap().next().is_none());
    }

    #[test]
    fn removes_ephemeral_tensor_files_when_the_visitor_fails() {
        let checkpoint = Arc::new(test_checkpoint());
        let endpoint = start_checkpoint_server(checkpoint, Arc::new(Mutex::new(Vec::new())));
        let cache = tempfile::tempdir().unwrap();
        let cache_root = cache.path().join("cache");
        let materializer =
            SafetensorsStageMaterializer::new(cache_root.clone(), Some(&endpoint), None).unwrap();

        let error = materializer
            .prepare_tensor_visit(test_request())
            .unwrap()
            .visit_tensor_files(|_| anyhow::bail!("stop after first tensor"))
            .unwrap_err();

        assert!(error.to_string().contains("stop after first tensor"));
        assert!(fs::read_dir(cache_root).unwrap().next().is_none());
    }

    #[test]
    fn cancelled_tensor_visit_stops_before_the_first_payload_request() {
        let checkpoint = Arc::new(test_checkpoint());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_checkpoint_server(checkpoint, Arc::clone(&requests));
        let cache = tempfile::tempdir().unwrap();
        let cache_root = cache.path().join("cache");
        let materializer =
            SafetensorsStageMaterializer::new(cache_root.clone(), Some(&endpoint), None).unwrap();

        let visit = materializer.prepare_tensor_visit(test_request()).unwrap();
        let metadata_request_count = requests.lock().unwrap().len();
        let error = visit
            .visit_tensor_files_cancellable(|| true, |_| Ok(()))
            .unwrap_err();

        assert!(error.to_string().contains("tensor visit cancelled"));
        assert_eq!(requests.lock().unwrap().len(), metadata_request_count);
        assert!(
            !cache_root.exists() || fs::read_dir(cache_root).unwrap().next().is_none(),
            "cancelled visit retained an ephemeral tensor directory"
        );
    }

    #[test]
    fn serializes_concurrent_materialization_of_the_same_cache_key() {
        let checkpoint = Arc::new(test_checkpoint());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_checkpoint_server(Arc::clone(&checkpoint), Arc::clone(&requests));
        let cache = tempfile::tempdir().unwrap();
        let cache_root = cache.path().join("cache");
        let barrier = Arc::new(Barrier::new(2));
        let handles = (0..2)
            .map(|_| {
                let endpoint = endpoint.clone();
                let cache_root = cache_root.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let materializer =
                        SafetensorsStageMaterializer::new(cache_root, Some(&endpoint), None)
                            .unwrap();
                    barrier.wait();
                    materializer.materialize(test_request()).unwrap()
                })
            })
            .collect::<Vec<_>>();

        let mut artifacts = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        artifacts.sort_by_key(|artifact| artifact.cache_hit);

        assert!(!artifacts[0].cache_hit);
        assert!(artifacts[1].cache_hit);
        assert_eq!(artifacts[0].path, artifacts[1].path);
        assert!(fs::read_dir(cache_root).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".partial")
        }));
    }

    #[test]
    fn rejects_checkpoint_shard_without_an_etag() {
        let checkpoint = Arc::new(test_checkpoint());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint =
            start_checkpoint_server_with_etag(checkpoint, requests, ModelEtagMode::Static(None));
        let cache = tempfile::tempdir().unwrap();
        let materializer =
            SafetensorsStageMaterializer::new(cache.path().join("cache"), Some(&endpoint), None)
                .unwrap();

        let error = materializer.materialize(test_request()).unwrap_err();

        assert!(format!("{error:#}").contains("omitted ETag"));
    }

    #[test]
    fn rejects_checkpoint_that_changes_etag_before_payload() {
        let checkpoint = Arc::new(test_checkpoint());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_checkpoint_server_with_etag(
            checkpoint,
            requests,
            ModelEtagMode::ChangeAfterHeader,
        );
        let cache = tempfile::tempdir().unwrap();
        let materializer =
            SafetensorsStageMaterializer::new(cache.path().join("cache"), Some(&endpoint), None)
                .unwrap();

        let error = materializer.materialize(test_request()).unwrap_err();

        assert!(format!("{error:#}").contains("changed identity"));
    }

    #[test]
    fn shares_checkpoint_identity_across_distinct_layer_ranges() {
        let checkpoint = Arc::new(test_checkpoint());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_checkpoint_server(checkpoint, requests);
        let cache = tempfile::tempdir().unwrap();
        let materializer =
            SafetensorsStageMaterializer::new(cache.path().join("cache"), Some(&endpoint), None)
                .unwrap();
        let first_request = test_request();
        let mut final_request = first_request.clone();
        final_request.layer_start = 1;
        final_request.layer_end = 2;

        let first = materializer.materialize(first_request).unwrap();
        let final_stage = materializer.materialize(final_request).unwrap();

        assert_eq!(
            first.manifest.checkpoint_sha256,
            final_stage.manifest.checkpoint_sha256
        );
        assert_ne!(first.path, final_stage.path);
        assert_eq!(final_stage.plan.selected_tensor_count, 3);
    }

    fn test_request() -> SafetensorsStageRequest {
        SafetensorsStageRequest {
            repo: "org/model".to_string(),
            revision: "0123456789012345678901234567890123456789".to_string(),
            layer_start: 0,
            layer_end: 1,
            include_prefixes: Vec::new(),
        }
    }

    fn test_checkpoint() -> Vec<u8> {
        let mut offset = 0_u64;
        let mut header = BTreeMap::new();
        for name in [
            "model.embed_tokens.weight",
            "model.layers.0.weight",
            "model.layers.1.weight",
            "model.norm.weight",
        ] {
            header.insert(
                name,
                TensorHeader {
                    dtype: "U8".to_string(),
                    shape: vec![2],
                    data_offsets: [offset, offset + 2],
                },
            );
            offset += 2;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut checkpoint = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        checkpoint.extend(header);
        checkpoint.extend(0_u8..8);
        checkpoint
    }

    fn start_checkpoint_server(
        checkpoint: Arc<Vec<u8>>,
        requests: Arc<Mutex<Vec<String>>>,
    ) -> String {
        start_checkpoint_server_with_etag(
            checkpoint,
            requests,
            ModelEtagMode::Static(Some("\"model-id\"")),
        )
    }

    #[derive(Clone, Copy)]
    enum ModelEtagMode {
        Static(Option<&'static str>),
        ChangeAfterHeader,
    }

    fn start_checkpoint_server_with_etag(
        checkpoint: Arc<Vec<u8>>,
        requests: Arc<Mutex<Vec<String>>>,
        etag_mode: ModelEtagMode,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let model_requests = Arc::new(AtomicUsize::new(0));
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else {
                    return;
                };
                let checkpoint = Arc::clone(&checkpoint);
                let requests = Arc::clone(&requests);
                let model_requests = Arc::clone(&model_requests);
                thread::spawn(move || {
                    handle_checkpoint_request(
                        &mut stream,
                        &checkpoint,
                        &requests,
                        etag_mode,
                        &model_requests,
                    )
                });
            }
        });
        format!("http://{address}")
    }

    fn handle_checkpoint_request(
        stream: &mut TcpStream,
        checkpoint: &[u8],
        requests: &Mutex<Vec<String>>,
        etag_mode: ModelEtagMode,
        model_requests: &AtomicUsize,
    ) {
        let mut bytes = vec![0_u8; 8192];
        let Ok(read) = stream.read(&mut bytes) else {
            return;
        };
        let request = String::from_utf8_lossy(&bytes[..read]).into_owned();
        requests.lock().unwrap().push(request.clone());
        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");
        let response = if path.ends_with("/config.json") {
            http_response(
                "200 OK",
                &[("ETag", "\"config-id\"")],
                br#"{"model_type":"llama","hidden_size":2,"num_hidden_layers":2,"tie_word_embeddings":true}"#,
            )
        } else if path.ends_with("/model.safetensors.index.json") {
            http_response("404 Not Found", &[], b"")
        } else if path.ends_with("/model.safetensors") {
            let request_index = model_requests.fetch_add(1, Ordering::SeqCst);
            let etag = match etag_mode {
                ModelEtagMode::Static(etag) => etag,
                ModelEtagMode::ChangeAfterHeader if request_index < 2 => Some("\"model-id\""),
                ModelEtagMode::ChangeAfterHeader => Some("\"changed-id\""),
            };
            checkpoint_range_response(&request, checkpoint, etag)
        } else {
            http_response("404 Not Found", &[], b"")
        };
        let _ = stream.write_all(&response);
    }

    fn checkpoint_range_response(request: &str, checkpoint: &[u8], etag: Option<&str>) -> Vec<u8> {
        let range = request
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("range").then_some(value.trim())
            })
            .unwrap();
        let bounds = range.strip_prefix("bytes=").unwrap();
        let (start, end) = bounds.split_once('-').unwrap();
        let start = start.parse::<usize>().unwrap();
        let end = end.parse::<usize>().unwrap();
        let content_range = format!("bytes {start}-{end}/{}", checkpoint.len());
        let mut headers = vec![("Content-Range", content_range.as_str())];
        if let Some(etag) = etag {
            headers.push(("ETag", etag));
        }
        http_response("206 Partial Content", &headers, &checkpoint[start..=end])
    }

    fn http_response(status: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
        let headers = headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>();
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{headers}Connection: close\r\n\r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect()
    }
}
