use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result, ensure};

use super::{
    layout,
    locking::AdvisoryFileLock,
    materialize::{SafetensorsStageMaterializer, ensure_source_identity},
    types::{
        PreparedStage, SafetensorsSourceShard, SafetensorsStagePlan, SafetensorsStageRequest,
        SafetensorsStageTensorFile, SafetensorsStageTensorVisitReport, SelectedTensor,
    },
};

static TENSOR_VISIT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

impl SafetensorsStageMaterializer {
    /// Prepares a verified sequential visit before any tensor payload is fetched.
    pub fn prepare_tensor_visit(
        &self,
        request: SafetensorsStageRequest,
    ) -> Result<SafetensorsStageTensorVisit<'_>> {
        let request = request.normalized()?;
        let prepared = layout::prepare(&self.remote, &request)?;
        Ok(SafetensorsStageTensorVisit {
            materializer: self,
            prepared,
        })
    }
}

/// A verified stage selection ready for sequential tensor-range consumption.
///
/// The plan, checkpoint identity, and source config are available before
/// `visit_tensor_files` starts fetching tensor payloads, so a consumer can
/// construct its destination model or derived-cache metadata first.
pub struct SafetensorsStageTensorVisit<'a> {
    materializer: &'a SafetensorsStageMaterializer,
    prepared: PreparedStage,
}

impl SafetensorsStageTensorVisit<'_> {
    pub fn plan(&self) -> &SafetensorsStagePlan {
        &self.prepared.plan
    }

    pub fn checkpoint_sha256(&self) -> &str {
        &self.prepared.checkpoint_sha256
    }

    pub fn config(&self) -> &[u8] {
        &self.prepared.config
    }

    pub fn config_sha256(&self) -> &str {
        &self.prepared.config_sha256
    }

    /// Visits selected tensors as ephemeral one-tensor SafeTensors files.
    ///
    /// Each file is removed immediately after `visitor` returns. Consumers
    /// using mmap or lazy device graphs must evaluate and synchronize all work
    /// that reads the file before returning from the callback. This lets a
    /// backend quantize or transform exact HTTP ranges sequentially without
    /// retaining the complete BF16 stage artifact on disk.
    pub fn visit_tensor_files<F>(self, visitor: F) -> Result<SafetensorsStageTensorVisitReport>
    where
        F: FnMut(&SafetensorsStageTensorFile) -> Result<()>,
    {
        self.visit_tensor_files_cancellable(|| false, visitor)
    }

    /// Visits selected tensors while cooperatively checking for cancellation.
    ///
    /// Cancellation is checked before each payload request and immediately
    /// before and after the visitor. An in-flight HTTP response or visitor call
    /// is allowed to finish; its ephemeral file is removed when this method
    /// returns.
    pub fn visit_tensor_files_cancellable<F, C>(
        self,
        mut is_cancelled: C,
        mut visitor: F,
    ) -> Result<SafetensorsStageTensorVisitReport>
    where
        F: FnMut(&SafetensorsStageTensorFile) -> Result<()>,
        C: FnMut() -> bool,
    {
        ensure!(!is_cancelled(), "SafeTensors tensor visit cancelled");
        let directory = EphemeralTensorDirectory::create(&self.materializer.cache_root)?;
        let source_shards = self
            .prepared
            .source_shards
            .iter()
            .map(|shard| (shard.file.as_str(), shard))
            .collect::<BTreeMap<_, _>>();
        let mut tensors = self.prepared.tensors.iter().collect::<Vec<_>>();
        tensors.sort_by(|left, right| {
            (&left.source_file, left.source_range.start)
                .cmp(&(&right.source_file, right.source_range.start))
        });

        let mut visited_tensor_bytes = 0_u64;
        let mut temporary_file_peak_bytes = 0_u64;
        for (index, tensor) in tensors.iter().enumerate() {
            ensure!(!is_cancelled(), "SafeTensors tensor visit cancelled");
            let source = source_shards
                .get(tensor.source_file.as_str())
                .with_context(|| format!("missing source identity for {}", tensor.source_file))?;
            let path = directory
                .path()
                .join(format!("tensor-{index:06}.safetensors"));
            let file_bytes =
                self.materializer
                    .write_tensor_file(&path, &self.prepared.plan, tensor, source)?;
            temporary_file_peak_bytes = temporary_file_peak_bytes.max(file_bytes);
            ensure!(!is_cancelled(), "SafeTensors tensor visit cancelled");
            visitor(&tensor_file(tensor, path.clone(), file_bytes))?;
            ensure!(!is_cancelled(), "SafeTensors tensor visit cancelled");
            fs::remove_file(&path)
                .with_context(|| format!("remove ephemeral tensor file {}", path.display()))?;
            visited_tensor_bytes = visited_tensor_bytes
                .checked_add(tensor.source_range.len())
                .context("visited tensor byte count overflow")?;
        }

        ensure!(
            tensors.len() == self.prepared.plan.selected_tensor_count
                && visited_tensor_bytes == self.prepared.plan.selected_tensor_bytes,
            "sequential tensor visit did not cover the selected stage"
        );
        Ok(SafetensorsStageTensorVisitReport {
            plan: self.prepared.plan,
            visited_tensor_count: tensors.len(),
            visited_tensor_bytes,
            source_range_request_count: tensors.len(),
            temporary_file_peak_bytes,
        })
    }
}

impl SafetensorsStageMaterializer {
    fn write_tensor_file(
        &self,
        path: &Path,
        plan: &super::types::SafetensorsStagePlan,
        tensor: &SelectedTensor,
        source: &SafetensorsSourceShard,
    ) -> Result<u64> {
        let (header, header_len) = one_tensor_header(tensor)?;
        let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header)?;

        let url = self
            .remote
            .url(&plan.repo, &plan.revision, &tensor.source_file)?;
        let expected_etag = source
            .etag
            .as_deref()
            .context("planned SafeTensors shard has no ETag")?;
        let response = self.remote.exact_range_if_range(
            url,
            tensor.source_range.start..tensor.source_range.end_exclusive,
            expected_etag,
        )?;
        ensure!(
            response.total_file_bytes == source.file_bytes,
            "SafeTensors shard {} changed size during tensor visit",
            tensor.source_file
        );
        ensure_source_identity(source, response.etag())?;
        let copied = response.copy_to(&mut writer)?;
        ensure!(
            copied == tensor.source_range.len(),
            "ephemeral SafeTensors tensor payload length mismatch"
        );
        writer.flush()?;
        let file_bytes = 8_u64
            .checked_add(header_len)
            .and_then(|bytes| bytes.checked_add(copied))
            .context("ephemeral SafeTensors file length overflow")?;
        ensure!(
            fs::metadata(path)?.len() == file_bytes,
            "ephemeral SafeTensors file length mismatch"
        );
        Ok(file_bytes)
    }
}

fn one_tensor_header(tensor: &SelectedTensor) -> Result<(Vec<u8>, u64)> {
    let tensor_bytes = tensor.source_range.len();
    let mut header = tensor.header.clone();
    header.data_offsets = [0, tensor_bytes];
    let mut header = serde_json::to_vec(&BTreeMap::from([(tensor.name.clone(), header)]))?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let header_len = u64::try_from(header.len()).context("one-tensor header is too large")?;
    Ok((header, header_len))
}

fn tensor_file(
    tensor: &SelectedTensor,
    path: PathBuf,
    file_bytes: u64,
) -> SafetensorsStageTensorFile {
    SafetensorsStageTensorFile {
        name: tensor.name.clone(),
        dtype: tensor.header.dtype.clone(),
        shape: tensor.header.shape.clone(),
        source_file: tensor.source_file.clone(),
        source_range: tensor.source_range.clone(),
        path,
        file_bytes,
    }
}

struct EphemeralTensorDirectory {
    path: PathBuf,
    lock_path: PathBuf,
    _lock: AdvisoryFileLock,
}

impl EphemeralTensorDirectory {
    fn create(cache_root: &Path) -> Result<Self> {
        fs::create_dir_all(cache_root)
            .with_context(|| format!("create SafeTensors cache root {}", cache_root.display()))?;
        remove_abandoned_tensor_directories(cache_root)?;
        for _ in 0..100 {
            let sequence = TENSOR_VISIT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let base = format!(".tensor-visit.{}.{}", std::process::id(), sequence);
            let path = cache_root.join(format!("{base}.partial"));
            let lock_path = cache_root.join(format!("{base}.lock"));
            let lock = AdvisoryFileLock::acquire(&lock_path)?;
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        lock_path,
                        _lock: lock,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    drop(lock);
                    let _ = fs::remove_file(&lock_path);
                }
                Err(error) => {
                    drop(lock);
                    let _ = fs::remove_file(&lock_path);
                    return Err(error)
                        .with_context(|| format!("create tensor visit dir {}", path.display()));
                }
            }
        }
        anyhow::bail!("could not allocate a unique SafeTensors tensor visit directory")
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for EphemeralTensorDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
        let _ = fs::remove_file(&self.lock_path);
    }
}

#[cfg(unix)]
fn remove_abandoned_tensor_directories(cache_root: &Path) -> Result<()> {
    for entry in fs::read_dir(cache_root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(base) = name.strip_suffix(".partial") else {
            continue;
        };
        if !base.starts_with(".tensor-visit.") || !entry.file_type()?.is_dir() {
            continue;
        }
        let lock_path = cache_root.join(format!("{base}.lock"));
        let Some(lock) = AdvisoryFileLock::try_acquire(&lock_path)? else {
            continue;
        };
        fs::remove_dir_all(entry.path())?;
        drop(lock);
        fs::remove_file(lock_path)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn remove_abandoned_tensor_directories(_cache_root: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use model_artifact::safetensors::TensorHeader;

    use super::*;

    #[test]
    fn one_tensor_header_rebases_payload_offsets() {
        let tensor = SelectedTensor {
            name: "model.layers.2.weight".to_string(),
            source_file: "model.safetensors".to_string(),
            source_range: super::super::types::ByteRange {
                start: 100,
                end_exclusive: 108,
            },
            header: TensorHeader {
                dtype: "F32".to_string(),
                shape: vec![2],
                data_offsets: [92, 100],
            },
        };

        let (header, _) = one_tensor_header(&tensor).unwrap();
        let parsed: BTreeMap<String, TensorHeader> = serde_json::from_slice(&header).unwrap();

        assert_eq!(parsed[&tensor.name].data_offsets, [0, 8]);
    }

    #[test]
    #[ignore = "reads pinned Nemotron-H config, index, and one shard header"]
    fn plans_real_nemotron_h_moe_layer_without_tensor_payloads() {
        let cache = tempfile::tempdir().unwrap();
        let materializer =
            SafetensorsStageMaterializer::new(cache.path().join("cache"), None, None).unwrap();
        let visit = materializer
            .prepare_tensor_visit(SafetensorsStageRequest {
                repo: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16".to_string(),
                revision: "97ab8012882a655dc38df4fee47422aca9caca07".to_string(),
                layer_start: 1,
                layer_end: 2,
                include_prefixes: Vec::new(),
            })
            .unwrap();
        let plan = visit.plan();

        assert_eq!(plan.selected_tensor_count, 261);
        assert_eq!(plan.selected_tensor_bytes, 2_594_936_576);
        assert_eq!(plan.largest_selected_tensor_bytes, 19_955_712);
        assert_eq!(plan.source_shard_count, 1);
        assert_eq!(plan.source_shard_bytes, 4_991_210_024);
        assert_eq!(plan.range_request_count, 2);
        assert!(plan.source_shard_bytes_avoided > 2_396_000_000);
        assert_eq!(visit.checkpoint_sha256().len(), 64);
    }

    #[test]
    #[ignore = "downloads one layer from a pinned Hugging Face SafeTensors checkpoint"]
    fn visits_real_smollm2_layer_without_retaining_source_shard() {
        let cache = tempfile::tempdir().unwrap();
        let cache_root = cache.path().join("cache");
        let materializer =
            SafetensorsStageMaterializer::new(cache_root.clone(), None, None).unwrap();
        let mut visited = Vec::new();
        let request = SafetensorsStageRequest {
            repo: "HuggingFaceTB/SmolLM2-135M-Instruct".to_string(),
            revision: "12fd25f77366fa6b3b4b768ec3050bf629380bac".to_string(),
            layer_start: 14,
            layer_end: 15,
            include_prefixes: Vec::new(),
        };

        let visit = materializer.prepare_tensor_visit(request).unwrap();
        assert_eq!(visit.checkpoint_sha256(), visit.plan().checkpoint_sha256);
        assert_eq!(visit.config_sha256().len(), 64);
        assert!(!visit.config().is_empty());
        let report = visit
            .visit_tensor_files(|tensor| {
                assert!(tensor.name.starts_with("model.layers.14."));
                assert!(tensor.path.is_file());
                assert_eq!(fs::read_dir(tensor.path.parent().unwrap())?.count(), 1);
                visited.push((tensor.name.clone(), tensor.source_range.len()));
                Ok(())
            })
            .unwrap();

        assert_eq!(report.visited_tensor_count, visited.len());
        assert_eq!(report.source_range_request_count, visited.len());
        assert_eq!(
            report.visited_tensor_bytes,
            visited.iter().map(|(_, bytes)| bytes).sum::<u64>()
        );
        assert!(report.visited_tensor_bytes < report.plan.source_shard_bytes);
        assert!(fs::read_dir(cache_root).unwrap().next().is_none());
        eprintln!(
            "visited {} tensors / {} bytes; source shard {} bytes; peak temporary file {} bytes",
            report.visited_tensor_count,
            report.visited_tensor_bytes,
            report.plan.source_shard_bytes,
            report.temporary_file_peak_bytes
        );
    }

    #[cfg(unix)]
    #[test]
    fn removes_abandoned_tensor_visit_directory() {
        let cache = tempfile::tempdir().unwrap();
        let cache_root = cache.path().join("cache");
        let abandoned = cache_root.join(".tensor-visit.999.0.partial");
        fs::create_dir_all(&abandoned).unwrap();
        fs::write(abandoned.join("tensor-000000.safetensors"), b"abandoned").unwrap();

        let active = EphemeralTensorDirectory::create(&cache_root).unwrap();

        assert!(!abandoned.exists());
        drop(active);
        assert!(fs::read_dir(cache_root).unwrap().next().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn keeps_concurrent_tensor_visit_directory_locked() {
        let cache = tempfile::tempdir().unwrap();
        let cache_root = cache.path().join("cache");
        let first = EphemeralTensorDirectory::create(&cache_root).unwrap();
        let first_path = first.path().to_path_buf();

        let second = EphemeralTensorDirectory::create(&cache_root).unwrap();

        assert!(first_path.is_dir());
        drop(second);
        assert!(first_path.is_dir());
        drop(first);
        assert!(fs::read_dir(cache_root).unwrap().next().is_none());
    }
}
