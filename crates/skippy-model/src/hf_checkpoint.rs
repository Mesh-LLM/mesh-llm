//! Validated SafeTensors checkpoint discovery and tensor access.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::ConvertOutputType;

#[derive(Debug, Serialize)]
pub struct HfCheckpointPlan {
    pub source: PathBuf,
    pub safetensor_count: usize,
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub largest_tensor_bytes: u64,
    pub source_windows: Vec<HfSourceWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_verification: Option<HfStreamVerification>,
}

#[derive(Debug, Serialize)]
pub struct HfSourceWindow {
    pub index: u32,
    pub files: Vec<PathBuf>,
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub largest_tensor_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct HfStreamVerification {
    pub safetensor_count: usize,
    pub tensor_count: usize,
    pub streamed_bytes: u64,
    pub buffer_size: usize,
}

#[derive(Debug)]
struct SafetensorSummary {
    path: PathBuf,
    tensor_count: usize,
    total_tensor_bytes: u64,
    largest_tensor_bytes: u64,
}

#[derive(Debug)]
pub struct SafetensorFile {
    path: PathBuf,
    data_start: u64,
    mapping: Mmap,
    tensors: BTreeMap<String, SafetensorTensorInfo>,
}

impl SafetensorFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the file is opened read-only and Skippy never mutates or truncates checkpoint
        // files while a SafetensorFile exists. The mapping is owned by this object and therefore
        // outlives every tensor slice exposed through its methods.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;
        let parsed = SafeTensors::deserialize(&mapping)
            .with_context(|| format!("validate safetensors container {}", path.display()))?;
        let data_start = parsed
            .tensors()
            .into_iter()
            .map(|(_, tensor)| tensor.data().as_ptr() as usize - mapping.as_ptr() as usize)
            .min()
            .unwrap_or(mapping.len()) as u64;
        let mut tensors = BTreeMap::new();
        for (name, tensor) in parsed.tensors() {
            let absolute_start = tensor.data().as_ptr() as usize - mapping.as_ptr() as usize;
            let absolute_end = absolute_start
                .checked_add(tensor.data().len())
                .with_context(|| format!("data range overflow for tensor {name}"))?;
            ensure!(
                absolute_start >= data_start as usize,
                "tensor {name} precedes data section"
            );
            let shape = tensor
                .shape()
                .iter()
                .map(|&dim| u64::try_from(dim).context("tensor dimension does not fit u64"))
                .collect::<Result<Vec<_>>>()?;
            tensors.insert(
                name.to_string(),
                SafetensorTensorInfo {
                    name: name.to_string(),
                    dtype: format!("{:?}", tensor.dtype()),
                    shape,
                    relative_data_offsets: [
                        (absolute_start - data_start as usize) as u64,
                        (absolute_end - data_start as usize) as u64,
                    ],
                    absolute_data_start: absolute_start,
                    byte_len: tensor.data().len(),
                },
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_start,
            mapping,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn tensors(&self) -> &BTreeMap<String, SafetensorTensorInfo> {
        &self.tensors
    }

    pub fn stream_tensor<W: Write>(
        &self,
        name: &str,
        writer: &mut W,
        buffer_size: usize,
    ) -> Result<u64> {
        self.stream_tensor_chunks(name, buffer_size, |chunk| {
            writer.write_all(chunk).context("write tensor bytes")
        })
    }

    pub fn stream_tensor_chunks<F>(
        &self,
        name: &str,
        buffer_size: usize,
        mut on_chunk: F,
    ) -> Result<u64>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor {name} not found in {}", self.path.display()))?;
        ensure!(buffer_size > 0, "buffer_size must be greater than zero");
        let range = tensor.absolute_data_range();
        let data = self
            .mapping
            .get(range.clone())
            .with_context(|| format!("tensor {name} range is outside {}", self.path.display()))?;
        for chunk in data.chunks(buffer_size) {
            on_chunk(chunk)?;
        }
        Ok(data.len() as u64)
    }
}

#[derive(Debug)]
pub struct SafetensorTensorInfo {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    relative_data_offsets: [u64; 2],
    absolute_data_start: usize,
    byte_len: usize,
}

impl SafetensorTensorInfo {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn relative_data_offsets(&self) -> [u64; 2] {
        self.relative_data_offsets
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len as u64
    }

    fn absolute_data_range(&self) -> Range<usize> {
        self.absolute_data_start..self.absolute_data_start + self.byte_len
    }
}

pub fn inspect_hf_checkpoint(
    source: &Path,
    max_memory_bytes: Option<u64>,
    staging_fraction: f64,
) -> Result<HfCheckpointPlan> {
    ensure!(
        staging_fraction > 0.0 && staging_fraction <= 1.0,
        "--staging-fraction must be in the range (0, 1]"
    );
    let safetensors = discover_safetensors(source)?;
    ensure!(
        !safetensors.is_empty(),
        "no safetensors files found under {}",
        source.display()
    );
    let mut summaries = safetensors
        .iter()
        .map(|path| summarize_safetensor(path))
        .collect::<Result<Vec<_>>>()?;
    summaries.sort_by(|a, b| a.path.cmp(&b.path));
    let tensor_count = summaries.iter().map(|summary| summary.tensor_count).sum();
    let total_tensor_bytes = summaries
        .iter()
        .map(|summary| summary.total_tensor_bytes)
        .sum();
    let largest_tensor_bytes = summaries
        .iter()
        .map(|summary| summary.largest_tensor_bytes)
        .max()
        .unwrap_or(0);
    let source_windows = plan_source_windows(&summaries, max_memory_bytes, staging_fraction)?;
    Ok(HfCheckpointPlan {
        source: source.to_path_buf(),
        safetensor_count: summaries.len(),
        tensor_count,
        total_tensor_bytes,
        largest_tensor_bytes,
        source_windows,
        stream_verification: None,
    })
}

pub fn verify_hf_checkpoint_tensor_streams(
    source: &Path,
    buffer_size: usize,
) -> Result<HfStreamVerification> {
    let safetensors = discover_safetensors(source)?;
    let mut sink = std::io::sink();
    let mut tensor_count = 0_usize;
    let mut streamed_bytes = 0_u64;
    for path in &safetensors {
        let safetensor = SafetensorFile::open(path)?;
        ensure!(
            safetensor.path().is_file(),
            "safetensor path is not a file: {}",
            safetensor.path().display()
        );
        ensure!(
            safetensor.data_start() >= 8,
            "invalid safetensors data start in {}",
            safetensor.path().display()
        );
        for tensor in safetensor.tensors().values() {
            ensure!(
                !tensor.name().is_empty(),
                "safetensor tensor has empty name"
            );
            ensure!(
                dtype_size(tensor.dtype()).is_some(),
                "unsupported safetensors dtype {}",
                tensor.dtype()
            );
            let offsets = tensor.relative_data_offsets();
            ensure!(
                offsets[0] <= offsets[1],
                "invalid safetensors offsets for {}",
                tensor.name()
            );
            let _rank = tensor.shape().len();
            streamed_bytes += safetensor.stream_tensor(tensor.name(), &mut sink, buffer_size)?;
            tensor_count += 1;
        }
    }
    Ok(HfStreamVerification {
        safetensor_count: safetensors.len(),
        tensor_count,
        streamed_bytes,
        buffer_size,
    })
}

pub fn open_safetensor_files(source: &Path) -> Result<Vec<SafetensorFile>> {
    discover_safetensors(source)?
        .iter()
        .map(|path| SafetensorFile::open(path))
        .collect()
}

pub fn resolve_auto_output_type(
    source: &Path,
    requested: ConvertOutputType,
) -> Result<ConvertOutputType> {
    if requested != ConvertOutputType::Auto {
        return Ok(requested);
    }
    for safetensor in open_safetensor_files(source)? {
        for tensor in safetensor.tensors().values() {
            if tensor.shape().len() < 2 {
                continue;
            }
            match tensor.dtype() {
                "BF16" => return Ok(ConvertOutputType::Bf16),
                "F16" => return Ok(ConvertOutputType::F16),
                _ => {}
            }
        }
    }
    Ok(ConvertOutputType::F16)
}

pub fn discover_safetensors(source: &Path) -> Result<Vec<PathBuf>> {
    ensure!(
        source.is_dir(),
        "HF checkpoint source must be a directory: {}",
        source.display()
    );
    let mut indexed = discover_indexed_safetensors(source)?;
    if !indexed.is_empty() {
        let mtp_sidecar = source.join("mtp.safetensors");
        if mtp_sidecar.is_file() && !indexed.contains(&mtp_sidecar) {
            indexed.push(mtp_sidecar);
            indexed.sort();
        }
        return Ok(indexed);
    }
    indexed = fs::read_dir(source)
        .with_context(|| format!("read checkpoint directory {}", source.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    indexed.sort();
    Ok(indexed)
}

fn discover_indexed_safetensors(source: &Path) -> Result<Vec<PathBuf>> {
    let index_path = source.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Ok(Vec::new());
    }
    let index: SafetensorIndex = serde_json::from_slice(
        &fs::read(&index_path).with_context(|| format!("read {}", index_path.display()))?,
    )
    .with_context(|| format!("parse {}", index_path.display()))?;
    let mut files = index
        .weight_map
        .values()
        .map(|name| indexed_shard_path(source, name))
        .collect::<Result<BTreeSet<_>>>()?
        .into_iter()
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn indexed_shard_path(source: &Path, name: &str) -> Result<PathBuf> {
    let relative = Path::new(name);
    ensure!(
        !name.is_empty()
            && !relative.is_absolute()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "SafeTensors index shard path must remain within the checkpoint directory: {name:?}"
    );
    ensure!(
        relative
            .extension()
            .is_some_and(|extension| extension == "safetensors"),
        "SafeTensors index shard must have a .safetensors extension: {name:?}"
    );
    let path = source.join(relative);
    ensure!(
        path.is_file(),
        "SafeTensors index shard does not exist: {}",
        path.display()
    );
    Ok(path)
}

#[derive(Debug, Deserialize)]
struct SafetensorIndex {
    weight_map: BTreeMap<String, String>,
}

fn summarize_safetensor(path: &Path) -> Result<SafetensorSummary> {
    let safetensor = SafetensorFile::open(path)?;
    let tensor_count = safetensor.tensors.len();
    let total_tensor_bytes = safetensor
        .tensors
        .values()
        .map(SafetensorTensorInfo::byte_len)
        .sum();
    let largest_tensor_bytes = safetensor
        .tensors
        .values()
        .map(SafetensorTensorInfo::byte_len)
        .max()
        .unwrap_or(0);
    Ok(SafetensorSummary {
        path: path.to_path_buf(),
        tensor_count,
        total_tensor_bytes,
        largest_tensor_bytes,
    })
}

fn dtype_size(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "I8" | "U8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

fn plan_source_windows(
    summaries: &[SafetensorSummary],
    max_memory_bytes: Option<u64>,
    staging_fraction: f64,
) -> Result<Vec<HfSourceWindow>> {
    let budget = max_memory_bytes
        .map(|memory| ((memory as f64) * staging_fraction).floor() as u64)
        .unwrap_or(u64::MAX)
        .max(1);
    let mut windows = Vec::new();
    let mut current = SourceWindowBuilder::new(1);
    for summary in summaries {
        if !current.is_empty() && current.total_tensor_bytes + summary.total_tensor_bytes > budget {
            windows.push(current.finish());
            current = SourceWindowBuilder::new(windows.len() as u32 + 1);
        }
        current.push(summary);
    }
    if !current.is_empty() {
        windows.push(current.finish());
    }
    Ok(windows)
}

struct SourceWindowBuilder {
    index: u32,
    files: Vec<PathBuf>,
    tensor_count: usize,
    total_tensor_bytes: u64,
    largest_tensor_bytes: u64,
}

impl SourceWindowBuilder {
    fn new(index: u32) -> Self {
        Self {
            index,
            files: Vec::new(),
            tensor_count: 0,
            total_tensor_bytes: 0,
            largest_tensor_bytes: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn push(&mut self, summary: &SafetensorSummary) {
        self.files.push(summary.path.clone());
        self.tensor_count += summary.tensor_count;
        self.total_tensor_bytes += summary.total_tensor_bytes;
        self.largest_tensor_bytes = self.largest_tensor_bytes.max(summary.largest_tensor_bytes);
    }

    fn finish(self) -> HfSourceWindow {
        HfSourceWindow {
            index: self.index,
            files: self.files,
            tensor_count: self.tensor_count,
            total_tensor_bytes: self.total_tensor_bytes,
            largest_tensor_bytes: self.largest_tensor_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn plans_unindexed_safetensors_under_memory_budget() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        write_safetensor(
            &root.join("model-00001-of-00002.safetensors"),
            &[("a.weight", "F32", &[2], &[1, 2, 3, 4, 5, 6, 7, 8])],
        );
        write_safetensor(
            &root.join("model-00002-of-00002.safetensors"),
            &[("b.weight", "BF16", &[4], &[1, 2, 3, 4, 5, 6, 7, 8])],
        );

        let plan = inspect_hf_checkpoint(&root, Some(12), 1.0).unwrap();

        assert_eq!(plan.safetensor_count, 2);
        assert_eq!(plan.tensor_count, 2);
        assert_eq!(plan.total_tensor_bytes, 16);
        assert_eq!(plan.source_windows.len(), 2);
        assert_eq!(plan.source_windows[0].total_tensor_bytes, 8);
        assert_eq!(plan.source_windows[1].total_tensor_bytes, 8);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn uses_index_weight_map_when_present() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        write_safetensor(
            &root.join("shard-b.safetensors"),
            &[("b.weight", "F32", &[1], &[1, 2, 3, 4])],
        );
        write_safetensor(
            &root.join("shard-a.safetensors"),
            &[("a.weight", "F32", &[1], &[1, 2, 3, 4])],
        );
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"a.weight":"shard-a.safetensors","b.weight":"shard-b.safetensors"}}"#,
        )
        .unwrap();

        let plan = inspect_hf_checkpoint(&root, None, 1.0).unwrap();

        assert_eq!(plan.safetensor_count, 2);
        assert_eq!(plan.source_windows.len(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_index_shards_that_escape_checkpoint_directory() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"a.weight":"../outside.safetensors"}}"#,
        )
        .unwrap();

        let error = discover_safetensors(&root).unwrap_err().to_string();

        assert!(error.contains("must remain within the checkpoint directory"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn includes_unindexed_mtp_sidecar_with_indexed_checkpoint() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        write_safetensor(
            &root.join("shard-a.safetensors"),
            &[("a.weight", "F32", &[1], &[1, 2, 3, 4])],
        );
        write_safetensor(
            &root.join("mtp.safetensors"),
            &[("model.mtp.layers.0.weight", "F32", &[1], &[5, 6, 7, 8])],
        );
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"a.weight":"shard-a.safetensors"}}"#,
        )
        .unwrap();

        let files = discover_safetensors(&root).unwrap();

        assert_eq!(
            files,
            vec![
                root.join("mtp.safetensors"),
                root.join("shard-a.safetensors")
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn streams_tensor_bytes_without_reading_neighbor_tensors() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let path = root.join("model.safetensors");
        write_safetensor(
            &path,
            &[
                ("a.weight", "U8", &[4], &[1, 2, 3, 4]),
                ("b.weight", "U8", &[3], &[9, 8, 7]),
            ],
        );

        let safetensor = SafetensorFile::open(&path).unwrap();
        let tensor = safetensor.tensors().get("b.weight").unwrap();
        let mut output = Vec::new();
        let copied = safetensor
            .stream_tensor("b.weight", &mut output, 2)
            .unwrap();

        assert_eq!(safetensor.path(), path);
        assert!(safetensor.data_start() > 8);
        assert_eq!(tensor.name(), "b.weight");
        assert_eq!(tensor.dtype(), "U8");
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.relative_data_offsets(), [4, 7]);
        assert_eq!(tensor.byte_len(), 3);
        assert_eq!(copied, 3);
        assert_eq!(output, vec![9, 8, 7]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_auto_output_type_from_first_rank_two_float_tensor() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        write_safetensor(
            &root.join("model.safetensors"),
            &[
                ("a.bias", "BF16", &[4], &[1, 2, 3, 4, 5, 6, 7, 8]),
                ("b.weight", "F16", &[2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]),
            ],
        );

        let output_type = resolve_auto_output_type(&root, ConvertOutputType::Auto).unwrap();

        assert_eq!(output_type, ConvertOutputType::F16);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_auto_output_type_to_bf16_when_rank_two_bf16_appears_first() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        write_safetensor(
            &root.join("model.safetensors"),
            &[("a.weight", "BF16", &[2, 2], &[1, 2, 3, 4, 5, 6, 7, 8])],
        );

        let output_type = resolve_auto_output_type(&root, ConvertOutputType::Auto).unwrap();

        assert_eq!(output_type, ConvertOutputType::Bf16);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_auto_output_type_to_f16_when_checkpoint_has_no_float_matrix() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        write_safetensor(
            &root.join("model.safetensors"),
            &[
                ("a.bias", "BF16", &[4], &[1, 2, 3, 4, 5, 6, 7, 8]),
                (
                    "b.count",
                    "I32",
                    &[2, 2],
                    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                ),
            ],
        );

        let output_type = resolve_auto_output_type(&root, ConvertOutputType::Auto).unwrap();

        assert_eq!(output_type, ConvertOutputType::F16);
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "skippy-hf-checkpoint-{}-{nanos}-{counter}",
            std::process::id()
        ))
    }

    fn write_safetensor(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
        let mut offset = 0_u64;
        let mut entries = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let end = offset + bytes.len() as u64;
            entries.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let header = serde_json::Value::Object(entries).to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for (_, _, _, tensor_bytes) in tensors {
            bytes.extend_from_slice(tensor_bytes);
        }
        fs::write(path, bytes).unwrap();
    }
}
