use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::memory_budget::MemorySize;
use crate::types::ConvertOutputType;

#[derive(Debug, Serialize)]
pub(crate) struct HfCheckpointPlan {
    pub(crate) source: PathBuf,
    pub(crate) safetensor_count: usize,
    pub(crate) tensor_count: usize,
    pub(crate) total_tensor_bytes: u64,
    pub(crate) largest_tensor_bytes: u64,
    pub(crate) source_windows: Vec<HfSourceWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream_verification: Option<HfStreamVerification>,
}

#[derive(Debug, Serialize)]
pub(crate) struct HfSourceWindow {
    pub(crate) index: u32,
    pub(crate) files: Vec<PathBuf>,
    pub(crate) tensor_count: usize,
    pub(crate) total_tensor_bytes: u64,
    pub(crate) largest_tensor_bytes: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct HfStreamVerification {
    pub(crate) safetensor_count: usize,
    pub(crate) tensor_count: usize,
    pub(crate) streamed_bytes: u64,
    pub(crate) buffer_size: usize,
}

#[derive(Debug)]
struct SafetensorSummary {
    path: PathBuf,
    tensor_count: usize,
    total_tensor_bytes: u64,
    largest_tensor_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct SafetensorTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
pub(crate) struct SafetensorFile {
    path: PathBuf,
    data_start: u64,
    tensors: BTreeMap<String, SafetensorTensorInfo>,
}

impl SafetensorFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let (data_start, raw_tensors) = read_safetensor_header(path)?;
        let file_len = fs::metadata(path)
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        let mut tensors = BTreeMap::new();
        for (name, tensor) in raw_tensors {
            tensors.insert(
                name.clone(),
                SafetensorTensorInfo::from_raw(name, tensor, data_start, file_len)?,
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_start,
            tensors,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn data_start(&self) -> u64 {
        self.data_start
    }

    pub(crate) fn tensors(&self) -> &BTreeMap<String, SafetensorTensorInfo> {
        &self.tensors
    }

    pub(crate) fn stream_tensor<W: Write>(
        &self,
        name: &str,
        writer: &mut W,
        buffer_size: usize,
    ) -> Result<u64> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor {name} not found in {}", self.path.display()))?;
        stream_file_range(
            &self.path,
            tensor.absolute_data_range(),
            writer,
            buffer_size,
        )
    }

    pub(crate) fn stream_tensor_chunks<F>(
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
        stream_file_range_chunks(
            &self.path,
            tensor.absolute_data_range(),
            buffer_size,
            |chunk| on_chunk(chunk),
        )
    }
}

#[derive(Debug)]
pub(crate) struct SafetensorTensorInfo {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    relative_data_offsets: [u64; 2],
    absolute_data_start: u64,
    byte_len: u64,
}

impl SafetensorTensorInfo {
    fn from_raw(
        name: String,
        tensor: SafetensorTensor,
        data_start: u64,
        file_len: u64,
    ) -> Result<Self> {
        let relative_start = tensor.data_offsets[0];
        let relative_end = tensor.data_offsets[1];
        let byte_len = relative_end
            .checked_sub(relative_start)
            .with_context(|| format!("invalid data_offsets for tensor {name}"))?;
        let shape_bytes = tensor_shape_bytes(&tensor)
            .with_context(|| format!("validate shape for tensor {name}"))?;
        ensure!(
            byte_len == shape_bytes,
            "tensor {name} byte length {byte_len} does not match dtype/shape byte length {shape_bytes}"
        );
        let absolute_data_start = data_start
            .checked_add(relative_start)
            .with_context(|| format!("absolute data offset overflow for tensor {name}"))?;
        let absolute_data_end = absolute_data_start
            .checked_add(byte_len)
            .with_context(|| format!("absolute data end overflow for tensor {name}"))?;
        ensure!(
            absolute_data_end <= file_len,
            "tensor {name} extends past end of safetensors file"
        );
        Ok(Self {
            name,
            dtype: tensor.dtype,
            shape: tensor.shape,
            relative_data_offsets: tensor.data_offsets,
            absolute_data_start,
            byte_len,
        })
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn dtype(&self) -> &str {
        &self.dtype
    }

    pub(crate) fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub(crate) fn relative_data_offsets(&self) -> [u64; 2] {
        self.relative_data_offsets
    }

    pub(crate) fn byte_len(&self) -> u64 {
        self.byte_len
    }

    fn absolute_data_range(&self) -> Range<u64> {
        self.absolute_data_start..self.absolute_data_start + self.byte_len
    }
}

pub(crate) fn inspect_hf_checkpoint(
    source: &Path,
    max_memory: Option<MemorySize>,
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
    let source_windows = plan_source_windows(&summaries, max_memory, staging_fraction)?;
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

pub(crate) fn verify_hf_checkpoint_tensor_streams(
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

pub(crate) fn open_safetensor_files(source: &Path) -> Result<Vec<SafetensorFile>> {
    discover_safetensors(source)?
        .iter()
        .map(|path| SafetensorFile::open(path))
        .collect()
}

pub(crate) fn resolve_auto_output_type(
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

fn discover_safetensors(source: &Path) -> Result<Vec<PathBuf>> {
    ensure!(
        source.is_dir(),
        "HF checkpoint source must be a directory: {}",
        source.display()
    );
    let mut indexed = discover_indexed_safetensors(source)?;
    if !indexed.is_empty() {
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
        .map(|name| source.join(name))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
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

fn read_safetensor_header(path: &Path) -> Result<(u64, BTreeMap<String, SafetensorTensor>)> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("read safetensors header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes);
    ensure!(
        header_len <= 256 * 1024 * 1024,
        "safetensors header is unexpectedly large in {}: {header_len} bytes",
        path.display()
    );
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("read safetensors header from {}", path.display()))?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
        .with_context(|| format!("parse safetensors header {}", path.display()))?;
    let mut tensors = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        tensors.insert(name, serde_json::from_value(value)?);
    }
    let data_start = 8_u64
        .checked_add(header_len)
        .with_context(|| format!("safetensors data start overflow in {}", path.display()))?;
    Ok((data_start, tensors))
}

fn tensor_shape_bytes(tensor: &SafetensorTensor) -> Result<u64> {
    let element_size = dtype_size(&tensor.dtype)
        .ok_or_else(|| anyhow!("unsupported safetensors dtype {}", tensor.dtype))?;
    let elements = tensor.shape.iter().try_fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim).context("tensor shape overflow")
    })?;
    elements
        .checked_mul(element_size)
        .context("tensor byte size overflow")
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
    max_memory: Option<MemorySize>,
    staging_fraction: f64,
) -> Result<Vec<HfSourceWindow>> {
    let budget = max_memory
        .map(|memory| ((memory.bytes() as f64) * staging_fraction).floor() as u64)
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

fn stream_file_range<W: Write>(
    path: &Path,
    range: Range<u64>,
    writer: &mut W,
    buffer_size: usize,
) -> Result<u64> {
    stream_file_range_chunks(path, range, buffer_size, |chunk| {
        writer.write_all(chunk).context("write tensor bytes")
    })
}

fn stream_file_range_chunks<F>(
    path: &Path,
    range: Range<u64>,
    buffer_size: usize,
    mut on_chunk: F,
) -> Result<u64>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    ensure!(buffer_size > 0, "buffer_size must be greater than zero");
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    file.seek(SeekFrom::Start(range.start))
        .with_context(|| format!("seek {}", path.display()))?;
    let mut remaining = range.end - range.start;
    let mut copied = 0_u64;
    let mut buffer = vec![0_u8; buffer_size];
    while remaining > 0 {
        let read_len = buffer.len().min(remaining as usize);
        file.read_exact(&mut buffer[..read_len])
            .with_context(|| format!("read tensor bytes from {}", path.display()))?;
        on_chunk(&buffer[..read_len])?;
        remaining -= read_len as u64;
        copied += read_len as u64;
    }
    Ok(copied)
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

        let plan =
            inspect_hf_checkpoint(&root, Some(MemorySize::from_bytes_for_tests(12)), 1.0).unwrap();

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
