use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_ALIGNMENT_DEFAULT: u64 = 32;
const GGML_TYPE_F32: u32 = 0;
const TENSOR_SUMS_SUFFIX: &str = ".in_sum2";
const TENSOR_COUNTS_SUFFIX: &str = ".counts";

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

const KV_GENERAL_ALIGNMENT: &str = "general.alignment";
const KV_IMATRIX_DATASETS: &str = "imatrix.datasets";
const KV_IMATRIX_CHUNK_COUNT: &str = "imatrix.chunk_count";

pub(crate) struct NativeImatrix {
    _names: Vec<CString>,
    _values: Vec<Vec<f32>>,
    entries: Vec<skippy_ffi::LlamaModelImatrixData>,
    source_path: PathBuf,
    dataset: Option<String>,
    chunk_count: i32,
}

impl NativeImatrix {
    pub(crate) fn load(
        path: &Path,
        include_weights: &[String],
        exclude_weights: &[String],
    ) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("read imatrix file {}", path.display()))?;
        let loaded = if bytes.starts_with(GGUF_MAGIC) {
            load_gguf_imatrix(&bytes, path)?
        } else {
            load_legacy_imatrix(&bytes, path)?
        };
        Self::from_loaded(path, loaded, include_weights, exclude_weights)
    }

    pub(crate) fn as_ptr(&self) -> *const skippy_ffi::LlamaModelImatrixData {
        self.entries.as_ptr()
    }

    pub(crate) fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub(crate) fn dataset(&self) -> Option<&str> {
        self.dataset.as_deref()
    }

    pub(crate) fn chunk_count(&self) -> i32 {
        self.chunk_count
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len().saturating_sub(1)
    }

    fn from_loaded(
        path: &Path,
        loaded: LoadedImatrix,
        include_weights: &[String],
        exclude_weights: &[String],
    ) -> Result<Self> {
        let mut selected = loaded
            .entries
            .into_iter()
            .filter(|entry| include_exclude_match(&entry.name, include_weights, exclude_weights))
            .collect::<Vec<_>>();
        selected.sort_by(|a, b| a.name.cmp(&b.name));
        ensure!(
            !selected.is_empty(),
            "imatrix filters removed all entries from {}",
            path.display()
        );

        let mut names = Vec::with_capacity(selected.len());
        let mut values = Vec::with_capacity(selected.len());
        for entry in selected {
            names.push(CString::new(entry.name)?);
            values.push(entry.values);
        }
        let mut entries = names
            .iter()
            .zip(values.iter())
            .map(|(name, value)| skippy_ffi::LlamaModelImatrixData {
                name: name.as_ptr(),
                data: value.as_ptr(),
                size: value.len(),
            })
            .collect::<Vec<_>>();
        entries.push(skippy_ffi::LlamaModelImatrixData {
            name: std::ptr::null(),
            data: std::ptr::null(),
            size: 0,
        });
        Ok(Self {
            _names: names,
            _values: values,
            entries,
            source_path: path.to_path_buf(),
            dataset: loaded.dataset,
            chunk_count: loaded.chunk_count,
        })
    }
}

struct LoadedImatrix {
    entries: Vec<ImatrixEntry>,
    dataset: Option<String>,
    chunk_count: i32,
}

struct ImatrixEntry {
    name: String,
    values: Vec<f32>,
}

fn load_legacy_imatrix(bytes: &[u8], path: &Path) -> Result<LoadedImatrix> {
    let mut reader = Cursor::new(bytes.to_vec());
    let entry_count = read_i32(&mut reader)
        .with_context(|| format!("read imatrix entry count from {}", path.display()))?;
    ensure!(
        entry_count > 0,
        "imatrix file has no entries: {}",
        path.display()
    );
    let mut entries = Vec::with_capacity(entry_count as usize);
    for index in 0..entry_count {
        entries.push(read_legacy_imatrix_entry(&mut reader, index)?);
    }
    let (chunk_count, dataset) = read_legacy_imatrix_trailer(&mut reader)?;
    Ok(LoadedImatrix {
        entries,
        dataset,
        chunk_count,
    })
}

fn read_legacy_imatrix_entry(reader: &mut Cursor<Vec<u8>>, index: i32) -> Result<ImatrixEntry> {
    let name_len = read_i32(reader).with_context(|| format!("read imatrix name length {index}"))?;
    ensure!(name_len > 0, "imatrix entry {index} has empty name");
    let mut name_bytes = vec![0_u8; name_len as usize];
    reader
        .read_exact(&mut name_bytes)
        .with_context(|| format!("read imatrix name {index}"))?;
    let name = String::from_utf8(name_bytes).with_context(|| format!("imatrix name {index}"))?;
    let ncall = read_i32(reader).with_context(|| format!("read imatrix ncall for {name}"))?;
    let value_count =
        read_i32(reader).with_context(|| format!("read imatrix value count for {name}"))?;
    ensure!(value_count > 0, "imatrix entry {name} has no values");
    let mut values = (0..value_count)
        .map(|_| read_f32(reader))
        .collect::<Result<Vec<_>>>()?;
    if ncall > 0 {
        let denom = ncall as f32;
        for value in &mut values {
            *value /= denom;
        }
    }
    Ok(ImatrixEntry { name, values })
}

fn read_legacy_imatrix_trailer(reader: &mut Cursor<Vec<u8>>) -> Result<(i32, Option<String>)> {
    if reader.position() as usize >= reader.get_ref().len() {
        return Ok((0, None));
    }
    let chunk_count = read_i32(reader)?;
    if reader.position() as usize >= reader.get_ref().len() {
        return Ok((chunk_count, None));
    }
    let len = read_i32(reader)?;
    if len <= 0 {
        return Ok((chunk_count, None));
    }
    let mut dataset = vec![0_u8; len as usize];
    reader.read_exact(&mut dataset)?;
    Ok((chunk_count, Some(String::from_utf8(dataset)?)))
}

fn load_gguf_imatrix(bytes: &[u8], path: &Path) -> Result<LoadedImatrix> {
    let mut reader = Cursor::new(bytes.to_vec());
    let header = GgufHeader::read(&mut reader)?;
    ensure!(
        header.version >= 2,
        "unsupported GGUF imatrix version {} in {}",
        header.version,
        path.display()
    );
    let mut metadata = GgufMetadata::default();
    for _ in 0..header.metadata_count {
        let key = read_gguf_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        metadata.read_value(&mut reader, &key, value_type)?;
    }

    let mut tensors = Vec::with_capacity(header.tensor_count as usize);
    for _ in 0..header.tensor_count {
        tensors.push(GgufTensorInfo::read(&mut reader)?);
    }
    let data_start = align_to(reader.position(), metadata.alignment);
    let entries = read_gguf_entries(bytes, &tensors, data_start)?;
    ensure!(
        !entries.is_empty(),
        "GGUF imatrix has no paired tensors in {}",
        path.display()
    );
    Ok(LoadedImatrix {
        entries,
        dataset: metadata.datasets.into_iter().next(),
        chunk_count: metadata.chunk_count.unwrap_or(0) as i32,
    })
}

struct GgufHeader {
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
}

impl GgufHeader {
    fn read(reader: &mut Cursor<Vec<u8>>) -> Result<Self> {
        let mut magic = [0_u8; 4];
        reader.read_exact(&mut magic)?;
        ensure!(&magic == GGUF_MAGIC, "not a GGUF file");
        Ok(Self {
            version: read_u32(reader)?,
            tensor_count: read_u64(reader)?,
            metadata_count: read_u64(reader)?,
        })
    }
}

#[derive(Default)]
struct GgufMetadata {
    alignment: u64,
    datasets: Vec<String>,
    chunk_count: Option<u32>,
}

impl GgufMetadata {
    fn read_value(
        &mut self,
        reader: &mut Cursor<Vec<u8>>,
        key: &str,
        value_type: u32,
    ) -> Result<()> {
        match (key, value_type) {
            (KV_GENERAL_ALIGNMENT, GGUF_TYPE_UINT32) => {
                self.alignment = read_u32(reader)? as u64;
            }
            (KV_GENERAL_ALIGNMENT, GGUF_TYPE_UINT64) => {
                self.alignment = read_u64(reader)?;
            }
            (KV_IMATRIX_CHUNK_COUNT, GGUF_TYPE_UINT32) => {
                self.chunk_count = Some(read_u32(reader)?);
            }
            (KV_IMATRIX_DATASETS, GGUF_TYPE_ARRAY) => {
                self.datasets = read_string_array(reader)?;
            }
            _ => skip_gguf_value(reader, value_type)?,
        }
        if self.alignment == 0 {
            self.alignment = GGUF_ALIGNMENT_DEFAULT;
        }
        Ok(())
    }
}

struct GgufTensorInfo {
    name: String,
    dims: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

impl GgufTensorInfo {
    fn read(reader: &mut Cursor<Vec<u8>>) -> Result<Self> {
        let name = read_gguf_string(reader)?;
        let n_dims = read_u32(reader)?;
        ensure!(n_dims > 0, "GGUF tensor {name} has no dimensions");
        let dims = (0..n_dims)
            .map(|_| read_u64(reader))
            .collect::<Result<Vec<_>>>()?;
        let tensor_type = read_u32(reader)?;
        let offset = read_u64(reader)?;
        Ok(Self {
            name,
            dims,
            tensor_type,
            offset,
        })
    }

    fn element_count(&self) -> Result<usize> {
        self.dims.iter().try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim as usize)
                .with_context(|| format!("GGUF tensor {} is too large", self.name))
        })
    }
}

fn read_gguf_entries(
    bytes: &[u8],
    tensors: &[GgufTensorInfo],
    data_start: u64,
) -> Result<Vec<ImatrixEntry>> {
    let mut sums = BTreeMap::<String, Vec<f32>>::new();
    let mut counts = BTreeMap::<String, Vec<f32>>::new();
    for tensor in tensors {
        if tensor.tensor_type != GGML_TYPE_F32 {
            continue;
        }
        let Some((base_name, kind)) = imatrix_tensor_name(&tensor.name) else {
            continue;
        };
        let values = read_gguf_f32_tensor(bytes, tensor, data_start)?;
        match kind {
            ImatrixTensorKind::Sums => {
                sums.insert(base_name.to_string(), values);
            }
            ImatrixTensorKind::Counts => {
                counts.insert(base_name.to_string(), values);
            }
        }
    }
    let mut entries = Vec::new();
    for (name, sum_values) in sums {
        let count_values = counts
            .remove(&name)
            .with_context(|| format!("GGUF imatrix tensor {name} is missing counts"))?;
        entries.push(ImatrixEntry {
            name,
            values: normalize_gguf_imatrix_values(&sum_values, &count_values)?,
        });
    }
    Ok(entries)
}

enum ImatrixTensorKind {
    Sums,
    Counts,
}

fn imatrix_tensor_name(name: &str) -> Option<(&str, ImatrixTensorKind)> {
    if let Some(base) = name.strip_suffix(TENSOR_SUMS_SUFFIX) {
        return Some((base, ImatrixTensorKind::Sums));
    }
    name.strip_suffix(TENSOR_COUNTS_SUFFIX)
        .map(|base| (base, ImatrixTensorKind::Counts))
}

fn normalize_gguf_imatrix_values(sums: &[f32], counts: &[f32]) -> Result<Vec<f32>> {
    ensure!(!counts.is_empty(), "GGUF imatrix entry has no counts");
    ensure!(
        sums.len().is_multiple_of(counts.len()),
        "GGUF imatrix sums/counts shape mismatch"
    );
    let values_per_count = sums.len() / counts.len();
    let mut output = vec![1.0_f32; sums.len()];
    for (count_index, count) in counts.iter().enumerate() {
        if *count <= 0.0 {
            continue;
        }
        let offset = count_index * values_per_count;
        for i in 0..values_per_count {
            output[offset + i] = sums[offset + i] / count;
        }
    }
    Ok(output)
}

fn read_gguf_f32_tensor(
    bytes: &[u8],
    tensor: &GgufTensorInfo,
    data_start: u64,
) -> Result<Vec<f32>> {
    let element_count = tensor.element_count()?;
    let byte_len = element_count
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("GGUF tensor {} is too large", tensor.name))?;
    let offset = data_start
        .checked_add(tensor.offset)
        .with_context(|| format!("GGUF tensor {} offset overflow", tensor.name))?
        as usize;
    let end = offset
        .checked_add(byte_len)
        .with_context(|| format!("GGUF tensor {} byte range overflow", tensor.name))?;
    ensure!(
        end <= bytes.len(),
        "GGUF tensor {} extends past end of file",
        tensor.name
    );
    bytes[offset..end]
        .chunks_exact(4)
        .map(|chunk| Ok(f32::from_le_bytes(chunk.try_into()?)))
        .collect::<Result<Vec<_>>>()
}

fn include_exclude_match(
    name: &str,
    include_weights: &[String],
    exclude_weights: &[String],
) -> bool {
    if !exclude_weights.is_empty() {
        return !exclude_weights.iter().any(|filter| name.contains(filter));
    }
    if !include_weights.is_empty() {
        return include_weights.iter().any(|filter| name.contains(filter));
    }
    true
}

fn read_string_array(reader: &mut Cursor<Vec<u8>>) -> Result<Vec<String>> {
    let element_type = read_u32(reader)?;
    ensure!(
        element_type == GGUF_TYPE_STRING,
        "expected GGUF string array, found type {element_type}"
    );
    let len = read_u64(reader)?;
    (0..len).map(|_| read_gguf_string(reader)).collect()
}

fn skip_gguf_value(reader: &mut Cursor<Vec<u8>>, value_type: u32) -> Result<()> {
    match value_type {
        0 | 1 => skip_bytes(reader, 1),
        2 | 3 => skip_bytes(reader, 2),
        GGUF_TYPE_UINT32 | 5 | GGUF_TYPE_FLOAT32 => skip_bytes(reader, 4),
        GGUF_TYPE_BOOL => skip_bytes(reader, 1),
        GGUF_TYPE_STRING => {
            let _ = read_gguf_string(reader)?;
            Ok(())
        }
        GGUF_TYPE_ARRAY => skip_gguf_array(reader),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => skip_bytes(reader, 8),
        _ => Err(anyhow!("unsupported GGUF metadata type {value_type}")),
    }
}

fn skip_gguf_array(reader: &mut Cursor<Vec<u8>>) -> Result<()> {
    let element_type = read_u32(reader)?;
    let len = read_u64(reader)?;
    for _ in 0..len {
        skip_gguf_value(reader, element_type)?;
    }
    Ok(())
}

fn skip_bytes(reader: &mut Cursor<Vec<u8>>, len: u64) -> Result<()> {
    let position = reader.position();
    let next = position
        .checked_add(len)
        .context("GGUF metadata offset overflow")?;
    ensure!(
        next <= reader.get_ref().len() as u64,
        "GGUF metadata extends past end of file"
    );
    reader.set_position(next);
    Ok(())
}

fn read_gguf_string(reader: &mut Cursor<Vec<u8>>) -> Result<String> {
    let len = read_u64(reader)?;
    let mut bytes = vec![0_u8; len as usize];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("GGUF string is not UTF-8")
}

fn read_i32(reader: &mut Cursor<Vec<u8>>) -> Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u32(reader: &mut Cursor<Vec<u8>>) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut Cursor<Vec<u8>>) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut Cursor<Vec<u8>>) -> Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_legacy_imatrix_with_include_filter() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let imatrix_path = root.join("imatrix.dat");
        write_legacy_imatrix(
            &imatrix_path,
            &[
                ("blk.0.attn_q.weight", 2, &[2.0, 4.0]),
                ("blk.0.ffn_down.weight", 1, &[9.0, 12.0]),
            ],
        );

        let imatrix =
            NativeImatrix::load(&imatrix_path, &["attn_q".to_string()], &Vec::new()).unwrap();

        assert_eq!(imatrix.entry_count(), 1);
        assert_eq!(imatrix._values[0], vec![1.0, 2.0]);
        assert_eq!(imatrix.dataset(), Some("dataset.txt"));
        assert_eq!(imatrix.chunk_count(), 3);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_gguf_imatrix_with_normalized_counts() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let imatrix_path = root.join("imatrix.gguf");
        write_gguf_imatrix(&imatrix_path);

        let imatrix =
            NativeImatrix::load(&imatrix_path, &["attn_q".to_string()], &Vec::new()).unwrap();

        assert_eq!(imatrix.entry_count(), 1);
        assert_eq!(imatrix._values[0], vec![1.0, 2.0, 1.0, 1.0]);
        assert_eq!(imatrix.dataset(), Some("calibration.txt"));
        assert_eq!(imatrix.chunk_count(), 7);
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir() -> PathBuf {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("skippy-imatrix-{nanos}-{id}"))
    }

    fn write_legacy_imatrix(path: &Path, entries: &[(&str, i32, &[f32])]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(entries.len() as i32).to_le_bytes());
        for (name, ncall, values) in entries {
            bytes.extend_from_slice(&(name.len() as i32).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&ncall.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as i32).to_le_bytes());
            for value in *values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&3_i32.to_le_bytes());
        let dataset = "dataset.txt";
        bytes.extend_from_slice(&(dataset.len() as i32).to_le_bytes());
        bytes.extend_from_slice(dataset.as_bytes());
        fs::write(path, bytes).unwrap();
    }

    fn write_gguf_imatrix(path: &Path) {
        let sums_name = "blk.0.attn_q.weight.in_sum2";
        let counts_name = "blk.0.attn_q.weight.counts";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&4_u64.to_le_bytes());
        write_gguf_kv_string(&mut bytes, "general.type", "imatrix");
        write_gguf_kv_u32(&mut bytes, KV_GENERAL_ALIGNMENT, 32);
        write_gguf_kv_u32(&mut bytes, KV_IMATRIX_CHUNK_COUNT, 7);
        write_gguf_kv_string_array(&mut bytes, KV_IMATRIX_DATASETS, &["calibration.txt"]);
        write_gguf_tensor_info(&mut bytes, sums_name, &[2, 2], GGML_TYPE_F32, 0);
        write_gguf_tensor_info(&mut bytes, counts_name, &[1, 2], GGML_TYPE_F32, 16);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        for value in [2.0_f32, 4.0, 9.0, 11.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [2.0_f32, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    fn write_gguf_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        write_gguf_string(bytes, value);
    }

    fn write_gguf_kv_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_gguf_kv_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        write_gguf_string(bytes, key);
        bytes.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            write_gguf_string(bytes, value);
        }
    }

    fn write_gguf_tensor_info(
        bytes: &mut Vec<u8>,
        name: &str,
        dims: &[u64],
        tensor_type: u32,
        offset: u64,
    ) {
        write_gguf_string(bytes, name);
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&tensor_type.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn write_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
