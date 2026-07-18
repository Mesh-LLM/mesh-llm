use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

pub(crate) const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_ARRAY_DEPTH: usize = 64;
const MAX_GGUF_HEADER_KV_COUNT: u64 = 1_000_000;
const MAX_GGUF_TENSOR_COUNT: u64 = 1_000_000;

pub(crate) fn activation_width(model_path: &Path) -> Result<u32> {
    let mut file = File::open(model_path)
        .with_context(|| format!("open GGUF metadata source {}", model_path.display()))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .with_context(|| format!("read GGUF magic from {}", model_path.display()))?;
    anyhow::ensure!(
        &magic == b"GGUF",
        "not a GGUF file: {}",
        model_path.display()
    );

    let version = read_gguf_u32(&mut file)?;
    anyhow::ensure!(
        version >= 2,
        "unsupported GGUF version {version} in {}",
        model_path.display()
    );
    let _tensor_count = read_gguf_header_count(&mut file, MAX_GGUF_TENSOR_COUNT, "tensor")?;
    let kv_count = read_gguf_header_count(&mut file, MAX_GGUF_HEADER_KV_COUNT, "metadata")?;

    let mut architecture = None;
    let mut embedding_lengths = BTreeMap::<String, u32>::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut file)?;
        let value_type = GgufValueType::from_u32(read_gguf_u32(&mut file)?)?;
        if key == "general.architecture" {
            architecture = read_gguf_string_value(&mut file, value_type)?;
        } else if let Some(arch) = key.strip_suffix(".embedding_length") {
            if let Some(value) = read_gguf_u32_value(&mut file, value_type)? {
                embedding_lengths.insert(arch.to_string(), value);
            }
        } else {
            skip_gguf_value(&mut file, value_type)?;
        }
    }

    let architecture = architecture.with_context(|| {
        format!(
            "GGUF metadata for {} does not contain general.architecture",
            model_path.display()
        )
    })?;
    let width = embedding_lengths.remove(&architecture).with_context(|| {
        format!(
            "GGUF metadata for {} does not contain {}.embedding_length",
            model_path.display(),
            architecture
        )
    })?;
    anyhow::ensure!(
        width > 0,
        "GGUF metadata for {} has invalid {}.embedding_length 0",
        model_path.display(),
        architecture
    );
    Ok(width)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl GgufValueType {
    fn from_u32(value: u32) -> Result<Self> {
        Ok(match value {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            other => bail!("unsupported GGUF metadata value type {other}"),
        })
    }

    fn fixed_width(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

fn read_gguf_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_gguf_i32(reader: &mut impl Read) -> Result<i32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF i32")?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_gguf_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read GGUF u64")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_gguf_i64(reader: &mut impl Read) -> Result<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read GGUF i64")?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_gguf_u16(reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes).context("read GGUF u16")?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_gguf_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes).context("read GGUF u8")?;
    Ok(bytes[0])
}

fn read_gguf_header_count(reader: &mut impl Read, max: u64, label: &str) -> Result<u64> {
    let count = read_gguf_i64(reader)?;
    ensure!(count >= 0, "GGUF {label} count is negative: {count}");
    let count = u64::try_from(count).context("GGUF header count does not fit u64")?;
    ensure!(
        count <= max,
        "GGUF {label} count {count} exceeds safety limit {max}"
    );
    Ok(count)
}

fn read_gguf_string(reader: &mut impl Read) -> Result<String> {
    let len = read_gguf_u64(reader)?;
    ensure!(
        len <= MAX_GGUF_STRING_BYTES,
        "GGUF string length {len} exceeds safety limit {MAX_GGUF_STRING_BYTES}"
    );
    let len = usize::try_from(len).context("GGUF string length does not fit usize")?;
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .context("read GGUF string bytes")?;
    String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
}

fn read_gguf_string_value(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
) -> Result<Option<String>> {
    if value_type == GgufValueType::String {
        return Ok(Some(read_gguf_string(reader)?));
    }
    skip_gguf_value(reader, value_type)?;
    Ok(None)
}

fn read_gguf_u32_value(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
) -> Result<Option<u32>> {
    Ok(match value_type {
        GgufValueType::Uint32 => Some(read_gguf_u32(reader)?),
        GgufValueType::Int32 => {
            let value = read_gguf_i32(reader)?;
            Some(u32::try_from(value).context("GGUF embedding_length is negative")?)
        }
        GgufValueType::Uint16 => Some(u32::from(read_gguf_u16(reader)?)),
        GgufValueType::Uint8 => Some(u32::from(read_gguf_u8(reader)?)),
        _ => {
            skip_gguf_value(reader, value_type)?;
            None
        }
    })
}

fn skip_gguf_value(reader: &mut (impl Read + Seek), value_type: GgufValueType) -> Result<()> {
    skip_gguf_value_with_depth(reader, value_type, 0)
}

fn skip_gguf_value_with_depth(
    reader: &mut (impl Read + Seek),
    value_type: GgufValueType,
    depth: usize,
) -> Result<()> {
    ensure!(
        depth <= MAX_GGUF_ARRAY_DEPTH,
        "GGUF array nesting exceeds safety limit {MAX_GGUF_ARRAY_DEPTH}"
    );
    if let Some(width) = value_type.fixed_width() {
        skip_gguf_bytes(reader, width)
    } else if value_type == GgufValueType::String {
        let len = read_gguf_u64(reader)?;
        ensure!(
            len <= MAX_GGUF_STRING_BYTES,
            "GGUF string length {len} exceeds safety limit {MAX_GGUF_STRING_BYTES}"
        );
        skip_gguf_bytes(reader, len)
    } else {
        let item_type = GgufValueType::from_u32(read_gguf_u32(reader)?)?;
        let len = read_gguf_u64(reader)?;
        ensure!(
            len <= MAX_GGUF_ARRAY_ELEMENTS,
            "GGUF array length {len} exceeds safety limit {MAX_GGUF_ARRAY_ELEMENTS}"
        );
        if let Some(width) = item_type.fixed_width() {
            let bytes = width
                .checked_mul(len)
                .context("GGUF array byte size overflows u64")?;
            skip_gguf_bytes(reader, bytes)
        } else {
            for _ in 0..len {
                skip_gguf_value_with_depth(reader, item_type, depth + 1)?;
            }
            Ok(())
        }
    }
}

fn skip_gguf_bytes(reader: &mut impl Seek, len: u64) -> Result<()> {
    let offset = i64::try_from(len).context("GGUF value is too large to seek over")?;
    reader
        .seek(SeekFrom::Current(offset))
        .context("skip GGUF metadata value")?;
    Ok(())
}
