use crate::command::DynResult;
#[path = "dimensions.rs"]
pub(in crate::automation) mod dimensions;
#[cfg(test)]
#[path = "model_dimensions_tests.rs"]
mod dimensions_tests;
#[path = "tensor_descriptors.rs"]
pub(in crate::automation) mod tensor_descriptors;
#[path = "tensor_layouts.rs"]
pub(in crate::automation) mod tensor_layouts;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Serialize)]
pub(in crate::automation) struct Identity {
    sha256: String,
    architecture: String,
    native_context_tokens: u64,
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix model-preflight --model PATH --sha256 DIGEST --minimum-context-tokens N --output PATH",
        values: &[
            "--model",
            "--sha256",
            "--minimum-context-tokens",
            "--output",
        ],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let path = Path::new(parsed.last("--model").ok_or("missing --model")?);
    let expected = parsed.last("--sha256").ok_or("missing --sha256")?;
    let minimum = parsed
        .last("--minimum-context-tokens")
        .ok_or("missing --minimum-context-tokens")?
        .parse::<u64>()?;
    let output = Path::new(parsed.last("--output").ok_or("missing --output")?);
    crate::command::write_json_file(output, &verify(path, expected, minimum)?)
}

pub(in crate::automation) fn verify(
    path: &Path,
    expected: &str,
    minimum: u64,
) -> DynResult<Identity> {
    if expected.len() != 64
        || !expected.bytes().all(|byte| byte.is_ascii_hexdigit())
        || minimum == 0
    {
        return Err(
            "model preflight requires a SHA-256 digest and positive context requirement".into(),
        );
    }
    let actual = crate::product::digest::file_sha256(path).map_err(|error| error.error)?;
    if actual != expected {
        return Err("model SHA-256 mismatch".into());
    }
    let (architecture, context) = inspect(path)?;
    if context < minimum {
        return Err("actual GGUF does not declare the required native context".into());
    }
    Ok(Identity {
        sha256: actual,
        architecture,
        native_context_tokens: context,
    })
}

/// Single-file stage admission; split-shard closure requires a separate producer.
pub(in crate::automation) fn require_single_file(path: &Path) -> DynResult<()> {
    let mut reader = Reader {
        file: File::open(path)?,
    };
    if reader.bytes::<4>()? != *b"GGUF" || !matches!(reader.u32()?, 2 | 3) {
        return Err("requires GGUF2/3".into());
    }
    reader.u64()?;
    let count = reader.u64()?;
    if count > 1_000_000 {
        return Err("GGUF metadata count exceeds bound".into());
    }
    let mut split_count = None;
    let mut split_no = None;
    for _ in 0..count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        if key == "split.count" || key == "split.no" {
            let target = if key == "split.count" {
                &mut split_count
            } else {
                &mut split_no
            };
            if target.replace(reader.integer(kind)?).is_some() {
                return Err("duplicate GGUF shard field".into());
            }
        } else {
            reader.skip(kind, 0)?;
        }
    }
    if split_count.is_some_and(|n| n != 1) || split_no.is_some_and(|n| n != 0) {
        return Err("split GGUF requires complete shard custody".into());
    }
    Ok(())
}

fn inspect(path: &Path) -> DynResult<(String, u64)> {
    let mut reader = Reader {
        file: File::open(path)?,
    };
    if reader.bytes::<4>()? != *b"GGUF" {
        return Err("not a GGUF file".into());
    }
    if !matches!(reader.u32()?, 2 | 3) {
        return Err("unsupported GGUF version".into());
    }
    reader.u64()?;
    let count = reader.u64()?;
    if count > 1_000_000 {
        return Err("GGUF metadata entry count exceeds bound".into());
    }
    let mut architecture = None;
    let mut contexts = BTreeMap::new();
    for _ in 0..count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        if key == "general.architecture" {
            if kind != 8 {
                return Err("GGUF architecture must be a string".into());
            }
            architecture = Some(reader.string()?);
        } else if key.ends_with(".context_length") {
            let value = match kind {
                0 => u64::from(reader.bytes::<1>()?[0]),
                2 => u64::from(u16::from_le_bytes(reader.bytes()?)),
                4 => u64::from(reader.u32()?),
                10 => reader.u64()?,
                5 => u64::try_from(i32::from_le_bytes(reader.bytes()?))?,
                11 => u64::try_from(i64::from_le_bytes(reader.bytes()?))?,
                _ => return Err("GGUF context length must be an integer".into()),
            };
            contexts.insert(key, value);
        } else {
            reader.skip(kind, 0)?;
        }
    }
    let architecture = architecture.ok_or("missing GGUF architecture")?;
    let context = contexts
        .get(&format!("{architecture}.context_length"))
        .copied()
        .ok_or("missing native GGUF context length")?;
    Ok((architecture, context))
}

struct Reader {
    file: File,
}
impl Reader {
    fn integer(&mut self, kind: u32) -> DynResult<u64> {
        Ok(match kind {
            0 => u64::from(self.bytes::<1>()?[0]),
            1 => u64::try_from(i8::from_le_bytes(self.bytes()?))?,
            2 => u64::from(u16::from_le_bytes(self.bytes()?)),
            3 => u64::try_from(i16::from_le_bytes(self.bytes()?))?,
            4 => u64::from(self.u32()?),
            5 => u64::try_from(i32::from_le_bytes(self.bytes()?))?,
            10 => self.u64()?,
            11 => u64::try_from(i64::from_le_bytes(self.bytes()?))?,
            _ => return Err("GGUF dimension must be an integer".into()),
        })
    }
    fn bytes<const LENGTH: usize>(&mut self) -> DynResult<[u8; LENGTH]> {
        let mut bytes = [0; LENGTH];
        self.file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> DynResult<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> DynResult<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> DynResult<String> {
        let length = self.u64()?;
        if length > 16 * 1024 * 1024 {
            return Err("GGUF string exceeds 16 MiB".into());
        }
        let mut bytes = vec![0; usize::try_from(length)?];
        self.file.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }
    fn advance(&mut self, count: u64) -> DynResult<()> {
        let position = self.file.stream_position()?;
        let end = position
            .checked_add(count)
            .ok_or("GGUF metadata position overflow")?;
        if end > self.file.metadata()?.len() {
            return Err("short GGUF metadata read".into());
        }
        self.file.seek(SeekFrom::Start(end))?;
        Ok(())
    }
    fn skip(&mut self, kind: u32, depth: usize) -> DynResult<()> {
        if depth > 32 {
            return Err("GGUF metadata nesting exceeds bound".into());
        }
        match kind {
            0 | 1 | 7 => self.advance(1),
            2 | 3 => self.advance(2),
            4..=6 => self.advance(4),
            10..=12 => self.advance(8),
            8 => {
                let count = self.u64()?;
                self.advance(count)
            }
            9 => {
                let item = self.u32()?;
                let count = self.u64()?;
                let width = match item {
                    0 | 1 | 7 => Some(1),
                    2 | 3 => Some(2),
                    4..=6 => Some(4),
                    10..=12 => Some(8),
                    8 | 9 => None,
                    _ => return Err("unsupported GGUF metadata type".into()),
                };
                if let Some(width) = width {
                    self.advance(count.checked_mul(width).ok_or("GGUF array size overflow")?)
                } else {
                    if count > 10_000_000 {
                        return Err("GGUF variable array count exceeds bound".into());
                    }
                    for _ in 0..count {
                        self.skip(item, depth + 1)?;
                    }
                    Ok(())
                }
            }
            _ => Err("unsupported GGUF metadata type".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    #[test]
    fn selects_architecture_context_after_skipping_unrelated_metadata() {
        let state = tempfile::tempdir().unwrap();
        let path = state.path().join("fixture.gguf");
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        bytes.extend(4_u64.to_le_bytes());
        string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend(9_u32.to_le_bytes());
        bytes.extend(8_u32.to_le_bytes());
        bytes.extend(2_u64.to_le_bytes());
        string(&mut bytes, "first");
        string(&mut bytes, "second");
        string(&mut bytes, "other.context_length");
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(4096_u32.to_le_bytes());
        string(&mut bytes, "fixture.context_length");
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(131072_u32.to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend(8_u32.to_le_bytes());
        string(&mut bytes, "fixture");
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(inspect(&path).unwrap(), ("fixture".into(), 131072));
        bytes.truncate(bytes.len() - 1);
        std::fs::write(&path, bytes).unwrap();
        assert!(inspect(&path).is_err());
    }
}
