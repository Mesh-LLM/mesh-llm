//! Parse arm64 Mach-O imports without executing an untrusted package binary.
use crate::command::DynResult;
use std::io::{Read, Seek, SeekFrom};

fn read_at(
    reader: &mut (impl Read + Seek),
    start: u64,
    length: u64,
    offset: u64,
    bytes: &mut [u8],
) -> DynResult<()> {
    let size = u64::try_from(bytes.len())?;
    if offset.checked_add(size).is_none_or(|end| end > length) {
        return Err("Mach-O field exceeds executable member".into());
    }
    reader.seek(SeekFrom::Start(
        start.checked_add(offset).ok_or("Mach-O offset overflow")?,
    ))?;
    reader.read_exact(bytes)?;
    Ok(())
}

fn u32le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte field"))
}
fn u32be(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().expect("four-byte field"))
}
fn u64be(bytes: &[u8]) -> u64 {
    u64::from_be_bytes(bytes.try_into().expect("eight-byte field"))
}

pub(super) fn inspect(reader: &mut (impl Read + Seek), start: u64, length: u64) -> DynResult<()> {
    let mut prefix = [0; 8];
    read_at(reader, start, length, 0, &mut prefix)?;
    let magic = u32be(&prefix[..4]);
    match magic {
        0xcafebabe | 0xcafebabf => {
            if u32be(&prefix[4..]) != 1 {
                return Err("certification executable must contain exactly arm64".into());
            }
            let wide = magic == 0xcafebabf;
            let mut architecture = [0; 32];
            let record_size = if wide { 32 } else { 20 };
            read_at(reader, start, length, 8, &mut architecture[..record_size])?;
            if u32be(&architecture[..4]) != 0x0100000c {
                return Err("non-arm64 fat executable".into());
            }
            let (offset, size) = if wide {
                (u64be(&architecture[8..16]), u64be(&architecture[16..24]))
            } else {
                (
                    u64::from(u32be(&architecture[8..12])),
                    u64::from(u32be(&architecture[12..16])),
                )
            };
            if offset < u64::try_from(8 + record_size)?
                || offset.checked_add(size).is_none_or(|end| end > length)
            {
                return Err("invalid fat executable extent".into());
            }
            thin(
                reader,
                start.checked_add(offset).ok_or("fat offset overflow")?,
                size,
            )
        }
        _ => thin(reader, start, length),
    }
}

fn thin(reader: &mut (impl Read + Seek), start: u64, length: u64) -> DynResult<()> {
    let mut header = [0; 32];
    read_at(reader, start, length, 0, &mut header)?;
    if u32le(&header[..4]) != 0xfeedfacf
        || u32le(&header[4..8]) != 0x0100000c
        || u32le(&header[12..16]) != 2
    {
        return Err("not an arm64 Mach-O executable".into());
    }
    let commands = u32le(&header[16..20]);
    let total = u64::from(u32le(&header[20..24]));
    let end = 32_u64
        .checked_add(total)
        .ok_or("Mach-O command extent overflow")?;
    if end > length || commands == 0 || u64::from(commands) > total / 8 {
        return Err("invalid Mach-O command table".into());
    }
    let mut offset = 32;
    for _ in 0..commands {
        let mut command = [0; 8];
        read_at(reader, start, length, offset, &mut command)?;
        let kind = u32le(&command[..4]) & 0x7fffffff;
        let size = u64::from(u32le(&command[4..]));
        if size < 8 || size % 8 != 0 || offset.checked_add(size).is_none_or(|next| next > end) {
            return Err("invalid Mach-O load command".into());
        }
        if matches!(kind, 0xc | 0x18 | 0x1f | 0x20 | 0x23) {
            dependency(reader, start, length, offset, size)?;
        }
        offset += size;
    }
    if offset != end {
        return Err("Mach-O command count/size mismatch".into());
    }
    Ok(())
}

fn dependency(
    reader: &mut (impl Read + Seek),
    start: u64,
    length: u64,
    offset: u64,
    size: u64,
) -> DynResult<()> {
    if size < 24 {
        return Err("truncated dylib command".into());
    }
    let mut name_offset = [0; 4];
    read_at(reader, start, length, offset + 8, &mut name_offset)?;
    let name_offset = u64::from(u32le(&name_offset));
    if name_offset < 24 || name_offset >= size || size - name_offset > 4096 {
        return Err("invalid dylib import name".into());
    }
    let mut bytes = vec![0; usize::try_from(size - name_offset)?];
    read_at(reader, start, length, offset + name_offset, &mut bytes)?;
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or("unterminated dylib import")?;
    let name = std::str::from_utf8(&bytes[..end])?;
    if !name.starts_with("/usr/lib/") && !name.starts_with("/System/Library/") {
        return Err(format!("unpackaged dynamic dependency: {name}").into());
    }
    if name.split('/').any(|part| part == ".." || part == ".") || name.contains(['\n', '\r']) {
        return Err("unsafe system dylib import".into());
    }
    Ok(())
}

pub(super) fn test_binary(bytes: &[u8]) -> DynResult<std::path::PathBuf> {
    let mut selected = Vec::new();
    for line in bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let row: serde_json::Value = serde_json::from_slice(line)?;
        if row["reason"] == "compiler-artifact"
            && row["target"]["name"] == "skippy_serving"
            && row["profile"]["test"] == true
            && let Some(path) = row["executable"].as_str().filter(|path| !path.is_empty())
        {
            selected.push(std::path::PathBuf::from(path));
        }
    }
    if selected.len() != 1 {
        return Err("expected exactly one prebuilt skippy_serving library-test artifact".into());
    }
    Ok(selected.remove(0))
}
