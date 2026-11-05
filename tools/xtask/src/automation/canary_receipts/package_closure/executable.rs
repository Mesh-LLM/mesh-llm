//! Inspect native arm64 imports without executing an untrusted package binary.
use crate::command::DynResult;
use goblin::elf::{Elf, header};
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
    if prefix.starts_with(b"\x7fELF") {
        return elf(reader, start, length);
    }
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

fn elf(reader: &mut (impl Read + Seek), start: u64, length: u64) -> DynResult<()> {
    const MAX_ELF_BYTES: u64 = 512 * 1024 * 1024;
    if length > MAX_ELF_BYTES {
        return Err("certification ELF exceeds inspection bound".into());
    }
    let mut bytes = vec![0; usize::try_from(length)?];
    read_at(reader, start, length, 0, &mut bytes)?;
    let image = Elf::parse(&bytes)?;
    if !image.is_64
        || !image.little_endian
        || image.header.e_machine != header::EM_AARCH64
        || !matches!(image.header.e_type, header::ET_EXEC | header::ET_DYN)
    {
        return Err("not an arm64 Linux ELF executable".into());
    }
    if image.interpreter.is_some_and(|path| {
        path != "/lib/ld-linux-aarch64.so.1" && path != "/lib64/ld-linux-aarch64.so.1"
    }) || !image.rpaths.is_empty()
        || !image.runpaths.is_empty()
    {
        return Err("ELF executable uses an unapproved loader or search path".into());
    }
    for library in image.libraries {
        if library.is_empty()
            || !library
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._+-".contains(&byte))
        {
            return Err("unsafe ELF dynamic dependency".into());
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn arm64_elf() -> Vec<u8> {
        let mut bytes = vec![0; 64];
        bytes[..16].copy_from_slice(b"\x7fELF\x02\x01\x01\0\0\0\0\0\0\0\0\0");
        bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&header::EM_AARCH64.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[52..54].copy_from_slice(&64_u16.to_le_bytes());
        bytes
    }

    #[test]
    fn elf_accepts_arm64_and_rejects_foreign_architecture() {
        let mut bytes = arm64_elf();
        assert!(inspect(&mut Cursor::new(&bytes), 0, bytes.len() as u64).is_ok());
        bytes[18..20].copy_from_slice(&header::EM_X86_64.to_le_bytes());
        assert!(inspect(&mut Cursor::new(&bytes), 0, bytes.len() as u64).is_err());
    }
}
