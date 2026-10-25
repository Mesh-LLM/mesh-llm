//! Read-only plain tar admission; no archive path is ever extracted here.
use super::executable;
use super::process;
use crate::{automation::canary_receipts::Digest, command::DynResult};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Clone)]
pub(super) struct Member {
    pub(super) start: u64,
    pub(super) size: u64,
    pub(super) sha256: Digest,
    pub(super) executable: bool,
    pub(super) modified: u128,
}

pub(super) fn path(name: &str) -> DynResult<()> {
    if name.is_empty()
        || !name.as_bytes()[0].is_ascii_alphanumeric()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._/-".contains(&byte))
        || name
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err("unsafe package member path".into());
    }
    Ok(())
}

fn text(bytes: &[u8]) -> DynResult<&str> {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    if bytes[end..].iter().any(|byte| *byte != 0) {
        return Err("nonzero bytes after tar field terminator".into());
    }
    Ok(std::str::from_utf8(&bytes[..end])?)
}

fn octal(bytes: &[u8]) -> DynResult<u64> {
    let value = std::str::from_utf8(bytes)?.trim_matches(['\0', ' ']);
    if value.is_empty() {
        return Ok(0);
    }
    if !value.bytes().all(|byte| (b'0'..=b'7').contains(&byte)) {
        return Err("invalid tar octal field".into());
    }
    Ok(u64::from_str_radix(value, 8)?)
}

fn header(bytes: &[u8; 512]) -> DynResult<(String, u64, u64, u64)> {
    if &bytes[257..263] != b"ustar\0" && &bytes[257..263] != b"ustar " {
        return Err("unsupported tar header".into());
    }
    let expected = octal(&bytes[148..156])?;
    let sum: u64 = bytes
        .iter()
        .enumerate()
        .map(|(index, byte)| {
            if (148..156).contains(&index) {
                u64::from(b' ')
            } else {
                u64::from(*byte)
            }
        })
        .sum();
    if sum != expected {
        return Err("tar header checksum mismatch".into());
    }
    let suffix = text(&bytes[..100])?;
    let prefix = text(&bytes[345..500])?;
    let name = if prefix.is_empty() {
        suffix.to_owned()
    } else {
        format!("{prefix}/{suffix}")
    };
    Ok((
        name,
        octal(&bytes[124..136])?,
        octal(&bytes[100..108])?,
        octal(&bytes[136..148])?,
    ))
}

fn pax(bytes: &[u8]) -> DynResult<BTreeMap<String, String>> {
    let mut at = 0;
    let mut values = BTreeMap::new();
    while at < bytes.len() {
        let separator = bytes[at..]
            .iter()
            .position(|byte| *byte == b' ')
            .ok_or("invalid PAX record length")?
            + at;
        let length: usize = std::str::from_utf8(&bytes[at..separator])?.parse()?;
        let end = at
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or("PAX record overrun")?;
        if end <= separator + 1 || bytes[end - 1] != b'\n' {
            return Err("invalid PAX record framing".into());
        }
        let record = std::str::from_utf8(&bytes[separator + 1..end - 1])?;
        let (key, value) = record.split_once('=').ok_or("invalid PAX key/value")?;
        if ![
            "path", "size", "mtime", "atime", "ctime", "uid", "gid", "uname", "gname",
        ]
        .contains(&key)
            || value.contains(['\0', '\n', '\r'])
            || values.insert(key.to_owned(), value.to_owned()).is_some()
        {
            return Err("unsupported or duplicate PAX metadata".into());
        }
        at = end;
    }
    Ok(values)
}

fn timestamp(value: &str) -> DynResult<u128> {
    let (seconds, fraction) = value.split_once('.').unwrap_or((value, ""));
    if seconds.is_empty()
        || !seconds.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.len() > 9
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err("invalid positive archive modification time".into());
    }
    let seconds: u128 = seconds.parse()?;
    let nanos: u128 = if fraction.is_empty() {
        0
    } else {
        fraction.parse::<u128>()? * 10_u128.pow(u32::try_from(9 - fraction.len())?)
    };
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or_else(|| "archive timestamp overflow".into())
}

fn data(file: &mut impl Read, size: u64) -> DynResult<Digest> {
    let mut remaining = size;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    while remaining != 0 {
        process::check()?;
        let count = usize::try_from(remaining.min(u64::try_from(buffer.len())?))?;
        file.read_exact(&mut buffer[..count])?;
        hash.update(&buffer[..count]);
        remaining -= u64::try_from(count)?;
    }
    let padding = usize::try_from((512 - size % 512) % 512)?;
    file.read_exact(&mut buffer[..padding])?;
    if buffer[..padding].iter().any(|byte| *byte != 0) {
        return Err("nonzero archive member padding".into());
    }
    Ok(Digest::try_from(hex::encode(hash.finalize()))?)
}

pub(super) fn scan(file: &mut File) -> DynResult<BTreeMap<String, Member>> {
    file.seek(SeekFrom::Start(0))?;
    let mut members = BTreeMap::new();
    let mut extended = BTreeMap::new();
    loop {
        process::check()?;
        let mut block = [0; 512];
        file.read_exact(&mut block)?;
        if block.iter().all(|byte| *byte == 0) {
            return finish(file, members, extended);
        }
        let (mut name, mut size, mode, modified) = header(&block)?;
        if block[156] == b'x' {
            if !extended.is_empty() || size == 0 || size > 64 * 1024 {
                return Err("invalid repeated or oversized PAX header".into());
            }
            let start = file.stream_position()?;
            let mut bytes = vec![0; usize::try_from(size)?];
            file.read_exact(&mut bytes)?;
            extended = pax(&bytes)?;
            file.seek(SeekFrom::Start(start))?;
            data(file, size)?;
            continue;
        }
        if !matches!(block[156], b'0' | 0) || block[157..257].iter().any(|byte| *byte != 0) {
            return Err("package archive contains a nonregular or linked member".into());
        }
        if let Some(value) = extended.remove("path") {
            name = value;
        }
        if let Some(value) = extended.remove("size") {
            size = value.parse()?;
        }
        let modified = timestamp(
            &extended
                .remove("mtime")
                .unwrap_or_else(|| modified.to_string()),
        )?;
        extended.clear();
        path(&name)?;
        let start = file.stream_position()?;
        let member = Member {
            start,
            size,
            sha256: data(file, size)?,
            executable: mode & 0o111 != 0,
            modified,
        };
        if members.insert(name, member).is_some() || members.len() > 4096 {
            return Err("duplicate or excessive package archive members".into());
        }
    }
}

fn finish(
    file: &mut File,
    members: BTreeMap<String, Member>,
    extended: BTreeMap<String, String>,
) -> DynResult<BTreeMap<String, Member>> {
    if !extended.is_empty() {
        return Err("PAX header without file member".into());
    }
    let mut block = [0; 512];
    file.read_exact(&mut block)?;
    if block.iter().any(|byte| *byte != 0) {
        return Err("archive has no second end marker".into());
    }
    loop {
        process::check()?;
        let count = file.read(&mut block[..1])?;
        if count == 0 {
            break;
        }
        file.read_exact(&mut block[1..])?;
        if block.iter().any(|byte| *byte != 0) {
            return Err("nonzero or truncated archive trailer".into());
        }
    }
    Ok(members)
}

pub(super) fn bytes(file: &mut File, member: &Member, limit: u64) -> DynResult<Vec<u8>> {
    process::check()?;
    if member.size > limit {
        return Err("package metadata member exceeds bound".into());
    }
    file.seek(SeekFrom::Start(member.start))?;
    let mut bytes = vec![0; usize::try_from(member.size)?];
    file.read_exact(&mut bytes)?;
    if Digest::of_bytes(&bytes) != member.sha256 {
        return Err("archive member changed after scan".into());
    }
    Ok(bytes)
}

pub(super) const BINARIES: [&str; 5] = [
    "skippy-correctness",
    "skippy",
    "skippy-package-builder",
    "skippy-topology-plan",
    "skippy-mm-test",
];

pub(super) fn binaries(path: &Path) -> DynResult<()> {
    let mut file = File::open(path)?;
    let members = scan(&mut file)?;
    if members.len() != BINARIES.len() || BINARIES.iter().any(|name| !members.contains_key(*name)) {
        return Err("incomplete or unexpected certification executable archive".into());
    }
    for member in members.values() {
        if !member.executable {
            return Err("packaged binary is not executable".into());
        }
        executable::inspect(&mut file, member.start, member.size)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "archive_stream_tests.rs"]
mod stream_tests;
