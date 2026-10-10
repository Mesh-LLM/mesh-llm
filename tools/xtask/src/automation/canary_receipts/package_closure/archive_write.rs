use super::{archive, process, producer_receipt};
use crate::{automation::canary_receipts::Digest, command::DynResult};
use sha2::{Digest as _, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub(super) enum Content {
    File(PathBuf),
    Bytes(Vec<u8>),
}
pub(super) struct Input {
    pub(super) name: String,
    pub(super) content: Content,
    pub(super) digest: Digest,
    pub(super) executable: bool,
}

fn octal(bytes: &mut [u8], value: u64) -> DynResult<()> {
    let digits = format!("{value:o}");
    if digits.len() >= bytes.len() {
        return Err("tar field exceeds ustar limit".into());
    }
    bytes.fill(b'0');
    let start = bytes.len() - digits.len() - 1;
    bytes[start..start + digits.len()].copy_from_slice(digits.as_bytes());
    bytes[bytes.len() - 1] = 0;
    Ok(())
}

fn header(name: &str, size: u64, mode: u64, modified: u64, kind: u8) -> DynResult<[u8; 512]> {
    let mut bytes = [0; 512];
    if name.len() <= 100 {
        bytes[..name.len()].copy_from_slice(name.as_bytes());
    } else {
        let (prefix, suffix) = name
            .match_indices('/')
            .filter_map(|(at, _)| {
                (at <= 155 && name.len() - at - 1 <= 100).then_some((&name[..at], &name[at + 1..]))
            })
            .next_back()
            .ok_or("tar path exceeds ustar extent")?;
        bytes[..suffix.len()].copy_from_slice(suffix.as_bytes());
        bytes[345..345 + prefix.len()].copy_from_slice(prefix.as_bytes());
    }
    octal(&mut bytes[100..108], mode)?;
    octal(&mut bytes[124..136], size)?;
    octal(&mut bytes[136..148], modified)?;
    bytes[148..156].fill(b' ');
    bytes[156] = kind;
    bytes[257..263].copy_from_slice(b"ustar\0");
    bytes[263..265].copy_from_slice(b"00");
    let sum = bytes.iter().map(|byte| u64::from(*byte)).sum();
    octal(&mut bytes[148..156], sum)?;
    Ok(bytes)
}

fn pax_time(value: SystemTime) -> DynResult<(u64, Vec<u8>)> {
    let time = value.duration_since(UNIX_EPOCH)?;
    let body = format!(" mtime={}.{:09}\n", time.as_secs(), time.subsec_nanos());
    let mut length = body.len() + 1;
    loop {
        let next = body.len() + length.to_string().len();
        if next == length {
            break;
        }
        length = next;
    }
    Ok((time.as_secs(), format!("{length}{body}").into_bytes()))
}

fn padding(output: &mut File, size: u64) -> DynResult<()> {
    let length = usize::try_from((512 - size % 512) % 512)?;
    output.write_all(&[0; 512][..length])?;
    Ok(())
}

pub(super) fn write(path: &Path, mut inputs: Vec<Input>) -> DynResult<()> {
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    if inputs.windows(2).any(|pair| pair[0].name == pair[1].name) {
        return Err("duplicate archive writer member".into());
    }
    producer_receipt::create_file(path, b"")?;
    let mut output = File::options().write(true).open(path)?;
    for input in inputs {
        process::check()?;
        member(&mut output, &input)?;
    }
    output.write_all(&[0; 1024])?;
    output.sync_all()?;
    Ok(())
}

fn member(output: &mut File, input: &Input) -> DynResult<()> {
    process::check()?;
    archive::path(&input.name)?;
    let (size, modified) = match &input.content {
        Content::File(path) => {
            if !fs::symlink_metadata(path)?.is_file() {
                return Err("archive source must be a regular file".into());
            }
            let metadata = fs::metadata(path)?;
            (metadata.len(), metadata.modified()?)
        }
        Content::Bytes(bytes) => (u64::try_from(bytes.len())?, SystemTime::now()),
    };
    let (seconds, pax) = pax_time(modified)?;
    output.write_all(&header(
        "PaxHeader",
        u64::try_from(pax.len())?,
        0o644,
        seconds,
        b'x',
    )?)?;
    output.write_all(&pax)?;
    padding(output, u64::try_from(pax.len())?)?;
    output.write_all(&header(
        &input.name,
        size,
        if input.executable { 0o755 } else { 0o644 },
        seconds,
        b'0',
    )?)?;
    let mut hash = Sha256::new();
    let mut count = 0_u64;
    match &input.content {
        Content::Bytes(bytes) => {
            for chunk in bytes.chunks(65536) {
                process::check()?;
                output.write_all(chunk)?;
                hash.update(chunk);
            }
            count = u64::try_from(bytes.len())?;
        }
        Content::File(path) => {
            let mut file = File::open(path)?;
            let mut buffer = [0; 65536];
            loop {
                process::check()?;
                let size = file.read(&mut buffer)?;
                if size == 0 {
                    break;
                }
                output.write_all(&buffer[..size])?;
                hash.update(&buffer[..size]);
                count = count
                    .checked_add(u64::try_from(size)?)
                    .ok_or("archive input size overflow")?;
            }
        }
    }
    if count != size || Digest::try_from(hex::encode(hash.finalize()))? != input.digest {
        return Err("archive input changed during packaging".into());
    }
    padding(output, size)
}
