use super::archive::Entry;
use flate2::Compression;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;

fn octal(field: &mut [u8], value: u64) -> Result<(), String> {
    let digits = format!("{value:o}");
    if digits.len() >= field.len() {
        return Err("tar member field exceeds ustar limit".into());
    }
    field.fill(b'0');
    let start = field.len() - digits.len() - 1;
    field[start..start + digits.len()].copy_from_slice(digits.as_bytes());
    field[field.len() - 1] = 0;
    Ok(())
}

fn header(entry: &Entry) -> Result<[u8; 512], String> {
    let mut bytes = [0_u8; 512];
    let name = entry.name.as_bytes();
    if name.len() <= 100 {
        bytes[..name.len()].copy_from_slice(name);
    } else {
        let (prefix, suffix) = entry
            .name
            .match_indices('/')
            .filter_map(|(index, _)| {
                let prefix = &name[..index];
                let suffix = &name[index + 1..];
                (prefix.len() <= 155 && !suffix.is_empty() && suffix.len() <= 100)
                    .then_some((prefix, suffix))
            })
            .next_back()
            .ok_or_else(|| format!("tar member name exceeds ustar limit: {}", entry.name))?;
        bytes[..suffix.len()].copy_from_slice(suffix);
        bytes[345..345 + prefix.len()].copy_from_slice(prefix);
    }
    octal(&mut bytes[100..108], u64::from(entry.mode))?;
    octal(&mut bytes[108..116], 0)?;
    octal(&mut bytes[116..124], 0)?;
    octal(
        &mut bytes[124..136],
        if entry.directory { 0 } else { entry.size },
    )?;
    octal(&mut bytes[136..148], 0)?;
    bytes[148..156].fill(b' ');
    bytes[156] = if entry.directory { b'5' } else { b'0' };
    bytes[257..263].copy_from_slice(b"ustar\0");
    bytes[263..265].copy_from_slice(b"00");
    let checksum: u64 = bytes.iter().map(|byte| u64::from(*byte)).sum();
    octal(&mut bytes[148..156], checksum)?;
    Ok(bytes)
}

pub(super) fn write(file: File, entries: &[Entry]) -> Result<(), String> {
    let mut encoder = GzEncoder::new(file, Compression::default());
    for entry in entries {
        encoder
            .write_all(&header(entry)?)
            .map_err(|error| error.to_string())?;
        if !entry.directory {
            let mut input = File::open(&entry.path).map_err(|error| error.to_string())?;
            let mut digest = Sha256::new();
            let mut copied = 0_u64;
            let mut buffer = [0_u8; 65536];
            loop {
                let count = std::io::Read::read(&mut input, &mut buffer)
                    .map_err(|error| error.to_string())?;
                if count == 0 {
                    break;
                }
                encoder
                    .write_all(&buffer[..count])
                    .map_err(|error| error.to_string())?;
                digest.update(&buffer[..count]);
                copied += u64::try_from(count).map_err(|error| error.to_string())?;
            }
            if copied != entry.size || Some(hex::encode(digest.finalize())) != entry.sha256 {
                return Err(format!("archive input changed: {}", entry.path.display()));
            }
            let pad = (512 - entry.size % 512) % 512;
            encoder
                .write_all(&vec![
                    0;
                    usize::try_from(pad)
                        .map_err(|error| error.to_string())?
                ])
                .map_err(|error| error.to_string())?;
        }
    }
    encoder
        .write_all(&[0; 1024])
        .map_err(|error| error.to_string())?;
    encoder.finish().map_err(|error| error.to_string())?;
    Ok(())
}
