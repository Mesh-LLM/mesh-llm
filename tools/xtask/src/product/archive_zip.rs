use super::archive::Entry;
use flate2::Compression;
use flate2::write::DeflateEncoder;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};

fn put16(output: &mut File, value: u16) -> Result<(), String> {
    output
        .write_all(&value.to_le_bytes())
        .map_err(|error| error.to_string())
}

fn put32(output: &mut File, value: u32) -> Result<(), String> {
    output
        .write_all(&value.to_le_bytes())
        .map_err(|error| error.to_string())
}

struct Central {
    name: String,
    compressed: u32,
    original: u32,
    crc: u32,
    offset: u32,
    directory: bool,
    mode: u32,
}

pub(super) fn write(mut file: File, entries: &[Entry]) -> Result<(), String> {
    let mut directory = Vec::new();
    let mut position = 0_u64;
    for entry in entries {
        let mut data = Vec::new();
        if !entry.directory {
            File::open(&entry.path)
                .map_err(|error| error.to_string())?
                .read_to_end(&mut data)
                .map_err(|error| error.to_string())?;
            if u64::try_from(data.len()).map_err(|error| error.to_string())? != entry.size
                || Some(hex::encode(Sha256::digest(&data))) != entry.sha256
            {
                return Err(format!("archive input changed: {}", entry.path.display()));
            }
        }
        let mut crc = 0xffff_ffff_u32;
        for byte in &data {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (if crc & 1 == 1 { 0xedb8_8320 } else { 0 });
            }
        }
        let crc = !crc;
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&data)
            .map_err(|error| error.to_string())?;
        let compressed = if entry.directory {
            Vec::new()
        } else {
            encoder.finish().map_err(|error| error.to_string())?
        };
        let name = entry.name.as_bytes();
        let length =
            u16::try_from(name.len()).map_err(|_| "ZIP member name exceeds format limit")?;
        let compressed_len =
            u32::try_from(compressed.len()).map_err(|_| "ZIP member exceeds 4 GiB")?;
        let original = u32::try_from(data.len()).map_err(|_| "ZIP member exceeds 4 GiB")?;
        let offset = u32::try_from(position).map_err(|_| "ZIP archive exceeds 4 GiB")?;
        put32(&mut file, 0x0403_4b50)?;
        for value in [20, 0, if entry.directory { 0 } else { 8 }, 0, 0] {
            put16(&mut file, value)?;
        }
        for value in [crc, compressed_len, original] {
            put32(&mut file, value)?;
        }
        put16(&mut file, length)?;
        put16(&mut file, 0)?;
        file.write_all(name).map_err(|error| error.to_string())?;
        file.write_all(&compressed)
            .map_err(|error| error.to_string())?;
        position += 30 + u64::from(length) + u64::from(compressed_len);
        directory.push(Central {
            name: entry.name.clone(),
            compressed: compressed_len,
            original,
            crc,
            offset,
            directory: entry.directory,
            mode: entry.mode,
        });
    }
    let central_start = u32::try_from(position).map_err(|_| "ZIP archive exceeds 4 GiB")?;
    for entry in &directory {
        put32(&mut file, 0x0201_4b50)?;
        put16(&mut file, 0x0314)?;
        put16(&mut file, 20)?;
        for value in [0, if entry.directory { 0 } else { 8 }, 0, 0] {
            put16(&mut file, value)?;
        }
        for value in [entry.crc, entry.compressed, entry.original] {
            put32(&mut file, value)?;
        }
        let length =
            u16::try_from(entry.name.len()).map_err(|_| "ZIP member name exceeds format limit")?;
        for value in [length, 0, 0, 0, 0] {
            put16(&mut file, value)?;
        }
        put32(
            &mut file,
            ((if entry.directory { 0o040000 } else { 0o100000 } | entry.mode) << 16)
                | if entry.directory { 0x10 } else { 0 },
        )?;
        put32(&mut file, entry.offset)?;
        file.write_all(entry.name.as_bytes())
            .map_err(|error| error.to_string())?;
        position += 46 + u64::from(length);
    }
    put32(&mut file, 0x0605_4b50)?;
    put16(&mut file, 0)?;
    put16(&mut file, 0)?;
    let count = u16::try_from(directory.len()).map_err(|_| "ZIP archive exceeds entry limit")?;
    put16(&mut file, count)?;
    put16(&mut file, count)?;
    put32(
        &mut file,
        u32::try_from(position - u64::from(central_start))
            .map_err(|_| "ZIP archive exceeds 4 GiB")?,
    )?;
    put32(&mut file, central_start)?;
    put16(&mut file, 0)?;
    Ok(())
}
