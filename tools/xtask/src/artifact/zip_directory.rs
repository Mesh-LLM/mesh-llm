//! ZIP central directory parsing with Python 3.13 `zipfile` semantics: the
//! same end-of-central-directory search, ZIP64 extra decoding, filename
//! decoding and the exception text a legacy traceback ends with.

use super::zip_text::{InfoView, cp437, info_repr};

/// Why extraction stopped: a legacy `SystemExit` diagnostic or the final
/// line of an uncaught Python exception.
#[derive(Debug)]
pub(super) enum Failure {
    Unsafe(String),
    Raised(String),
}

pub(super) fn bad_zip(message: &str) -> Failure {
    Failure::Raised(format!("zipfile.BadZipFile: {message}"))
}

/// A central-directory record (`ZipInfo`).
pub(super) struct Info {
    pub(super) filename: String,
    pub(super) orig_filename: String,
    pub(super) flags: u16,
    pub(super) method: u16,
    pub(super) crc: u32,
    pub(super) compress_size: u64,
    pub(super) file_size: u64,
    pub(super) external: u32,
    pub(super) header_offset: u64,
    pub(super) end_offset: u64,
}

impl Info {
    pub(super) fn is_dir(&self) -> bool {
        self.filename.ends_with('/')
    }

    /// `repr(ZipInfo)`.
    pub(super) fn repr(&self) -> String {
        info_repr(&InfoView {
            filename: &self.filename,
            method: self.method,
            external: self.external,
            file_size: self.file_size,
            compress_size: self.compress_size,
        })
    }
}

pub(super) struct Archive {
    pub(super) data: Vec<u8>,
    pub(super) infos: Vec<Info>,
}

pub(super) fn u16_at(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([data[at], data[at + 1]])
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

pub(super) fn decode_name(raw: &[u8], flags: u16) -> Result<String, Failure> {
    if flags & 0x800 == 0 {
        return Ok(cp437(raw));
    }
    String::from_utf8(raw.to_vec()).map_err(|error| {
        let at = error.utf8_error().valid_up_to();
        let reason = match error.utf8_error().error_len() {
            None => "unexpected end of data",
            Some(_) if raw[at] >= 0xc2 && raw[at] <= 0xf4 => "invalid continuation byte",
            Some(_) => "invalid start byte",
        };
        Failure::Raised(format!(
            "UnicodeDecodeError: 'utf-8' codec can't decode byte {:#04x} in position {at}: {reason}",
            raw[at]
        ))
    })
}

/// `_EndRecData`: (`offset of the record`, `size_cd`, `offset_cd`).
fn end_record(data: &[u8]) -> Result<(usize, usize, usize), Failure> {
    let not_zip = || bad_zip("File is not a zip file");
    let size = data.len();
    if size < 22 {
        return Err(not_zip());
    }
    let tail = size - 22;
    let location = if data[tail..tail + 4] == *b"PK\x05\x06" && data[size - 2..] == [0, 0] {
        tail
    } else {
        let floor = size.saturating_sub((1 << 16) + 22);
        let found = data[floor..]
            .windows(4)
            .rposition(|window| window == b"PK\x05\x06")
            .ok_or_else(not_zip)?;
        if floor + found + 22 > size {
            return Err(not_zip());
        }
        floor + found
    };
    if location >= 20 && data[location - 20..location - 16] == *b"PK\x06\x07" {
        return Err(bad_zip("zip64 archives are not supported"));
    }
    let size_cd = u32_at(data, location + 12) as usize;
    let offset_cd = u32_at(data, location + 16) as usize;
    Ok((location, size_cd, offset_cd))
}

/// `_decodeExtra`: structural checks plus ZIP64 size and offset fields.
fn decode_extra(info: &mut Info, mut extra: &[u8]) -> Result<(), Failure> {
    while extra.len() >= 4 {
        let (kind, len) = (u16_at(extra, 0), usize::from(u16_at(extra, 2)));
        if len + 4 > extra.len() {
            return Err(bad_zip(&format!(
                "Corrupt extra field {kind:04x} (size={len})"
            )));
        }
        if kind == 1 {
            let mut field = &extra[4..len + 4];
            let wide = [0xffff_ffff, u64::MAX];
            let slots = [
                (&mut info.file_size, "File size"),
                (&mut info.compress_size, "Compress size"),
                (&mut info.header_offset, "Header offset"),
            ];
            for (slot, name) in slots {
                if wide.contains(slot) {
                    let bytes: [u8; 8] = field
                        .get(..8)
                        .and_then(|b| b.try_into().ok())
                        .ok_or_else(|| {
                            bad_zip(&format!("Corrupt zip64 extra field. {name} not found."))
                        })?;
                    *slot = u64::from_le_bytes(bytes);
                    field = &field[8..];
                }
            }
        }
        extra = &extra[len + 4..];
    }
    Ok(())
}

/// `_RealGetContents`.
pub(super) fn open(data: Vec<u8>) -> Result<Archive, Failure> {
    let (location, size_cd, offset_cd) = end_record(&data)?;
    let concat = location as i64 - size_cd as i64 - offset_cd as i64;
    let start_dir = offset_cd as i64 + concat;
    if start_dir < 0 {
        return Err(bad_zip("Bad offset for central directory"));
    }
    let start = usize::try_from(start_dir)
        .unwrap_or(usize::MAX)
        .min(data.len());
    let central = &data[start..(start + size_cd).min(data.len())];
    let mut infos = Vec::new();
    let mut at = 0;
    while at < size_cd {
        let header = central
            .get(at..at + 46)
            .ok_or_else(|| bad_zip("Truncated central directory"))?;
        if header[..4] != *b"PK\x01\x02" {
            return Err(bad_zip("Bad magic number for central directory"));
        }
        let lengths = [28, 30, 32].map(|field| usize::from(u16_at(header, field)));
        let clip = |from: usize, len: usize| {
            &central[from.min(central.len())..(from + len).min(central.len())]
        };
        let raw_name = clip(at + 46, lengths[0]);
        let extra = clip(at + 46 + lengths[0], lengths[1]);
        let flags = u16_at(header, 8);
        let orig_filename = decode_name(raw_name, flags)?;
        if header[6] > 63 {
            let version = f64::from(header[6]) / 10.0;
            return Err(Failure::Raised(format!(
                "NotImplementedError: zip file version {version:.1}"
            )));
        }
        let mut info = Info {
            filename: orig_filename
                .split('\0')
                .next()
                .unwrap_or_default()
                .to_owned(),
            orig_filename,
            flags,
            method: u16_at(header, 10),
            crc: u32_at(header, 16),
            compress_size: u64::from(u32_at(header, 20)),
            file_size: u64::from(u32_at(header, 24)),
            external: u32_at(header, 38),
            header_offset: u64::from(u32_at(header, 42)),
            end_offset: 0,
        };
        decode_extra(&mut info, extra)?;
        info.header_offset = u64::try_from(info.header_offset as i64 + concat).unwrap_or(u64::MAX);
        infos.push(info);
        at += 46 + lengths.iter().sum::<usize>();
    }
    let mut order: Vec<usize> = (0..infos.len()).collect();
    order.sort_by_key(|&index| infos[index].header_offset);
    let mut end = start_dir as u64;
    for &index in order.iter().rev() {
        infos[index].end_offset = end;
        end = infos[index].header_offset;
    }
    Ok(Archive { data, infos })
}
