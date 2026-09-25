//! Python 3.13 `tarfile.open(mode="r:*")` + `getmembers()` over an in-memory
//! archive: the gz/bz2/xz/tar probe order and its combined diagnostic, ustar,
//! v7, pax (`x`/`g`) and GNU long-name headers, and the end-of-archive rules
//! (a bad header after the first member ends the archive silently).

use crate::repository::python_text::repr;
use std::io::Read;

use super::tar_header::{BLOCK, HeaderError, Member, apply_pax, frombuf, nts, pax_records};

/// A decoded archive: the (decompressed) bytes and every member.
pub(super) struct Archive {
    pub(super) data: Vec<u8>,
    pub(super) members: Vec<Member>,
}

/// `tarfile.open(bytes, "r:*")` followed by `getmembers()`. The error is
/// `str(ReadError)`.
pub(super) fn open(mut raw: Vec<u8>) -> Result<Archive, String> {
    let mut failures = Vec::new();
    for method in ["gz", "bz2", "xz", "tar"] {
        let data = match method {
            "gz" => gunzip(&raw),
            "bz2" => Err("not a bzip2 file".to_owned()),
            "xz" => Err("not an lzma file".to_owned()),
            _ => Ok(std::mem::take(&mut raw)),
        };
        let mut reader = Reader::default();
        match data.and_then(|data| reader.next(&data).map(|first| (data, first))) {
            Ok((data, first)) => {
                let mut members: Vec<Member> = first.into_iter().collect();
                if !members.is_empty() {
                    while let Some(member) = reader.next(&data)? {
                        members.push(member);
                    }
                }
                return Ok(Archive { data, members });
            }
            Err(message) => {
                failures.push(format!("- method {method}: ReadError({})", repr(&message)))
            }
        }
    }
    Err(format!(
        "file could not be opened successfully:\n{}",
        failures.join("\n")
    ))
}

fn gunzip(raw: &[u8]) -> Result<Vec<u8>, String> {
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let mut data = Vec::new();
    flate2::read::MultiGzDecoder::new(raw)
        .read_to_end(&mut data)
        .map_err(|_| "not a gzip file".to_owned())?;
    Ok(data)
}

#[derive(Default)]
struct Reader {
    offset: usize,
    tell: usize,
    global: Vec<(String, String)>,
}

impl Reader {
    /// `TarFile.next()`.
    fn next(&mut self, data: &[u8]) -> Result<Option<Member>, String> {
        if self.offset != self.tell {
            if self.offset == 0 {
                return Ok(None);
            }
            if self.offset > data.len() {
                return Err("unexpected end of data".to_owned());
            }
            self.tell = self.offset;
        }
        let first = self.offset == 0;
        match self.member(data, true) {
            Ok(member) => Ok(Some(member)),
            Err(HeaderError::EndOfFile) => Ok(None),
            Err(HeaderError::Empty) if first => Err("empty file".to_owned()),
            Err(HeaderError::Truncated) if first => Err("truncated header".to_owned()),
            Err(HeaderError::Invalid(message)) if first => Err(message.to_owned()),
            Err(HeaderError::Empty | HeaderError::Truncated | HeaderError::Invalid(_)) => Ok(None),
            Err(HeaderError::Subsequent(message) | HeaderError::Read(message)) => Err(message),
        }
    }

    /// `TarInfo.fromtarfile`: one header block plus its extension records.
    fn member(&mut self, data: &[u8], dircheck: bool) -> Result<Member, HeaderError> {
        let start = self.tell.min(data.len());
        let buf = &data[start..(start + BLOCK).min(data.len())];
        self.tell = start + buf.len();
        let mut member = frombuf(buf, dircheck)?;
        match member.kind {
            b'x' | b'g' | b'X' => self.pax(data, &member),
            b'L' | b'K' => self.gnu_long(data, &member),
            _ => {
                member.data_start = self.tell;
                self.offset = self.tell + self.payload_blocks(&member);
                apply_pax(&mut member, &self.global);
                if member.is_dir() {
                    member.name = member.name.trim_end_matches('/').to_owned();
                }
                Ok(member)
            }
        }
    }

    fn payload_blocks(&self, member: &Member) -> usize {
        if member.is_reg() || !member.is_supported() {
            member.size.div_ceil(BLOCK) * BLOCK
        } else {
            0
        }
    }

    fn read_block_payload<'a>(
        &mut self,
        data: &'a [u8],
        size: usize,
    ) -> Result<&'a [u8], HeaderError> {
        let length = size.div_ceil(BLOCK) * BLOCK;
        let end = self.tell + length;
        let buf = data
            .get(self.tell..end)
            .ok_or_else(|| HeaderError::Read("unexpected end of data".to_owned()))?;
        self.tell = end;
        Ok(buf)
    }

    fn following(&mut self, data: &[u8]) -> Result<Member, HeaderError> {
        self.member(data, false).map_err(|error| match error {
            HeaderError::Subsequent(message) | HeaderError::Read(message) => {
                HeaderError::Subsequent(message)
            }
            HeaderError::Empty => HeaderError::Subsequent("empty header".to_owned()),
            HeaderError::Truncated => HeaderError::Subsequent("truncated header".to_owned()),
            HeaderError::EndOfFile => HeaderError::Subsequent("end of file header".to_owned()),
            HeaderError::Invalid(message) => HeaderError::Subsequent(message.to_owned()),
        })
    }

    /// `_proc_pax`: extended (`x`) records patch the next member; global
    /// (`g`) records patch every later one.
    fn pax(&mut self, data: &[u8], header: &Member) -> Result<Member, HeaderError> {
        let records = pax_records(self.read_block_payload(data, header.size)?)?;
        if header.kind == b'g' {
            self.global.extend(records.iter().cloned());
        }
        let mut next = self.following(data)?;
        if header.kind != b'g' {
            apply_pax(&mut next, &records);
            if records.iter().any(|(key, _)| key == "size") {
                self.offset = next.data_start + self.payload_blocks(&next);
            }
        }
        Ok(next)
    }

    fn gnu_long(&mut self, data: &[u8], header: &Member) -> Result<Member, HeaderError> {
        let value = nts(self.read_block_payload(data, header.size)?);
        let mut next = self.following(data)?;
        if header.kind == b'L' {
            next.name = value;
        } else {
            next.link = value;
        }
        if next.is_dir() {
            next.name = next.name.strip_suffix('/').unwrap_or(&next.name).to_owned();
        }
        Ok(next)
    }
}
