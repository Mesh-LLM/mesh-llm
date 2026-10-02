//! Plain and gzip tar decoding, including ustar, v7, pax and GNU name records.
//! Header and payload failures are archive errors, independent of interpreters.

use std::io::Read;

use super::tar_header::{BLOCK, HeaderError, Member, apply_pax, frombuf, nts, pax_records};

/// A decoded archive: the (decompressed) bytes and every member.
pub(super) struct Archive {
    pub(super) data: Vec<u8>,
    pub(super) members: Vec<Member>,
}

/// Decode an accepted tar container before extraction can write any members.
pub(super) fn open(raw: Vec<u8>) -> Result<Archive, String> {
    let mut gzip_reader = Reader::default();
    let gzip_failure = match gunzip(&raw) {
        Ok(data) => match gzip_reader.next(&data) {
            Ok(first) => return collect_members(data, gzip_reader, first),
            Err(message) => message,
        },
        Err(message) => message,
    };
    let mut reader = Reader::default();
    match reader.next(&raw) {
        Ok(first) => collect_members(raw, reader, first),
        Err(_) if raw.starts_with(&[0x1f, 0x8b]) => {
            Err(format!("corrupt gzip tar archive: {gzip_failure}"))
        }
        Err(message) => Err(format!("invalid or unsupported tar archive: {message}")),
    }
}

fn collect_members(
    data: Vec<u8>,
    mut reader: Reader,
    first: Option<Member>,
) -> Result<Archive, String> {
    let mut members: Vec<Member> = first.into_iter().collect();
    if !members.is_empty() {
        while let Some(member) = reader.next(&data)? {
            members.push(member);
        }
    }
    Ok(Archive { data, members })
}

fn gunzip(raw: &[u8]) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    flate2::read::MultiGzDecoder::new(raw)
        .read_to_end(&mut data)
        .map_err(|error| format!("gzip decoding failed: {error}"))?;
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
                apply_pax(&mut member, &self.global)?;
                self.offset = self.payload_end(&member)?;
                if member.is_dir() {
                    member.name = member.name.trim_end_matches('/').to_owned();
                }
                Ok(member)
            }
        }
    }

    fn payload_end(&self, member: &Member) -> Result<usize, HeaderError> {
        let length = if member.is_reg() || !member.is_supported() {
            rounded_payload(member.size)?
        } else {
            0
        };
        member
            .data_start
            .checked_add(length)
            .ok_or_else(offset_overflow)
    }

    fn read_block_payload<'a>(
        &mut self,
        data: &'a [u8],
        size: usize,
    ) -> Result<&'a [u8], HeaderError> {
        let length = rounded_payload(size)?;
        let end = self.tell.checked_add(length).ok_or_else(offset_overflow)?;
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
            apply_pax(&mut next, &records)?;
            if records.iter().any(|(key, _)| key == "size") {
                self.offset = self.payload_end(&next)?;
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

fn rounded_payload(size: usize) -> Result<usize, HeaderError> {
    size.div_ceil(BLOCK)
        .checked_mul(BLOCK)
        .ok_or_else(offset_overflow)
}

fn offset_overflow() -> HeaderError {
    HeaderError::Read("tar payload offset overflow".to_owned())
}
