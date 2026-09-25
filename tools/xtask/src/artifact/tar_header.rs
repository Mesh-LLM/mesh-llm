//! One tar header block as Python 3.13's `TarInfo._frombuf` decodes it, plus
//! the pax record parser and the member type predicates `tarfile` exposes.

pub(super) const BLOCK: usize = 512;

/// One archive member after pax and GNU long-name headers are applied.
pub(super) struct Member {
    pub(super) name: String,
    pub(super) kind: u8,
    pub(super) mode: i64,
    pub(super) link: String,
    pub(super) data_start: usize,
    pub(super) size: usize,
}

impl Member {
    pub(super) fn is_dir(&self) -> bool {
        self.kind == b'5'
    }

    pub(super) fn is_reg(&self) -> bool {
        matches!(self.kind, b'0' | b'\0' | b'7' | b'S')
    }

    pub(super) fn is_sym(&self) -> bool {
        self.kind == b'2'
    }

    pub(super) fn is_lnk(&self) -> bool {
        self.kind == b'1'
    }

    pub(super) fn is_supported(&self) -> bool {
        self.is_reg() || matches!(self.kind, b'1'..=b'6' | b'L' | b'K')
    }
}

/// `tarfile.HeaderError` subclasses, which `next()` maps by offset.
pub(super) enum HeaderError {
    Empty,
    Truncated,
    EndOfFile,
    Invalid(&'static str),
    Subsequent(String),
    Read(String),
}

/// `_apply_pax_info` for the fields the extractor reads.
pub(super) fn apply_pax(member: &mut Member, records: &[(String, String)]) {
    for (key, value) in records {
        match key.as_str() {
            "path" => member.name = value.trim_end_matches('/').to_owned(),
            "linkpath" => member.link.clone_from(value),
            "size" => member.size = value.parse().unwrap_or(0),
            _ => {}
        }
    }
}

/// `"%d %s=%s\n"` records up to the first NUL.
pub(super) fn pax_records(buf: &[u8]) -> Result<Vec<(String, String)>, HeaderError> {
    const INVALID: HeaderError = HeaderError::Invalid("invalid header");
    let mut records = Vec::new();
    let mut pos = 0;
    while pos < buf.len() && buf[pos] != 0 {
        let digits = buf[pos..].iter().take_while(|b| b.is_ascii_digit()).count();
        if !(1..=20).contains(&digits) || buf.get(pos + digits) != Some(&b' ') {
            return Err(INVALID);
        }
        let length: usize = String::from_utf8_lossy(&buf[pos..pos + digits])
            .parse()
            .map_err(|_| INVALID)?;
        if length < 5 || pos + length > buf.len() || pos + digits + 1 > pos + length - 1 {
            return Err(INVALID);
        }
        let body = &buf[pos + digits + 1..pos + length - 1];
        let split = body.iter().position(|b| *b == b'=');
        let (Some(split), b'\n') = (split, buf[pos + length - 1]) else {
            return Err(INVALID);
        };
        if split == 0 {
            return Err(INVALID);
        }
        let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
        records.push((text(&body[..split]), text(&body[split + 1..])));
        pos += length;
    }
    Ok(records)
}

/// `TarInfo._frombuf`.
pub(super) fn frombuf(buf: &[u8], dircheck: bool) -> Result<Member, HeaderError> {
    if buf.is_empty() {
        return Err(HeaderError::Empty);
    }
    if buf.len() != BLOCK {
        return Err(HeaderError::Truncated);
    }
    if buf.iter().all(|b| *b == 0) {
        return Err(HeaderError::EndOfFile);
    }
    let checksum = nti(&buf[148..156])?;
    let unsigned: i64 = buf[..148]
        .iter()
        .chain(&buf[156..])
        .map(|b| i64::from(*b))
        .sum();
    let signed: i64 = buf[..148]
        .iter()
        .chain(&buf[156..])
        .map(|b| i64::from(i8::from_ne_bytes([*b])))
        .sum();
    if checksum != 256 + unsigned && checksum != 256 + signed {
        return Err(HeaderError::Invalid("bad checksum"));
    }
    for field in [108..116, 116..124, 136..148, 329..337, 337..345] {
        nti(&buf[field])?;
    }
    let size = usize::try_from(nti(&buf[124..136])?)
        .map_err(|_| HeaderError::Invalid("invalid offset"))?;
    let mut name = nts(&buf[..100]);
    let mut kind = buf[156];
    if dircheck && kind == b'\0' && name.ends_with('/') {
        kind = b'5';
    }
    if kind == b'5' {
        name = name.trim_end_matches('/').to_owned();
    }
    let prefix = nts(&buf[345..500]);
    if !prefix.is_empty() && !matches!(kind, b'L' | b'K' | b'S') {
        name = format!("{prefix}/{name}");
    }
    Ok(Member {
        name,
        kind,
        mode: nti(&buf[100..108])?,
        link: nts(&buf[157..257]),
        data_start: 0,
        size,
    })
}

/// A NUL-terminated string field.
pub(super) fn nts(field: &[u8]) -> String {
    let end = field.iter().position(|b| *b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).into_owned()
}

/// `nti`: base-256 when the first byte is 0o200/0o377, else stripped octal.
fn nti(field: &[u8]) -> Result<i64, HeaderError> {
    const INVALID: HeaderError = HeaderError::Invalid("invalid header");
    if matches!(field[0], 0o200 | 0o377) {
        let magnitude = field[1..]
            .iter()
            .try_fold(0_i64, |acc, byte| {
                acc.checked_mul(256)?.checked_add(i64::from(*byte))
            })
            .ok_or(INVALID)?;
        return Ok(if field[0] == 0o377 {
            magnitude - 256_i64.pow(7)
        } else {
            magnitude
        });
    }
    let end = field.iter().position(|b| *b == 0).unwrap_or(field.len());
    let text = std::str::from_utf8(&field[..end])
        .map_err(|_| INVALID)?
        .trim();
    if text.is_empty() {
        return Ok(0);
    }
    i64::from_str_radix(text, 8).map_err(|_| INVALID)
}
