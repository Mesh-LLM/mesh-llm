//! Byte-level ZIP builders for slice C (`artifact extract-zip`). Enabled
//! together with `zip_extract.rs` in `migration_archives.rs`.

use std::io::Write;

/// One ZIP entry: name, `external_attr` (mode in the upper 16 bits), payload
/// and whether the payload is deflated.
pub(crate) struct ZipEntry {
    pub(crate) name: String,
    pub(crate) external: u32,
    pub(crate) data: Vec<u8>,
    pub(crate) deflate: bool,
}

pub(crate) const S_IFREG: u32 = 0o100_000;
pub(crate) const S_IFDIR: u32 = 0o040_000;
pub(crate) const S_IFLNK: u32 = 0o120_000;

impl ZipEntry {
    pub(crate) fn file(name: &str, mode: u32, data: &[u8]) -> Self {
        Self::raw(name, (S_IFREG | mode) << 16, data)
    }

    pub(crate) fn symlink(name: &str, target: &str) -> Self {
        Self::raw(name, (S_IFLNK | 0o777) << 16, target.as_bytes())
    }

    pub(crate) fn raw(name: &str, external: u32, data: &[u8]) -> Self {
        Self {
            name: name.to_owned(),
            external,
            data: data.to_vec(),
            deflate: false,
        }
    }

    pub(crate) fn deflated(mut self) -> Self {
        self.deflate = true;
        self
    }

    fn payload(&self) -> Vec<u8> {
        if !self.deflate {
            return self.data.clone();
        }
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&self.data).expect("in-memory deflate");
        encoder.finish().expect("in-memory deflate")
    }
}

fn le16(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&u16::try_from(value).expect("u16 field").to_le_bytes());
}

fn le32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn size32(value: usize) -> u32 {
    u32::try_from(value).expect("u32 field")
}

/// A ZIP archive with UNIX `version made by`, every entry flagged UTF-8.
/// `crc_delta` is added to each recorded CRC to build corrupt archives.
pub(crate) fn zip_with(entries: &[ZipEntry], crc_delta: u32) -> Vec<u8> {
    let mut archive = Vec::new();
    let mut central = Vec::new();
    for entry in entries {
        let payload = entry.payload();
        let mut crc = flate2::Crc::new();
        crc.update(&entry.data);
        let method = if entry.deflate { 8 } else { 0 };
        let offset = size32(archive.len());
        let mut common = Vec::new();
        le16(&mut common, 0x0800);
        le16(&mut common, method);
        le32(&mut common, 0);
        le32(&mut common, crc.sum().wrapping_add(crc_delta));
        le32(&mut common, size32(payload.len()));
        le32(&mut common, size32(entry.data.len()));
        le16(&mut common, entry.name.len());
        le16(&mut common, 0);
        le32(&mut archive, 0x0403_4b50);
        le16(&mut archive, 20);
        archive.extend_from_slice(&common);
        archive.extend_from_slice(entry.name.as_bytes());
        archive.extend_from_slice(&payload);
        le32(&mut central, 0x0201_4b50);
        le16(&mut central, 0x031e);
        le16(&mut central, 20);
        central.extend_from_slice(&common);
        le16(&mut central, 0);
        le16(&mut central, 0);
        le16(&mut central, 0);
        le32(&mut central, entry.external);
        le32(&mut central, offset);
        central.extend_from_slice(entry.name.as_bytes());
    }
    let central_offset = size32(archive.len());
    archive.extend_from_slice(&central);
    le32(&mut archive, 0x0605_4b50);
    le32(&mut archive, 0);
    le16(&mut archive, entries.len());
    le16(&mut archive, entries.len());
    le32(&mut archive, size32(central.len()));
    le32(&mut archive, central_offset);
    le16(&mut archive, 0);
    archive
}

pub(crate) fn zip(entries: &[ZipEntry]) -> Vec<u8> {
    zip_with(entries, 0)
}
