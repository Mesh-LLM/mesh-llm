//! Byte-level archive builders. They write exactly the headers a test names,
//! including ones no well-behaved archiver emits (traversal, device nodes,
//! duplicates), so fixtures need no checked-in binaries.

use std::io::Write;

/// One ustar member: name, type flag, permission bits, payload and link name.
pub(crate) struct TarMember {
    pub(crate) name: String,
    pub(crate) kind: u8,
    pub(crate) mode: u32,
    pub(crate) data: Vec<u8>,
    pub(crate) link: String,
}

impl TarMember {
    pub(crate) fn file(name: &str, mode: u32, data: &[u8]) -> Self {
        Self::new(name, b'0', mode).with_data(data)
    }

    pub(crate) fn dir(name: &str, mode: u32) -> Self {
        Self::new(name, b'5', mode)
    }

    pub(crate) fn symlink(name: &str, target: &str) -> Self {
        Self::new(name, b'2', 0o777).with_link(target)
    }

    pub(crate) fn hardlink(name: &str, target: &str) -> Self {
        Self::new(name, b'1', 0o644).with_link(target)
    }

    pub(crate) fn new(name: &str, kind: u8, mode: u32) -> Self {
        Self {
            name: name.to_owned(),
            kind,
            mode,
            data: Vec::new(),
            link: String::new(),
        }
    }

    pub(crate) fn with_data(mut self, data: &[u8]) -> Self {
        self.data = data.to_vec();
        self
    }

    fn with_link(mut self, link: &str) -> Self {
        self.link = link.to_owned();
        self
    }

    pub(crate) fn header_bytes(&self) -> [u8; 512] {
        let mut header = [0_u8; 512];
        put(&mut header, 0, self.name.as_bytes());
        put(&mut header, 100, format!("{:07o}\0", self.mode).as_bytes());
        put(&mut header, 108, b"0000000\0");
        put(&mut header, 116, b"0000000\0");
        put(
            &mut header,
            124,
            format!("{:011o}\0", self.data.len()).as_bytes(),
        );
        put(&mut header, 136, b"00000000000\0");
        header[156] = self.kind;
        put(&mut header, 157, self.link.as_bytes());
        put(&mut header, 257, b"ustar\x0000");
        put(&mut header, 148, b"        ");
        let sum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
        put(&mut header, 148, format!("{sum:06o}\0 ").as_bytes());
        header
    }
}

fn put(header: &mut [u8], at: usize, bytes: &[u8]) {
    header[at..at + bytes.len()].copy_from_slice(bytes);
}

/// An uncompressed ustar archive with the two terminating zero blocks.
pub(crate) fn tar(members: &[TarMember]) -> Vec<u8> {
    let mut archive = Vec::new();
    for member in members {
        archive.extend_from_slice(&member.header_bytes());
        archive.extend_from_slice(&member.data);
        let padding = (512 - member.data.len() % 512) % 512;
        archive.extend(std::iter::repeat_n(0, padding));
    }
    archive.extend(std::iter::repeat_n(0, 1024));
    archive
}

/// `tar -czf`: the ustar archive in one gzip member.
pub(crate) fn tar_gz(members: &[TarMember]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&tar(members)).expect("in-memory gzip");
    encoder.finish().expect("in-memory gzip")
}
