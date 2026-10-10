use anyhow::{Result, bail};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    time::Instant,
};
/// A supplied owned regular-file handle, with an expected SHA256 and byte size.
/// No pathname open, ambient acquisition, GGUF parsing or model admission occurs here.
pub struct Object {
    pub file: File,
    pub oid: String,
    pub size: u64,
}
pub(super) fn check(until: Instant) -> Result<()> {
    if Instant::now() >= until {
        bail!("LFS deadline expired");
    }
    Ok(())
}
pub(super) fn repo(value: &str) -> Result<()> {
    let parts = value.split('/').collect::<Vec<_>>();
    if parts.len() != 2
        || parts.iter().any(|p| {
            p.is_empty()
                || matches!(*p, "." | "..")
                || p.len() > 96
                || !p
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
    {
        bail!("LFS model repository refused");
    }
    Ok(())
}
impl Object {
    pub(in crate::snapshot_promotion) fn verify(&mut self, until: Instant) -> Result<()> {
        check(until)?;
        if self.size == 0
            || self.size > 1024_u64 * 1024 * 1024 * 1024
            || self.oid.len() != 64
            || !self
                .oid
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || !self.file.metadata()?.is_file()
            || self.file.metadata()?.len() != self.size
        {
            bail!("LFS source type/size/hash admission refused");
        }
        self.file.seek(SeekFrom::Start(0))?;
        let mut hash = Sha256::new();
        let mut seen = 0;
        let mut bytes = [0u8; 65536];
        loop {
            check(until)?;
            let n = self.file.read(&mut bytes)?;
            if n == 0 {
                break;
            }
            seen += n as u64;
            if seen > self.size {
                bail!("LFS source grew");
            }
            hash.update(&bytes[..n]);
        }
        if seen != self.size
            || hash
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
                != self.oid
        {
            bail!("LFS source byte identity changed");
        }
        check(until)?;
        self.file.seek(SeekFrom::Start(0))?;
        Ok(())
    }
    pub(super) fn body(&mut self, start: u64, size: u64, until: Instant) -> Result<reqwest::Body> {
        self.file.seek(SeekFrom::Start(start))?;
        let input = self.file.try_clone()?.take(size);
        let stream =
            futures::stream::try_unfold((input, size), move |(mut input, left)| async move {
                check(until).map_err(|_| std::io::Error::other("LFS source deadline"))?;
                if left == 0 {
                    return Ok::<_, std::io::Error>(None);
                }
                let mut bytes = vec![0u8; left.min(65536) as usize];
                let n = input.read(&mut bytes)?;
                if n == 0 {
                    return Err(std::io::Error::other("LFS source ended early"));
                }
                bytes.truncate(n);
                Ok(Some((bytes::Bytes::from(bytes), (input, left - n as u64))))
            });
        Ok(reqwest::Body::wrap_stream(stream))
    }
}
