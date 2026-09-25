//! Member access over a parsed ZIP directory with Python 3.13 `zipfile`
//! semantics: local header checks, stored and deflate members only, and CRC
//! and size verification, failing with the exception text a legacy
//! traceback ends with.

use super::zip_directory::{Archive, Failure, Info, bad_zip, decode_name, u16_at};
use super::zip_text::bytes_repr;
use crate::repository::python_text::repr;
use std::io::Read;

impl Archive {
    /// `ZipFile.open`: the compressed byte range, or the exception raised
    /// before any member byte is read.
    pub(super) fn open_member(&self, info: &Info) -> Result<(usize, usize), Failure> {
        let start = usize::try_from(info.header_offset).unwrap_or(usize::MAX);
        let header = self
            .data
            .get(start..)
            .and_then(|rest| rest.get(..30))
            .ok_or_else(|| bad_zip("Truncated file header"))?;
        if header[..4] != *b"PK\x03\x04" {
            return Err(bad_zip("Bad magic number for file header"));
        }
        let (name_len, extra_len) = (
            usize::from(u16_at(header, 26)),
            usize::from(u16_at(header, 28)),
        );
        let name_end = (start + 30 + name_len).min(self.data.len());
        let raw_name = &self.data[start + 30..name_end];
        if info.flags & 0x20 != 0 {
            return Err(Failure::Raised(
                "NotImplementedError: compressed patched data (flag bit 5)".to_owned(),
            ));
        }
        if info.flags & 0x40 != 0 {
            return Err(Failure::Raised(
                "NotImplementedError: strong encryption (flag bit 6)".to_owned(),
            ));
        }
        if decode_name(raw_name, u16_at(header, 6))? != info.orig_filename {
            return Err(bad_zip(&format!(
                "File name in directory {} and header {} differ.",
                repr(&info.orig_filename),
                bytes_repr(raw_name)
            )));
        }
        let data_start = name_end + extra_len;
        let end = data_start as u64 + info.compress_size;
        if end > info.end_offset && info.end_offset != info.header_offset {
            let name = repr(&info.orig_filename);
            return Err(bad_zip(&format!(
                "Overlapped entries: {name} (possible zip bomb)"
            )));
        }
        if info.flags & 1 != 0 {
            return Err(Failure::Raised(format!(
                "RuntimeError: File {} is encrypted, password required for extraction",
                info.repr()
            )));
        }
        if !matches!(info.method, 0 | 8) {
            return Err(Failure::Raised(
                "NotImplementedError: That compression method is not supported".to_owned(),
            ));
        }
        let available = self.data.len().saturating_sub(data_start);
        let length = usize::try_from(info.compress_size)
            .unwrap_or(usize::MAX)
            .min(available);
        Ok((data_start.min(self.data.len()), length))
    }

    /// `ZipExtFile.read()` over an opened member.
    pub(super) fn decode(
        &self,
        info: &Info,
        (start, length): (usize, usize),
    ) -> Result<Vec<u8>, Failure> {
        let compressed = &self.data[start..start + length];
        let limit = usize::try_from(info.file_size).unwrap_or(usize::MAX);
        let truncated = (length as u64) < info.compress_size;
        let mut output = Vec::new();
        if info.method == 0 {
            output.extend_from_slice(&compressed[..compressed.len().min(limit)]);
        } else {
            let mut decoder = flate2::read::DeflateDecoder::new(compressed).take(limit as u64);
            if let Err(error) = decoder.read_to_end(&mut output) {
                let message = format!("zlib.error: Error -3 while decompressing data: {error}");
                return Err(Failure::Raised(message));
            }
        }
        if truncated && output.len() < limit {
            return Err(Failure::Raised("EOFError".to_owned()));
        }
        let mut crc = flate2::Crc::new();
        crc.update(&output);
        if crc.sum() != info.crc {
            let message = format!("Bad CRC-32 for file {}", repr(&info.filename));
            return Err(bad_zip(&message));
        }
        Ok(output)
    }
}
