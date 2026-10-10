//! Bounds on what a media part's header says it decodes to.
//!
//! The native media helper sizes its buffers from the file header before it
//! decodes any data, and nothing checked those headers. A 750 KB PNG described
//! a 16000x16000 image that decoded to 1.4 GB, and a 42 byte FLAC claimed 2^36
//! samples, which made the helper allocate 256 GiB. Each part's header is read
//! here first, the way the native decoders read it, and parts that claim more
//! than these limits are rejected before the native decoder sees them.

mod audio;
mod image;
mod wav;

use super::MediaRejected;

/// Most pixels a media image may declare, about 67 megapixels (8192x8192).
pub(super) const MAX_IMAGE_PIXELS: u64 = 1 << 26;
/// Longest a media audio part may declare itself to be.
pub(super) const MAX_AUDIO_SECONDS: u64 = 60 * 60;

/// Fails if `bytes` declares an image or audio clip larger than the limits.
pub(super) fn check_declared_media_size(bytes: &[u8]) -> Result<(), MediaRejected> {
    // The native helper treats anything that looks like audio as audio and
    // everything else as an image.
    if audio::is_audio(bytes) {
        for claim in audio::declared_lengths(bytes)? {
            if !claim.within_seconds(MAX_AUDIO_SECONDS) {
                return Err(MediaRejected::new(format!(
                    "audio declares {} frames at {} Hz, longer than {} minutes",
                    claim.frames,
                    claim.sample_rate,
                    MAX_AUDIO_SECONDS / 60
                )));
            }
        }
        return Ok(());
    }
    if let Some((width, height)) = image::declared_dimensions(bytes)
        && width.saturating_mul(height) > MAX_IMAGE_PIXELS
    {
        return Err(MediaRejected::new(format!(
            "image declares {width}x{height} pixels, more than {MAX_IMAGE_PIXELS}"
        )));
    }
    Ok(())
}

/// Reads a header the way the native decoders do: past the end of the input
/// every byte reads as zero.
struct HeaderReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> HeaderReader<'a> {
    fn new(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn skip(&mut self, count: usize) {
        self.pos = self.pos.saturating_add(count);
    }

    fn u8(&mut self) -> u8 {
        let byte = self.bytes.get(self.pos).copied().unwrap_or(0);
        self.pos = self.pos.saturating_add(1);
        byte
    }

    fn be16(&mut self) -> u16 {
        u16::from_be_bytes([self.u8(), self.u8()])
    }

    fn le16(&mut self) -> u16 {
        u16::from_le_bytes([self.u8(), self.u8()])
    }

    fn be32(&mut self) -> u32 {
        u32::from_be_bytes([self.u8(), self.u8(), self.u8(), self.u8()])
    }

    fn le32(&mut self) -> u32 {
        u32::from_le_bytes([self.u8(), self.u8(), self.u8(), self.u8()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_media_and_unknown_bytes_pass() {
        check_declared_media_size(&image::tests::png_header(640, 480)).unwrap();
        check_declared_media_size(b"not a media file").unwrap();
        check_declared_media_size(&[]).unwrap();
    }

    #[test]
    fn rejects_images_that_declare_too_many_pixels() {
        check_declared_media_size(&image::tests::png_header(8192, 8192)).unwrap();
        let error = check_declared_media_size(&image::tests::png_header(16000, 16000)).unwrap_err();
        assert!(error.to_string().contains("16000x16000"), "{error}");
    }

    #[test]
    fn rejects_audio_that_declares_more_than_an_hour() {
        let hour = 16_000 * MAX_AUDIO_SECONDS;
        check_declared_media_size(&audio::tests::flac_header(16_000, hour)).unwrap();
        let error =
            check_declared_media_size(&audio::tests::flac_header(16_000, hour + 1)).unwrap_err();
        assert!(
            error.to_string().contains("longer than 60 minutes"),
            "{error}"
        );
    }
}
