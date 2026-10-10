//! Audio lengths as miniaudio reads them from the file header.
//!
//! The native helper sizes its sample buffer from the length the decoder
//! reports before decoding, and that length comes from the header: FLAC
//! STREAMINFO, an MP3 Xing or Info tag, or a WAV data size. The buffer holds
//! the declared duration at the model's sample rate, so a low declared sample
//! rate inflates it as well: a 1 MB WAV at 1 Hz asked for 31 GiB, and an
//! 8.7 KB MP3 whose Xing tag claims 2^24 frames for 2.9 GiB.
//!
//! miniaudio tries every decoder on every input, so each place one of them
//! could find a length counts as a claim, and every claim must be in bounds.

use super::HeaderReader;
use crate::media::MediaRejected;

/// A length an audio header declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DeclaredLength {
    pub(super) frames: u64,
    pub(super) sample_rate: u32,
}

impl DeclaredLength {
    pub(super) fn within_seconds(self, seconds: u64) -> bool {
        self.frames == 0
            || (self.sample_rate > 0
                && self.frames <= seconds.saturating_mul(u64::from(self.sample_rate)))
    }
}

/// Mirrors the native helper's `is_audio_file`: inputs it routes to the
/// audio decoders instead of the image decoder.
pub(super) fn is_audio(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    let wav = bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE";
    let mp3 = bytes.starts_with(b"ID3") || (bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0);
    wav || mp3 || bytes.starts_with(b"fLaC")
}

/// Every length a miniaudio decoder could read from `bytes`, or an error for
/// a header the decoder would stall on.
pub(super) fn declared_lengths(bytes: &[u8]) -> Result<Vec<DeclaredLength>, MediaRejected> {
    let mut lengths: Vec<DeclaredLength> = flac_stream_info(bytes).collect();
    lengths.extend(mp3_frame_count_tags(bytes));
    lengths.extend(super::wav::declared_length(bytes)?);
    Ok(lengths)
}

fn find_all<'a>(bytes: &'a [u8], needle: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
    bytes
        .windows(needle.len())
        .enumerate()
        .filter(move |(_, window)| *window == needle)
        .map(|(pos, _)| pos)
}

/// FLAC STREAMINFO blocks. Native FLAC puts one right after `fLaC`, following
/// any ID3 tags, and Ogg FLAC carries `fLaC` and STREAMINFO inside its first
/// packet, so every `fLaC` followed by a STREAMINFO block header counts.
fn flac_stream_info(bytes: &[u8]) -> impl Iterator<Item = DeclaredLength> + '_ {
    find_all(bytes, b"fLaC").filter_map(|pos| {
        let mut header = HeaderReader::new(bytes, pos + 4);
        let block_type = header.u8() & 0x7F;
        let block_size = u32::from_be_bytes([0, header.u8(), header.u8(), header.u8()]);
        if block_type != 0 || block_size != 34 || pos + 8 + 34 > bytes.len() {
            return None;
        }
        // Block sizes and frame sizes come first; then 20 bits of sample
        // rate, 3 of channels, 5 of bits per sample and 36 of total samples.
        header.skip(10);
        let properties = u64::from(header.be32()) << 32 | u64::from(header.be32());
        Some(DeclaredLength {
            frames: properties & ((1 << 36) - 1),
            sample_rate: (properties >> 44) as u32,
        })
    })
}

/// Xing and Info tags in an MPEG audio frame. miniaudio reads the frame
/// count from the tag in the first frame it decodes, after skipping any
/// leading junk, so a tag counts wherever a valid frame header precedes it at
/// the offset the frame's side information puts it.
fn mp3_frame_count_tags(bytes: &[u8]) -> impl Iterator<Item = DeclaredLength> + '_ {
    find_all(bytes, b"Xing")
        .chain(find_all(bytes, b"Info"))
        .flat_map(move |tag| {
            [0usize, 2].into_iter().flat_map(move |crc| {
                [9usize, 17, 32]
                    .into_iter()
                    .filter_map(move |side_info| mp3_tag_length(bytes, tag, crc, side_info))
            })
        })
}

fn mp3_tag_length(
    bytes: &[u8],
    tag: usize,
    crc: usize,
    side_info: usize,
) -> Option<DeclaredLength> {
    let frame = tag.checked_sub(4 + crc + side_info)?;
    let header: [u8; 4] = bytes.get(frame..frame + 4)?.try_into().ok()?;
    if !mp3_header_valid(&header)
        || mp3_has_crc(&header) != (crc == 2)
        || mp3_side_info_len(&header) != side_info
    {
        return None;
    }
    let mut tag_data = HeaderReader::new(bytes, tag + 7);
    if tag_data.u8() & 0x01 == 0 {
        return None;
    }
    let frames = tag_data.be32();
    // miniaudio treats an all-ones count as no count.
    if frames == u32::MAX {
        return None;
    }
    // miniaudio multiplies these in 32 bits, so its length can wrap to less
    // than this product but never exceed it.
    Some(DeclaredLength {
        frames: u64::from(frames) * u64::from(mp3_frame_samples(&header)),
        sample_rate: mp3_sample_rate(&header),
    })
}

/// miniaudio's `hdr_valid`.
fn mp3_header_valid(header: &[u8; 4]) -> bool {
    header[0] == 0xFF
        && (header[1] & 0xF0 == 0xF0 || header[1] & 0xFE == 0xE2)
        && (header[1] >> 1) & 3 != 0
        && header[2] >> 4 != 15
        && (header[2] >> 2) & 3 != 3
}

fn mp3_has_crc(header: &[u8; 4]) -> bool {
    header[1] & 1 == 0
}

fn mp3_is_mpeg1(header: &[u8; 4]) -> bool {
    header[1] & 0x08 != 0
}

fn mp3_side_info_len(header: &[u8; 4]) -> usize {
    let mono = header[3] & 0xC0 == 0xC0;
    match (mp3_is_mpeg1(header), mono) {
        (true, true) => 17,
        (true, false) => 32,
        (false, true) => 9,
        (false, false) => 17,
    }
}

fn mp3_frame_samples(header: &[u8; 4]) -> u32 {
    if header[1] & 6 == 6 {
        384
    } else if header[1] & 14 == 2 {
        576
    } else {
        1152
    }
}

fn mp3_sample_rate(header: &[u8; 4]) -> u32 {
    let base = [44_100, 48_000, 32_000][usize::from((header[2] >> 2) & 3)];
    let not_mpeg1 = u32::from(!mp3_is_mpeg1(header));
    let mpeg25 = u32::from(header[1] & 0x10 == 0);
    base >> not_mpeg1 >> mpeg25
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// `fLaC` and a STREAMINFO block declaring `frames` samples.
    pub(in crate::media::declared_size) fn flac_header(sample_rate: u32, frames: u64) -> Vec<u8> {
        let mut flac = b"fLaC".to_vec();
        flac.extend_from_slice(&[0x80, 0, 0, 34]);
        flac.extend_from_slice(&[0x10, 0x00, 0x10, 0x00, 0, 0, 0, 0, 0, 0]);
        let properties = u64::from(sample_rate) << 44 | 15 << 36 | frames;
        flac.extend_from_slice(&properties.to_be_bytes());
        flac.extend_from_slice(&[0; 16]); // MD5
        flac
    }

    #[test]
    fn reads_flac_stream_info_after_id3_tags() {
        let mut flac = b"ID3\x04\0\0\0\0\0\x02xx".to_vec();
        flac.extend_from_slice(&flac_header(16_000, (1 << 36) - 1));
        assert!(is_audio(&flac));
        assert_eq!(
            declared_lengths(&flac).unwrap(),
            [DeclaredLength {
                frames: (1 << 36) - 1,
                sample_rate: 16_000
            }]
        );
    }

    #[test]
    fn reads_mp3_frame_count_tags_behind_valid_frame_headers() {
        // MPEG-1 layer III, 44.1 kHz, stereo, no CRC: 32 bytes of side info.
        let mut mp3 = vec![0xFF, 0xFB, 0x90, 0x00];
        mp3.resize(4 + 32, 0);
        mp3.extend_from_slice(b"Xing\0\0\0\x01");
        mp3.extend_from_slice(&0x0100_0000u32.to_be_bytes());
        mp3.resize(417, 0);
        assert!(is_audio(&mp3));
        let length = DeclaredLength {
            frames: 0x0100_0000 * 1152,
            sample_rate: 44_100,
        };
        assert_eq!(declared_lengths(&mp3).unwrap(), [length]);
        assert!(!length.within_seconds(3600));

        // The same tag behind junk the decoder skips still counts.
        let mut junk = b"ID3\x04\0\0\0\0\0\0".to_vec();
        junk.extend_from_slice(&[0x55; 300]);
        junk.extend_from_slice(&mp3);
        assert_eq!(declared_lengths(&junk).unwrap(), [length]);

        // A tag without a frame header in front of it is not read.
        assert!(
            declared_lengths(b"ID3\x04\0\0\0\0\0\0 Xing\0\0\0\x01\xff\xff\xff\xfe")
                .unwrap()
                .is_empty()
        );
    }
}
