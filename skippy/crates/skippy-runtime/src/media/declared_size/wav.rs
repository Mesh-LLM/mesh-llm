//! WAV lengths as miniaudio's WAV decoder (dr_wav) works them out.
//!
//! The media helper decodes audio with `ma_decoder_init_memory`, which opens
//! a RIFF/WAVE file twice if it has to: first with dr_wav's own memory
//! reader, where a seek past either end fails, and, when every decoder
//! rejected the file that way, again through miniaudio's memory stream,
//! where such seeks stop at the end instead. dr_wav also keeps its own count
//! of how far into the file it is, which it uses for the data position and
//! which drifts from the real position after some seeks. Lengths therefore
//! come from replaying dr_wav's reads and seeks in both modes rather than
//! from what the headers describe.
//!
//! Some headers make dr_wav loop, or skip about 2^64 bytes in 2 GiB steps,
//! which tied up the runtime for many seconds on a 24 byte file. Those are
//! rejected outright.

use super::audio::DeclaredLength;
use crate::media::MediaRejected;

const WAVE_FORMAT_ADPCM: u16 = 0x0002;
const WAVE_FORMAT_IMA_ADPCM: u16 = 0x0011;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// dr_wav's `MA_DR_WAV_MAX_SAMPLE_RATE`, `MAX_CHANNELS` and
/// `MAX_BITS_PER_SAMPLE`.
const MAX_SAMPLE_RATE: u32 = 384_000;
const MAX_CHANNELS: u16 = 256;
const MAX_BITS_PER_SAMPLE: u16 = 64;

/// Chunks read before treating the walk as a loop. Real files hold a few.
const MAX_CHUNKS: usize = 4096;
/// dr_wav seeks forward in steps of this size.
const SEEK_STEP: u64 = 0x7FFF_FFFF;
/// Seek steps one skip may take before it counts as a stall.
const MAX_SEEK_STEPS: u64 = 16;

/// The length dr_wav would report for `bytes`, `None` when it would not
/// open them, or an error when it would stall on them.
pub(super) fn declared_length(bytes: &[u8]) -> Result<Option<DeclaredLength>, MediaRejected> {
    if !(bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE")) {
        return Ok(None);
    }
    match open(bytes, Seeks::FailPastEnds)? {
        Some(length) => Ok(Some(length)),
        // The decoder only gets here when every decoder failed the first
        // way; a WAV header that failed is checked the second way too.
        None => open(bytes, Seeks::StopAtEnds),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Seeks {
    /// dr_wav's memory reader: seeking past either end fails.
    FailPastEnds,
    /// miniaudio's memory stream: seeking past either end stops there.
    StopAtEnds,
}

/// A memory stream with dr_wav's read and seek behavior.
struct Stream<'a> {
    bytes: &'a [u8],
    pos: u64,
    seeks: Seeks,
}

impl<'a> Stream<'a> {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Reads exactly `count` bytes, or consumes what is left and returns
    /// `None`.
    fn read(&mut self, count: usize) -> Option<&'a [u8]> {
        let start = self.pos as usize;
        let available = self.bytes.len() - start;
        if available < count {
            self.pos = self.len();
            return None;
        }
        self.pos += count as u64;
        Some(&self.bytes[start..start + count])
    }

    fn seek_to(&mut self, target: i128) -> bool {
        let len = i128::from(self.len());
        match self.seeks {
            Seeks::FailPastEnds if !(0..=len).contains(&target) => false,
            _ => {
                self.pos = target.clamp(0, len) as u64;
                true
            }
        }
    }

    /// One relative `onSeek` call.
    fn seek_by(&mut self, offset: i64) -> bool {
        self.seek_to(i128::from(self.pos) + i128::from(offset))
    }

    /// Mirrors `ma_dr_wav__seek_forward`, which seeks in steps.
    fn seek_forward(&mut self, offset: u64) -> Result<bool, MediaRejected> {
        if self.seeks == Seeks::StopAtEnds && offset / SEEK_STEP > MAX_SEEK_STEPS {
            return Err(stall("skips past the end of the file in billions of steps"));
        }
        // Inside an in-memory file the first step either fits or fails.
        Ok(self.seek_to(i128::from(self.pos) + i128::from(offset)))
    }

    /// Mirrors `ma_dr_wav__seek_from_start`.
    fn seek_from_start(&mut self, offset: u64) -> Result<bool, MediaRejected> {
        if offset <= SEEK_STEP {
            return Ok(self.seek_to(i128::from(offset)));
        }
        if !self.seek_to(i128::from(SEEK_STEP)) {
            return Ok(false);
        }
        self.seek_forward(offset - SEEK_STEP)
    }
}

fn stall(reason: &str) -> MediaRejected {
    MediaRejected::new(format!("WAV header would stall the decoder: it {reason}"))
}

#[derive(Clone, Copy, Default)]
struct Format {
    tag: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    sub_format_tag: u16,
}

/// Replays `ma_dr_wav_init` as the media decoder calls it (no metadata,
/// not sequential) over a file that starts with `RIFF....WAVE`.
fn open(bytes: &[u8], seeks: Seeks) -> Result<Option<DeclaredLength>, MediaRejected> {
    let mut stream = Stream {
        bytes,
        pos: 12,
        seeks,
    };
    // dr_wav's count of bytes consumed, which it uses as the data position.
    let mut cursor: u64 = 12;
    let mut format = None;
    let mut data = None;
    let mut chunks = 0;
    loop {
        if chunks == MAX_CHUNKS {
            return Err(stall("reads the same chunks over and over"));
        }
        chunks += 1;
        let Some(id) = stream.read(4) else {
            break;
        };
        cursor += 4;
        let Some(size) = stream.read(4) else {
            break;
        };
        cursor += 4;
        let size = u64::from(u32::from_le_bytes(size.try_into().expect("4 bytes")));
        let padding = size & 1;
        match id {
            b"fmt " => {
                let Some(parsed) = read_format(&mut stream, &mut cursor, size) else {
                    return Ok(None);
                };
                format = Some(parsed);
                if padding > 0 {
                    if !stream.seek_forward(padding)? {
                        break;
                    }
                    cursor += padding;
                }
            }
            b"data" => {
                data = Some((cursor, size));
                break;
            }
            b"fact" => {
                if stream.read(4).is_none() {
                    return Ok(None);
                }
                cursor += 4;
                let skip = size.wrapping_sub(4).wrapping_add(padding);
                if !stream.seek_forward(skip)? {
                    break;
                }
                cursor = cursor.wrapping_add(skip);
            }
            _ => {
                if !stream.seek_forward(size + padding)? {
                    break;
                }
                cursor = cursor.wrapping_add(size + padding);
            }
        }
    }
    let (Some(format), Some((data_pos, declared))) = (format, data) else {
        return Ok(None);
    };
    length(&mut stream, format, data_pos, declared)
}

/// Mirrors dr_wav's `fmt ` chunk read: 16 bytes always, and past a
/// 16 byte chunk an extension size and extension, followed by a seek
/// back to where the chunk said it ends.
fn read_format(stream: &mut Stream<'_>, cursor: &mut u64, size: u64) -> Option<Format> {
    let fields = stream.read(16)?;
    *cursor += 16;
    let le16 = |at: usize| u16::from_le_bytes([fields[at], fields[at + 1]]);
    let mut format = Format {
        tag: le16(0),
        channels: le16(2),
        sample_rate: u32::from_le_bytes(fields[4..8].try_into().expect("4 bytes")),
        block_align: le16(12),
        bits_per_sample: le16(14),
        sub_format_tag: 0,
    };
    if size > 16 {
        let extension_len = stream.read(2)?;
        *cursor += 2;
        let extension_len = u16::from_le_bytes([extension_len[0], extension_len[1]]);
        let mut read_so_far: u64 = 18;
        if extension_len > 0 {
            if format.tag == WAVE_FORMAT_EXTENSIBLE {
                if extension_len != 22 {
                    return None;
                }
                let extension = stream.read(22)?;
                format.sub_format_tag = u16::from_le_bytes([extension[6], extension[7]]);
            } else if !stream.seek_by(i64::from(extension_len)) {
                return None;
            }
            *cursor += u64::from(extension_len);
            read_so_far += u64::from(extension_len);
        }
        // dr_wav seeks by `(int)(size - read_so_far)`, which goes backwards
        // when the extension ran past the chunk or the size exceeds
        // `i32::MAX`, while its count moves forward by the unsigned
        // difference.
        let rest = size.wrapping_sub(read_so_far);
        if !stream.seek_by(i64::from(rest as u32 as i32)) {
            return None;
        }
        *cursor = cursor.wrapping_add(rest);
    }
    Some(format)
}

/// The rest of `ma_dr_wav_init` once both chunks are found: validation,
/// the data size, and the frame count.
fn length(
    stream: &mut Stream<'_>,
    format: Format,
    data_pos: u64,
    declared: u64,
) -> Result<Option<DeclaredLength>, MediaRejected> {
    if !(1..=MAX_SAMPLE_RATE).contains(&format.sample_rate)
        || !(1..=MAX_CHANNELS).contains(&format.channels)
        || !(1..=MAX_BITS_PER_SAMPLE).contains(&format.bits_per_sample)
        || format.block_align == 0
    {
        return Ok(None);
    }
    let tag = if format.tag == WAVE_FORMAT_EXTENSIBLE {
        format.sub_format_tag
    } else {
        format.tag
    };
    if !stream.seek_from_start(data_pos)? {
        return Ok(None);
    }
    // The data is capped at the end of the file, in unsigned arithmetic
    // against dr_wav's count, which can be past the end.
    let file_len = stream.len();
    let mut data_len = declared;
    if data_len.wrapping_add(data_pos) > file_len {
        data_len = file_len.wrapping_sub(data_pos);
    }
    if data_len == 0xFFFF_FFFF {
        data_len = file_len - stream.pos;
    }
    let Some(frames) = frames(&format, tag, data_len) else {
        return Ok(None);
    };
    Ok(Some(DeclaredLength {
        frames,
        sample_rate: format.sample_rate,
    }))
}

/// dr_wav's frame count, or `None` where it rejects the format.
fn frames(format: &Format, tag: u16, mut data_len: u64) -> Option<u64> {
    let channels = u64::from(format.channels);
    let block_align = u64::from(format.block_align);
    let bytes_per_frame = if format.bits_per_sample.is_multiple_of(8) {
        u64::from(format.bits_per_sample) * channels / 8
    } else {
        block_align
    };
    if bytes_per_frame == 0 {
        return None;
    }
    let header_bytes_per_channel = match tag {
        WAVE_FORMAT_ADPCM => 6,
        WAVE_FORMAT_IMA_ADPCM => 4,
        _ => {
            data_len -= data_len % bytes_per_frame;
            return Some(data_len / bytes_per_frame);
        }
    };
    if channels > 2 {
        return None;
    }
    let block_count = data_len.div_ceil(block_align);
    let header_bytes = block_count.wrapping_mul(header_bytes_per_channel * channels);
    let frames = data_len.wrapping_sub(header_bytes).wrapping_mul(2) / channels;
    Some(if tag == WAVE_FORMAT_IMA_ADPCM {
        frames.wrapping_add(block_count)
    } else {
        frames
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(bytes: &[u8]) -> Option<DeclaredLength> {
        declared_length(bytes).unwrap()
    }

    fn wav(tag: u16, sample_rate: u32, data: usize) -> Vec<u8> {
        let mut wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&tag.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&u32::MAX.to_le_bytes());
        wav.resize(wav.len() + data, 0);
        wav
    }

    #[test]
    fn wav_length_is_capped_by_the_data_present() {
        // The data chunk claims 4 GiB but holds 1 KiB of 16-bit mono.
        let lengths = length(&wav(1, 16_000, 1024));
        assert_eq!(
            lengths,
            Some(DeclaredLength {
                frames: 512,
                sample_rate: 16_000
            })
        );
        // A 1 Hz sample rate makes a 1 MB file last for days.
        assert!(!length(&wav(1, 1, 1 << 20)).unwrap().within_seconds(3600));
    }

    fn riff_wave(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut wav = b"RIFF\0\0\0\0WAVE".to_vec();
        for (id, body) in chunks {
            wav.extend_from_slice(*id);
            wav.extend_from_slice(&(body.len() as u32).to_le_bytes());
            wav.extend_from_slice(body);
            if body.len() % 2 == 1 {
                wav.push(0);
            }
        }
        wav
    }

    fn fmt_chunk(tag: u16, channels: u16, rate: u32, block_align: u16, bits: u16) -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt
    }

    #[test]
    fn wav_length_comes_from_the_first_data_chunk() {
        // miniaudio stops at the first data chunk: here 1 MiB at 1 Hz, which
        // it reports as 8.4 billion frames once resampled to 16 kHz.
        let wav = riff_wave(&[
            (b"fmt ", fmt_chunk(1, 1, 1, 2, 16)),
            (b"data", vec![0; 1 << 20]),
            (b"data", Vec::new()),
            (b"fmt ", fmt_chunk(1, 1, 48_000, 2, 16)),
        ]);
        let lengths = length(&wav);
        assert!(
            lengths.iter().any(|length| !length.within_seconds(3600)),
            "{lengths:?}"
        );
    }

    #[test]
    fn format_chunks_advance_the_way_the_decoder_reads_them() {
        // The decoder always reads 16 format bytes, even from a chunk that
        // declares fewer, so the data chunk starts right after them.
        let mut undersized = b"RIFF\0\0\0\0WAVEfmt \0\0\0\0".to_vec();
        undersized.extend_from_slice(&fmt_chunk(1, 1, 1, 2, 16));
        undersized.extend_from_slice(b"data");
        undersized.extend_from_slice(&(1u32 << 20).to_le_bytes());
        undersized.resize(undersized.len() + (1 << 20), 0);
        let lengths = length(&undersized);
        assert!(
            lengths.iter().any(|length| !length.within_seconds(3600)),
            "{lengths:?}"
        );

        // A WAVE_FORMAT_EXTENSIBLE chunk declaring 18 bytes but a 22 byte
        // extension: the decoder reads the sub-format past the chunk, here
        // IMA ADPCM, and the following JUNK chunk header overlaps it.
        let mut short = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        short.extend_from_slice(&18u32.to_le_bytes());
        short.extend_from_slice(&fmt_chunk(WAVE_FORMAT_EXTENSIBLE, 1, 16_000, 256, 4));
        short.extend_from_slice(&22u16.to_le_bytes());
        short.extend_from_slice(b"JUNK\0\0");
        short.extend_from_slice(&WAVE_FORMAT_IMA_ADPCM.to_le_bytes());
        let junk_body = 0x0011_0000;
        short.extend_from_slice(&[0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        short.resize(short.len() + junk_body - 14, 0);
        short.extend_from_slice(b"data\x01\0\0\0\0\0");
        assert_eq!(length(&short).unwrap().frames, u64::MAX - 4);
    }

    #[test]
    fn short_fact_chunks_are_rejected() {
        // The decoder spent 17 seconds skipping past this 24 byte file.
        let stall = b"RIFF\x10\0\0\0WAVEfact\x02\0\0\0\xd6\x9e\xd6\xf1";
        let error = declared_length(stall).unwrap_err();
        assert!(error.to_string().contains("stall"), "{error}");

        let fine = riff_wave(&[
            (b"fact", vec![0; 4]),
            (b"fmt ", fmt_chunk(1, 1, 16_000, 2, 16)),
        ]);
        declared_length(&fine).unwrap();
    }

    #[test]
    fn adpcm_lengths_follow_the_decoder_arithmetic() {
        // One byte of IMA ADPCM is smaller than its block header, and the
        // decoder's unsigned arithmetic wraps to 2^64 - 5 frames.
        let ima = riff_wave(&[
            (b"fmt ", fmt_chunk(WAVE_FORMAT_IMA_ADPCM, 1, 16_000, 256, 4)),
            (b"data", vec![0]),
        ]);
        assert_eq!(length(&ima).unwrap().frames, u64::MAX - 4);

        let ms = riff_wave(&[
            (b"fmt ", fmt_chunk(WAVE_FORMAT_ADPCM, 1, 16_000, 256, 4)),
            (b"data", vec![0]),
        ]);
        assert!(!length(&ms).unwrap().within_seconds(3600));

        // A whole block decodes to a normal length.
        let whole = riff_wave(&[
            (b"fmt ", fmt_chunk(WAVE_FORMAT_IMA_ADPCM, 1, 16_000, 256, 4)),
            (b"data", vec![0; 256]),
        ]);
        assert_eq!(length(&whole).unwrap().frames, 505);

        // The decoder rejects zero channels or block alignment before it
        // would divide by them.
        for fmt in [
            fmt_chunk(WAVE_FORMAT_IMA_ADPCM, 0, 16_000, 256, 4),
            fmt_chunk(WAVE_FORMAT_ADPCM, 1, 16_000, 0, 8),
        ] {
            let wav = riff_wave(&[(b"fmt ", fmt), (b"data", vec![0; 512])]);
            assert_eq!(length(&wav), None);
        }
    }

    #[test]
    fn a_count_pushed_past_the_end_wraps_the_data_size() {
        // A fmt size near 2^32 seeks back to offset 0 while dr_wav's count
        // jumps 4 GB ahead; the RIFF header then reads as a chunk whose size
        // lands on a data chunk. Capping the data against the count wraps,
        // and miniaudio reported 9.2e18 frames for these 72 bytes.
        let mut wav = b"RIFF\x28\0\0\0WAVEfmt ".to_vec();
        wav.extend_from_slice(&0xFFFF_FFECu32.to_le_bytes());
        wav.extend_from_slice(&fmt_chunk(1, 1, 16_000, 2, 16));
        wav.extend_from_slice(&0u16.to_le_bytes());
        wav.resize(48, 0);
        wav.extend_from_slice(b"data\x10\0\0\0");
        wav.resize(72, 0);
        assert!(!length(&wav).unwrap().within_seconds(3600));
    }

    #[test]
    fn chunks_that_lead_back_to_themselves_are_rejected() {
        // This fmt size seeks back onto its own header, and dr_wav loops.
        let mut wav = b"RIFF\x40\0\0\0WAVEfmt ".to_vec();
        wav.extend_from_slice(&0xFFFF_FFF8u32.to_le_bytes());
        wav.extend_from_slice(&fmt_chunk(1, 1, 16_000, 2, 16));
        wav.extend_from_slice(&[0; 18]);
        let error = declared_length(&wav).unwrap_err();
        assert!(error.to_string().contains("over and over"), "{error}");
    }
}
