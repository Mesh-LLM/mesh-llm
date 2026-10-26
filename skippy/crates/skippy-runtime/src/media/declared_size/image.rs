//! Image dimensions as stb_image reads them from the file header.
//!
//! The native helper decodes images with stb_image, which picks the first
//! format whose test accepts the data, in a fixed order, and then trusts that
//! format's header. Formats are tried here in the same order with the same
//! tests, so the dimensions found are the ones stb_image would allocate for.

use super::HeaderReader;

/// Returns the width and height the image header declares, or `None` when
/// stb_image would not decode the data as an image at all.
pub(super) fn declared_dimensions(bytes: &[u8]) -> Option<(u64, u64)> {
    // stb_image's `stbi__load_main` order.
    if is_png(bytes) {
        return png(bytes);
    }
    if is_bmp(bytes) {
        return Some(bmp(bytes));
    }
    if is_gif(bytes) {
        return Some(gif(bytes));
    }
    if bytes.starts_with(b"8BPS") {
        return Some(psd(bytes));
    }
    if is_pic(bytes) {
        return Some(pic(bytes));
    }
    if is_jpeg(bytes) {
        return jpeg(bytes);
    }
    if is_pnm(bytes) {
        return Some(pnm(bytes));
    }
    if is_hdr(bytes) {
        return hdr(bytes);
    }
    if is_tga(bytes) {
        return Some(tga(bytes));
    }
    None
}

const PNG_SIGNATURE: &[u8] = &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

fn is_png(bytes: &[u8]) -> bool {
    bytes.starts_with(PNG_SIGNATURE)
}

/// The IHDR chunk must come first, after any Apple `CgBI` chunks.
fn png(bytes: &[u8]) -> Option<(u64, u64)> {
    let mut header = HeaderReader::new(bytes, PNG_SIGNATURE.len());
    loop {
        let length = header.be32();
        match &header.be32().to_be_bytes() {
            b"CgBI" => {
                // stb_image skips with a signed count, so a length past
                // `i32::MAX` jumps to the end.
                let skip = i32::try_from(length).map_or(bytes.len(), |length| length as usize);
                header.skip(skip);
                header.skip(4); // CRC
            }
            b"IHDR" => {
                let width = header.be32();
                let height = header.be32();
                return Some((width.into(), height.into()));
            }
            _ => return None,
        }
    }
}

fn is_bmp(bytes: &[u8]) -> bool {
    let mut header = HeaderReader::new(bytes, 0);
    if header.u8() != b'B' || header.u8() != b'M' {
        return false;
    }
    header.skip(12);
    matches!(header.le32(), 12 | 40 | 56 | 108 | 124)
}

fn bmp(bytes: &[u8]) -> (u64, u64) {
    let mut header = HeaderReader::new(bytes, 14);
    if header.le32() == 12 {
        (header.le16().into(), header.le16().into())
    } else {
        let width = header.le32();
        // A negative height marks a top-down bitmap.
        let height = i32::from_le_bytes(header.le32().to_le_bytes()).unsigned_abs();
        (width.into(), height.into())
    }
}

fn is_gif(bytes: &[u8]) -> bool {
    bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")
}

/// stb_image allocates the logical screen size for every GIF.
fn gif(bytes: &[u8]) -> (u64, u64) {
    let mut header = HeaderReader::new(bytes, 6);
    (header.le16().into(), header.le16().into())
}

fn psd(bytes: &[u8]) -> (u64, u64) {
    let mut header = HeaderReader::new(bytes, 14);
    let height = header.be32();
    let width = header.be32();
    (width.into(), height.into())
}

fn is_pic(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x53, 0x80, 0xF6, 0x34]) && bytes.get(88..92) == Some(b"PICT")
}

fn pic(bytes: &[u8]) -> (u64, u64) {
    let mut header = HeaderReader::new(bytes, 92);
    (header.be16().into(), header.be16().into())
}

const JPEG_NO_MARKER: u8 = 0xFF;
const JPEG_SOI: u8 = 0xD8;

/// Mirrors `stbi__get_marker`: a marker is `0xFF`, any fill `0xFF`s, then the
/// marker byte. Anything else is no marker.
fn jpeg_marker(header: &mut HeaderReader<'_>) -> u8 {
    if header.u8() != 0xFF {
        return JPEG_NO_MARKER;
    }
    let mut marker = 0xFF;
    while marker == 0xFF {
        marker = header.u8();
    }
    marker
}

fn is_jpeg(bytes: &[u8]) -> bool {
    jpeg_marker(&mut HeaderReader::new(bytes, 0)) == JPEG_SOI
}

/// Mirrors `stbi__decode_jpeg_header`: walk the markers after SOI until the
/// first baseline, extended or progressive frame header.
fn jpeg(bytes: &[u8]) -> Option<(u64, u64)> {
    let mut header = HeaderReader::new(bytes, 0);
    jpeg_marker(&mut header);
    let mut marker = jpeg_marker(&mut header);
    loop {
        match marker {
            0xC0..=0xC2 => {
                header.skip(3); // length and sample precision
                let height = header.be16();
                let width = header.be16();
                return Some((width.into(), height.into()));
            }
            // Restart interval, quantization and Huffman tables, comments
            // and application data are the segments stb_image accepts
            // before the frame header.
            0xDD | 0xDB | 0xC4 | 0xE0..=0xEF | 0xFE => {
                let length = usize::from(header.be16());
                header.skip(length.saturating_sub(2));
            }
            _ => return None,
        }
        marker = jpeg_marker(&mut header);
        // Padding between segments is skipped up to the next marker.
        while marker == JPEG_NO_MARKER {
            if header.at_end() {
                return None;
            }
            marker = jpeg_marker(&mut header);
        }
    }
}

fn is_pnm(bytes: &[u8]) -> bool {
    bytes.starts_with(b"P5") || bytes.starts_with(b"P6")
}

/// Mirrors `stbi__pnm_info`, including how it reads past the last byte.
fn pnm(bytes: &[u8]) -> (u64, u64) {
    let mut header = HeaderReader::new(bytes, 2);
    let mut current = header.u8();
    pnm_skip_whitespace(&mut header, &mut current);
    let width = pnm_integer(&mut header, &mut current);
    pnm_skip_whitespace(&mut header, &mut current);
    let height = pnm_integer(&mut header, &mut current);
    (width, height)
}

fn pnm_skip_whitespace(header: &mut HeaderReader<'_>, current: &mut u8) {
    loop {
        while !header.at_end() && matches!(*current, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r') {
            *current = header.u8();
        }
        if header.at_end() || *current != b'#' {
            return;
        }
        while !header.at_end() && *current != b'\n' && *current != b'\r' {
            *current = header.u8();
        }
    }
}

fn pnm_integer(header: &mut HeaderReader<'_>, current: &mut u8) -> u64 {
    let mut value = 0u64;
    while !header.at_end() && current.is_ascii_digit() {
        value = value * 10 + u64::from(*current - b'0');
        *current = header.u8();
        // stb_image rejects values that overflow an `int`.
        if value > i32::MAX as u64 / 10 {
            return 0;
        }
    }
    value
}

fn is_hdr(bytes: &[u8]) -> bool {
    bytes.starts_with(b"#?RADIANCE\n") || bytes.starts_with(b"#?RGBE\n")
}

/// Mirrors `stbi__hdr_load`'s header parse: the format line, a blank line,
/// then `-Y <height> +X <width>`.
fn hdr(bytes: &[u8]) -> Option<(u64, u64)> {
    let mut header = HeaderReader::new(bytes, 0);
    hdr_line(&mut header);
    let mut supported = false;
    loop {
        let line = hdr_line(&mut header);
        if line.is_empty() {
            break;
        }
        supported |= line == b"FORMAT=32-bit_rle_rgbe";
    }
    if !supported {
        return None;
    }
    let line = hdr_line(&mut header);
    let rest = line.strip_prefix(b"-Y ")?;
    let (height, mut rest) = c_strtol(rest);
    while let Some(after_space) = rest.strip_prefix(b" ") {
        rest = after_space;
    }
    let rest = rest.strip_prefix(b"+X ")?;
    let (width, _) = c_strtol(rest);
    // stb_image stores the results in an `int`.
    let width = u64::try_from(width as i32).ok()?;
    let height = u64::try_from(height as i32).ok()?;
    Some((width, height))
}

/// Mirrors `stbi__hdr_gettoken`: up to 1023 bytes of a line, without the
/// newline.
fn hdr_line(header: &mut HeaderReader<'_>) -> Vec<u8> {
    let mut line = Vec::new();
    let mut current = header.u8();
    while !header.at_end() && current != b'\n' {
        line.push(current);
        if line.len() == 1023 {
            while !header.at_end() && header.u8() != b'\n' {}
            break;
        }
        current = header.u8();
    }
    line
}

/// Parses a leading integer like C `strtol`, saturating on overflow, and
/// returns it with the unparsed rest.
fn c_strtol(text: &[u8]) -> (i64, &[u8]) {
    let start = text
        .iter()
        .position(|byte| !matches!(byte, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r'))
        .unwrap_or(text.len());
    let text = &text[start..];
    let (negative, digits) = match text.first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let count = digits
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if count == 0 {
        return (0, text);
    }
    let magnitude = digits[..count].iter().fold(0i64, |value, digit| {
        value
            .saturating_mul(10)
            .saturating_add(i64::from(digit - b'0'))
    });
    let value = if negative {
        magnitude.saturating_neg()
    } else {
        magnitude
    };
    (value, &digits[count..])
}

/// Mirrors `stbi__tga_test`; TGA has no magic number, so it is tried last.
fn is_tga(bytes: &[u8]) -> bool {
    let mut header = HeaderReader::new(bytes, 1);
    let colormap_type = header.u8();
    let image_type = header.u8();
    match colormap_type {
        1 => {
            if !matches!(image_type, 1 | 9) {
                return false;
            }
            header.skip(4);
            if !matches!(header.u8(), 8 | 15 | 16 | 24 | 32) {
                return false;
            }
            header.skip(4);
        }
        0 => {
            if !matches!(image_type, 2 | 3 | 10 | 11) {
                return false;
            }
            header.skip(9);
        }
        _ => return false,
    }
    if header.le16() < 1 || header.le16() < 1 {
        return false;
    }
    let bits_per_pixel = header.u8();
    if colormap_type == 1 && !matches!(bits_per_pixel, 8 | 16) {
        return false;
    }
    matches!(bits_per_pixel, 8 | 15 | 16 | 24 | 32)
}

fn tga(bytes: &[u8]) -> (u64, u64) {
    let mut header = HeaderReader::new(bytes, 12);
    (header.le16().into(), header.le16().into())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A PNG signature and IHDR chunk; stb_image allocates for these
    /// dimensions before it reads any image data.
    pub(in crate::media::declared_size) fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut png = PNG_SIGNATURE.to_vec();
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&width.to_be_bytes());
        png.extend_from_slice(&height.to_be_bytes());
        png.extend_from_slice(&[8, 2, 0, 0, 0]);
        png.extend_from_slice(&[0; 4]); // CRC
        png
    }

    #[test]
    fn reads_png_dimensions_after_apple_chunks() {
        assert_eq!(declared_dimensions(&png_header(3, 4)), Some((3, 4)));
        let mut iphone = PNG_SIGNATURE.to_vec();
        iphone.extend_from_slice(&4u32.to_be_bytes());
        iphone.extend_from_slice(b"CgBI");
        iphone.extend_from_slice(&[0; 8]);
        iphone.extend_from_slice(&png_header(5, 6)[PNG_SIGNATURE.len()..]);
        assert_eq!(declared_dimensions(&iphone), Some((5, 6)));
    }

    #[test]
    fn reads_jpeg_frame_header_after_other_segments() {
        let mut jpeg = vec![0xFF, 0xD8];
        // APP0 with four bytes of data, then padding before the next marker.
        jpeg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x06, b'J', b'F', b'I', b'F', 0x00]);
        jpeg.extend_from_slice(&[0xFF, 0xFF, 0xC2, 0x00, 0x11, 0x08]);
        jpeg.extend_from_slice(&40_000u16.to_be_bytes()); // height
        jpeg.extend_from_slice(&30_000u16.to_be_bytes()); // width
        assert_eq!(declared_dimensions(&jpeg), Some((30_000, 40_000)));
        // stb_image rejects an unknown marker before the frame header.
        assert_eq!(declared_dimensions(&[0xFF, 0xD8, 0xFF, 0x01]), None);
    }

    #[test]
    fn reads_the_other_stb_image_formats() {
        let mut bmp = b"BM".to_vec();
        bmp.extend_from_slice(&[0; 12]);
        bmp.extend_from_slice(&40u32.to_le_bytes());
        bmp.extend_from_slice(&7u32.to_le_bytes());
        bmp.extend_from_slice(&(-9i32).to_le_bytes());
        assert_eq!(declared_dimensions(&bmp), Some((7, 9)));

        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&[0xFF, 0xFF, 0x02, 0x00]);
        assert_eq!(declared_dimensions(&gif), Some((65_535, 2)));

        let mut psd = b"8BPS".to_vec();
        psd.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0, 0, 3]);
        psd.extend_from_slice(&11u32.to_be_bytes());
        psd.extend_from_slice(&12u32.to_be_bytes());
        assert_eq!(declared_dimensions(&psd), Some((12, 11)));

        let mut pic = vec![0x53, 0x80, 0xF6, 0x34];
        pic.resize(88, 0);
        pic.extend_from_slice(b"PICT");
        pic.extend_from_slice(&[0x01, 0x00, 0x00, 0x02]);
        assert_eq!(declared_dimensions(&pic), Some((256, 2)));

        assert_eq!(
            declared_dimensions(b"P6\n# comment\n 1200 34\n255\n"),
            Some((1200, 34))
        );
        assert_eq!(
            declared_dimensions(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 20 +X 30\n"),
            Some((30, 20))
        );
        assert_eq!(
            declared_dimensions(b"#?RADIANCE\nFORMAT=other\n\n-Y 20 +X 30\n"),
            None
        );

        let mut tga = vec![0, 0, 2];
        tga.extend_from_slice(&[0; 9]);
        tga.extend_from_slice(&640u16.to_le_bytes());
        tga.extend_from_slice(&480u16.to_le_bytes());
        tga.push(24);
        assert_eq!(declared_dimensions(&tga), Some((640, 480)));
    }

    #[test]
    fn unknown_data_has_no_dimensions() {
        assert_eq!(declared_dimensions(b"hello world"), None);
        assert_eq!(declared_dimensions(&[]), None);
    }
}
