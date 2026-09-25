//! Python 3.13 renderings the ZIP port reproduces: `cp437` filename
//! decoding, `repr(bytes)`, `stat.filemode` and `repr(ZipInfo)`.

use crate::repository::python_text::repr;

/// `bytes.decode("cp437")` for 0x80..=0xFF; the lower half is ASCII.
const CP437_HIGH: &str = "ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\u{a0}";

pub(super) fn cp437(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&byte| match byte {
            0..=0x7f => char::from(byte),
            _ => CP437_HIGH
                .chars()
                .nth(usize::from(byte - 0x80))
                .unwrap_or('\u{fffd}'),
        })
        .collect()
}

/// `repr(bytes)`.
pub(super) fn bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        '"'
    } else {
        '\''
    };
    let mut rendered = format!("b{quote}");
    for &byte in bytes {
        match byte {
            b'\\' => rendered.push_str("\\\\"),
            b'\t' => rendered.push_str("\\t"),
            b'\n' => rendered.push_str("\\n"),
            b'\r' => rendered.push_str("\\r"),
            _ if char::from(byte) == quote => {
                rendered.push('\\');
                rendered.push(quote);
            }
            0x20..=0x7e => rendered.push(char::from(byte)),
            _ => rendered.push_str(&format!("\\x{byte:02x}")),
        }
    }
    rendered.push(quote);
    rendered
}

/// `stat.filemode(mode)`.
pub(super) fn filemode(mode: u32) -> String {
    let kind = match mode & 0o170_000 {
        0o120_000 => 'l',
        0o140_000 => 's',
        0o100_000 => '-',
        0o060_000 => 'b',
        0o040_000 => 'd',
        0o020_000 => 'c',
        0o010_000 => 'p',
        _ => '?',
    };
    let triad = |shift: u32, special: u32, set: char| {
        let bits = (mode >> shift) & 0o7;
        let exec = match (mode & special != 0, bits & 1 != 0) {
            (true, true) => set,
            (true, false) => set.to_ascii_uppercase(),
            (false, true) => 'x',
            (false, false) => '-',
        };
        let read = if bits & 4 != 0 { 'r' } else { '-' };
        let write = if bits & 2 != 0 { 'w' } else { '-' };
        format!("{read}{write}{exec}")
    };
    format!(
        "{kind}{}{}{}",
        triad(6, 0o4000, 's'),
        triad(3, 0o2000, 's'),
        triad(0, 0o1000, 't')
    )
}

fn compressor_name(method: u16) -> String {
    let name = match method {
        1 => "shrink",
        2..=5 => "reduce",
        6 | 10 => "implode",
        7 => "tokenize",
        8 => "deflate",
        9 => "deflate64",
        12 => "bzip2",
        14 => "lzma",
        18 => "terse",
        19 => "lz77",
        97 => "wavpack",
        98 => "ppmd",
        _ => return method.to_string(),
    };
    name.to_owned()
}

/// The `ZipInfo` fields its `repr` shows.
pub(super) struct InfoView<'a> {
    pub(super) filename: &'a str,
    pub(super) method: u16,
    pub(super) external: u32,
    pub(super) file_size: u64,
    pub(super) compress_size: u64,
}

/// `repr(ZipInfo)`.
pub(super) fn info_repr(info: &InfoView<'_>) -> String {
    let mut out = format!("<ZipInfo filename={}", repr(info.filename));
    if info.method != 0 {
        out.push_str(&format!(" compress_type={}", compressor_name(info.method)));
    }
    let (hi, lo) = (info.external >> 16, info.external & 0xffff);
    if hi != 0 {
        out.push_str(&format!(" filemode={}", repr(&filemode(hi))));
    }
    if lo != 0 {
        out.push_str(&format!(" external_attr={lo:#x}"));
    }
    let is_dir = info.filename.ends_with('/');
    if !is_dir || info.file_size != 0 {
        out.push_str(&format!(" file_size={}", info.file_size));
    }
    let differs = info.method != 0 || info.file_size != info.compress_size;
    if (!is_dir || info.compress_size != 0) && differs {
        out.push_str(&format!(" compress_size={}", info.compress_size));
    }
    out.push('>');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_artifact_zip_text_matches_python() {
        assert_eq!(CP437_HIGH.chars().count(), 128);
        assert_eq!(cp437(b"a\x80\xff"), "aÇ\u{a0}");
        assert_eq!(filemode(0o104_755), "-rwsr-xr-x");
        assert_eq!(filemode(0o041_777), "drwxrwxrwt");
        assert_eq!(bytes_repr(b"a'\x00"), "b\"a'\\x00\"");
    }
}
