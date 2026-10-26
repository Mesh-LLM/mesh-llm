//! ZIP legacy filename decoding required by the archive format.

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zip_legacy_filename_encoding_preserves_ascii_and_extended_characters() {
        assert_eq!(CP437_HIGH.chars().count(), 128);
        assert_eq!(cp437(b"a\x80\xff"), "aÇ\u{a0}");
        assert_eq!(cp437(b"path/file.txt"), "path/file.txt");
    }
}
