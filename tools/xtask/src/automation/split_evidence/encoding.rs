use crate::automation::codepoint_json::{parser, value::Value};

pub(super) fn snapshot(raw: &[u8]) -> Result<Value, String> {
    let encoding = if raw.starts_with(b"\x00\x00\xfe\xff") {
        Some((4, false, 4))
    } else if raw.starts_with(b"\xff\xfe\x00\x00") {
        Some((4, true, 4))
    } else if raw.starts_with(b"\xfe\xff") {
        Some((2, false, 2))
    } else if raw.starts_with(b"\xff\xfe") {
        Some((2, true, 2))
    } else if raw.len() >= 4 && raw[0..3] == [0, 0, 0] {
        Some((4, false, 0))
    } else if raw.len() >= 4 && raw[1..4] == [0, 0, 0] {
        Some((4, true, 0))
    } else if raw.len() >= 2 && raw[0] == 0 {
        Some((2, false, 0))
    } else if raw.len() >= 2 && raw[1] == 0 {
        Some((2, true, 0))
    } else {
        None
    };
    let Some((width, little, skip)) = encoding else {
        return parser::parse(raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(raw));
    };
    let mut text = String::new();
    let mut units = raw[skip..].chunks_exact(width);
    let mut codes = Vec::new();
    for unit in &mut units {
        let code = if width == 2 {
            let bytes = [unit[0], unit[1]];
            u32::from(if little {
                u16::from_le_bytes(bytes)
            } else {
                u16::from_be_bytes(bytes)
            })
        } else {
            let bytes = [unit[0], unit[1], unit[2], unit[3]];
            if little {
                u32::from_le_bytes(bytes)
            } else {
                u32::from_be_bytes(bytes)
            }
        };
        codes.push(code);
    }
    if !units.remainder().is_empty() {
        return Err("truncated Unicode snapshot".into());
    }
    let mut codes = codes.into_iter().peekable();
    while let Some(mut code) = codes.next() {
        if width == 2
            && (0xd800..0xdc00).contains(&code)
            && let Some(low) = codes
                .peek()
                .copied()
                .filter(|low| (0xdc00..0xe000).contains(low))
        {
            codes.next();
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
        }
        match char::from_u32(code) {
            Some(character) => text.push(character),
            None if (0xd800..0xe000).contains(&code) => text.push_str(&format!("\\u{code:04x}")),
            None => return Err("Unicode snapshot codepoint out of range".into()),
        }
    }
    parser::parse(text.as_bytes())
}
