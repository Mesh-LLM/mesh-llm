use crate::command::DynResult;
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub(super) fn encode(bytes: &[u8]) -> String {
    let mut output = String::new();
    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        output.push(char::from(ALPHABET[usize::from(first >> 2)]));
        output.push(char::from(
            ALPHABET[usize::from((first & 3) << 4 | second >> 4)],
        ));
        output.push(if chunk.len() > 1 {
            char::from(ALPHABET[usize::from((second & 15) << 2 | third >> 6)])
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            char::from(ALPHABET[usize::from(third & 63)])
        } else {
            '='
        });
    }
    output
}

pub(super) fn decode(text: &str) -> DynResult<Vec<u8>> {
    if !text.len().is_multiple_of(4) {
        return Err("invalid base64 length".into());
    }
    let mut result = Vec::new();
    for (index, chunk) in text.as_bytes().as_chunks::<4>().0.iter().enumerate() {
        let mut values = [0_u8; 4];
        let padding = chunk.iter().rev().take_while(|byte| **byte == b'=').count();
        if padding > 2 || (padding > 0 && index + 1 != text.len() / 4) {
            return Err("invalid base64 padding".into());
        }
        for position in 0..4 - padding {
            values[position] = u8::try_from(
                ALPHABET
                    .iter()
                    .position(|byte| *byte == chunk[position])
                    .ok_or("invalid base64 character")?,
            )?;
        }
        result.push(values[0] << 2 | values[1] >> 4);
        if padding < 2 {
            result.push(values[1] << 4 | values[2] >> 2);
        }
        if padding == 0 {
            result.push(values[2] << 6 | values[3]);
        }
    }
    Ok(result)
}
