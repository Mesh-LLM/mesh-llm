pub(super) use super::EncodingError;
use super::{Argument, Value, kind};
use std::ffi::OsString;
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;

#[cfg(unix)]
pub(super) fn unix_utf8_argv(arguments: &[Argument<'_>]) -> Result<Vec<OsString>, EncodingError> {
    arguments
        .iter()
        .enumerate()
        .map(|(index, argument)| {
            let value = match argument {
                Argument::Os(value) => value.clone(),
                Argument::Text(text) => encode(text.codepoints(), index)?,
                Argument::Decoded(Value::Str(text)) => encode(text.codepoints(), index)?,
                Argument::Decoded(value) => {
                    return Err(EncodingError::NonString {
                        index,
                        kind: kind(value),
                    });
                }
            };
            if value.as_encoded_bytes().contains(&0) {
                return Err(EncodingError::EmbeddedNul { index });
            }
            Ok(value)
        })
        .collect()
}

#[cfg(unix)]
fn encode(codes: impl Iterator<Item = u32>, index: usize) -> Result<OsString, EncodingError> {
    let mut bytes = Vec::new();
    for codepoint in codes {
        match char::from_u32(codepoint) {
            Some(character) => {
                let mut encoded = [0; 4];
                bytes.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
            }
            None if (0xdc80..=0xdcff).contains(&codepoint) => {
                bytes.push(codepoint.to_le_bytes()[0])
            }
            None => return Err(EncodingError::UnencodableCodepoint { index, codepoint }),
        }
    }
    Ok(OsString::from_vec(bytes))
}

#[cfg(windows)]
pub(super) fn windows_argv(arguments: &[Argument<'_>]) -> Result<Vec<OsString>, EncodingError> {
    arguments
        .iter()
        .enumerate()
        .map(|(index, argument)| {
            let value = match argument {
                Argument::Os(value) => value.clone(),
                Argument::Text(text) => scalar_os(text.codepoints(), index)?,
                Argument::Decoded(Value::Str(text)) => scalar_os(text.codepoints(), index)?,
                Argument::Decoded(value) => {
                    return Err(EncodingError::NonString {
                        index,
                        kind: kind(value),
                    });
                }
            };
            if value.as_encoded_bytes().contains(&0) {
                return Err(EncodingError::EmbeddedNul { index });
            }
            Ok(value)
        })
        .collect()
}

#[cfg(windows)]
fn scalar_os(codes: impl Iterator<Item = u32>, index: usize) -> Result<OsString, EncodingError> {
    let mut encoded = String::new();
    for codepoint in codes {
        encoded.push(
            char::from_u32(codepoint)
                .ok_or(EncodingError::UnencodableCodepoint { index, codepoint })?,
        );
    }
    Ok(encoded.into())
}
