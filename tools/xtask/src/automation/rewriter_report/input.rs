use super::Mode;
use super::types::{Field, Report, ReportValue};
use serde::Deserialize;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek};
use std::path::Path;

const MAX_CONTAINER_DEPTH: usize = 256;

#[derive(Debug)]
pub(in crate::automation) enum ReportInputError {
    Load(io::Error),
    Syntax(serde_json::Error),
    Encoding { offset: usize },
    Depth,
    Root,
}

impl fmt::Display for ReportInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(error) => write!(formatter, "{error}"),
            Self::Syntax(error) => write!(formatter, "{error}"),
            Self::Encoding { offset } => write!(formatter, "invalid UTF-8 at byte {offset}"),
            Self::Depth => write!(
                formatter,
                "report JSON nesting exceeds safety limit of {MAX_CONTAINER_DEPTH} containers"
            ),
            Self::Root => formatter.write_str("report must be a JSON object"),
        }
    }
}

impl std::error::Error for ReportInputError {}

pub(super) fn load(path: &Path, mode: Mode) -> Result<Report, ReportInputError> {
    let file = File::open(path).map_err(ReportInputError::Load)?;
    parse(BufReader::new(file), mode)
}
pub(super) fn load_projection<T: ReportValue>(path: &Path) -> Result<T, ReportInputError> {
    let file = File::open(path).map_err(ReportInputError::Load)?;
    let mut reader = BufReader::new(file);
    admit(&mut reader)?;
    reader.rewind().map_err(ReportInputError::Load)?;
    let mut decoder = serde_json::Deserializer::from_reader(reader);
    let report = root(&mut decoder)?;
    decoder.end().map_err(ReportInputError::Syntax)?;
    Ok(report)
}

fn parse(mut reader: impl BufRead + Seek, mode: Mode) -> Result<Report, ReportInputError> {
    admit(&mut reader)?;
    reader.rewind().map_err(ReportInputError::Load)?;
    let mut decoder = serde_json::Deserializer::from_reader(reader);
    let report = match mode {
        Mode::Validate => Report::Validate(Box::new(root(&mut decoder)?)),
        Mode::Idempotence => Report::Idempotence(root(&mut decoder)?),
    };
    decoder.end().map_err(ReportInputError::Syntax)?;
    Ok(report)
}

fn root<'de, R: serde_json::de::Read<'de>, T: ReportValue>(
    decoder: &mut serde_json::Deserializer<R>,
) -> Result<T, ReportInputError> {
    match Field::<T>::deserialize(decoder).map_err(ReportInputError::Syntax)? {
        Field::Present(report) => Ok(report),
        Field::Missing | Field::Null | Field::Invalid(_) => Err(ReportInputError::Root),
    }
}

fn admit(reader: &mut impl BufRead) -> Result<(), ReportInputError> {
    let mut depth = 0_usize;
    let mut quoted = false;
    let mut escaped = false;
    let mut utf8 = Utf8::default();
    loop {
        let bytes = reader.fill_buf().map_err(ReportInputError::Load)?;
        if bytes.is_empty() {
            break;
        }
        for &byte in bytes {
            utf8.push(byte)?;
            if quoted {
                match (escaped, byte) {
                    (true, _) => escaped = false,
                    (false, b'\\') => escaped = true,
                    (false, b'"') => quoted = false,
                    (false, _) => {}
                }
            } else {
                match byte {
                    b'"' => quoted = true,
                    b'[' | b'{' => {
                        depth += 1;
                        if depth > MAX_CONTAINER_DEPTH {
                            return Err(ReportInputError::Depth);
                        }
                    }
                    b']' | b'}' => depth = depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
        let length = bytes.len();
        reader.consume(length);
    }
    if utf8.remaining != 0 {
        return Err(ReportInputError::Encoding {
            offset: utf8.offset,
        });
    }
    Ok(())
}

#[derive(Default)]
struct Utf8 {
    remaining: u8,
    lower: u8,
    upper: u8,
    offset: usize,
}

impl Utf8 {
    fn push(&mut self, byte: u8) -> Result<(), ReportInputError> {
        if self.remaining != 0 {
            if byte < self.lower || byte > self.upper {
                return Err(ReportInputError::Encoding {
                    offset: self.offset,
                });
            }
            self.remaining -= 1;
            self.lower = 0x80;
            self.upper = 0xbf;
        } else {
            (self.remaining, self.lower, self.upper) = match byte {
                0x00..=0x7f => (0, 0, 0),
                0xc2..=0xdf => (1, 0x80, 0xbf),
                0xe0 => (2, 0xa0, 0xbf),
                0xe1..=0xec | 0xee..=0xef => (2, 0x80, 0xbf),
                0xed => (2, 0x80, 0x9f),
                0xf0 => (3, 0x90, 0xbf),
                0xf1..=0xf3 => (3, 0x80, 0xbf),
                0xf4 => (3, 0x80, 0x8f),
                _ => {
                    return Err(ReportInputError::Encoding {
                        offset: self.offset,
                    });
                }
            };
        }
        self.offset += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, parse};
    use std::io::{BufReader, Cursor};

    #[test]
    fn boundary_decode_conversion_and_drop_fit_a_one_mib_stack() {
        std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(|| {
                for (open, close) in [("[", "]"), (r#"{"nested":"#, "}")] {
                    let nested = format!("{}0{}", open.repeat(255), close.repeat(255));
                    for mode in [Mode::Validate, Mode::Idempotence] {
                        let body = format!(r#"{{"ignored":{nested}}}"#);
                        drop(
                            parse(Cursor::new(body.as_bytes()), mode)
                                .expect("decode boundary report"),
                        );
                        let partial = format!(r#"{{"ignored":{nested},"broken":"#);
                        assert!(parse(Cursor::new(partial.as_bytes()), mode).is_err());
                    }
                }
            })
            .expect("spawn bounded-stack test")
            .join()
            .expect("bounded-stack decoding and cleanup");
    }

    #[test]
    fn streamed_utf8_validates_across_buffer_boundaries() {
        let valid = "{\"ignored\":\"\u{fffd}\u{1f600}\"}";
        assert!(
            parse(
                BufReader::with_capacity(1, Cursor::new(valid)),
                Mode::Idempotence
            )
            .is_ok()
        );
        for bytes in [
            &b"{\"ignored\":\"\xff\"}"[..],
            &b"{\"ignored\":\"\xed\xa0\x80\"}"[..],
            &b"{\"ignored\":\"\xf0\x80\x80\x80\"}"[..],
        ] {
            assert!(
                parse(
                    BufReader::with_capacity(1, Cursor::new(bytes)),
                    Mode::Idempotence
                )
                .is_err()
            );
        }
    }
}
