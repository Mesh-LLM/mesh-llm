//! Readable single-line JSON inside trajectory prompts. Separator spaces are
//! part of the pinned prompt text, which participates in manifest hashes.
use std::io::{self, Write};

use serde::Serialize;
use serde_json::{Value, ser::Formatter};

use crate::command::DynResult;

struct ReadableJson;

impl Formatter for ReadableJson {
    fn begin_array_value<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

pub(super) fn render(value: &Value) -> DynResult<String> {
    let mut bytes = Vec::new();
    value.serialize(&mut serde_json::Serializer::with_formatter(
        &mut bytes,
        ReadableJson,
    ))?;
    Ok(String::from_utf8(bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn structured_prompt_content_preserves_spaces_sorted_keys_and_utf8() {
        assert_eq!(
            render(&json!({"path":"src/lib.rs"})).unwrap(),
            r#"{"path": "src/lib.rs"}"#
        );
        assert_eq!(
            render(&json!({"z":[1,null,"é"],"a":true})).unwrap(),
            r#"{"a": true, "z": [1, null, "é"]}"#
        );
        assert_eq!(render(&json!([])).unwrap(), "[]");
        assert_eq!(render(&json!({})).unwrap(), "{}");
        assert_eq!(
            render(&json!("line\nquote\"\\")).unwrap(),
            r#""line\nquote\"\\""#
        );
    }
}
