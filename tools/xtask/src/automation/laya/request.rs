use crate::ci_plan::document::Json;
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

pub(super) struct Ordered<'a>(pub &'a Json);

impl Serialize for Ordered<'_> {
    fn serialize<Encoder: Serializer>(
        &self,
        encoder: Encoder,
    ) -> Result<Encoder::Ok, Encoder::Error> {
        match self.0 {
            Json::Null => encoder.serialize_unit(),
            Json::Bool(value) => encoder.serialize_bool(*value),
            Json::Number(value) => value.serialize(encoder),
            Json::String(value) => encoder.serialize_str(value),
            Json::Array(values) => {
                let mut sequence = encoder.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(&Ordered(value))?;
                }
                sequence.end()
            }
            Json::Object(entries) => {
                let mut map = encoder.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, &Ordered(value))?;
                }
                map.end()
            }
        }
    }
}

pub(super) fn body(
    model: &str,
    golden: &super::parity::Golden,
) -> Result<Vec<u8>, serde_json::Error> {
    #[derive(Serialize)]
    struct Request<'a> {
        model: &'a str,
        state: Ordered<'a>,
        questions: Ordered<'a>,
    }
    serde_json::to_vec(&Request {
        model,
        state: Ordered(&golden.state),
        questions: Ordered(&golden.questions),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_options_keep_source_order() {
        let source = br#"{"state":{"z":1,"a":2},"questions":{"q":{"criteria":{"z":"last alphabetically","a":"first alphabetically"}}},"answers":{},"per_question":{}}"#;
        let golden = serde_json::from_slice(source).unwrap();
        let bytes = body("fixture", &golden).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            r#"{"model":"fixture","state":{"z":1,"a":2},"questions":{"q":{"criteria":{"z":"last alphabetically","a":"first alphabetically"}}}}"#
        );
    }
}
