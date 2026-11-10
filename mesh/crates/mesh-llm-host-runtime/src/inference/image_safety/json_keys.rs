//! Reject ambiguous JSON before screening a view of unchanged forwarding bytes.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt;

use serde::Deserializer;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};

/// Check decoded object keys before using serde_json's normal value parser.
/// This leaves its number, string, and recursion-limit behavior authoritative.
pub(super) fn parse(bytes: &[u8]) -> serde_json::Result<serde_json::Value> {
    let duplicate = Cell::new(false);
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    UniqueKeys {
        duplicate: &duplicate,
    }
    .deserialize(&mut deserializer)?;
    deserializer.end()?;
    serde_json::from_slice(bytes)
}

/// Detect ambiguity without requiring unrelated binary workloads to be JSON.
pub(super) fn has_duplicate_keys(bytes: &[u8]) -> bool {
    let duplicate = Cell::new(false);
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let _ = UniqueKeys {
        duplicate: &duplicate,
    }
    .deserialize(&mut deserializer);
    duplicate.get() && serde_json::from_slice::<de::IgnoredAny>(bytes).is_ok()
}

struct UniqueKeys<'a> {
    duplicate: &'a Cell<bool>,
}

impl<'de> DeserializeSeed<'de> for UniqueKeys<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for UniqueKeys<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object keys")
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E: de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        while sequence
            .next_element_seed(UniqueKeys {
                duplicate: self.duplicate,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key) {
                self.duplicate.set(true);
                return Err(de::Error::custom("duplicate JSON object key"));
            }
            map.next_value_seed(UniqueKeys {
                duplicate: self.duplicate,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{has_duplicate_keys, parse};

    #[test]
    fn rejects_duplicate_decoded_keys_at_every_depth() {
        for body in [
            r#"{"prompt":"first","prompt":"second"}"#,
            r#"{"messages":[{"content":"first","content":"second"}]}"#,
            r#"{"messages":[{"content":"first","cont\u0065nt":"second"}]}"#,
            r#"{"template":{"instruction":"first","instruction":"second"}}"#,
        ] {
            assert!(parse(body.as_bytes()).is_err());
            assert!(has_duplicate_keys(body.as_bytes()));
        }
    }

    #[test]
    fn preserves_normal_values_and_separate_object_key_scopes() {
        let body = br#"{"messages":[{"content":"first"},{"content":"second"}],"values":[null,true,false,0,-1,18446744073709551615,1.25,1e30,"\u0063at"]}"#;
        assert_eq!(
            parse(body).unwrap(),
            serde_json::from_slice::<serde_json::Value>(body).unwrap()
        );
    }

    #[test]
    fn rejects_trailing_data_and_invalid_json() {
        for body in ["{} {}", "{", "[1,]", r#"{"key":"\uD800"}"#] {
            assert!(parse(body.as_bytes()).is_err());
        }
    }

    #[test]
    fn duplicate_probe_preserves_non_json_workloads() {
        for body in [
            b"\xff\xd8\xff".as_slice(),
            b"--audio\r\nContent-Type: audio/wav\r\n",
            b"not JSON",
            b"{",
            b"[1,]",
            b"{\"key\":0,\"key\":1}\0binary",
        ] {
            assert!(!has_duplicate_keys(body));
        }
    }
}
