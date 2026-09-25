//! Order-preserving JSON for planner input and the JSON-compatible YAML
//! catalogs. The legacy planner iterates Python dicts in insertion order, and
//! that order decides which diagnostic a malformed catalog reports first, so
//! objects keep their source order (a repeated key keeps its first position
//! and its last value, like `json.loads`). Plans themselves are emitted with
//! sorted keys, which [`Json::to_value`] provides.

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Python `dict.get`: `None` for a missing key or a non-object.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// Python `dict.get` that also maps JSON `null` to `None`.
    pub(crate) fn get_present(&self, key: &str) -> Option<&Json> {
        self.get(key).filter(|value| **value != Json::Null)
    }

    pub(crate) fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(text) => Some(text),
            _ => None,
        }
    }

    /// Python `type(value) is int`: an integral JSON number, never a bool
    /// or a float such as `1.0`.
    pub(crate) fn as_int(&self) -> Option<i128> {
        match self {
            Json::Number(number) => number
                .as_i64()
                .map(i128::from)
                .or_else(|| number.as_u64().map(i128::from)),
            _ => None,
        }
    }

    /// Python `value == 1`, which `1.0` and `True` also satisfy.
    pub(crate) fn equals_one(&self) -> bool {
        match self {
            Json::Bool(flag) => *flag,
            Json::Number(number) => number.as_f64() == Some(1.0),
            _ => false,
        }
    }

    pub(crate) fn to_value(&self) -> Value {
        match self {
            Json::Null => Value::Null,
            Json::Bool(flag) => Value::Bool(*flag),
            Json::Number(number) => Value::Number(number.clone()),
            Json::String(text) => Value::String(text.clone()),
            Json::Array(items) => Value::Array(items.iter().map(Json::to_value).collect()),
            Json::Object(entries) => Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_value()))
                    .collect::<Map<_, _>>(),
            ),
        }
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Json, E> {
        Ok(Json::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Json, E> {
        Ok(Json::Number(value.into()))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Json, E> {
        Ok(Json::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Json, E> {
        Number::from_f64(value)
            .map(Json::Number)
            .ok_or_else(|| E::custom("non-finite number"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Json, E> {
        Ok(Json::String(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Json, E> {
        Ok(Json::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Json::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut entries: Vec<(String, Json)> = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, Json>()? {
            match entries.iter_mut().find(|(name, _)| *name == key) {
                Some(entry) => entry.1 = value,
                None => entries.push((key, value)),
            }
        }
        Ok(Json::Object(entries))
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_plan_document_keeps_first_position_and_last_value() {
        let parsed = Json::parse(br#"{"b":1,"a":2,"b":3}"#).expect("valid JSON");
        let keys = parsed
            .as_object()
            .expect("object")
            .iter()
            .map(|(key, _)| key.as_str())
            .collect::<Vec<_>>();
        assert_eq!(keys, ["b", "a"]);
        assert_eq!(parsed.get("b").and_then(Json::as_int), Some(3));
    }

    #[test]
    fn migration_ci_plan_document_follows_python_numeric_identity() {
        let one = |text: &str| Json::parse(text.as_bytes()).expect("valid JSON");
        assert!(one("1").equals_one() && one("1.0").equals_one() && one("true").equals_one());
        assert!(!one("2").equals_one() && !one("\"1\"").equals_one());
        assert_eq!(one("1.0").as_int(), None);
        assert_eq!(one("true").as_int(), None);
        assert_eq!(one("7").as_int(), Some(7));
    }
}
