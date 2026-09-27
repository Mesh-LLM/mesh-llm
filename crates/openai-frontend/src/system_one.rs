use std::collections::BTreeMap;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Number, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemOneRequest {
    pub state: SystemOneJson,
    pub model: String,
    pub questions: BTreeMap<String, SystemOneQuestion>,
    #[serde(default)]
    pub images: Option<Vec<Value>>,
    #[serde(default)]
    pub steps: Option<u8>,
    #[serde(default)]
    pub samples: Option<u8>,
    #[serde(default)]
    pub think: Option<u32>,
    #[serde(default)]
    pub sequential: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneQuestion {
    Noul {
        #[serde(default)]
        instructions: Option<Value>,
        #[serde(default)]
        criteria: Option<SystemOneNoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Option<Value>,
        /// Options in request order; see [`SystemOneJson`].
        criteria: SystemOneJsonObject,
    },
    Score {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: Vec<SystemOneJson>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SystemOneNoulCriteria {
    #[serde(default)]
    pub r#true: Option<Value>,
    #[serde(default)]
    pub r#false: Option<Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneAnswer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        legend: BTreeMap<String, Value>,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct SystemOneUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A JSON value that remembers the order its object keys arrived in.
///
/// Jev clients and the Python reference implementations present `state` and
/// choice options in request order, and order-sensitive backends (Laya) read
/// them that way. It serializes, compares, and converts through
/// [`SystemOneJson::to_value`] like an ordinary `serde_json::Value`, so
/// consumers that want the canonical sorted form see exactly that.
#[derive(Debug, Clone, Default)]
pub enum SystemOneJson {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<SystemOneJson>),
    Object(SystemOneJsonObject),
}

/// JSON object entries in request order. A repeated key keeps its first
/// position and its last value, as a Python `dict` built from the same text.
#[derive(Debug, Clone, Default)]
pub struct SystemOneJsonObject(Vec<(String, SystemOneJson)>);

impl SystemOneJson {
    pub fn to_value(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(value.clone()),
            Self::String(value) => Value::String(value.clone()),
            Self::Array(values) => Value::Array(values.iter().map(Self::to_value).collect()),
            Self::Object(object) => Value::Object(object.to_map()),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

impl SystemOneJsonObject {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Entries in request order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SystemOneJson)> {
        self.0.iter().map(|(key, value)| (key.as_str(), value))
    }

    pub fn to_map(&self) -> Map<String, Value> {
        self.0
            .iter()
            .map(|(key, value)| (key.clone(), value.to_value()))
            .collect()
    }

    /// Entries in sorted key order, as an ordinary JSON object map presents them.
    pub fn to_sorted_values(&self) -> BTreeMap<String, Value> {
        self.0
            .iter()
            .map(|(key, value)| (key.clone(), value.to_value()))
            .collect()
    }

    fn insert(&mut self, key: String, value: SystemOneJson) {
        match self.0.iter_mut().find(|(existing, _)| *existing == key) {
            Some(entry) => entry.1 = value,
            None => self.0.push((key, value)),
        }
    }
}

impl<const N: usize> From<[(&str, SystemOneJson); N]> for SystemOneJsonObject {
    fn from(entries: [(&str, SystemOneJson); N]) -> Self {
        let mut object = Self::default();
        for (key, value) in entries {
            object.insert(key.to_string(), value);
        }
        object
    }
}

impl From<Value> for SystemOneJson {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => Self::Number(value),
            Value::String(value) => Self::String(value),
            Value::Array(values) => Self::Array(values.into_iter().map(Self::from).collect()),
            Value::Object(map) => Self::Object(SystemOneJsonObject(
                map.into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            )),
        }
    }
}

impl From<&str> for SystemOneJson {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl PartialEq for SystemOneJson {
    fn eq(&self, other: &Self) -> bool {
        self.to_value() == other.to_value()
    }
}

impl PartialEq for SystemOneJsonObject {
    fn eq(&self, other: &Self) -> bool {
        self.to_map() == other.to_map()
    }
}

impl Serialize for SystemOneJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_value().serialize(serializer)
    }
}

impl Serialize for SystemOneJsonObject {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_map().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SystemOneJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(SystemOneJsonVisitor)
    }
}

impl<'de> Deserialize<'de> for SystemOneJsonObject {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match SystemOneJson::deserialize(deserializer)? {
            SystemOneJson::Object(object) => Ok(object),
            _ => Err(de::Error::custom("expected a JSON object")),
        }
    }
}

struct SystemOneJsonVisitor;

impl<'de> Visitor<'de> for SystemOneJsonVisitor {
    type Value = SystemOneJson;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(SystemOneJson::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(SystemOneJson::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        SystemOneJson::deserialize(deserializer)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(SystemOneJson::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(SystemOneJson::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(SystemOneJson::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        Number::from_f64(value)
            .map(SystemOneJson::Number)
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(SystemOneJson::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(SystemOneJson::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(value) = seq.next_element()? {
            values.push(value);
        }
        Ok(SystemOneJson::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut object = SystemOneJsonObject::default();
        while let Some((key, value)) = map.next_entry::<String, SystemOneJson>()? {
            object.insert(key, value);
        }
        Ok(SystemOneJson::Object(object))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_keep_request_order_and_python_duplicate_semantics() {
        let state: SystemOneJson =
            serde_json::from_str(r#"{"role":"user","content":"hi","role":"assistant"}"#)
                .expect("parse");
        let SystemOneJson::Object(object) = &state else {
            panic!("expected an object");
        };
        let entries = object
            .iter()
            .map(|(key, value)| (key, value.as_str().unwrap_or_default()))
            .collect::<Vec<_>>();
        assert_eq!(entries, vec![("role", "assistant"), ("content", "hi")]);
    }

    #[test]
    fn serializes_as_the_canonical_sorted_value() {
        let raw = r#"{"b":[1,2.5,{"z":null,"a":true}],"a":"x"}"#;
        let state: SystemOneJson = serde_json::from_str(raw).expect("parse");
        let value: Value = serde_json::from_str(raw).expect("parse value");
        assert_eq!(state.to_value(), value);
        assert_eq!(
            serde_json::to_string(&state).expect("serialize"),
            serde_json::to_string(&value).expect("serialize value")
        );
    }

    #[test]
    fn choice_criteria_keep_request_order() {
        let question: SystemOneQuestion = serde_json::from_str(
            r#"{"type":"choice","criteria":{"zeta":"last letter","alpha":"first letter"}}"#,
        )
        .expect("parse");
        let SystemOneQuestion::Choice { criteria, .. } = question else {
            panic!("expected a choice question");
        };
        assert_eq!(
            criteria.iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec!["zeta", "alpha"]
        );
        assert_eq!(
            criteria.to_sorted_values().keys().collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
    }
}
