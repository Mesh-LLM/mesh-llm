use std::collections::{BTreeMap, HashMap};

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
        instructions: Option<SystemOneJson>,
        #[serde(default)]
        criteria: Option<SystemOneNoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Option<SystemOneJson>,
        /// Options in request order; see [`SystemOneJson`].
        criteria: SystemOneJsonObject,
    },
    Score {
        #[serde(default)]
        instructions: Option<SystemOneJson>,
        criteria: Vec<SystemOneJson>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SystemOneNoulCriteria {
    #[serde(default)]
    pub r#true: Option<SystemOneJson>,
    #[serde(default)]
    pub r#false: Option<SystemOneJson>,
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

    /// Builds an object from entries in document order. A repeated key keeps
    /// its first position and its last value, as a Python `dict` would.
    pub(crate) fn from_entries(entries: impl IntoIterator<Item = (String, SystemOneJson)>) -> Self {
        let mut ordered: Vec<(String, SystemOneJson)> = Vec::new();
        let mut positions: HashMap<String, usize> = HashMap::new();
        for (key, value) in entries {
            match positions.get(&key) {
                Some(&position) => ordered[position].1 = value,
                None => {
                    positions.insert(key.clone(), ordered.len());
                    ordered.push((key, value));
                }
            }
        }
        Self(ordered)
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

/// Serializes the canonical form: object keys sorted, whatever order the
/// request used and whether or not `serde_json` preserves insertion order.
/// System One request digests are taken over this form.
impl Serialize for SystemOneJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => value.serialize(serializer),
            Self::String(value) => serializer.serialize_str(value),
            Self::Array(values) => values.serialize(serializer),
            Self::Object(object) => object.serialize(serializer),
        }
    }
}

impl Serialize for SystemOneJsonObject {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut entries = self.0.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut map = serializer.serialize_map(Some(entries.len()))?;
        for (key, value) in entries {
            map.serialize_entry(key, value)?;
        }
        map.end()
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
        let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
        while let Some(entry) = map.next_entry::<String, SystemOneJson>()? {
            entries.push(entry);
        }
        Ok(SystemOneJson::Object(SystemOneJsonObject::from_entries(
            entries,
        )))
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
    fn canonical_serialization_matches_main_byte_for_byte() {
        // System One request digests are taken over this serialization.
        // Expected bytes captured from origin/main (7d571a91d); object keys
        // are sorted explicitly, not by serde_json's default map, and struct
        // fields keep their declaration order.
        let request: SystemOneRequest = serde_json::from_str(
            r#"{"model":"m","state":{"zeta":{"b":[1,2.5,"x"],"a":null},"alpha":"hi"},"questions":{"team":{"type":"choice","instructions":{"y":1,"x":2},"criteria":{"support":"faults","billing":{"z":true,"a":false}}},"billing":{"type":"noul","instructions":"Billing?","criteria":{"true":"a charge","false":{"k":2,"j":1}}},"urgency":{"type":"score","criteria":["low",{"m":1,"l":0},"high"]}}}"#,
        )
        .expect("request");
        assert_eq!(
            serde_json::to_string(&(&request.state, &request.questions)).unwrap(),
            r#"[{"alpha":"hi","zeta":{"a":null,"b":[1,2.5,"x"]}},{"billing":{"type":"noul","instructions":"Billing?","criteria":{"true":"a charge","false":{"j":1,"k":2}}},"team":{"type":"choice","instructions":{"x":2,"y":1},"criteria":{"billing":{"a":false,"z":true},"support":"faults"}},"urgency":{"type":"score","instructions":null,"criteria":["low",{"l":0,"m":1},"high"]}}]"#
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
