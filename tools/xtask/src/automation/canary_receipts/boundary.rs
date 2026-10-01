use serde::{
    Deserialize, Deserializer,
    de::{DeserializeOwned, Error as _, MapAccess, Visitor, value::MapDeserializer},
};
use std::fmt;

type OrderedFields = Vec<(String, serde_json::Value)>;
type OrderedDeserializer<'de> =
    MapDeserializer<'de, std::vec::IntoIter<(String, serde_json::Value)>, serde_json::Error>;

pub(super) fn ordered_object_last_wins<'de, D: Deserializer<'de>, T>(
    deserializer: D,
    decode: impl FnOnce(OrderedDeserializer<'de>) -> Result<T, serde_json::Error>,
) -> Result<T, D::Error> {
    let fields = deserializer.deserialize_map(OrderedObject)?;
    decode(MapDeserializer::new(fields.into_iter())).map_err(D::Error::custom)
}

struct OrderedObject;

impl<'de> Visitor<'de> for OrderedObject {
    type Value = OrderedFields;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut fields: OrderedFields = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
            match fields.iter_mut().find(|(name, _)| *name == key) {
                Some(field) => field.1 = value,
                None => fields.push((key, value)),
            }
        }
        Ok(fields)
    }
}

pub(super) fn object_last_wins<'de, D: Deserializer<'de>, T: DeserializeOwned>(
    deserializer: D,
) -> Result<T, D::Error> {
    let fields = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
    serde_json::from_value(serde_json::Value::Object(fields)).map_err(D::Error::custom)
}

pub(super) fn object_rows_last_wins<'de, D: Deserializer<'de>, T: DeserializeOwned>(
    deserializer: D,
) -> Result<Vec<T>, D::Error> {
    let rows = Vec::<serde_json::Map<String, serde_json::Value>>::deserialize(deserializer)?;
    rows.into_iter()
        .map(|fields| {
            serde_json::from_value(serde_json::Value::Object(fields)).map_err(D::Error::custom)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Truth(pub(super) bool);

impl<'de> Deserialize<'de> for Truth {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Ok(Self(match value {
            serde_json::Value::Null => false,
            serde_json::Value::Bool(flag) => flag,
            serde_json::Value::Number(number) => number.as_f64() != Some(0.0),
            serde_json::Value::String(text) => !text.is_empty(),
            serde_json::Value::Array(items) => !items.is_empty(),
            serde_json::Value::Object(items) => !items.is_empty(),
        }))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ExitSuccess(pub(super) bool);

impl<'de> Deserialize<'de> for ExitSuccess {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Ok(Self(match value {
            serde_json::Value::Bool(flag) => !flag,
            serde_json::Value::Number(number) => number.as_f64() == Some(0.0),
            serde_json::Value::Null
            | serde_json::Value::String(_)
            | serde_json::Value::Array(_)
            | serde_json::Value::Object(_) => false,
        }))
    }
}

#[derive(Debug, Deserialize)]
#[serde(from = "String")]
pub(super) enum ModelClass {
    Causal,
    Workload(String),
}

impl From<String> for ModelClass {
    fn from(value: String) -> Self {
        match value.as_str() {
            "causal_generation" => Self::Causal,
            _ => Self::Workload(value),
        }
    }
}
