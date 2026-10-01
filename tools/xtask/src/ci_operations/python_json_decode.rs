use crate::ci_plan::document::Json;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use std::fmt;

pub(crate) type PairsHook = fn(Vec<(String, Json)>) -> Result<Json, String>;

pub(crate) struct Hooks {
    pub(crate) pairs: PairsHook,
    pub(crate) constant: fn(&str) -> Result<Json, String>,
}

pub(crate) enum DecodeError {
    Value(String),
    Recursion,
}

pub(crate) const EXACT_NUMBER: &str = "\u{0}exact-number";

pub(crate) fn loads(raw: &[u8], hooks: &Hooks) -> Result<Json, DecodeError> {
    let _ = hooks.constant;
    let mut decoder = serde_json::Deserializer::from_slice(raw);
    let parsed = Seed(hooks).deserialize(&mut decoder).map_err(|error| {
        if error.to_string().starts_with("recursion limit exceeded") {
            DecodeError::Recursion
        } else {
            DecodeError::Value(error.to_string())
        }
    })?;
    decoder
        .end()
        .map_err(|error| DecodeError::Value(error.to_string()))?;
    Ok(parsed)
}

pub(crate) fn loads_exact(raw: &[u8], hooks: &Hooks) -> Result<Json, DecodeError> {
    loads(raw, hooks)
}

struct Seed<'a>(&'a Hooks);

impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value = Json;
    fn deserialize<D: de::Deserializer<'de>>(self, decoder: D) -> Result<Json, D::Error> {
        decoder.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Seed<'_> {
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
        serde_json::Number::from_f64(value)
            .map(Json::Number)
            .ok_or_else(|| E::custom("nonfinite number"))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Json, E> {
        Ok(Json::String(value.to_owned()))
    }
    fn visit_string<E: de::Error>(self, value: String) -> Result<Json, E> {
        Ok(Json::String(value))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Json, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(Seed(self.0))? {
            values.push(value);
        }
        Ok(Json::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut values = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            values.push((key, map.next_value_seed(Seed(self.0))?));
        }
        (self.0.pairs)(values).map_err(de::Error::custom)
    }
}
