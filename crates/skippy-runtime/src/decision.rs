//! Typed System One decisions over any model that can make them.
//!
//! A decision request is Jev-shaped: shared `state` plus named `noul`,
//! `choice`, and `score` questions. Each backend turns that into its own native
//! input — a DiffusionGemma answer canvas, a Laya encoder sequence per
//! question — and returns one probability distribution per question, so a
//! caller never needs to know which kind of model it holds.

use serde_json::{Map, Number, Value};

mod diffusion;
mod laya;

/// A JSON value that keeps the order its object keys were given in.
///
/// Backends differ in how they render structured criteria: the Laya reference
/// uses request order with Python `json.dumps` separators, while the
/// DiffusionGemma read uses compact JSON with sorted keys. Carrying the order
/// lets each render the form it was built for.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionValue {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<DecisionValue>),
    Object(Vec<(String, DecisionValue)>),
}

impl DecisionValue {
    /// The canonical JSON value, with object keys sorted.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(value.clone()),
            Self::String(value) => Value::String(value.clone()),
            Self::Array(values) => Value::Array(values.iter().map(Self::to_json).collect()),
            Self::Object(entries) => Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_json()))
                    .collect::<Map<_, _>>(),
            ),
        }
    }
}

impl From<Value> for DecisionValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => Self::Number(value),
            Value::String(value) => Self::String(value),
            Value::Array(values) => Self::Array(values.into_iter().map(Self::from).collect()),
            Value::Object(map) => Self::Object(
                map.into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            ),
        }
    }
}

impl<'de> serde::Deserialize<'de> for DecisionValue {
    /// Parses JSON keeping object keys in document order. A repeated key keeps
    /// its first position and its last value, as a Python `dict` would.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(DecisionValueVisitor)
    }
}

struct DecisionValueVisitor;

impl<'de> serde::de::Visitor<'de> for DecisionValueVisitor {
    type Value = DecisionValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(DecisionValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(DecisionValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(DecisionValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(DecisionValue::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(DecisionValue::Number(value.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
        Number::from_f64(value)
            .map(DecisionValue::Number)
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(DecisionValue::String(value.to_string()))
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element()? {
            values.push(value);
        }
        Ok(DecisionValue::Array(values))
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entries: Vec<(String, DecisionValue)> = Vec::new();
        let mut positions: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        while let Some((key, value)) = map.next_entry::<String, DecisionValue>()? {
            match positions.get(&key) {
                Some(&position) => entries[position].1 = value,
                None => {
                    positions.insert(key.clone(), entries.len());
                    entries.push((key, value));
                }
            }
        }
        Ok(DecisionValue::Object(entries))
    }
}

/// What a question asks, with its options in request order.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionQuestionKind {
    /// Yes/no. The distribution is `[p(true), p(false)]`.
    Noul {
        when_true: Option<DecisionValue>,
        when_false: Option<DecisionValue>,
    },
    /// One of several named options, as `(name, description)` pairs. The
    /// distribution follows this order.
    Choice {
        options: Vec<(String, DecisionValue)>,
    },
    /// Ordered levels, lowest first. The distribution follows this order.
    Score { levels: Vec<DecisionValue> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionQuestion {
    pub key: String,
    pub instructions: Option<DecisionValue>,
    pub kind: DecisionQuestionKind,
}

impl DecisionQuestion {
    /// How many probabilities a backend returns for this question.
    pub fn option_count(&self) -> usize {
        match &self.kind {
            DecisionQuestionKind::Noul { .. } => 2,
            DecisionQuestionKind::Choice { options } => options.len(),
            DecisionQuestionKind::Score { levels } => levels.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRequest {
    pub state: DecisionValue,
    pub questions: Vec<DecisionQuestion>,
    /// Deterministic per-request seed. Backends that randomize part of their
    /// input derive it from this, so identical requests read identically.
    pub seed: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOutput {
    /// One distribution per question, in request order; see
    /// [`DecisionQuestionKind`] for each kind's option order.
    pub probabilities: Vec<Vec<f32>>,
    /// Tokens the model read to answer, across every question.
    pub input_tokens: usize,
}

#[derive(Debug)]
pub enum DecisionError {
    /// The request cannot be expressed for this model, such as too many
    /// options or labels the tokenizer cannot place.
    InvalidRequest(String),
    /// The model cannot make decisions at all.
    Unsupported(String),
    /// The native read failed.
    Backend(anyhow::Error),
}

impl std::fmt::Display for DecisionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::Unsupported(message) => {
                formatter.write_str(message)
            }
            Self::Backend(error) => write!(formatter, "{error:#}"),
        }
    }
}

impl std::error::Error for DecisionError {}

impl From<anyhow::Error> for DecisionError {
    fn from(error: anyhow::Error) -> Self {
        Self::Backend(error)
    }
}

/// A model that answers typed System One questions.
pub trait DecisionModel {
    fn decide(&self, request: &DecisionRequest) -> Result<DecisionOutput, DecisionError>;
}

/// Checks that a backend returned one well-sized distribution per question.
fn check_output(
    request: &DecisionRequest,
    probabilities: &[Vec<f32>],
) -> Result<(), DecisionError> {
    if probabilities.len() != request.questions.len() {
        return Err(DecisionError::Backend(anyhow::anyhow!(
            "decision output has {} distributions for {} questions",
            probabilities.len(),
            request.questions.len()
        )));
    }
    for (question, distribution) in request.questions.iter().zip(probabilities) {
        if distribution.len() != question.option_count() {
            return Err(DecisionError::Backend(anyhow::anyhow!(
                "decision output for {:?} has {} probabilities for {} options",
                question.key,
                distribution.len(),
                question.option_count()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_json_sorts_keys_but_the_value_keeps_request_order() {
        let value = DecisionValue::Object(vec![
            ("zeta".into(), DecisionValue::Bool(true)),
            ("alpha".into(), DecisionValue::Null),
        ]);
        assert_eq!(
            serde_json::to_string(&value.to_json()).unwrap(),
            r#"{"alpha":null,"zeta":true}"#
        );
        let DecisionValue::Object(entries) = &value else {
            unreachable!()
        };
        assert_eq!(entries[0].0, "zeta");
    }

    #[test]
    fn option_counts_follow_the_kind() {
        let noul = DecisionQuestion {
            key: "q".into(),
            instructions: None,
            kind: DecisionQuestionKind::Noul {
                when_true: None,
                when_false: None,
            },
        };
        assert_eq!(noul.option_count(), 2);
    }
}
