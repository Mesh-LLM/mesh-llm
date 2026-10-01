use crate::automation::codepoint_json::value::Value;
use num_bigint::{BigInt, ToBigInt};

pub(super) fn equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        (Value::Float(left), Value::Float(right)) => left == right,
        (
            Value::Bool(_) | Value::Int(_) | Value::BigInt(_) | Value::Float(_),
            Value::Bool(_) | Value::Int(_) | Value::BigInt(_) | Value::Float(_),
        ) => integral(left)
            .zip(integral(right))
            .is_some_and(|(left, right)| left == right),
        (Value::Str(left), Value::Str(right)) => left == right,
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| equal(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right
                        .iter()
                        .find(|(other, _)| other == key)
                        .is_some_and(|(_, other)| equal(value, other))
                })
        }
        (
            Value::Null
            | Value::Bool(_)
            | Value::Int(_)
            | Value::BigInt(_)
            | Value::Float(_)
            | Value::Str(_)
            | Value::Array(_)
            | Value::Object(_),
            Value::Null
            | Value::Bool(_)
            | Value::Int(_)
            | Value::BigInt(_)
            | Value::Float(_)
            | Value::Str(_)
            | Value::Array(_)
            | Value::Object(_),
        ) => false,
    }
}

fn integral(value: &Value) -> Option<BigInt> {
    match value {
        Value::Bool(flag) => Some(BigInt::from(u8::from(*flag))),
        Value::Int(integer) => Some(BigInt::from(*integer)),
        Value::BigInt(decimal) => BigInt::parse_bytes(decimal.as_bytes(), 10),
        Value::Float(number) if number.is_finite() && number.fract() == 0.0 => number.to_bigint(),
        Value::Null | Value::Float(_) | Value::Str(_) | Value::Array(_) | Value::Object(_) => None,
    }
}
