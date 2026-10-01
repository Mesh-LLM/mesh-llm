use super::strings::JsonString;

#[derive(Clone)]
pub(in crate::automation) enum Value {
    Null,
    Bool(bool),
    Int(i128),
    BigInt(String),
    Float(f64),
    Str(JsonString),
    Array(Vec<Value>),
    Object(Vec<(JsonString, Value)>),
}

impl Value {
    pub(in crate::automation) fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            Self::Null
            | Self::Bool(_)
            | Self::Int(_)
            | Self::BigInt(_)
            | Self::Float(_)
            | Self::Str(_)
            | Self::Array(_) => None,
        }
    }
}
