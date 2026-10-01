use super::document::Json;

pub(super) fn equal(left: &Json, right: &Json) -> bool {
    match (left, right) {
        (Json::Null, Json::Null) => true,
        (Json::Bool(left), Json::Bool(right)) => left == right,
        (Json::Integer(left), Json::Integer(right)) => left == right,
        (Json::Float(left), Json::Float(right)) => left == right,
        (Json::String(left), Json::String(right)) => left == right,
        (Json::Array(left), Json::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| equal(left, right))
        }
        (Json::Object(entries), Json::Object(other)) => {
            entries.len() == other.len()
                && entries
                    .iter()
                    .all(|(key, value)| right.get_key(key).is_some_and(|other| equal(value, other)))
        }
        _ => false,
    }
}
