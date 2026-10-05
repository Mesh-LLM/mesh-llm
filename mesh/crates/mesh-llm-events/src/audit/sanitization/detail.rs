use super::{
    MAX_AUDIT_DETAIL_RAW_BYTES, MAX_AUDIT_METADATA_DEPTH, MAX_AUDIT_METADATA_NODES,
    MAX_AUDIT_TEXT_LEN, is_invite_token_key, is_sensitive_key, sanitize_audit_key,
    sanitize_audit_text,
};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// JSON audit detail that has passed the shared privacy and size policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SanitizedAuditDetailJson(String);

impl SanitizedAuditDetailJson {
    /// Bound raw input, sanitize valid JSON recursively, and replace malformed input completely.
    pub fn sanitize(raw: &str) -> Self {
        if raw.len() > MAX_AUDIT_DETAIL_RAW_BYTES {
            return Self(Value::String("[TRUNCATED]".to_owned()).to_string());
        }
        let Ok(mut value) = serde_json::from_str(raw) else {
            return Self(Value::String("[REDACTED]".to_owned()).to_string());
        };
        sanitize_root(&mut value);
        Self(value.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub(super) fn sanitize_metadata(metadata: BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut object = metadata.into_iter().collect::<Map<_, _>>();
    let mut remaining_nodes = MAX_AUDIT_METADATA_NODES;
    sanitize_object(&mut object, 0, &mut remaining_nodes);
    object.into_iter().collect()
}

fn sanitize_root(value: &mut Value) {
    let mut remaining_nodes = MAX_AUDIT_METADATA_NODES;
    sanitize_json_value(None, value, 0, &mut remaining_nodes);
}

fn sanitize_json_value(
    key: Option<&str>,
    value: &mut Value,
    depth: usize,
    remaining_nodes: &mut usize,
) {
    if let Value::String(token) = value
        && key.is_some_and(is_invite_token_key)
    {
        *token = sanitize_audit_text(token, false);
        return;
    }
    if key.is_some_and(is_sensitive_key) {
        *value = Value::String("[REDACTED]".to_owned());
        return;
    }
    if depth >= MAX_AUDIT_METADATA_DEPTH && matches!(value, Value::Object(_) | Value::Array(_)) {
        *value = Value::String("[TRUNCATED]".to_owned());
        return;
    }
    match value {
        Value::Object(object) => sanitize_object(object, depth, remaining_nodes),
        Value::Array(values) => sanitize_array(values, depth, remaining_nodes),
        Value::String(string) => *string = sanitize_audit_text(string, true),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn sanitize_object(object: &mut Map<String, Value>, depth: usize, remaining_nodes: &mut usize) {
    let mut original = std::mem::take(object).into_iter().collect::<Vec<_>>();
    original.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    let mut entries = Vec::new();
    for (original_key, mut nested_value) in original {
        if *remaining_nodes == 0 {
            break;
        }
        *remaining_nodes -= 1;
        sanitize_json_value(
            Some(&original_key),
            &mut nested_value,
            depth + 1,
            remaining_nodes,
        );
        let canonical_key = sanitize_audit_key(&original_key);
        entries.push((original_key, canonical_key, nested_value));
    }
    insert_collision_safe(object, entries);
}

fn sanitize_array(values: &mut Vec<Value>, depth: usize, remaining_nodes: &mut usize) {
    let original = std::mem::take(values);
    for mut nested_value in original {
        if *remaining_nodes == 0 {
            break;
        }
        *remaining_nodes -= 1;
        sanitize_json_value(None, &mut nested_value, depth + 1, remaining_nodes);
        values.push(nested_value);
    }
}

fn insert_collision_safe(object: &mut Map<String, Value>, entries: Vec<(String, String, Value)>) {
    let reserved_keys = entries
        .iter()
        .filter(|(original, canonical, _)| original == canonical)
        .map(|(_, canonical, _)| canonical.clone())
        .collect::<BTreeSet<_>>();
    let mut used_keys = BTreeSet::new();
    let mut collision_indices = BTreeMap::new();
    for (original_key, canonical_key, value) in entries {
        let owns_canonical = original_key == canonical_key;
        let final_key = if (owns_canonical || !reserved_keys.contains(&canonical_key))
            && used_keys.insert(canonical_key.clone())
        {
            canonical_key
        } else {
            next_collision_key(
                &canonical_key,
                &reserved_keys,
                &mut used_keys,
                &mut collision_indices,
            )
        };
        object.insert(final_key, value);
    }
}

fn next_collision_key(
    canonical_key: &str,
    reserved_keys: &BTreeSet<String>,
    used_keys: &mut BTreeSet<String>,
    collision_indices: &mut BTreeMap<String, usize>,
) -> String {
    let next_index = collision_indices
        .entry(canonical_key.to_owned())
        .or_insert(2);
    loop {
        let suffix = format!("#{next_index}");
        let prefix_len = MAX_AUDIT_TEXT_LEN.saturating_sub(suffix.chars().count());
        let prefix = canonical_key.chars().take(prefix_len).collect::<String>();
        let candidate = format!("{prefix}{suffix}");
        *next_index += 1;
        if !reserved_keys.contains(&candidate) && used_keys.insert(candidate.clone()) {
            return candidate;
        }
    }
}

#[cfg(test)]
mod tests;
