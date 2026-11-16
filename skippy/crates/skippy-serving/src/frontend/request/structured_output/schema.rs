//! Conservative schema shapes whose assertions the packaged GBNF converter enforces.
use super::*;

pub(super) fn validate(schema: &Value) -> InferenceResult<()> {
    visit(schema, "#", 0, &mut 4096)
}

fn invalid(path: &str, message: &str) -> InferenceError {
    InferenceError::invalid_request(format!("JSON schema at {path}: {message}"))
        .with_param("response_format")
}

fn visit(schema: &Value, path: &str, depth: usize, budget: &mut usize) -> InferenceResult<()> {
    if depth > 32 || *budget == 0 {
        return Err(invalid(path, "schema exceeds depth or node limit"));
    }
    *budget -= 1;
    let object = schema
        .as_object()
        .ok_or_else(|| invalid(path, "expected a schema object"))?;
    if object
        .iter()
        .all(|(key, value)| matches!(key.as_str(), "title" | "description") && value.is_string())
    {
        return Ok(());
    }
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(path, "an explicit single type is required"))?;
    let assertions: &[&str] = match kind {
        "object" => &["properties", "required", "additionalProperties"],
        "array" => &["items"],
        "string" | "number" | "integer" | "boolean" | "null" => &["const", "enum"],
        _ => return Err(invalid(path, "unsupported schema type")),
    };
    for (key, value) in object {
        match key.as_str() {
            "type" => {}
            "title" | "description" if value.is_string() => {}
            _ if assertions.contains(&key.as_str()) => {}
            _ => {
                return Err(invalid(
                    path,
                    &format!("unsupported or malformed keyword {key:?}"),
                ));
            }
        }
    }
    match kind {
        "object" => object_schema(object, path, depth, budget),
        "array" => visit(
            object
                .get("items")
                .ok_or_else(|| invalid(path, "items is required"))?,
            &format!("{path}/items"),
            depth + 1,
            budget,
        ),
        _ => scalar_schema(object, kind, path, budget),
    }
}

fn object_schema(
    object: &serde_json::Map<String, Value>,
    path: &str,
    depth: usize,
    budget: &mut usize,
) -> InferenceResult<()> {
    if object.get("additionalProperties") != Some(&Value::Bool(false)) {
        return Err(invalid(
            path,
            "only closed objects (additionalProperties:false) are supported",
        ));
    }
    let properties = object
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(path, "properties must be an object"))?;
    if let Some(required) = object.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| invalid(path, "required must be an array"))?;
        let mut names = std::collections::HashSet::new();
        for name in required {
            let name = name
                .as_str()
                .ok_or_else(|| invalid(path, "required entries must be strings"))?;
            if !properties.contains_key(name) || !names.insert(name) {
                return Err(invalid(
                    path,
                    "required names must be unique declared properties",
                ));
            }
        }
    }
    for (name, child) in properties {
        let escaped = name.replace('~', "~0").replace('/', "~1");
        visit(
            child,
            &format!("{path}/properties/{escaped}"),
            depth + 1,
            budget,
        )?;
    }
    Ok(())
}

fn scalar_schema(
    object: &serde_json::Map<String, Value>,
    kind: &str,
    path: &str,
    budget: &mut usize,
) -> InferenceResult<()> {
    if object.contains_key("const") && object.contains_key("enum") {
        return Err(invalid(path, "const and enum cannot be combined"));
    }
    if let Some(value) = object.get("const") {
        literal(value, kind, path)?;
    }
    if let Some(values) = object.get("enum") {
        let values = values
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| invalid(path, "enum must be a nonempty array"))?;
        if values.len() > *budget {
            return Err(invalid(path, "enum exceeds schema node limit"));
        }
        *budget -= values.len();
        for value in values {
            literal(value, kind, path)?;
        }
    }
    Ok(())
}

fn literal(value: &Value, kind: &str, path: &str) -> InferenceResult<()> {
    let matches = match kind {
        "string" => value.as_str().is_some_and(|s| s.len() <= 8192),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    };
    if !matches {
        return Err(invalid(
            path,
            "literal must match its declared type and size limit",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn admits_closed_objects_arrays_and_typed_scalar_literals() {
        validate(&json!({"type":"object","additionalProperties":false,
            "required":["answer"], "properties":{"answer":{"type":"integer","const":7},
                "tags":{"type":"array","items":{"type":"string","enum":["a","b"]}}}}))
        .unwrap();
    }

    #[test]
    fn admits_nested_unconstrained_and_annotated_schemas() {
        for child in [
            json!({}),
            json!({"title":"Any JSON","description":"Any value"}),
        ] {
            validate(&json!({"type":"array","items":child})).unwrap();
            validate(&json!({"type":"object","additionalProperties":false,
                "properties":{"value":child},"required":["value"]}))
            .unwrap();
        }
    }

    #[test]
    fn bounds_depth_nodes_enums_and_literals() {
        let mut schema = json!({"type":"null"});
        for _ in 0..32 {
            schema = json!({"type":"array","items":schema});
        }
        validate(&schema).unwrap();
        assert!(validate(&json!({"type":"array","items":schema})).is_err());
        validate(&json!({"type":"string","const":"x".repeat(8192)})).unwrap();
        assert!(validate(&json!({"type":"string","const":"x".repeat(8193)})).is_err());
        validate(&json!({"type":"integer","enum":vec![0;4095]})).unwrap();
        assert!(validate(&json!({"type":"integer","enum":vec![0;4096]})).is_err());
        assert!(
            validate(&json!({"type":"array","items":{
                "type":"integer","enum":vec![0;4095]
            }}))
            .is_err()
        );
        assert!(
            validate(&json!({"type":"array","items":{
                "type":"integer","minimum":100
            }}))
            .is_err()
        );
    }

    #[test]
    fn rejects_malformed_and_unenforced_assertions() {
        for schema in [
            json!({"type":42}),
            json!({"type":"not-a-type"}),
            json!({"type":"number","minimum":100}),
            json!({"type":"integer","const":"wrong"}),
            json!({"type":"string","pattern":"^a$"}),
            json!({"type":"integer","enum":[]}),
            json!({"type":"integer","const":1,"enum":[2]}),
            json!({"type":"array","items":[{"type":"integer"}]}),
            json!({"type":"object","properties":[],"additionalProperties":false}),
            json!({"type":"object","properties":{},"required":["x"],"additionalProperties":false}),
            json!({"type":"object","properties":{},"additionalProperties":true}),
            json!({"oneOf":[{"type":"integer"},{"type":"number"}]}),
        ] {
            assert!(validate(&schema).is_err(), "{schema}");
        }
    }
}
