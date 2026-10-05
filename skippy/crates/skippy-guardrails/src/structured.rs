use serde_json::{Map, Value};

/// Supported subset for validated structured-output emulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredOutputSpec {
    JsonObject,
    JsonSchema { schema: Value },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedStructuredSchema;

impl StructuredOutputSpec {
    pub fn from_response_format_object(
        object: &Map<String, Value>,
    ) -> Result<Self, UnsupportedStructuredSchema> {
        match object.get("type").and_then(Value::as_str) {
            Some("json_object") => Ok(Self::JsonObject),
            Some("json_schema") => {
                let schema = object
                    .get("json_schema")
                    .and_then(Value::as_object)
                    .and_then(|json_schema| json_schema.get("schema"))
                    .cloned()
                    .ok_or(UnsupportedStructuredSchema)?;
                validate_supported_schema(&schema)?;
                Ok(Self::JsonSchema { schema })
            }
            _ => Err(UnsupportedStructuredSchema),
        }
    }

    pub fn tool_parameters(&self) -> Value {
        match self {
            Self::JsonObject => serde_json::json!({
                "type": "object",
                "additionalProperties": true
            }),
            Self::JsonSchema { schema } => schema.clone(),
        }
    }

    pub fn validate_payload(&self, payload: &Value) -> Result<(), UnsupportedStructuredSchema> {
        match self {
            Self::JsonObject => payload
                .as_object()
                .map(|_| ())
                .ok_or(UnsupportedStructuredSchema),
            Self::JsonSchema { schema } => validate_payload_against_schema(schema, payload),
        }
    }
}

fn validate_supported_schema(schema: &Value) -> Result<(), UnsupportedStructuredSchema> {
    let definitions = schema.get("$defs").and_then(Value::as_object);
    validate_supported_schema_inner(schema, definitions, 0)
}

fn validate_supported_schema_inner(
    schema: &Value,
    definitions: Option<&Map<String, Value>>,
    depth: usize,
) -> Result<(), UnsupportedStructuredSchema> {
    if depth > 32 {
        return Err(UnsupportedStructuredSchema);
    }
    let object = schema.as_object().ok_or(UnsupportedStructuredSchema)?;
    if let Some(resolved) = resolve_schema_ref(object, definitions) {
        return validate_supported_schema_inner(resolved?, definitions, depth + 1);
    }
    let schema_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(UnsupportedStructuredSchema)?;

    reject_unsupported_keywords(object)?;

    match schema_type {
        "object" => validate_object_schema(object, definitions, depth),
        "array" => validate_array_schema(object, definitions, depth),
        "string" | "number" | "integer" | "boolean" | "null" => validate_scalar_schema(object),
        _ => Err(UnsupportedStructuredSchema),
    }
}

fn reject_unsupported_keywords(
    object: &Map<String, Value>,
) -> Result<(), UnsupportedStructuredSchema> {
    const UNSUPPORTED_KEYS: &[&str] = &[
        "allOf",
        "anyOf",
        "const",
        "enum",
        "format",
        "maximum",
        "maxItems",
        "minimum",
        "minItems",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
    ];
    if UNSUPPORTED_KEYS.iter().any(|key| object.contains_key(*key)) {
        Err(UnsupportedStructuredSchema)
    } else {
        Ok(())
    }
}

fn validate_object_schema(
    object: &Map<String, Value>,
    definitions: Option<&Map<String, Value>>,
    depth: usize,
) -> Result<(), UnsupportedStructuredSchema> {
    validate_definitions(object, definitions, depth)?;
    let properties = match object.get("properties") {
        Some(Value::Object(properties)) => Some(properties),
        Some(_) => return Err(UnsupportedStructuredSchema),
        None => None,
    };
    if let Some(required) = object.get("required") {
        let required_entries = required.as_array().ok_or(UnsupportedStructuredSchema)?;
        for entry in required_entries {
            let name = entry.as_str().ok_or(UnsupportedStructuredSchema)?;
            if !properties.is_some_and(|properties| properties.contains_key(name)) {
                return Err(UnsupportedStructuredSchema);
            }
        }
    }
    if let Some(additional_properties) = object.get("additionalProperties")
        && !additional_properties.is_boolean()
    {
        return Err(UnsupportedStructuredSchema);
    }
    if let Some(properties) = properties {
        for schema in properties.values() {
            validate_supported_schema_inner(schema, definitions, depth + 1)?;
        }
    }
    Ok(())
}

fn validate_array_schema(
    object: &Map<String, Value>,
    definitions: Option<&Map<String, Value>>,
    depth: usize,
) -> Result<(), UnsupportedStructuredSchema> {
    validate_definitions(object, definitions, depth)?;
    let items = object.get("items").ok_or(UnsupportedStructuredSchema)?;
    if items.is_array() {
        return Err(UnsupportedStructuredSchema);
    }
    validate_supported_schema_inner(items, definitions, depth + 1)
}

fn validate_scalar_schema(object: &Map<String, Value>) -> Result<(), UnsupportedStructuredSchema> {
    let allowed = ["type", "description", "title"];
    if object.keys().all(|key| allowed.contains(&key.as_str())) {
        Ok(())
    } else {
        Err(UnsupportedStructuredSchema)
    }
}

fn validate_payload_against_schema(
    schema: &Value,
    payload: &Value,
) -> Result<(), UnsupportedStructuredSchema> {
    let definitions = schema.get("$defs").and_then(Value::as_object);
    validate_payload_against_schema_inner(schema, payload, definitions, 0)
}

fn validate_payload_against_schema_inner(
    schema: &Value,
    payload: &Value,
    definitions: Option<&Map<String, Value>>,
    depth: usize,
) -> Result<(), UnsupportedStructuredSchema> {
    if depth > 32 {
        return Err(UnsupportedStructuredSchema);
    }
    let object = schema.as_object().ok_or(UnsupportedStructuredSchema)?;
    if let Some(resolved) = resolve_schema_ref(object, definitions) {
        return validate_payload_against_schema_inner(resolved?, payload, definitions, depth + 1);
    }
    match object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(UnsupportedStructuredSchema)?
    {
        "object" => validate_object_payload(object, payload, definitions, depth),
        "array" => validate_array_payload(object, payload, definitions, depth),
        "string" => payload
            .as_str()
            .map(|_| ())
            .ok_or(UnsupportedStructuredSchema),
        "number" => payload
            .as_f64()
            .map(|_| ())
            .ok_or(UnsupportedStructuredSchema),
        "integer" => payload
            .as_i64()
            .or_else(|| payload.as_u64().and_then(|value| i64::try_from(value).ok()))
            .map(|_| ())
            .ok_or(UnsupportedStructuredSchema),
        "boolean" => payload
            .as_bool()
            .map(|_| ())
            .ok_or(UnsupportedStructuredSchema),
        "null" => payload
            .is_null()
            .then_some(())
            .ok_or(UnsupportedStructuredSchema),
        _ => Err(UnsupportedStructuredSchema),
    }
}

fn validate_object_payload(
    schema: &Map<String, Value>,
    payload: &Value,
    definitions: Option<&Map<String, Value>>,
    depth: usize,
) -> Result<(), UnsupportedStructuredSchema> {
    let payload = payload.as_object().ok_or(UnsupportedStructuredSchema)?;
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required_key in required {
        let key = required_key.as_str().ok_or(UnsupportedStructuredSchema)?;
        if !payload.contains_key(key) {
            return Err(UnsupportedStructuredSchema);
        }
    }
    let allow_additional = schema
        .get("additionalProperties")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    for (key, value) in payload {
        if let Some(property_schema) = properties.get(key) {
            validate_payload_against_schema_inner(property_schema, value, definitions, depth + 1)?;
        } else if !allow_additional {
            return Err(UnsupportedStructuredSchema);
        }
    }
    Ok(())
}

fn validate_array_payload(
    schema: &Map<String, Value>,
    payload: &Value,
    definitions: Option<&Map<String, Value>>,
    depth: usize,
) -> Result<(), UnsupportedStructuredSchema> {
    let payload = payload.as_array().ok_or(UnsupportedStructuredSchema)?;
    let item_schema = schema.get("items").ok_or(UnsupportedStructuredSchema)?;
    for item in payload {
        validate_payload_against_schema_inner(item_schema, item, definitions, depth + 1)?;
    }
    Ok(())
}

fn validate_definitions(
    object: &Map<String, Value>,
    definitions: Option<&Map<String, Value>>,
    depth: usize,
) -> Result<(), UnsupportedStructuredSchema> {
    let Some(Value::Object(local_definitions)) = object.get("$defs") else {
        return Ok(());
    };
    for definition in local_definitions.values() {
        validate_supported_schema_inner(definition, definitions, depth + 1)?;
    }
    Ok(())
}

fn resolve_schema_ref<'a>(
    object: &'a Map<String, Value>,
    definitions: Option<&'a Map<String, Value>>,
) -> Option<Result<&'a Value, UnsupportedStructuredSchema>> {
    let reference = object.get("$ref")?;
    let Some(reference) = reference.as_str() else {
        return Some(Err(UnsupportedStructuredSchema));
    };
    if object
        .keys()
        .any(|key| key != "$ref" && key != "description" && key != "title")
    {
        return Some(Err(UnsupportedStructuredSchema));
    }
    let Some(name) = reference.strip_prefix("#/$defs/") else {
        return Some(Err(UnsupportedStructuredSchema));
    };
    let name = decode_json_pointer_segment(name);
    Some(name.and_then(|name| {
        definitions
            .and_then(|definitions| definitions.get(&name))
            .ok_or(UnsupportedStructuredSchema)
    }))
}

fn decode_json_pointer_segment(segment: &str) -> Result<String, UnsupportedStructuredSchema> {
    let mut decoded = String::with_capacity(segment.len());
    let mut chars = segment.chars();
    while let Some(ch) = chars.next() {
        if ch != '~' {
            decoded.push(ch);
            continue;
        }
        match chars.next() {
            Some('0') => decoded.push('~'),
            Some('1') => decoded.push('/'),
            _ => return Err(UnsupportedStructuredSchema),
        }
    }
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::StructuredOutputSpec;

    #[test]
    fn json_schema_supports_local_defs_refs() {
        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "CommandBatchResponse",
                "strict": true,
                "schema": {
                    "$defs": {
                        "Command": {
                            "type": "object",
                            "properties": {
                                "keystrokes": {"type": "string"},
                                "is_blocking": {"type": "boolean"},
                                "timeout_sec": {"type": "number"}
                            },
                            "required": ["keystrokes", "is_blocking", "timeout_sec"],
                            "additionalProperties": false
                        }
                    },
                    "type": "object",
                    "properties": {
                        "state_analysis": {"type": "string"},
                        "commands": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Command"}
                        },
                        "is_task_complete": {"type": "boolean"}
                    },
                    "required": ["state_analysis", "commands", "is_task_complete"],
                    "additionalProperties": false
                }
            }
        });
        let spec =
            StructuredOutputSpec::from_response_format_object(response_format.as_object().unwrap())
                .expect("local refs should be supported");

        spec.validate_payload(&json!({
            "state_analysis": "ready",
            "commands": [{
                "keystrokes": "ls\n",
                "is_blocking": true,
                "timeout_sec": 1
            }],
            "is_task_complete": false
        }))
        .expect("payload should validate through ref");

        assert!(
            spec.validate_payload(&json!({
                "state_analysis": "ready",
                "commands": [{"keystrokes": "ls\n"}],
                "is_task_complete": false
            }))
            .is_err()
        );
    }
}
