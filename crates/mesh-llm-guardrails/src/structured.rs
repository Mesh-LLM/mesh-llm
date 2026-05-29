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
    let object = schema.as_object().ok_or(UnsupportedStructuredSchema)?;
    let schema_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(UnsupportedStructuredSchema)?;

    reject_unsupported_keywords(object)?;

    match schema_type {
        "object" => validate_object_schema(object),
        "array" => validate_array_schema(object),
        "string" | "number" | "integer" | "boolean" | "null" => validate_scalar_schema(object),
        _ => Err(UnsupportedStructuredSchema),
    }
}

fn reject_unsupported_keywords(
    object: &Map<String, Value>,
) -> Result<(), UnsupportedStructuredSchema> {
    const UNSUPPORTED_KEYS: &[&str] = &[
        "$ref",
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

fn validate_object_schema(object: &Map<String, Value>) -> Result<(), UnsupportedStructuredSchema> {
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
            validate_supported_schema(schema)?;
        }
    }
    Ok(())
}

fn validate_array_schema(object: &Map<String, Value>) -> Result<(), UnsupportedStructuredSchema> {
    let items = object.get("items").ok_or(UnsupportedStructuredSchema)?;
    if items.is_array() {
        return Err(UnsupportedStructuredSchema);
    }
    validate_supported_schema(items)
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
    let object = schema.as_object().ok_or(UnsupportedStructuredSchema)?;
    match object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(UnsupportedStructuredSchema)?
    {
        "object" => validate_object_payload(object, payload),
        "array" => validate_array_payload(object, payload),
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
            validate_payload_against_schema(property_schema, value)?;
        } else if !allow_additional {
            return Err(UnsupportedStructuredSchema);
        }
    }
    Ok(())
}

fn validate_array_payload(
    schema: &Map<String, Value>,
    payload: &Value,
) -> Result<(), UnsupportedStructuredSchema> {
    let payload = payload.as_array().ok_or(UnsupportedStructuredSchema)?;
    let item_schema = schema.get("items").ok_or(UnsupportedStructuredSchema)?;
    for item in payload {
        validate_payload_against_schema(item_schema, item)?;
    }
    Ok(())
}
