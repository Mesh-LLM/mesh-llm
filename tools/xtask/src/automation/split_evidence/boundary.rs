use super::{
    Error,
    types::{Identity, Integer, Observer, Text},
};
use crate::automation::codepoint_json::{strings::JsonString, value::Value};

pub(super) fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Value, Error> {
    match value {
        Value::Object(_) => Ok(value),
        _ => Err(Error::Contract(format!("{label} must be a JSON object"))),
    }
}

pub(super) fn array<'a>(value: Option<&'a Value>, label: &str) -> Result<&'a [Value], Error> {
    match value {
        Some(Value::Array(items)) => Ok(items),
        _ => Err(Error::Contract(format!("{label} must be a JSON array"))),
    }
}

pub(super) fn text(value: Option<&Value>, label: &str) -> Result<Text, Error> {
    match value {
        Some(Value::Str(text)) if text.codepoints().next().is_some() => Ok(Text(text.clone())),
        _ => Err(Error::Contract(format!(
            "{label} must be a non-empty string"
        ))),
    }
}

pub(super) fn digest(value: Option<&Value>, label: &str) -> Result<Text, Error> {
    let text = text(value, label)?;
    let ascii = text
        .0
        .codepoints()
        .map(char::from_u32)
        .collect::<Option<String>>();
    match ascii {
        Some(ascii) if ascii.len() == 64 && ascii.bytes().all(|byte| byte.is_ascii_hexdigit()) => {
            Ok(Text(JsonString::from(ascii.to_ascii_lowercase().as_str())))
        }
        _ => Err(Error::Contract(format!(
            "{label} must be a 64-character SHA-256"
        ))),
    }
}

pub(super) fn identity(value: &Value, label: &str) -> Result<Identity, Error> {
    Ok(Identity {
        topology_id: text(value.get("topology_id"), &format!("{label}.topology_id"))?,
        run_id: text(value.get("run_id"), &format!("{label}.run_id"))?,
        model_id: text(value.get("model_id"), &format!("{label}.model_id"))?,
        package_ref: text(value.get("package_ref"), &format!("{label}.package_ref"))?,
        manifest_sha256: digest(
            value.get("manifest_sha256"),
            &format!("{label}.manifest_sha256"),
        )?,
    })
}

pub(super) fn observer_identity(value: &Value, label: &str) -> Result<(Text, Text), Error> {
    Ok((
        text(value.get("node_id"), &format!("{label}.node_id"))?,
        text(value.get("mesh_id"), &format!("{label}.mesh_id"))?,
    ))
}

pub(super) fn observer(
    value: &Value,
    identity: (Text, Text),
    label: &str,
) -> Result<Observer, Error> {
    let peers = array(value.get("peers"), &format!("{label}.peers"))?;
    let peers = peers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let label = format!("{label}.peers[{index}]");
            let value = object(value, &label)?;
            text(value.get("id"), &format!("{label}.id"))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let count = peers.len();
    let [peer_node_id]: [Text; 1] = peers.try_into().map_err(|_| {
        Error::Contract(format!(
            "{label}.peers must contain exactly one peer, got {count}"
        ))
    })?;
    Ok(Observer {
        node_id: identity.0,
        mesh_id: identity.1,
        peer_node_id,
    })
}

pub(super) fn model(value: &Value, label: &str) -> Result<Text, Error> {
    let mut models = Vec::new();
    for (index, value) in array(value.get("data"), &format!("{label}.data"))?
        .iter()
        .enumerate()
    {
        let label = format!("{label}.data[{index}]");
        let value = object(value, &label)?;
        let id = text(value.get("id"), &format!("{label}.id"))?;
        if id.0 != *"mesh" {
            models.push(id);
        }
    }
    let count = models.len();
    let [model]: [Text; 1] = models.try_into().map_err(|_| {
        Error::Contract(format!(
            "{label}.data must contain exactly one concrete model, got {count}"
        ))
    })?;
    Ok(model)
}

pub(super) fn integer(value: &Value, field: &str, label: &str) -> Result<Integer, Error> {
    Integer::parse(value.get(field), &format!("{label}.{field}"))
}
