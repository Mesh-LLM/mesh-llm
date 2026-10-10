use crate::artifact::os_error_line;
use crate::ci_plan::document::Json;
use crate::prepared_input::value_format;
use std::path::Path;

pub(super) struct Raised(pub(super) String);

pub(super) fn load_manifest(path: &Path, shown: &str) -> Result<Json, Raised> {
    let raw = std::fs::read(path).map_err(|error| Raised(os_error_line(&error, shown)))?;
    Json::parse(&raw).map_err(|error| Raised(error.to_string()))
}

pub(super) fn subscript<'a>(value: &'a Json, key: &str) -> Result<&'a Json, Raised> {
    let entries = value
        .as_object()
        .ok_or_else(|| Raised("runtime manifest field must be an object".to_owned()))?;
    entries
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, child)| child)
        .ok_or_else(|| Raised(format!("missing runtime manifest field {key}")))
}

pub(super) fn toolkit_text(value: &Json, outer: &str, inner: &str) -> Result<String, Raised> {
    let Some(nested) = value.get(outer) else {
        return Ok(String::new());
    };
    if nested.as_object().is_none() {
        return Err(Raised(format!(
            "runtime manifest {outer} must be an object"
        )));
    }
    Ok(nested
        .get(inner)
        .map(|field| value_format::display(Some(field)))
        .unwrap_or_default())
}
