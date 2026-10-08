use crate::ci_plan::document::Json;
use std::path::Path;

pub(super) fn load(path: &Path, shown: &str) -> Result<Json, String> {
    let raw = std::fs::read(path).map_err(|error| format!("{shown}: {error}"))?;
    Json::parse(&raw).map_err(|error| format!("{shown}: {error}"))
}
