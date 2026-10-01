use super::input::Request;
use super::{Checked, Issue, reject};
use std::collections::BTreeMap;
use std::path::Path;

pub(super) fn validate(base: &Path, request: &Request) -> Checked<()> {
    let raw = request.bootstrap.report.read(base)?;
    let text =
        std::str::from_utf8(&raw).map_err(|error| reject(Issue::Bootstrap, error.to_string()))?;
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if matches!(key, "binary_path" | "target_directory" | "host")
            && (value.is_empty() || fields.insert(key, value).is_some())
        {
            return Err(reject(
                Issue::Bootstrap,
                format!("empty or duplicate {key}"),
            ));
        }
    }
    let field = |key| {
        fields
            .get(key)
            .copied()
            .ok_or_else(|| reject(Issue::Bootstrap, format!("missing {key}")))
    };
    let binary = Path::new(field("binary_path")?);
    let target = Path::new(field("target_directory")?);
    field("host")?;
    if !binary.is_absolute() || !target.is_absolute() {
        return Err(reject(Issue::Bootstrap, "bootstrap paths must be absolute"));
    }
    let binary = binary
        .canonicalize()
        .map_err(|error| reject(Issue::Bootstrap, error.to_string()))?;
    let target = target
        .canonicalize()
        .map_err(|error| reject(Issue::Bootstrap, error.to_string()))?;
    if binary.parent().and_then(Path::parent) != Some(target.as_path()) {
        return Err(reject(
            Issue::Bootstrap,
            "binary is not directly below a Cargo target profile",
        ));
    }
    if request.bootstrap.binary.verify_binary(base)? != binary {
        return Err(reject(
            Issue::Bootstrap,
            "bootstrap output names a different binary",
        ));
    }
    Ok(())
}
