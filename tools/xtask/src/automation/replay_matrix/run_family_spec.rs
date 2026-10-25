use super::InvocationContext;
use crate::process::{ProcessSpec, Value as ProcessValue};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;

pub(super) fn child_spec(
    context: &InvocationContext<'_>,
    reader: &Path,
    arguments: Vec<OsString>,
) -> ProcessSpec {
    ProcessSpec {
        executable: reader.to_path_buf(),
        arguments: arguments.into_iter().map(ProcessValue::Public).collect(),
        cwd: context.cwd.to_path_buf(),
        environment: child_environment(),
    }
}

fn child_environment() -> BTreeMap<OsString, ProcessValue> {
    std::env::vars_os()
        .filter(|(_, value)| !value.is_empty())
        .map(|(key, value)| {
            let name = key.to_string_lossy().to_ascii_lowercase();
            let sensitive = [
                "token",
                "password",
                "secret",
                "authorization",
                "api_key",
                "access_key",
            ]
            .iter()
            .any(|marker| name.contains(marker));
            let value = match sensitive {
                true => ProcessValue::Secret(value),
                false => ProcessValue::Public(value),
            };
            (key, value)
        })
        .collect()
}
