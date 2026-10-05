use crate::{
    CANONICAL_MODEL_REF_SEGMENT, ConfigControlSurface, ConfigRestartScope, ConfigSupportState,
    ConfigValueSchema, built_in_config_settings,
};
use std::collections::BTreeMap;

const CONFIG_REFERENCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../website/src/docs/pages/config-reference.md"
));

#[derive(Debug)]
struct DocumentedMetadata<'a> {
    value_type: &'a str,
    allowed_and_default: &'a str,
    scope: Option<&'a str>,
    restart: &'a str,
    status: &'a str,
    cli: &'a str,
}

fn normalize_schema_path(rendered: &str) -> String {
    let model_prefix = format!("models.{CANONICAL_MODEL_REF_SEGMENT}.");
    rendered
        .strip_prefix("defaults.")
        .or_else(|| rendered.strip_prefix(&model_prefix))
        .unwrap_or(rendered)
        .replace("<plugin-name>", "<name>")
        .to_string()
}

fn row_paths(cell: &str) -> Vec<&str> {
    cell.split("<br>")
        .filter_map(|part| {
            let start = part.find('`')? + 1;
            let end = part[start..].find('`')? + start;
            Some(&part[start..end])
        })
        .collect()
}

fn documented_metadata() -> BTreeMap<&'static str, DocumentedMetadata<'static>> {
    let mut rows = BTreeMap::new();
    let mut headers: Vec<&str> = Vec::new();
    for line in CONFIG_REFERENCE
        .lines()
        .filter(|line| line.starts_with('|'))
    {
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        if cells.first() == Some(&"Key path") {
            headers = cells;
            continue;
        }
        if cells.first().is_none_or(|cell| cell.starts_with("---")) || headers.is_empty() {
            continue;
        }
        let column = |name: &str| {
            headers
                .iter()
                .position(|header| *header == name)
                .and_then(|index| cells.get(index))
                .copied()
        };
        let Some(value_type) = column("Type") else {
            continue;
        };
        let Some(allowed_and_default) =
            column("Allowed values / default (`auto`)").or_else(|| column("Default / `auto`"))
        else {
            continue;
        };
        let Some(restart) = column("Restart") else {
            continue;
        };
        let Some(status) = column("Status") else {
            continue;
        };
        let Some(cli) = column("CLI equivalent") else {
            continue;
        };
        for path in row_paths(cells[0]) {
            rows.insert(
                path,
                DocumentedMetadata {
                    value_type,
                    allowed_and_default,
                    scope: column("`[defaults]` / `[[models]]`")
                        .or_else(|| column("[defaults] / [[models]]")),
                    restart,
                    status,
                    cli,
                },
            );
        }
    }
    rows
}

fn expected_type_token(path: &str, schema: &ConfigValueSchema) -> Option<&'static str> {
    match schema {
        ConfigValueSchema::Boolean => Some("boolean"),
        ConfigValueSchema::Integer => Some("integer"),
        ConfigValueSchema::Float => Some("float"),
        ConfigValueSchema::String => Some("string"),
        ConfigValueSchema::Path => Some("path"),
        ConfigValueSchema::Url => Some("URL"),
        ConfigValueSchema::SocketAddr => Some("socket address"),
        ConfigValueSchema::Enum { .. }
            if matches!(
                path,
                "mesh_requirements.min_node_version" | "mesh_requirements.max_node_version"
            ) =>
        {
            Some("string (semver)")
        }
        ConfigValueSchema::Enum { .. } => Some("enum"),
        ConfigValueSchema::Array { .. } => Some("array"),
        ConfigValueSchema::Object { .. } => Some("object"),
        ConfigValueSchema::OneOf { .. } => None,
    }
}

fn expected_restart(
    path: &str,
    scope: ConfigRestartScope,
    support: ConfigSupportState,
) -> &'static str {
    if matches!(
        support,
        ConfigSupportState::Unsupported | ConfigSupportState::Rejected
    ) {
        return "not applicable";
    }
    if path.starts_with("request_defaults.") && scope == ConfigRestartScope::None {
        return "request-time";
    }
    if path.starts_with("plugin.") && scope == ConfigRestartScope::ProcessRestart {
        return "plugin process restart";
    }
    match scope {
        ConfigRestartScope::None => "applies dynamically",
        ConfigRestartScope::ModelReload => "model reload",
        ConfigRestartScope::ProcessRestart => "process restart",
        ConfigRestartScope::MeshRestart => "mesh restart",
    }
}

fn assert_support(path: &str, support: ConfigSupportState, status: &str) {
    let expected = match support {
        ConfigSupportState::Unwired => Some("unwired"),
        ConfigSupportState::Unsupported | ConfigSupportState::Rejected => Some("rejected"),
        ConfigSupportState::Supported
        | ConfigSupportState::Experimental
        | ConfigSupportState::DeprecatedAlias => None,
    };
    if let Some(expected) = expected {
        assert!(
            status.starts_with(expected),
            "{path} has schema support {support:?}, but website status is {status:?}"
        );
    }
}

#[test]
fn website_metadata_matches_exported_schema_contract() {
    let documented = documented_metadata();
    let settings = built_in_config_settings();
    let mut schema_scopes = BTreeMap::<String, (bool, bool)>::new();
    for setting in &settings {
        let rendered = setting.path.render();
        let scope = schema_scopes
            .entry(normalize_schema_path(&rendered))
            .or_default();
        scope.0 |= rendered.starts_with("defaults.");
        scope.1 |= rendered.starts_with("models.");
    }

    for setting in settings {
        let rendered = setting.path.render();
        let path = normalize_schema_path(&rendered);
        let row = documented
            .get(path.as_str())
            .unwrap_or_else(|| panic!("missing canonical metadata row for {path}"));

        assert!(
            !row.allowed_and_default.is_empty(),
            "{path} has no default/auto metadata"
        );
        assert!(!row.cli.is_empty(), "{path} has no CLI-equivalent metadata");
        if let Some(token) = expected_type_token(&path, &setting.value_schema) {
            assert!(
                row.value_type.contains(token),
                "{path} schema type {token:?} disagrees with website type {:?}",
                row.value_type
            );
        }
        if let ConfigValueSchema::Enum { values } = &setting.value_schema
            && values.len() <= 16
        {
            for value in values {
                assert!(
                    row.allowed_and_default.contains(&format!("`{value}`")),
                    "{path} website metadata omits enum value {value:?}"
                );
            }
        }
        assert_eq!(
            row.restart,
            expected_restart(&path, setting.restart_scope, setting.support),
            "{path} restart drift"
        );
        assert_support(&path, setting.support, row.status);

        if rendered.starts_with("defaults.") || rendered.starts_with("models.") {
            let expected_scope = match schema_scopes[&path] {
                (true, true) => "both",
                (true, false) => "`[defaults]` only",
                (false, true) => "`[[models]]` only",
                (false, false) => unreachable!("model setting has no model scope"),
            };
            assert_eq!(row.scope, Some(expected_scope), "{path} scope drift");
        }
        let has_cli = setting
            .control_surfaces
            .contains(&ConfigControlSurface::Cli);
        if has_cli {
            assert_ne!(
                row.cli, "none",
                "{path} drops its exported CLI control surface"
            );
        }
    }
}
