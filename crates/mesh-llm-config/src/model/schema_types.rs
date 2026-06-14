use serde::{Deserialize, Serialize};
use std::iter::Peekable;
use std::str::Chars;

pub const CANONICAL_MODEL_REF_SEGMENT: &str = "<model-ref>";
pub const CANONICAL_PLUGIN_NAME_SEGMENT: &str = "<plugin-name>";

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigSchema {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub settings: Vec<ConfigSettingSchema>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigSettingSchema {
    pub path: ConfigPath,
    #[serde(default)]
    pub alias_policy: ConfigAliasPolicy,
    pub owner: ConfigSettingOwner,
    pub value_schema: ConfigValueSchema,
    pub support: ConfigSupportState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub control_surfaces: Vec<ConfigControlSurface>,
    pub apply_mode: ConfigApplyMode,
    pub restart_scope: ConfigRestartScope,
    pub visibility: ConfigVisibility,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<ConfigConstraint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConfigPath {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segments: Vec<ConfigPathSegment>,
}

impl ConfigPath {
    pub fn root() -> Self {
        Self::default()
    }

    pub fn field(name: impl Into<String>) -> Self {
        let mut path = Self::root();
        path.push_field(name);
        path
    }

    pub fn from_fields<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut path = Self::root();
        for field in fields {
            path.push_field(field);
        }
        path
    }

    pub fn push_field(&mut self, name: impl Into<String>) -> &mut Self {
        self.segments
            .push(ConfigPathSegment::Field { name: name.into() });
        self
    }

    pub fn push_index(&mut self, index: usize) -> &mut Self {
        self.segments.push(ConfigPathSegment::Index { index });
        self
    }

    pub fn push_key(&mut self, name: impl Into<String>) -> &mut Self {
        self.segments
            .push(ConfigPathSegment::Key { name: name.into() });
        self
    }

    pub fn render(&self) -> String {
        let mut rendered = String::new();
        for segment in &self.segments {
            match segment {
                ConfigPathSegment::Field { name } => {
                    if !rendered.is_empty() {
                        rendered.push('.');
                    }
                    rendered.push_str(name);
                }
                ConfigPathSegment::Index { index } => {
                    rendered.push('[');
                    rendered.push_str(&index.to_string());
                    rendered.push(']');
                }
                ConfigPathSegment::Key { name } => {
                    rendered.push('[');
                    rendered.push_str(&format!("{name:?}"));
                    rendered.push(']');
                }
            }
        }
        rendered
    }

    pub fn parse_rendered(rendered: &str) -> Result<Self, String> {
        let mut path = Self::root();
        let mut chars = rendered.chars().peekable();
        let mut field = String::new();

        while let Some(ch) = chars.next() {
            match ch {
                '.' => {
                    if field.is_empty() {
                        if path.segments.is_empty() {
                            return Err(format!("invalid config path `{rendered}`"));
                        }
                        continue;
                    }
                    path.push_field(std::mem::take(&mut field));
                }
                '[' => {
                    if !field.is_empty() {
                        path.push_field(std::mem::take(&mut field));
                    }
                    match chars.peek().copied() {
                        Some('"') => {
                            path.push_key(parse_rendered_key(&mut chars, rendered)?);
                        }
                        Some(next) if next.is_ascii_digit() => {
                            let mut index = String::new();
                            while let Some(next) = chars.peek().copied() {
                                if next == ']' {
                                    break;
                                }
                                if !next.is_ascii_digit() {
                                    return Err(format!("invalid config path `{rendered}`"));
                                }
                                index.push(next);
                                chars.next();
                            }
                            if chars.next() != Some(']') || index.is_empty() {
                                return Err(format!("invalid config path `{rendered}`"));
                            }
                            let index = index
                                .parse::<usize>()
                                .map_err(|_| format!("invalid config path `{rendered}`"))?;
                            path.push_index(index);
                        }
                        _ => return Err(format!("invalid config path `{rendered}`")),
                    }
                }
                other => field.push(other),
            }
        }

        if !field.is_empty() {
            path.push_field(field);
        }

        Ok(path)
    }

    pub fn normalize_builtin_layout(&self) -> Self {
        let mut normalized = Self::root();
        let root_field = self.segments.first().and_then(|segment| match segment {
            ConfigPathSegment::Field { name } => Some(name.as_str()),
            _ => None,
        });

        for (index, segment) in self.segments.iter().enumerate() {
            match (root_field, index, segment) {
                (Some("models"), 1, ConfigPathSegment::Index { .. }) => {
                    normalized.push_field(CANONICAL_MODEL_REF_SEGMENT);
                }
                (Some("plugin"), 1, ConfigPathSegment::Index { .. }) => {
                    normalized.push_field(CANONICAL_PLUGIN_NAME_SEGMENT);
                }
                _ => normalized.segments.push(segment.clone()),
            }
        }

        normalized
    }
}

fn parse_rendered_key(chars: &mut Peekable<Chars<'_>>, rendered: &str) -> Result<String, String> {
    if chars.next() != Some('"') {
        return Err(format!("invalid config path `{rendered}`"));
    }

    let mut key = String::new();
    while let Some(next) = chars.next() {
        match next {
            '"' => {
                if chars.next() != Some(']') {
                    return Err(format!("invalid config path `{rendered}`"));
                }
                return Ok(key);
            }
            '\\' => key.push(parse_rendered_escape(chars, rendered)?),
            other => key.push(other),
        }
    }

    Err(format!("invalid config path `{rendered}`"))
}

fn parse_rendered_escape(chars: &mut Peekable<Chars<'_>>, rendered: &str) -> Result<char, String> {
    match chars.next() {
        Some('"') => Ok('"'),
        Some('\\') => Ok('\\'),
        Some('n') => Ok('\n'),
        Some('r') => Ok('\r'),
        Some('t') => Ok('\t'),
        Some('0') => Ok('\0'),
        Some('u') => parse_rendered_unicode_escape(chars, rendered),
        _ => Err(format!("invalid config path `{rendered}`")),
    }
}

fn parse_rendered_unicode_escape(
    chars: &mut Peekable<Chars<'_>>,
    rendered: &str,
) -> Result<char, String> {
    if chars.next() != Some('{') {
        return Err(format!("invalid config path `{rendered}`"));
    }

    let mut codepoint = String::new();
    for next in chars.by_ref() {
        match next {
            '}' => {
                let codepoint = u32::from_str_radix(&codepoint, 16)
                    .map_err(|_| format!("invalid config path `{rendered}`"))?;
                return char::from_u32(codepoint)
                    .ok_or_else(|| format!("invalid config path `{rendered}`"));
            }
            hex if hex.is_ascii_hexdigit() => codepoint.push(hex),
            _ => return Err(format!("invalid config path `{rendered}`")),
        }
    }

    Err(format!("invalid config path `{rendered}`"))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigPathSegment {
    Field { name: String },
    Index { index: usize },
    Key { name: String },
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigAliasPolicy {
    #[serde(default)]
    pub mode: ConfigAliasMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<ConfigPathAlias>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConfigPathAlias {
    pub path: ConfigPath,
    pub kind: ConfigPathAliasKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigAliasMode {
    #[default]
    CanonicalOnly,
    CanonicalWithLegacyAliases,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigPathAliasKind {
    #[default]
    LegacyKey,
    LegacyLayout,
    LegacySection,
    LegacyShim,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSettingOwner {
    #[default]
    BuiltIn,
    Engine,
    Plugin,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigValueSchema {
    Boolean,
    Integer,
    Float,
    String,
    SocketAddr,
    Enum { values: Vec<String> },
    OneOf { variants: Vec<ConfigValueSchema> },
    Array { items: Box<ConfigValueSchema> },
    Object,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigSupportState {
    #[default]
    Supported,
    Experimental,
    DeprecatedAlias,
    Unwired,
    Unsupported,
    Rejected,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigControlSurface {
    ConfigFile,
    Cli,
    OwnerControl,
    Api,
    Ui,
    PluginManifest,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigApplyMode {
    #[default]
    StaticOnLoad,
    DynamicValidationOnly,
    DynamicApply,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigRestartScope {
    #[default]
    None,
    ModelReload,
    ProcessRestart,
    MeshRestart,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigVisibility {
    #[default]
    User,
    Advanced,
    Hidden,
    Internal,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigConstraint {
    NonEmpty,
    Positive,
    Range {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<String>,
    },
    Requires {
        path: ConfigPath,
    },
    AllowedValues {
        values: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rendered_accepts_canonical_placeholder_path() {
        let rendered = "models.<model-ref>.hardware.device";
        let path = ConfigPath::parse_rendered(rendered).expect("canonical path should parse");

        assert_eq!(path.render(), rendered);
    }

    #[test]
    fn parse_rendered_roundtrips_rendered_key_escapes() {
        let mut path = ConfigPath::field("plugin");
        path.push_key("plugin.with\nquote\"backslash\\escape\u{1b}");
        path.push_field("settings");

        let rendered = path.render();
        let parsed = ConfigPath::parse_rendered(&rendered).expect("rendered key should parse");

        assert_eq!(parsed, path);
        assert_eq!(parsed.render(), rendered);
    }
}
