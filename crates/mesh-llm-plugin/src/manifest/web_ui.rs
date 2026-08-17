use crate::proto;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path};

const INTEGRATIONS_PARENT_TAB: &str = "integrations";

#[derive(Clone, Debug)]
pub struct PluginWebUiBuilder {
    inner: proto::PluginWebUiManifest,
}

#[derive(Clone, Debug)]
pub struct PluginWebUiPageBuilder {
    inner: proto::PluginWebUiPageManifest,
}

#[derive(Clone, Debug)]
pub struct PluginWebUiConfigSectionBuilder {
    inner: proto::PluginWebUiConfigSectionManifest,
}

#[derive(Clone, Debug)]
pub struct PluginWebUiBundleBuilder {
    inner: proto::PluginWebUiBundleManifest,
}

pub fn web_ui() -> PluginWebUiBuilder {
    PluginWebUiBuilder {
        inner: proto::PluginWebUiManifest::default(),
    }
}

pub fn web_ui_page(
    id: impl Into<String>,
    label: impl Into<String>,
    route: impl Into<String>,
    entry_script: impl Into<String>,
) -> PluginWebUiPageBuilder {
    PluginWebUiPageBuilder {
        inner: proto::PluginWebUiPageManifest {
            id: id.into(),
            label: label.into(),
            icon: None,
            route: route.into(),
            bundle_id: String::new(),
            entry_script: entry_script.into(),
        },
    }
}

pub fn web_ui_config_section(
    id: impl Into<String>,
    title: impl Into<String>,
    entry_script: impl Into<String>,
) -> PluginWebUiConfigSectionBuilder {
    PluginWebUiConfigSectionBuilder {
        inner: proto::PluginWebUiConfigSectionManifest {
            id: id.into(),
            title: title.into(),
            entry_script: entry_script.into(),
            parent_tab: None,
            bundle_id: String::new(),
        },
    }
}

pub fn web_ui_bundle(
    id: impl Into<String>,
    root_path: impl Into<String>,
) -> PluginWebUiBundleBuilder {
    PluginWebUiBundleBuilder {
        inner: proto::PluginWebUiBundleManifest {
            id: id.into(),
            root_path: root_path.into(),
        },
    }
}

impl PluginWebUiBuilder {
    pub fn page<T: Into<proto::PluginWebUiPageManifest>>(mut self, page: T) -> Self {
        self.inner.pages.push(page.into());
        self
    }

    pub fn config_section<T: Into<proto::PluginWebUiConfigSectionManifest>>(
        mut self,
        section: T,
    ) -> Self {
        self.inner.config_sections.push(section.into());
        self
    }

    pub fn bundle<T: Into<proto::PluginWebUiBundleManifest>>(mut self, bundle: T) -> Self {
        self.inner.bundles.push(bundle.into());
        self
    }
}

impl PluginWebUiPageBuilder {
    pub fn icon(mut self, icon: impl Into<String>) -> Self {
        self.inner.icon = Some(icon.into());
        self
    }

    pub fn bundle_id(mut self, bundle_id: impl Into<String>) -> Self {
        self.inner.bundle_id = bundle_id.into();
        self
    }
}

impl PluginWebUiConfigSectionBuilder {
    pub fn parent_tab(mut self, parent_tab: impl Into<String>) -> Self {
        self.inner.parent_tab = Some(parent_tab.into());
        self
    }

    pub fn bundle_id(mut self, bundle_id: impl Into<String>) -> Self {
        self.inner.bundle_id = bundle_id.into();
        self
    }
}

impl From<PluginWebUiBuilder> for proto::PluginWebUiManifest {
    fn from(value: PluginWebUiBuilder) -> Self {
        value.inner
    }
}

impl From<PluginWebUiPageBuilder> for proto::PluginWebUiPageManifest {
    fn from(value: PluginWebUiPageBuilder) -> Self {
        value.inner
    }
}

impl From<PluginWebUiConfigSectionBuilder> for proto::PluginWebUiConfigSectionManifest {
    fn from(value: PluginWebUiConfigSectionBuilder) -> Self {
        value.inner
    }
}

impl From<PluginWebUiBundleBuilder> for proto::PluginWebUiBundleManifest {
    fn from(value: PluginWebUiBundleBuilder) -> Self {
        value.inner
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct PackagedPluginWebUi {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pages: Vec<PackagedPluginWebUiPage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_sections: Vec<PackagedPluginWebUiConfigSection>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bundles: Vec<PackagedPluginWebUiBundle>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct PackagedPluginWebUiPage {
    pub id: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub icon: Option<String>,
    pub route: String,
    pub bundle_id: String,
    pub entry_script: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct PackagedPluginWebUiConfigSection {
    pub id: String,
    pub title: String,
    pub entry_script: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent_tab: Option<String>,
    pub bundle_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct PackagedPluginWebUiBundle {
    pub id: String,
    pub root_path: String,
}

impl TryFrom<&proto::PluginWebUiManifest> for PackagedPluginWebUi {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginWebUiManifest) -> Result<Self> {
        let bundle_id = validate_v1_bundle_contract(value)?;
        let pages = value
            .pages
            .iter()
            .map(|page| {
                PackagedPluginWebUiPage::try_from_with_bundle_id(page, bundle_id.as_deref())
            })
            .collect::<Result<Vec<_>>>()?;
        let config_sections = value
            .config_sections
            .iter()
            .map(|section| {
                PackagedPluginWebUiConfigSection::try_from_with_bundle_id(
                    section,
                    bundle_id.as_deref(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let bundles = value
            .bundles
            .iter()
            .map(PackagedPluginWebUiBundle::try_from)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            pages,
            config_sections,
            bundles,
        })
    }
}

impl TryFrom<&proto::PluginWebUiPageManifest> for PackagedPluginWebUiPage {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginWebUiPageManifest) -> Result<Self> {
        Self::try_from_with_bundle_id(value, None)
    }
}

impl PackagedPluginWebUiPage {
    fn try_from_with_bundle_id(
        value: &proto::PluginWebUiPageManifest,
        expected_bundle_id: Option<&str>,
    ) -> Result<Self> {
        validate_non_empty("web UI page id", &value.id)?;
        validate_non_empty("web UI page label", &value.label)?;
        validate_route_slug("web UI page route", &value.route)?;
        validate_bundle_reference(
            "web UI page bundle_id",
            &value.bundle_id,
            expected_bundle_id,
        )?;
        validate_relative_path("web UI page entry_script", &value.entry_script)?;
        if let Some(icon) = &value.icon {
            validate_relative_path("web UI page icon", icon)?;
        }
        Ok(Self {
            id: value.id.clone(),
            label: value.label.clone(),
            icon: value.icon.clone(),
            route: value.route.clone(),
            bundle_id: value.bundle_id.clone(),
            entry_script: value.entry_script.clone(),
        })
    }
}

impl TryFrom<&proto::PluginWebUiConfigSectionManifest> for PackagedPluginWebUiConfigSection {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginWebUiConfigSectionManifest) -> Result<Self> {
        Self::try_from_with_bundle_id(value, None)
    }
}

impl PackagedPluginWebUiConfigSection {
    fn try_from_with_bundle_id(
        value: &proto::PluginWebUiConfigSectionManifest,
        expected_bundle_id: Option<&str>,
    ) -> Result<Self> {
        validate_non_empty("web UI config section id", &value.id)?;
        validate_non_empty("web UI config section title", &value.title)?;
        validate_bundle_reference(
            "web UI config section bundle_id",
            &value.bundle_id,
            expected_bundle_id,
        )?;
        validate_relative_path("web UI config section entry_script", &value.entry_script)?;
        if let Some(parent_tab) = &value.parent_tab {
            validate_config_parent_tab(parent_tab)?;
        }
        Ok(Self {
            id: value.id.clone(),
            title: value.title.clone(),
            entry_script: value.entry_script.clone(),
            parent_tab: value.parent_tab.clone(),
            bundle_id: value.bundle_id.clone(),
        })
    }
}

impl TryFrom<&proto::PluginWebUiBundleManifest> for PackagedPluginWebUiBundle {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginWebUiBundleManifest) -> Result<Self> {
        validate_non_empty("web UI bundle id", &value.id)?;
        validate_relative_path("web UI bundle root_path", &value.root_path)?;
        Ok(Self {
            id: value.id.clone(),
            root_path: value.root_path.clone(),
        })
    }
}

fn validate_v1_bundle_contract(value: &proto::PluginWebUiManifest) -> Result<Option<String>> {
    if value.pages.is_empty() && value.config_sections.is_empty() && value.bundles.is_empty() {
        return Ok(None);
    }
    let [bundle] = value.bundles.as_slice() else {
        bail!(
            "web UI v1 declarations with pages or config sections must declare exactly one bundle root"
        );
    };
    validate_non_empty("web UI bundle id", &bundle.id)?;
    Ok(Some(bundle.id.clone()))
}

fn validate_bundle_reference(
    field_name: &str,
    value: &str,
    expected_bundle_id: Option<&str>,
) -> Result<()> {
    validate_non_empty(field_name, value)?;
    if let Some(expected) = expected_bundle_id
        && value != expected
    {
        bail!("{field_name} must reference declared web UI bundle `{expected}`, got `{value}`");
    }
    Ok(())
}

fn validate_non_empty(field_name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field_name} must be non-empty");
    }
    Ok(())
}

fn validate_config_parent_tab(parent_tab: &str) -> Result<()> {
    if parent_tab != INTEGRATIONS_PARENT_TAB {
        bail!("web UI config section parent_tab must be `integrations`");
    }
    Ok(())
}

fn validate_route_slug(field_name: &str, value: &str) -> Result<()> {
    validate_non_empty(field_name, value)?;
    if has_remote_url_scheme(value) || value.contains("://") {
        bail!("{field_name} must be a slug, got URL-like value `{value}`");
    }
    if value.contains('/') || value.contains('\\') {
        bail!("{field_name} must be a slug without path separators `{value}`");
    }
    if value == "." || value == ".." || value.starts_with('.') {
        bail!("{field_name} must be a slug without traversal or hidden path syntax `{value}`");
    }
    Ok(())
}

fn validate_relative_path(field_name: &str, value: &str) -> Result<()> {
    validate_non_empty(field_name, value)?;
    if has_remote_url_scheme(value) {
        bail!("{field_name} must be a relative path, got remote URL `{value}`");
    }
    let path = Path::new(value);
    if path.is_absolute() {
        bail!("{field_name} must be a relative path, got absolute path `{value}`");
    }
    if path
        .components()
        .all(|component| matches!(component, Component::CurDir))
    {
        bail!("{field_name} must name a file or directory below the package root");
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        bail!("{field_name} must not contain traversal segments `{value}`");
    }
    if path.components().any(|component| match component {
        Component::Normal(name) => name.to_string_lossy().starts_with('.'),
        _ => false,
    }) {
        bail!("{field_name} must not contain hidden path segments `{value}`");
    }
    Ok(())
}

fn has_remote_url_scheme(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://") || value.starts_with("//")
}

#[cfg(test)]
mod tests;
