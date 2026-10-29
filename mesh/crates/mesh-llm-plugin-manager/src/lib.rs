mod archive;
pub mod asset;
pub mod bundled;
pub mod catalog;
pub mod defaults;
pub mod github;
pub mod install;
pub mod plugin_toml;
pub mod requirements;
pub mod skills;
pub mod source_ref;
pub mod store;
pub mod target;

pub use asset::{AssetMatchKind, PluginAsset, select_plugin_asset};
pub use bundled::install_bundled_default;
pub use catalog::{CatalogEntry, PinnedRelease, PluginCatalog};
pub use github::{GitHubRelease, GitHubReleaseAsset, GitHubReleaseClient};
pub use install::{
    InstallOutcome, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter,
    install_plugin, install_plugin_requirements, update_plugin,
};
pub use mesh_llm_skills::{
    SkillAgent, SkillInstallAction, SkillInstallReport, SkillInstallStatus, SkillPackage,
    SkillTarget,
};
pub use requirements::{
    MissingRequirement, MissingRequirementReason, missing_requirements, required_by,
};
pub use skills::{PluginSkillInstallOptions, discover_plugin_skills, install_available_skills};
pub use source_ref::{GitHubPluginSource, PluginInstallRef, PluginVersion, parse_install_ref};
pub use store::{
    InstalledPluginApplyMode, InstalledPluginConditionOperator, InstalledPluginConditionValue,
    InstalledPluginConditionalDisable, InstalledPluginConfigSchema, InstalledPluginConflictRule,
    InstalledPluginConstraint, InstalledPluginControlAvailability,
    InstalledPluginControlAvailabilitySource, InstalledPluginControlBehavior,
    InstalledPluginControlCondition, InstalledPluginDisabledWritePolicy,
    InstalledPluginManifestMetadata, InstalledPluginMetadata, InstalledPluginNumericControl,
    InstalledPluginObjectProperty, InstalledPluginOptionsSource,
    InstalledPluginPresentationMetadata, InstalledPluginRestartScope, InstalledPluginSettingSchema,
    InstalledPluginTextFormat, InstalledPluginValueKind, InstalledPluginValueSchema,
    InstalledPluginVisibility, PluginStore, SUPPORTED_PLUGIN_SCHEMA_VERSION, default_store_root,
};
pub use target::{ArchiveExt, PluginTarget, UnsupportedTarget};
