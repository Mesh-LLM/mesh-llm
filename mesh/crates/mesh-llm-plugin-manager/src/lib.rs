mod archive;
pub mod asset;
pub mod catalog;
mod catalog_pin;
pub mod compatibility;
pub mod github;
pub mod install;
pub mod legacy_defaults;
mod package_identity;
mod placement;
pub mod plugin_toml;
mod sets;
pub mod skills;
pub mod source_ref;
pub mod store;
pub mod target;
mod upgrade;

pub use asset::{AssetMatchKind, PluginAsset, infer_plugin_asset, select_plugin_asset};
pub use catalog::{CatalogEntry, CatalogSet, PinnedRelease, PluginCatalog, sets_url_for};
pub use compatibility::{
    HostCompatibility, Verdict as CompatibilityVerdict, evaluate as evaluate_compatibility,
};
pub use github::{GitHubRelease, GitHubReleaseAsset, GitHubReleaseClient};
pub use install::{
    InstallOutcome, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter,
    install_plugin, update_plugin, update_plugin_from_catalog,
};
pub use legacy_defaults::{
    NO_DEFAULT_PLUGINS_ENV, Surface, UNLISTED_DEFAULT_ALLOWS, allows, default_plugins_opted_out,
    migrated_allows, migrated_allows_keys, opts_out, parse_allows,
};
pub use mesh_llm_skills::{
    SkillAgent, SkillInstallAction, SkillInstallReport, SkillInstallStatus, SkillPackage,
    SkillTarget,
};
pub use sets::{
    MemberAvailability, MemberInstall, SetInstallOutcome, SetInstallResult, SetMemberStatus,
    SetResolution, install_named_set, install_set, plan_set_install, resolve_set,
    set_offer_summary,
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
pub use upgrade::{
    PluginUpgradeCheck, PluginUpgradePlan, PluginUpgradeVerdict, RepairOutcome,
    plan_plugin_upgrades, repair_incompatible_plugins,
};
