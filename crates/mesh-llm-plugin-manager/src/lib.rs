mod archive;
pub mod asset;
pub mod catalog;
pub mod github;
pub mod install;
pub mod source_ref;
pub mod store;
pub mod target;

pub use asset::{AssetMatchKind, PluginAsset, select_plugin_asset};
pub use catalog::{CatalogEntry, PluginCatalog};
pub use github::{GitHubRelease, GitHubReleaseAsset, GitHubReleaseClient};
pub use install::{
    InstallOutcome, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter,
    install_plugin, update_plugin,
};
pub use source_ref::{GitHubPluginSource, PluginInstallRef, PluginVersion, parse_install_ref};
pub use store::{InstalledPluginMetadata, PluginStore, default_store_root};
pub use target::{ArchiveExt, PluginTarget, UnsupportedTarget};
