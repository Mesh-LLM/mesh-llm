//! Web-console asset accessors.
//!
//! With the default `embed-assets` feature, `include_dir!` bundles the
//! built React console (`dist/`) into the crate at compile time and
//! [`index`] / [`asset`] serve those bytes.
//!
//! With `embed-assets` disabled, the crate compiles down to ~nothing and
//! both accessors return `None`. Callers (notably
//! `mesh-llm-host-runtime`'s console asset routes) treat that as "no UI
//! bundled" and surface 404s for the asset paths while keeping every
//! other management-API surface working. This lets lib-style consumers
//! of `mesh-llm-host-runtime` drop several MB of embedded payload by
//! opting out of `default-features`.
//!
//! SDKs can use [`FileSystemConsoleAssets`] to serve the same built console
//! from package resources without baking those assets into every native
//! library variant.

use std::{borrow::Cow, path::PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UiAsset {
    pub contents: Cow<'static, [u8]>,
    pub content_type: &'static str,
    pub cache_control: &'static str,
}

pub trait ConsoleAssetProvider: Send + Sync {
    fn index(&self) -> Option<UiAsset>;
    fn asset(&self, path: &str) -> Option<UiAsset>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EmbeddedConsoleAssets;

#[derive(Clone, Debug)]
pub struct FileSystemConsoleAssets {
    root: PathBuf,
}

impl FileSystemConsoleAssets {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

#[cfg(feature = "embed-assets")]
mod embedded {
    use include_dir::{Dir, include_dir};

    pub(super) static CONSOLE_DIST: Dir<'_> = include_dir!("$MESH_LLM_UI_DIST");
}

pub fn index() -> Option<UiAsset> {
    EmbeddedConsoleAssets.index()
}

pub fn asset(path: &str) -> Option<UiAsset> {
    EmbeddedConsoleAssets.asset(path)
}

impl ConsoleAssetProvider for EmbeddedConsoleAssets {
    fn index(&self) -> Option<UiAsset> {
        self.asset("index.html").map(|mut asset| {
            asset.cache_control = "public, max-age=3600";
            asset
        })
    }

    #[cfg(feature = "embed-assets")]
    fn asset(&self, path: &str) -> Option<UiAsset> {
        let rel = clean_relative_path(path)?;
        let file = embedded::CONSOLE_DIST.get_file(rel)?;
        Some(UiAsset {
            contents: Cow::Borrowed(file.contents()),
            content_type: content_type(rel),
            cache_control: cache_control(rel),
        })
    }

    #[cfg(not(feature = "embed-assets"))]
    fn asset(&self, _path: &str) -> Option<UiAsset> {
        None
    }
}

impl ConsoleAssetProvider for FileSystemConsoleAssets {
    fn index(&self) -> Option<UiAsset> {
        self.asset("index.html").map(|mut asset| {
            asset.cache_control = "public, max-age=3600";
            asset
        })
    }

    fn asset(&self, path: &str) -> Option<UiAsset> {
        let rel = clean_relative_path(path)?;
        let root = self.root.canonicalize().ok()?;
        let full_path = root.join(rel).canonicalize().ok()?;
        if !full_path.starts_with(&root) {
            return None;
        }
        let contents = std::fs::read(full_path).ok()?;
        Some(UiAsset {
            contents: Cow::Owned(contents),
            content_type: content_type(rel),
            cache_control: cache_control(rel),
        })
    }
}

fn clean_relative_path(path: &str) -> Option<&str> {
    let rel = path.trim_start_matches('/');
    if rel.is_empty() || rel.contains("..") || rel.starts_with('.') {
        return None;
    }
    Some(rel)
}

pub fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "woff2" => "font/woff2",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

pub fn cache_control(path: &str) -> &'static str {
    if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    }
}

#[cfg(all(test, feature = "embed-assets"))]
mod tests {
    use super::{asset, content_type};

    #[test]
    fn rejects_parent_directory_paths() {
        assert!(asset("../index.html").is_none());
    }

    #[test]
    fn maps_common_asset_content_types() {
        assert_eq!(content_type("index.html"), "text/html; charset=utf-8");
        assert_eq!(
            content_type("assets/app.js"),
            "text/javascript; charset=utf-8"
        );
        assert_eq!(content_type("assets/app.css"), "text/css; charset=utf-8");
        assert_eq!(
            content_type("manifest.json"),
            "application/json; charset=utf-8"
        );
    }
}

#[cfg(test)]
mod filesystem_tests {
    use super::{ConsoleAssetProvider, FileSystemConsoleAssets};
    use std::fs;

    #[test]
    fn filesystem_assets_read_from_root() {
        let root = std::env::temp_dir().join(format!("mesh-llm-ui-assets-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("assets")).expect("create temp asset root");
        fs::write(root.join("index.html"), "<html></html>").expect("write index");
        fs::write(root.join("assets/app.js"), "console.log('ok')").expect("write js");

        let assets = FileSystemConsoleAssets::new(&root);
        assert_eq!(
            assets.index().expect("index").contents.as_ref(),
            b"<html></html>"
        );
        assert_eq!(
            assets.asset("/assets/app.js").expect("asset").content_type,
            "text/javascript; charset=utf-8"
        );
        assert!(assets.asset("../secret").is_none());

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_assets_reject_symlinks_outside_root() {
        let temp =
            std::env::temp_dir().join(format!("mesh-llm-ui-assets-symlink-{}", std::process::id()));
        let root = temp.join("root");
        let secret = temp.join("secret.txt");
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(root.join("assets")).expect("create temp asset root");
        fs::write(root.join("index.html"), "<html></html>").expect("write index");
        fs::write(&secret, "secret").expect("write secret");
        std::os::unix::fs::symlink(&secret, root.join("assets/secret.txt"))
            .expect("create symlink");

        let assets = FileSystemConsoleAssets::new(&root);
        assert!(assets.asset("/assets/secret.txt").is_none());

        let _ = fs::remove_dir_all(temp);
    }
}

#[cfg(all(test, not(feature = "embed-assets")))]
mod stub_tests {
    use super::{asset, index};

    #[test]
    fn returns_none_when_assets_not_embedded() {
        assert!(index().is_none());
        assert!(asset("index.html").is_none());
        assert!(asset("assets/app.js").is_none());
    }
}
