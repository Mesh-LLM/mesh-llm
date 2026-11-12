use std::{fs, path::Path};

use flate2::{Compression, write::GzEncoder};

use super::*;
use crate::{PluginStore, install::install_plugin_archive, source_ref::GitHubPluginSource};

fn options(root: &Path) -> PluginInstallOptions {
    PluginInstallOptions {
        store_root: root.join("store"),
        install_root: root.join("installed"),
        catalog_url: "unused".into(),
        target: PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
        host_version: "0.79.0".into(),
    }
}

fn archive(root: &Path, package: &str) -> std::path::PathBuf {
    let path = root.join("demo.tar.gz");
    let encoder = GzEncoder::new(fs::File::create(&path).unwrap(), Compression::default());
    let mut builder = tar::Builder::new(encoder);
    for (name, bytes) in [
        ("plugin.toml", package.as_bytes()),
        ("demo", b"new executable".as_slice()),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o755);
        header.set_size(bytes.len() as u64);
        header.set_cksum();
        builder
            .append_data(&mut header, format!("demo/{name}"), bytes)
            .unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap();
    path
}

fn replacement(
    current: Option<&InstalledPluginMetadata>,
    plugin_toml: PluginToml,
) -> InstalledPluginMetadata {
    let resolved = ResolvedInstallSource {
        plugin_name: "demo".into(),
        source: GitHubPluginSource::from_url("https://github.com/mesh-llm/demo").unwrap(),
        version: None,
        pin: None,
        compatibility: Some(HostCompatibility {
            min: Some("0.1.0".into()),
            max: None,
        }),
    };
    build_installed_metadata(
        &resolved,
        "v1.0.0",
        &GitHubReleaseAsset {
            name: "demo.tar.gz".into(),
            browser_download_url: "unused".into(),
            size: None,
            digest: None,
        },
        &PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
        ExtractedPluginArchive {
            install_path: "unused".into(),
            manifest: None,
            plugin_toml,
        },
        current,
        Some("ab".repeat(32)),
    )
}

#[test]
fn packaged_bounds_override_the_catalog_before_files_are_replaced() {
    let root = tempfile::tempdir().unwrap();
    let options = options(root.path());
    let old_dir = options.install_root.join("demo");
    fs::create_dir_all(&old_dir).unwrap();
    fs::write(old_dir.join("demo"), b"old executable").unwrap();
    let archive = archive(root.path(), "name = 'demo'\nmax_host_version = '0.79.0'\n");
    let catalog = HostCompatibility {
        min: Some("0.1.0".into()),
        max: None,
    };

    let error = extract_for_install(
        &archive,
        ArchiveExt::TarGz,
        "demo",
        &options,
        Some(&catalog),
        "0.80.0",
    )
    .unwrap_err();

    assert!(error.to_string().contains("0.79.0 or older"), "{error:#}");
    assert_eq!(fs::read(old_dir.join("demo")).unwrap(), b"old executable");
    assert!(!old_dir.join("plugin.toml").exists());
}

#[test]
fn staging_uses_the_explicit_target_host_not_the_running_host() {
    let root = tempfile::tempdir().unwrap();
    let options = options(root.path());
    let archive = archive(root.path(), "name = 'demo'\nmin_host_version = '0.80.0'\n");

    let extracted = extract_for_install(
        &archive,
        ArchiveExt::TarGz,
        "demo",
        &options,
        None,
        "0.80.0",
    )
    .unwrap();

    assert_eq!(
        fs::read(extracted.install_path.join("demo")).unwrap(),
        b"new executable"
    );
    assert_eq!(
        extracted.plugin_toml.min_host_version.as_deref(),
        Some("0.80.0")
    );
}

#[test]
fn replacement_keeps_grants_and_disabled_state_but_not_old_binary_status() {
    for allowed in [None, Some(Vec::new()), Some(vec!["web_ui".into()])] {
        let mut current = replacement(None, PluginToml::default());
        current.enabled = false;
        current.default_managed = true;
        current.allowed_surfaces = allowed.clone();
        current.last_protocol_version = Some(999);
        current.last_status = Some("failed".into());
        current.last_error = Some("old executable".into());
        let package = PluginToml {
            min_host_version: Some("0.80.0".into()),
            ..PluginToml::default()
        };

        let installed = replacement(Some(&current), package);

        assert!(!installed.enabled);
        assert!(!installed.default_managed);
        assert_eq!(
            installed.allowed_surfaces,
            allowed.or_else(|| Some(crate::legacy_defaults::migrated_allows_keys("demo")))
        );
        assert_eq!(installed.last_protocol_version, None);
        assert_eq!(installed.last_status, None);
        assert_eq!(installed.last_error, None);
        assert_eq!(
            installed.host_compatibility.unwrap().min.as_deref(),
            Some("0.80.0")
        );
    }
}

#[test]
fn an_unchanged_incompatible_release_cannot_succeed_as_a_repair() {
    let current = replacement(
        None,
        PluginToml {
            max_host_version: Some("0.79.0".into()),
            ..PluginToml::default()
        },
    );
    let error = unchanged_install(Some(&current), "v1.0.0", "0.80.0").unwrap_err();
    assert!(
        error.to_string().contains("unchanged and incompatible"),
        "{error:#}"
    );
    assert!(
        !unchanged_install(Some(&current), "v1.0.0", "0.79.0")
            .unwrap()
            .unwrap()
            .changed
    );
    assert!(
        unchanged_install(Some(&current), "v2.0.0", "0.80.0")
            .unwrap()
            .is_none()
    );
}

#[test]
fn local_replacement_preserves_retained_grants_and_plugin_data() {
    let root = tempfile::tempdir().unwrap();
    let options = options(root.path());
    let archive = archive(root.path(), "name = 'demo'\n");
    let mut current = replacement(None, PluginToml::default());
    current.install_path = options.install_root.join("demo");
    current.enabled = false;
    current.allowed_surfaces = Some(vec!["web_ui".into()]);
    current.last_protocol_version = Some(999);
    current.last_status = Some("failed".into());
    let store = PluginStore::new(&options.store_root);
    store.save(&current).unwrap();
    let data = options.store_root.join("demo/data");
    fs::create_dir_all(&data).unwrap();
    fs::write(data.join("retained"), b"operator data").unwrap();

    let installed = install_plugin_archive("demo", "v2.0.0", &archive, &options, &mut |_| {})
        .unwrap()
        .metadata;

    assert!(!installed.enabled);
    assert_eq!(installed.allowed_surfaces, current.allowed_surfaces);
    assert_eq!(installed.last_protocol_version, None);
    assert_eq!(installed.last_status, None);
    assert_eq!(fs::read(data.join("retained")).unwrap(), b"operator data");
    assert_eq!(store.load("demo").unwrap(), installed);
}
