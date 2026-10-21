use super::*;

const ARCHIVE: &str = "capsules-0.79.0-x86_64-unknown-linux-gnu.tar.gz";

fn record(store: &Path, name: &str, asset: &str, enabled: bool, managed: bool) {
    let dir = store.join(name);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join(INSTALL_RECORD),
        serde_json::json!({
            "name": name,
            "downloaded_asset_name": asset,
            "enabled": enabled,
            "default_managed": managed,
            "installed_version": "0.79.0",
        })
        .to_string(),
    )
    .unwrap();
}

#[test]
fn capsules_release_commands_are_named_in_their_usage() {
    assert!(ARCHIVE_GRAMMAR.usage.contains("capsules-plugin-archive"));
    assert!(CHECK_GRAMMAR.usage.contains("capsules-bundle-check"));
}

#[test]
fn only_archives_in_plugins_are_placed() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join(ARCHIVE), b"x").unwrap();
    fs::write(dir.path().join(format!("{ARCHIVE}.sha256")), b"x").unwrap();
    fs::create_dir(dir.path().join("nested.tar.gz")).unwrap();
    assert_eq!(placed_archives(dir.path()).unwrap(), [ARCHIVE]);
}

#[test]
fn only_an_enabled_default_counts_as_installed() {
    let store = tempfile::tempdir().unwrap();
    record(store.path(), "capsules", ARCHIVE, true, true);
    record(store.path(), "operator", "operator.tar.gz", true, false);
    record(store.path(), "off", "off.tar.gz", false, true);
    assert_eq!(installed_defaults(store.path()).unwrap(), [ARCHIVE]);
    assert!(
        installed_defaults(&store.path().join("absent"))
            .unwrap()
            .is_empty()
    );
    let placed = [ARCHIVE.to_owned(), "off.tar.gz".to_owned()];
    assert_eq!(
        not_installed(&placed, &installed_defaults(store.path()).unwrap()),
        ["off.tar.gz"]
    );
}

#[test]
fn the_sidecar_sits_beside_the_archive() {
    assert_eq!(
        sidecar(Path::new("out").join(ARCHIVE).as_path()),
        Path::new("out").join(format!("{ARCHIVE}.sha256"))
    );
}

/// A stand-in release host: `install-defaults` records an install for every
/// archive it is given (`installs`), records none (`ignores`), or refuses
/// (`refuses`), as a host whose pin matches, pins nothing, or pins another
/// digest would.
#[cfg(unix)]
fn stand_in_host(dir: &Path, behaviour: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    let script = match behaviour {
        "installs" => {
            r#"[ -z "$MESH_LLM_BUNDLED_PLUGINS_DIR" ] || exit 8
for a in "$(dirname "$0")"/plugins/*.tar.gz; do
  n=$(basename "$a"); mkdir -p "$MESH_LLM_PLUGIN_DIR/${n%%-*}"
  printf '{"downloaded_asset_name":"%s","enabled":true,"default_managed":true}' "$n" \
    > "$MESH_LLM_PLUGIN_DIR/${n%%-*}/plugin-install.json"
done"#
        }
        "ignores" => "exit 0",
        _ => "echo 'the bundled archive is not the reviewed one' >&2; exit 1",
    };
    let host = dir.join(HOST);
    fs::write(
        &host,
        format!(
            "#!/bin/sh\n[ \"$*\" = '--log-format json plugins install-defaults' ] || exit 9\n{script}\n"
        ),
    )
    .unwrap();
    fs::set_permissions(&host, fs::Permissions::from_mode(0o755)).unwrap();
    host
}

#[cfg(unix)]
#[test]
fn the_bundle_check_passes_only_when_the_host_installs_every_archive() {
    let dir = tempfile::tempdir().unwrap();
    let plugins = dir.path().join(PLUGINS);
    fs::create_dir(&plugins).unwrap();
    stand_in_host(dir.path(), "installs");
    let empty = check_bundle(dir.path()).unwrap();
    assert_eq!(empty.code, 1, "an empty plugins/ is not a bundle");
    fs::write(plugins.join(ARCHIVE), b"archive").unwrap();
    for (behaviour, code, says) in [
        ("installs", 0, "installed every bundled plugin"),
        ("ignores", 1, "pins no archive by that name and digest"),
        ("refuses", 1, "not the reviewed one"),
    ] {
        stand_in_host(dir.path(), behaviour);
        let report = check_bundle(dir.path()).unwrap();
        assert_eq!(report.code, code, "{behaviour}: {report:?}");
        assert!(
            report.stdout.contains(says) || report.stderr.contains(says),
            "{behaviour}: {report:?}"
        );
    }
}

/// The host sees no `MESH_LLM_*` setting of the machine running the check:
/// a config naming its own `capsules` would make the default operator
/// managed and install nothing.
#[test]
fn the_host_runs_with_scratch_settings_only() {
    let parent = [
        ("PATH", "/usr/bin"),
        ("HOME", "/home/operator"),
        ("MESH_LLM_CONFIG", "/home/operator/.mesh-llm/config.toml"),
        ("MESH_LLM_NO_DEFAULT_PLUGINS", "1"),
        ("MESH_LLM_PLUGIN_DIR", "/home/operator/plugins"),
        ("MESH_LLM_BUNDLED_PLUGINS_DIR", "/elsewhere"),
    ]
    .map(|(k, v)| (OsString::from(k), OsString::from(v)));
    let env = sandbox_env(
        parent.into_iter(),
        Path::new("/scratch/home"),
        Path::new("/scratch/store"),
    );
    let get = |key: &str| {
        let values: Vec<_> = env
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v)
            .collect();
        assert!(values.len() <= 1, "{key} set twice");
        values.first().map(|v| v.to_string_lossy().into_owned())
    };
    assert_eq!(get("PATH").as_deref(), Some("/usr/bin"));
    assert_eq!(get("HOME").as_deref(), Some("/scratch/home"));
    assert_eq!(
        get("MESH_LLM_CONFIG").as_deref(),
        Some("/scratch/home/.mesh-llm/config.toml")
    );
    assert_eq!(
        get("MESH_LLM_PLUGIN_DIR").as_deref(),
        Some("/scratch/store")
    );
    assert_eq!(get("MESH_LLM_BUNDLED_PLUGINS_DIR"), None);
    assert_eq!(get("MESH_LLM_NO_DEFAULT_PLUGINS"), None);
}

#[test]
fn the_workspace_version_is_read_from_the_root_manifest() {
    assert_eq!(
        workspace_version("[workspace.package]\nversion = \"0.79.0-rc.1\"\n").unwrap(),
        "0.79.0-rc.1"
    );
    assert!(workspace_version("[package]\nversion = \"1.0.0\"\n").is_err());
    let root = include_str!("../../../../../Cargo.toml");
    let plugin = include_str!("../../../../../mesh/crates/mesh-llm-capsules-plugin/plugin.toml");
    assert_eq!(
        workspace_version(root).unwrap(),
        plugin_version(plugin).unwrap(),
        "plugin.toml must carry mesh-llm's version"
    );
}

/// A prebuilt Evidence page becomes the plugin's whole `bundle/`; anything
/// but the built entry is refused.
#[test]
fn a_prebuilt_web_bundle_replaces_the_bundle_directory() {
    let dir = tempfile::tempdir().unwrap();
    let plugin = dir.path().join("plugin");
    fs::create_dir_all(plugin.join("bundle")).unwrap();
    fs::write(plugin.join("bundle/stale.js"), b"old").unwrap();
    let built = dir.path().join(BUNDLE_ENTRY);
    fs::write(&built, b"page").unwrap();
    place_web_bundle(&plugin, &built).unwrap();
    let mut entries: Vec<_> = fs::read_dir(plugin.join("bundle"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    entries.sort();
    assert_eq!(entries, [BUNDLE_ENTRY]);
    assert_eq!(
        fs::read(plugin.join("bundle").join(BUNDLE_ENTRY)).unwrap(),
        b"page"
    );
    let other = dir.path().join("other.js");
    fs::write(&other, b"x").unwrap();
    assert!(place_web_bundle(&plugin, &other).is_err());
}
