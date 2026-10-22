use super::check::{
    CHECK_GRAMMAR, INSTALL_RECORD, PLUGINS, check_bundle, installed_defaults, not_installed,
    placed_archives, sandbox_env,
};
use super::pins::{PinnedDefault, parse};
use super::*;
use std::ffi::OsString;
use std::fs;

const DEFAULTS: &str =
    include_str!("../../../../../mesh/crates/mesh-llm-plugin-manager/src/defaults.rs");
const LINUX: &str = "x86_64-unknown-linux-gnu";
const DIGEST: &str = "abababababababababababababababababababababababababababababababab";

/// A `DEFAULT_PLUGINS` with one `notes` entry whose fields are `fields`.
fn fixture(fields: &str) -> String {
    format!("pub const DEFAULT_PLUGINS: &[DefaultPlugin] = &[DefaultPlugin {{ {fields} }}];")
}

fn notes(name: &str, repository: &str, pins: &str) -> String {
    fixture(&format!(
        "name: {name:?}, version: \"1.2.0\", repository: {repository:?}, sha256: &[{pins}], \
         allows: &[Surface::WebUi], capabilities: &[]"
    ))
}

fn linux_pin(digest: &str) -> String {
    format!("({LINUX:?}, {digest:?})")
}

#[test]
fn the_commands_are_named_in_their_usage() {
    assert!(LIST_GRAMMAR.usage.contains("repository bundled-plugins "));
    assert!(
        CHECK_GRAMMAR
            .usage
            .contains("repository bundled-plugins-check")
    );
}

/// The real pins parse, and every pinned target yields the archive on the
/// plugin's own release.
#[test]
fn the_shipped_defaults_parse() {
    let defaults = parse(DEFAULTS).unwrap();
    assert!(!defaults.is_empty());
    for default in &defaults {
        for (triple, digest) in &default.sha256 {
            let line = default.bundle_line(triple).unwrap();
            assert_eq!(
                line,
                format!(
                    "{} {} {digest} https://github.com/{}/releases/download/v{}/{}-{}-{triple}.tar.gz\n",
                    default.name,
                    default.version,
                    default.repository,
                    default.version,
                    default.name,
                    default.version
                )
            );
        }
        assert_eq!(default.bundle_line("x86_64-pc-windows-msvc"), None);
    }
    let capsules = defaults.iter().find(|d| d.name == "capsules").unwrap();
    assert_eq!(capsules.repository, "Mesh-LLM/capsules");
}

#[test]
fn a_well_formed_entry_lists_only_its_pinned_targets() {
    let source = notes("notes", "example/notes", &linux_pin(DIGEST));
    assert_eq!(
        parse(&source).unwrap(),
        [PinnedDefault {
            name: "notes".into(),
            version: "1.2.0".into(),
            repository: "example/notes".into(),
            sha256: vec![(LINUX.into(), DIGEST.into())],
        }]
    );
    let path = Path::new("defaults.rs");
    let report = list_report(&source, path, LINUX);
    assert_eq!(report.code, 0);
    assert_eq!(
        report.stdout,
        format!(
            "notes 1.2.0 {DIGEST} https://github.com/example/notes/releases/download/v1.2.0/\
             notes-1.2.0-{LINUX}.tar.gz\n"
        )
    );
    let none = list_report(&source, path, "aarch64-apple-darwin");
    assert_eq!((none.code, none.stdout.as_str()), (0, ""));
}

#[test]
fn a_malformed_list_fails_and_says_why() {
    let cases = [
        (
            "const OTHER: u8 = 0;".to_owned(),
            "no `const DEFAULT_PLUGINS`",
        ),
        (
            fixture("name: \"notes\", version: \"1.2.0\", sha256: &[]"),
            "no `repository`",
        ),
        (
            notes("notes", "example/notes", &linux_pin("ABAB")),
            "not a lowercase hex SHA-256",
        ),
        (
            notes("notes", "example/notes", &linux_pin(&DIGEST.to_uppercase())),
            "not a lowercase hex SHA-256",
        ),
        (
            notes("notes", "notes", &linux_pin(DIGEST)),
            "not a GitHub owner/name",
        ),
        (
            notes("notes", "example/../notes", &linux_pin(DIGEST)),
            "not a GitHub owner/name",
        ),
        (
            notes("../notes", "example/notes", &linux_pin(DIGEST)),
            "not a plugin name",
        ),
        (
            notes(
                "notes",
                "example/notes",
                &format!("{}, {}", linux_pin(DIGEST), linux_pin(DIGEST)),
            ),
            "pinned twice",
        ),
        (
            notes("notes", "example/notes", &format!("({LINUX:?})")),
            "not a (target, digest) pair",
        ),
        (
            fixture("name: NAME, version: \"1.2.0\""),
            "`name` is not a string literal",
        ),
    ];
    for (source, says) in cases {
        let report = list_report(&source, Path::new("defaults.rs"), LINUX);
        assert_eq!(report.code, 1, "{source}");
        assert!(report.stderr.contains(says), "{source}: {}", report.stderr);
        assert!(report.stdout.is_empty());
    }
}

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
            "installed_version": "1.2.0",
        })
        .to_string(),
    )
    .unwrap();
}

const ARCHIVE: &str = "notes-1.2.0-x86_64-unknown-linux-gnu.tar.gz";

#[test]
fn only_archives_in_plugins_are_placed() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join(ARCHIVE), b"x").unwrap();
    fs::write(dir.path().join(format!("{ARCHIVE}.sha256")), b"x").unwrap();
    fs::create_dir(dir.path().join("nested.tar.gz")).unwrap();
    assert_eq!(placed_archives(dir.path()).unwrap(), [ARCHIVE]);
    assert!(
        placed_archives(&dir.path().join("absent"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn only_an_enabled_default_counts_as_installed() {
    let store = tempfile::tempdir().unwrap();
    record(store.path(), "notes", ARCHIVE, true, true);
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

/// A stand-in release host: `install-defaults` records an install for every
/// archive it is given (`installs`), records none (`ignores`), or refuses
/// (`refuses`), as a host whose pin matches, pins nothing, or pins another
/// digest would.
#[cfg(unix)]
fn stand_in_host(dir: &Path, behaviour: &str) {
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
    let host = dir.join("mesh-llm");
    fs::write(
        &host,
        format!(
            "#!/bin/sh\n[ \"$*\" = '--log-format json plugins install-defaults' ] || exit 9\n{script}\n"
        ),
    )
    .unwrap();
    fs::set_permissions(&host, fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
#[test]
fn the_bundle_check_passes_only_when_the_host_installs_every_archive() {
    let dir = tempfile::tempdir().unwrap();
    stand_in_host(dir.path(), "installs");
    let absent = check_bundle(dir.path()).unwrap();
    assert_eq!(absent.code, 1, "a bundle with no plugins/ bundles nothing");
    let plugins = dir.path().join(PLUGINS);
    fs::create_dir(&plugins).unwrap();
    let empty = check_bundle(dir.path()).unwrap();
    assert_eq!(empty.code, 1, "an empty plugins/ bundles nothing");
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
/// a config naming its own copy of a default would make the default operator
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
