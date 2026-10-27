use super::check::{
    CHECK_GRAMMAR, INSTALL_RECORD, NOT_BUNDLED_LINE, installed_defaults, not_installed,
    placed_archives, sandbox_env,
};
// The stand-in host is a shell script, so the tests that run it are Unix-only.
#[cfg(unix)]
use super::check::{PLUGINS, check_bundle};
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
    assert_eq!(capsules.repository, "Mesh-LLM/capsules-plugin");
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
        (
            format!(
                "#[cfg(unix)]\n{}",
                notes("notes", "example/notes", &linux_pin(DIGEST))
            ),
            "`DEFAULT_PLUGINS` carries an attribute (#[cfg])",
        ),
        (
            fixture("#[cfg_attr(test, allow(unused))] name: \"notes\""),
            "an entry's field carries an attribute (#[cfg_attr])",
        ),
        (
            notes("notes", "example/notes", &linux_pin(DIGEST)).replace(
                "= &[DefaultPlugin",
                "= &[#[cfg(target_os = \"linux\")] DefaultPlugin",
            ),
            "an entry carries an attribute (#[cfg])",
        ),
        (
            format!(
                "{}\n{}",
                notes("notes", "example/notes", &linux_pin(DIGEST)),
                notes("other", "example/other", &linux_pin(DIGEST))
            ),
            "more than one `DEFAULT_PLUGINS`",
        ),
        (
            fixture(&format!(
                "name: \"notes\", version: \"1.2.0\", repository: \"example/notes\", \
                 sha256: &[{}], ..BASE",
                linux_pin(DIGEST)
            )),
            "takes fields from another",
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

/// A stand-in release host for `install-defaults`, as a host would answer:
/// - `installs`: its pins match, so it records an install for every archive
///   in `plugins/`;
/// - `ignores`: it pins nothing by those names and records none;
/// - `unsupported`: it pins nothing for this platform and says so;
/// - `unbundled`: it pins a default for this platform that the bundle lacks;
/// - `refuses`: it pins another digest and fails.
///
/// Every behaviour first insists on the sandbox: no bundled-directory
/// override, and the internal-helper marker that keeps analytics off.
#[cfg(unix)]
fn stand_in_host(dir: &Path, behaviour: &str) {
    use std::os::unix::fs::PermissionsExt as _;
    let script = match behaviour {
        "installs" => {
            r#"for a in "$(dirname "$0")"/plugins/*.tar.gz; do
  [ -f "$a" ] || continue
  n=$(basename "$a"); mkdir -p "$MESH_LLM_PLUGIN_DIR/${n%%-*}"
  printf '{"downloaded_asset_name":"%s","enabled":true,"default_managed":true}' "$n" \
    > "$MESH_LLM_PLUGIN_DIR/${n%%-*}/plugin-install.json"
done"#
        }
        "ignores" => "exit 0",
        "unsupported" => {
            "echo 'ℹ️  No reviewed notes release for this platform; not installed' >&2"
        }
        "unbundled" => {
            "echo 'ℹ️  Default notes not installed: this install carries no bundled copy of it, \
             and a default plugin is never downloaded' >&2"
        }
        _ => "echo 'the bundled archive is not the reviewed one' >&2; exit 1",
    };
    let host = dir.join("mesh-llm");
    fs::write(
        &host,
        format!(
            "#!/bin/sh\n[ \"$*\" = 'plugins install-defaults' ] || exit 9\n\
             [ -z \"$MESH_LLM_BUNDLED_PLUGINS_DIR\" ] || exit 8\n\
             [ \"$MESH_LLM_INTERNAL_HELPER\" = 1 ] || exit 7\n{script}\n"
        ),
    )
    .unwrap();
    fs::set_permissions(&host, fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
fn check(dir: &Path, behaviour: &str) -> CheckReport {
    stand_in_host(dir, behaviour);
    check_bundle(dir).unwrap()
}

#[cfg(unix)]
fn says(report: &CheckReport, text: &str) -> bool {
    report.stdout.contains(text) || report.stderr.contains(text)
}

/// With no archive placed, the bundle passes only when the host pins nothing
/// for this platform; a default it pins but the bundle lacks fails.
#[cfg(unix)]
#[test]
fn a_bundle_with_no_archive_passes_only_when_the_host_pins_nothing_here() {
    let dir = tempfile::tempdir().unwrap();
    for plugins_dir in [false, true] {
        if plugins_dir {
            fs::create_dir(dir.path().join(PLUGINS)).unwrap();
        }
        for (behaviour, code, text) in [
            ("ignores", 0, "pins no default plugin for this platform"),
            ("unsupported", 0, "pins no default plugin for this platform"),
            ("installs", 0, "pins no default plugin for this platform"),
            ("unbundled", 1, "that the bundle does not carry"),
        ] {
            let report = check(dir.path(), behaviour);
            assert_eq!(report.code, code, "{behaviour}: {report:?}");
            assert!(says(&report, text), "{behaviour}: {report:?}");
        }
    }
}

/// With an archive placed, the host must install it, and must report no
/// other pinned default missing.
#[cfg(unix)]
#[test]
fn a_placed_archive_passes_only_when_the_host_installs_it() {
    let dir = tempfile::tempdir().unwrap();
    let plugins = dir.path().join(PLUGINS);
    fs::create_dir(&plugins).unwrap();
    fs::write(plugins.join(ARCHIVE), b"archive").unwrap();
    for (behaviour, code, text) in [
        ("installs", 0, "installed every bundled plugin"),
        ("ignores", 1, "pins no archive by that name and digest"),
        ("unsupported", 1, "pins no archive by that name and digest"),
        ("unbundled", 1, "that the bundle does not carry"),
        ("refuses", 1, "not the reviewed one"),
    ] {
        let report = check(dir.path(), behaviour);
        assert_eq!(report.code, code, "{behaviour}: {report:?}");
        assert!(says(&report, text), "{behaviour}: {report:?}");
    }
}

/// The check recognises a missing bundled copy by the line the host prints
/// for it; this fails if that line changes in `mesh-llm-commands`.
#[test]
fn the_not_bundled_line_is_the_one_install_defaults_prints() {
    let source = include_str!("../../../../../mesh/crates/mesh-llm-commands/src/plugin.rs");
    let arm = source
        .find("DefaultPluginOutcome::NotBundled => Some(format!(")
        .expect("install-defaults reports NotBundled with its own line");
    let line = &source[arm..source.len().min(arm + 200)];
    assert!(line.contains(NOT_BUNDLED_LINE), "{line}");
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
        ("MESH_LLM_INTERNAL_HELPER", "0"),
        ("MESH_LLM_ANALYTICS", "1"),
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
    assert_eq!(get("MESH_LLM_INTERNAL_HELPER").as_deref(), Some("1"));
    assert_eq!(get("MESH_LLM_ANALYTICS"), None);
}
