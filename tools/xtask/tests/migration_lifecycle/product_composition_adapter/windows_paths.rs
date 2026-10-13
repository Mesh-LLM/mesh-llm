//! Actual Bash path conversion over finite cygpath boundaries, not Windows CI.
use super::{Fixture, executable, snapshot};
use std::{collections::BTreeMap, fs};
#[test]
fn product_composer_converts_crlf_canonical_inputs_and_publishes_workflow_paths() {
    let fixture = Fixture::new("1.2.3");
    executable(
        &fixture.root.join("bin/automation"),
        r#"#!/bin/sh
printf '%s %s\n' "$1" "$2" >> "$PRODUCT_EVENTS"
if [ "$1" = product ] && [ "$2" = canonical-inputs ]; then
  "$PRODUCT_REAL_XTASK" "$@" | while IFS= read -r path; do printf 'C:%s\r\n' "$path"; done
else
  exec "$PRODUCT_REAL_XTASK" "$@"
fi
"#,
    );
    executable(
        &fixture.root.join("bin/cygpath"),
        r#"#!/bin/sh
[ "$#" -eq 2 ] || exit 98
cr=$(printf '\r')
case "$2" in *"$cr"*) exit 97;; esac
case "$1" in
  -u) case "$2" in C:/*) printf '%s\n' "${2#C:}";; *) exit 98;; esac;;
  -m) case "$2" in /*) printf 'C:%s\n' "$2";; *) exit 98;; esac;;
  *) exit 98;;
esac
"#,
    );
    let before = snapshot(&fixture.root.join("inputs"));
    let output = format!("C:{}\r", fixture.root.join("github-output").display());
    let report = fixture.run_inputs("product", "", &[("GITHUB_OUTPUT", output)]);
    fixture.accepted(&report, &before);
    let text = fs::read_to_string(fixture.root.join("github-output")).unwrap();
    let fields = text
        .lines()
        .skip(1)
        .map(|line| line.split_once('=').unwrap())
        .collect::<BTreeMap<_, _>>();
    for (key, path) in [
        ("product_dir", "product"),
        ("binary_path", "product/mesh-llm"),
        ("runtime_root", "product/native-runtimes"),
        ("runtime_dir", "product/native-runtimes/runtime"),
        ("archive_path", "product.tar.gz"),
    ] {
        assert_eq!(
            fields.get(key).copied(),
            Some(format!("C:{}", fixture.root.join(path).display()).as_str())
        );
    }
    assert_eq!(fields.len(), 5);
    assert!(!text.contains('\r'));
}
