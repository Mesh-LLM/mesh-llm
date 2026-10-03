//! Native Just owns imported recipe parsing; these tests own default-shell portability.
//! Actual execution is restricted to finite synthetic recipes, never maintained build bodies.
use super::just_recipes::{Justfile, dump};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn default_bodies(parsed: &Justfile) -> BTreeMap<&str, String> {
    parsed
        .recipes
        .iter()
        .filter_map(|(name, recipe)| {
            if recipe
                .attributes
                .iter()
                .any(|attribute| attribute == "windows")
            {
                return None;
            }
            let lines = recipe
                .body
                .iter()
                .map(|line| {
                    line.iter()
                        .map(|part| part.as_str().unwrap_or("{{native interpolation}}"))
                        .collect::<String>()
                })
                .collect::<Vec<_>>();
            let first = lines.iter().find(|line| !line.trim().is_empty())?;
            if first.trim_start().starts_with("#!")
                || recipe.attributes.iter().any(|a| a == "script")
            {
                return None;
            }
            Some((
                name.as_str(),
                lines
                    .into_iter()
                    .filter(|line| !line.trim_start().starts_with('#'))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        })
        .collect()
}
// Finite direct syntax obligations of the original guard, not a shell parser or VM.
fn bash_construct(body: &str) -> Option<&'static str> {
    for (token, reason) in [
        ("[[", "bash conditional"),
        ("<<<", "here-string"),
        ("&>", "combined redirection"),
    ] {
        if body.contains(token) {
            return Some(reason);
        }
    }
    let words = body
        .split_whitespace()
        .map(|word| word.trim_matches([';', '(', ')', '{', '}']))
        .collect::<Vec<_>>();
    if words.contains(&"pipefail") {
        return Some("pipefail");
    }
    if words.windows(2).any(|w| w == ["echo", "-e"]) {
        return Some("echo escape flag");
    }
    if words.windows(2).any(|w| {
        w[0] == "function"
            && w[1]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    }) {
        return Some("function keyword");
    }
    // Other array names have the same literal ${name[@]} shape; arithmetic stays admitted.
    if body.split("${").skip(1).any(|s| {
        s.split('}')
            .next()
            .is_some_and(|part| part.ends_with("[@]") || part.ends_with("[*]"))
    }) {
        return Some("array expansion");
    }
    None
}
fn pid_variable(body: &str) -> bool {
    body.split("$$").skip(1).any(|rest| {
        rest.chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '{')
    })
}
struct Fixture(tempfile::TempDir);
impl Fixture {
    fn new(source: &str) -> Self {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("Justfile"), source).unwrap();
        Self(temp)
    }
    fn root(&self) -> &Path {
        self.0.path()
    }
    fn parsed(&self) -> Justfile {
        dump(self.root()).unwrap()
    }
    fn execute(&self, name: &str) -> String {
        let just = std::env::split_paths(&std::env::var_os("PATH").unwrap())
            .map(|p| p.join("just"))
            .find(|p| p.is_file() && fs::metadata(p).unwrap().permissions().mode() & 0o111 != 0)
            .expect("native Just component required")
            .canonicalize()
            .unwrap();
        let environment: BTreeMap<_, _> = [
            ("PATH", std::env::var_os("PATH").unwrap()),
            ("HOME", self.root().as_os_str().to_owned()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v)))
        .collect();
        let r = process::supervise_raw(
            &ProcessSpec {
                executable: just,
                cwd: self.root().to_owned(),
                environment,
                arguments: ["--justfile", "Justfile", name]
                    .into_iter()
                    .map(|s| Value::Public(s.into()))
                    .collect(),
            },
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(r.process.failure.is_none(), "{:?}", r.process);
        assert!(r.process.cleanup.complete);
        assert_eq!(
            r.process.status.unwrap().code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(r.stderr.unwrap().as_bytes())
        );
        String::from_utf8(r.stdout.unwrap().as_bytes().to_vec()).unwrap()
    }
}
#[test]
fn native_just_shell_portability_current_imported_graph_preserves_default_and_script_ownership() {
    let parsed = dump(&repo()).unwrap();
    let bodies = default_bodies(&parsed);
    assert!(parsed.recipes.contains_key("release-build-cuda"));
    assert!(
        !bodies.contains_key("release-build-cuda"),
        "bash script treated as default shell"
    );
    assert!(
        bodies.contains_key("llama-prepare"),
        "plain imported recipe disappeared"
    );
    let offenses = bodies
        .iter()
        .filter_map(|(name, body)| bash_construct(body).map(|reason| format!("{name}:{reason}")))
        .collect::<Vec<_>>();
    assert!(
        offenses.is_empty(),
        "default-shell recipe portability:{offenses:?}"
    );
    // Windows attributes remain a separate platform boundary; no broad recipe-name exemption.
    let pid = bodies
        .iter()
        .filter(|(_, body)| pid_variable(body))
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    assert!(
        pid.is_empty(),
        "default-shell recipe PID variable misuse:{pid:?}"
    );
}
#[test]
fn native_just_shell_portability_direct_bash_constructs_require_explicit_script_interpreter() {
    for body in [
        "[[ x == x ]]",
        "cat <<< x",
        "set -o pipefail",
        "echo -e x",
        "printf '%s' \"${arbitrary[@]}\"",
        "function action() { true; }",
        "true &> out",
    ] {
        let source = format!("plain:\n    {body}\nscript:\n    #!/usr/bin/env bash\n    {body}\n");
        let f = Fixture::new(&source);
        let parsed = f.parsed();
        let bodies = default_bodies(&parsed);
        assert!(bash_construct(&bodies["plain"]).is_some(), "{body}");
        assert!(!bodies.contains_key("script"));
    }
    let f = Fixture::new(
        "script:\n    #!/usr/bin/env bash\n    value=retained\n    [[ \"$value\" == retained ]] && printf '%s\\n' \"$value\"\n",
    );
    assert_eq!(f.execute("script"), "retained\n");
}
#[test]
fn native_just_shell_portability_pid_variable_misuse_is_not_just_escaping() {
    for body in [
        "printf '%s' \"$$value\"",
        "printf '%s' \"$${value}\"",
        "printf '%s' \"$$_value\"",
    ] {
        let f = Fixture::new(&format!("bad:\n    {body}\n"));
        assert!(pid_variable(&default_bodies(&f.parsed())["bad"]));
    }
    let f = Fixture::new(
        "bad:\n    value=retained; printf '%s\\n' \"$$value\"\ngood:\n    value=retained; printf '%s\\n' \"$value\"\n",
    );
    let bad = f.execute("bad");
    assert_ne!(bad.trim(), "retained");
    assert!(
        bad.trim()
            .strip_suffix("value")
            .is_some_and(|prefix| !prefix.is_empty() && prefix.bytes().all(|b| b.is_ascii_digit()))
    );
    assert_eq!(f.execute("good"), "retained\n");
}
#[test]
fn native_just_shell_portability_posix_arithmetic_remains_default_and_executes() {
    let f = Fixture::new("arithmetic:\n    value=41; printf '%s\\n' \"$((value + 1))\"\n");
    let parsed = f.parsed();
    let bodies = default_bodies(&parsed);
    assert!(bodies.contains_key("arithmetic"));
    assert!(bash_construct(&bodies["arithmetic"]).is_none());
    assert!(!pid_variable(&bodies["arithmetic"]));
    assert_eq!(f.execute("arithmetic"), "42\n");
}
