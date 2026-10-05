//! A dependency-free Rust editing exercise with independently owned verification.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, Readiness, Value,
};
use crate::{command::DynResult, product::digest::file_sha256};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

const INITIAL: &str = r#"use std::path::Path;

pub fn parse_codeword(path: &Path) -> String {
    let _ = path;
    todo!("ci smoke fixture")
}

pub fn prime_sum_from_matrix(path: &Path) -> i64 {
    let _ = path;
    todo!("ci smoke fixture")
}
"#;
const SIGNAL: &str = "# Runtime Signal\n\nCODEWORD=signal-7429\n\nQuestion seed:\nWhich file names contain the word signal?\n";
const MATRIX: &str = "checksum: FS-319-DELTA\nnumbers: 2 3 4 5\nhint: the prime sum is computed from the numbers line\n";
const VISIBLE: &str = r#"#[path = "../src/smoke_calc.rs"]
mod smoke_calc;

#[test]
fn parse_codeword() {
    assert_eq!(smoke_calc::parse_codeword(std::path::Path::new("facts/signal.md")), "signal-7429");
}

#[test]
fn prime_sum_from_matrix() {
    assert_eq!(smoke_calc::prime_sum_from_matrix(std::path::Path::new("src/matrix.txt")), 10);
}
"#;

pub(super) fn setup(root: &Path) -> DynResult<()> {
    for directory in ["facts", "src", "notes", "tests"] {
        fs::create_dir_all(root.join(directory))?;
    }
    // Refuse to overwrite an existing exercise or a caller's edited implementation.
    let visible_recipe = format!(
        "set shell := [\"bash\", \"-eu\", \"-o\", \"pipefail\", \"-c\"]\nset windows-shell := [\"bash\", \"-eu\", \"-o\", \"pipefail\", \"-c\"]\n\ntest:\n    rustc --edition=2024 --test tests/smoke_calc.rs -o .smoke-tests{}\n    ./.smoke-tests{}\n",
        std::env::consts::EXE_SUFFIX,
        std::env::consts::EXE_SUFFIX
    );
    for (relative, contents) in [
        (
            "README.md",
            "# OpenCode Smoke Fixture\n\nAnswer from files on disk. Implement the two public functions in src/smoke_calc.rs using only the Rust standard library. Run `just test`.\n",
        ),
        ("facts/signal.md", SIGNAL),
        ("src/matrix.txt", MATRIX),
        ("src/smoke_calc.rs", INITIAL),
        ("tests/smoke_calc.rs", VISIBLE),
        (
            "notes/manifest.txt",
            "tracked files:\n- facts/signal.md\n- src/matrix.txt\n- src/smoke_calc.rs\n- tests/smoke_calc.rs\n- README.md\n- Justfile\n",
        ),
        ("Justfile", visible_recipe.as_str()),
    ] {
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(relative))?;
        file.write_all(contents.as_bytes())?;
    }
    let digest = file_sha256(&root.join("src/smoke_calc.rs")).map_err(|failure| failure.error)?;
    println!("{digest}");
    Ok(())
}

fn source(root: &Path, initial: &str) -> DynResult<std::path::PathBuf> {
    if initial.len() != 64 || !initial.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("coding fixture initial digest must be a SHA-256 hex digest".into());
    }
    let path = root.join("src/smoke_calc.rs");
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.is_file() || metadata.len() > 8 * 1024 * 1024 {
        return Err("coding implementation must be a regular file at most 8 MiB".into());
    }
    let digest = file_sha256(&path).map_err(|failure| failure.error)?;
    if digest.eq_ignore_ascii_case(initial) {
        return Err("OpenCode left src/smoke_calc.rs unchanged".into());
    }
    let contents = fs::read_to_string(&path)?;
    for marker in ["todo!", "unimplemented!", "ci smoke fixture"] {
        if contents.contains(marker) {
            return Err(
                format!("OpenCode left placeholder marker in src/smoke_calc.rs: {marker}").into(),
            );
        }
    }
    Ok(path.canonicalize()?)
}

fn harness(source: &Path, visible: &Path, hidden: &Path) -> DynResult<String> {
    // Rust debug escaping produces a quoted Rust string literal, including unusual path characters.
    let literal = |path: &Path| -> DynResult<String> {
        Ok(format!(
            "{:?}",
            path.to_str().ok_or("fixture path must be UTF-8")?
        ))
    };
    Ok(format!(
        r#"#[path = {source}]
mod smoke_calc;

#[test]
fn visible_inputs() {{
    let root = std::path::Path::new({visible});
    assert_eq!(smoke_calc::parse_codeword(&root.join("signal.md")), "signal-7429");
    assert_eq!(smoke_calc::prime_sum_from_matrix(&root.join("matrix.txt")), 10);
}}

#[test]
fn hidden_inputs() {{
    let root = std::path::Path::new({hidden});
    assert_eq!(smoke_calc::parse_codeword(&root.join("other_signal.md")), "hidden-8842");
    assert_eq!(smoke_calc::prime_sum_from_matrix(&root.join("other_matrix.txt")), 31);
}}
"#,
        source = literal(source)?,
        visible = literal(visible)?,
        hidden = literal(hidden)?
    ))
}

pub(super) fn verify(root: &Path, initial: &str, just: &Path, rustc: &Path) -> DynResult<()> {
    if !just.is_absolute() || !rustc.is_absolute() {
        return Err("coding fixture Just and rustc must be absolute executable paths".into());
    }
    let source = source(&root.canonicalize()?, initial)?;
    let temporary = tempfile::tempdir()?;
    let state = temporary.path().canonicalize()?;
    let visible = state.join("visible");
    let hidden = state.join("hidden");
    fs::create_dir(&visible)?;
    fs::create_dir(&hidden)?;
    fs::write(visible.join("signal.md"), SIGNAL)?;
    fs::write(visible.join("matrix.txt"), MATRIX)?;
    fs::write(
        hidden.join("other_signal.md"),
        "# Hidden\n\nCODEWORD=hidden-8842\n",
    )?;
    fs::write(
        hidden.join("other_matrix.txt"),
        "checksum: hidden\nnumbers: 6 7 8 9 10 11 12 13\n",
    )?;
    fs::write(
        state.join("verification.rs"),
        harness(&source, &visible, &hidden)?,
    )?;
    // The verifier owns the recipe and tests; model edits to the project tests do not qualify.
    fs::write(
        state.join("Justfile"),
        format!(
            "set shell := [\"bash\", \"-eu\", \"-o\", \"pipefail\", \"-c\"]\nset windows-shell := [\"bash\", \"-eu\", \"-o\", \"pipefail\", \"-c\"]\n\nverify:\n    \"$FIXTURE_RUSTC\" --edition=2024 --test verification.rs -o verification{}\n    ./verification{} --test-threads=1\n",
            std::env::consts::EXE_SUFFIX,
            std::env::consts::EXE_SUFFIX
        ),
    )?;
    let mut environment: BTreeMap<_, _> = [
        "PATH",
        "RUSTUP_HOME",
        "HOME",
        "USERPROFILE",
        "CARGO_HOME",
        "RUSTUP_TOOLCHAIN",
        "SYSTEMROOT",
        "SystemRoot",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
    .collect();
    environment.insert(
        "FIXTURE_RUSTC".into(),
        Value::Public(rustc.to_path_buf().into_os_string()),
    );
    let report = process::supervise(
        &ProcessSpec {
            executable: just.canonicalize()?,
            arguments: vec![Value::Public("verify".into())],
            cwd: state,
            environment,
        },
        &Limits {
            execution: Duration::from_secs(60),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        Default::default(),
    )?;
    if report.outcome != Outcome::Exited || !report.success() {
        return Err(format!(
            "coding fixture verification failed: {:?}; stderr={}",
            report.outcome,
            String::from_utf8_lossy(&report.stderr.bytes_retained)
        )
        .into());
    }
    println!("Visible and hidden implementation validation passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOLUTION: &str = r#"use std::path::Path;
pub fn parse_codeword(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap().lines()
        .find_map(|line| line.strip_prefix("CODEWORD=").map(str::to_owned)).unwrap()
}
pub fn prime_sum_from_matrix(path: &Path) -> i64 {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines().find_map(|line| line.strip_prefix("numbers:")).unwrap()
        .split_whitespace().map(|word| word.parse::<i64>().unwrap())
        .filter(|&n| n >= 2 && !(2..n).any(|d| n % d == 0)).sum()
}
"#;

    fn tool(name: &str) -> std::path::PathBuf {
        let path = std::env::split_paths(&std::env::var_os("PATH").unwrap())
            .map(|root| root.join(format!("{name}{}", std::env::consts::EXE_SUFFIX)))
            .find(|path| path.is_file())
            .unwrap_or_else(|| panic!("coding fixture test requires {name}"));
        if path.is_absolute() {
            path
        } else {
            std::env::current_dir().unwrap().join(path)
        }
    }

    fn fixture() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        setup(root.path()).unwrap();
        root
    }

    #[test]
    fn setup_preserves_facts_and_refuses_to_overwrite_an_exercise() {
        let root = fixture();
        assert_eq!(
            fs::read_to_string(root.path().join("facts/signal.md")).unwrap(),
            SIGNAL
        );
        assert_eq!(
            fs::read_to_string(root.path().join("src/matrix.txt")).unwrap(),
            MATRIX
        );
        assert!(setup(root.path()).is_err());
        assert_eq!(
            fs::read_to_string(root.path().join("src/smoke_calc.rs")).unwrap(),
            INITIAL
        );
    }

    #[test]
    fn changed_implementation_must_remove_every_placeholder_marker() {
        let root = fixture();
        let path = root.path().join("src/smoke_calc.rs");
        let initial = file_sha256(&path).map_err(|failure| failure.error).unwrap();
        assert!(source(root.path(), &initial).is_err());
        for marker in ["todo!", "unimplemented!", "ci smoke fixture"] {
            fs::write(&path, format!("// {marker}\n")).unwrap();
            assert!(source(root.path(), &initial).is_err());
        }
        fs::write(&path, "// edited implementation\n").unwrap();
        assert!(source(root.path(), &initial).is_ok());
        assert!(source(root.path(), "bad-digest").is_err());
    }

    #[test]
    fn verifier_harness_checks_both_original_and_unseen_inputs() {
        let text = harness(
            Path::new("/source with spaces/smoke_calc.rs"),
            Path::new("/visible"),
            Path::new("/hidden"),
        )
        .unwrap();
        for required in [
            "signal-7429",
            "hidden-8842",
            "other_signal.md",
            "other_matrix.txt",
            "), 10)",
            "), 31)",
        ] {
            assert!(text.contains(required), "missing {required}");
        }
    }

    #[test]
    fn actual_rust_implementation_passes_independent_visible_and_hidden_checks() {
        let root = fixture();
        let path = root.path().join("src/smoke_calc.rs");
        let initial = file_sha256(&path).map_err(|failure| failure.error).unwrap();
        fs::write(path, SOLUTION).unwrap();
        // Tampering with visible tests cannot replace the controller-owned harness.
        fs::write(root.path().join("tests/smoke_calc.rs"), "").unwrap();
        verify(root.path(), &initial, &tool("just"), &tool("rustc")).unwrap();
    }

    #[test]
    fn visible_answer_constants_fail_actual_hidden_verification() {
        let root = fixture();
        let path = root.path().join("src/smoke_calc.rs");
        let initial = file_sha256(&path).map_err(|failure| failure.error).unwrap();
        fs::write(path, "use std::path::Path;\npub fn parse_codeword(_: &Path) -> String { \"signal-7429\".into() }\npub fn prime_sum_from_matrix(_: &Path) -> i64 { 10 }\n").unwrap();
        assert!(verify(root.path(), &initial, &tool("just"), &tool("rustc")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn compiler_multicall_shim_keeps_its_selected_invocation_name() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let root = fixture();
        let implementation = root.path().join("src/smoke_calc.rs");
        let initial = file_sha256(&implementation)
            .map_err(|failure| failure.error)
            .unwrap();
        fs::write(implementation, SOLUTION).unwrap();
        let compiler = tool("rustc");
        let quoted = format!("'{}'", compiler.to_str().unwrap().replace('\'', "'\\''"));
        let dispatcher = root.path().join("compiler-multicall");
        fs::write(
            &dispatcher,
            format!("#!/bin/sh\n[ \"${{0##*/}}\" = rustc ] || exit 93\nexec {quoted} \"$@\"\n"),
        )
        .unwrap();
        fs::set_permissions(&dispatcher, fs::Permissions::from_mode(0o700)).unwrap();
        let shim = root.path().join("rustc");
        symlink(&dispatcher, &shim).unwrap();
        verify(root.path(), &initial, &tool("just"), &shim).unwrap();
    }
}
