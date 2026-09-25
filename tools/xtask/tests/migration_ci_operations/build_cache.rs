//! `ci-ops build-cache` against `scripts/manage-build-cache.py`. Each case
//! builds a fake cache tree in a temporary workspace with stub `ps` and
//! `just` executables first on `PATH` (the test itself runs under Cargo, a
//! live compiler the real `ps` census would report), then compares exit
//! status and both streams with a golden in `fixtures/ci_operations/
//! build_cache/` and, when configured, a side-by-side legacy run.

use crate::support::{CAPTURE_ENV, LEGACY_ENV, Outcome, Stage, TestResult, fixture_dir, repo_root};
use serde_json::{Value, json};
use std::error::Error;
use std::fs::{self, File, FileTimes};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, UNIX_EPOCH};

const SCRIPT: &str = "scripts/manage-build-cache.py";
const OLD: u64 = 1_577_836_800;

struct Tree {
    stage: Stage,
}

impl Tree {
    /// `<root>/ws` is the workspace; `<root>/bin` holds the stubs;
    /// `<root>/outside` is a directory outside the workspace.
    fn new(label: &str, compilers: &str) -> Result<Self, Box<dyn Error>> {
        let stage = Stage::empty(&format!("build-cache-{label}"))?;
        let metadata = fixture_dir().join("build_cache/metadata.json");
        stage.write(
            "bin/just",
            format!(
                "#!/bin/sh\ncase \"$1\" in\n  cache-cargo-metadata) exec cat '{}' ;;\n  \
                 cache-cargo-clean) echo \"clean $MESH_LLM_CACHE_PACKAGE\" >> \"$MESH_LLM_CACHE_TARGET_DIR/../clean.log\"\n    \
                 rm -rf \"$MESH_LLM_CACHE_TARGET_DIR\"/debug/deps/demo_pkg-* ;;\n  *) exit 9 ;;\nesac\n",
                metadata.display()
            )
            .as_bytes(),
        )?;
        stage.write(
            "bin/ps",
            format!("#!/bin/sh\nprintf '%b' '{compilers}'\n").as_bytes(),
        )?;
        for stub in ["bin/just", "bin/ps"] {
            fs::set_permissions(stage.path().join(stub), fs::Permissions::from_mode(0o755))?;
        }
        let tree = Self { stage };
        tree.populate()?;
        Ok(tree)
    }

    fn path(&self, relative: &str) -> PathBuf {
        self.stage.path().join(relative)
    }

    fn file(&self, relative: &str, bytes: usize, old: bool) -> TestResult {
        self.stage.write(relative, &vec![b'x'; bytes])?;
        if old {
            let time = UNIX_EPOCH + Duration::from_secs(OLD);
            File::options()
                .write(true)
                .open(self.path(relative))?
                .set_times(FileTimes::new().set_modified(time).set_accessed(time))?;
        }
        Ok(())
    }

    /// Resets the workspace to the canonical fake cache tree.
    fn populate(&self) -> TestResult {
        let _missing = fs::remove_dir_all(self.path("ws"));
        let _missing = fs::remove_file(self.path("clean.log"));
        self.file("ws/target/debug/incremental/stale-1/s", 3000, true)?;
        self.file("ws/target/debug/incremental/live-2/l", 5000, false)?;
        self.file(
            "ws/target/aarch64-apple-darwin/release/incremental/cross-3/c",
            700,
            true,
        )?;
        self.file("ws/target/debug/deps/demo_pkg-abc.rlib", 4000, true)?;
        self.file("ws/target/debug/deps/libdemo_pkg-abc.rmeta", 1000, true)?;
        self.file("ws/target/debug/build/demo-pkg-123/out", 200, true)?;
        self.file("ws/target/debug/deps/fresh_pkg-1.rlib", 2500, false)?;
        self.file("outside/keep/secret", 64, true)?;
        std::os::unix::fs::symlink(
            "../../../../outside/keep",
            self.path("ws/target/debug/incremental/escape"),
        )?;
        self.age(&[
            "ws/target/debug/incremental/stale-1",
            "ws/target/debug/incremental/escape",
            "ws/target/aarch64-apple-darwin/release/incremental/cross-3",
            "ws/target/debug/build/demo-pkg-123",
            "outside/keep",
        ])
    }

    /// Backdates directories and the symlink itself (`touch -h`).
    fn age(&self, relatives: &[&str]) -> TestResult {
        let status = Command::new("touch")
            .args(["-h", "-t", "202001010000"])
            .args(relatives.iter().map(|relative| self.path(relative)))
            .status()?;
        assert!(status.success(), "touch failed");
        Ok(())
    }

    fn command(&self, program: &Path, prefix: &[PathBuf], args: &[&str]) -> Command {
        let mut command = Command::new(program);
        let path = format!(
            "{}:{}",
            self.path("bin").display(),
            std::env::var("PATH").unwrap_or_default()
        );
        command
            .current_dir(self.path("ws"))
            .args(prefix)
            .args(
                args.iter()
                    .map(|arg| arg.replace("{root}", &self.stage.root_arg())),
            )
            .env("PATH", path)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env_remove("CARGO_BUILD_BUILD_DIR")
            .env_remove("COLUMNS")
            .stdin(Stdio::null());
        command
    }

    fn run(
        &self,
        program: &Path,
        prefix: &[PathBuf],
        args: &[&str],
    ) -> Result<Outcome, Box<dyn Error>> {
        let output = self.command(program, prefix, args).output()?;
        let root = self.stage.root_arg();
        Ok(Outcome {
            code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).replace(&root, "{root}"),
            stderr: String::from_utf8_lossy(&output.stderr).replace(&root, "{root}"),
        })
    }

    fn rust(&self, args: &[&str]) -> Result<Outcome, Box<dyn Error>> {
        let prefix = ["ci-ops", "build-cache"].map(PathBuf::from);
        self.run(Path::new(env!("CARGO_BIN_EXE_xtask")), &prefix, args)
    }
}

fn golden_path(name: &str) -> PathBuf {
    fixture_dir()
        .join("build_cache")
        .join(format!("{name}.json"))
}

/// Runs the port, then (if configured) resets and runs the legacy script on
/// the same tree; both must match each other and the captured golden.
/// `inspect` sees the tree after the Rust run, before any reset.
fn case(
    name: &str,
    tree: &Tree,
    args: &[&str],
    inspect: &dyn Fn(&Tree) -> TestResult,
) -> Result<Outcome, Box<dyn Error>> {
    run_case(name, tree, args, inspect, false)
}

/// `case` for commands that change the tree: the legacy run starts from a
/// freshly populated copy.
fn mutating(
    name: &str,
    tree: &Tree,
    args: &[&str],
    inspect: &dyn Fn(&Tree) -> TestResult,
) -> Result<Outcome, Box<dyn Error>> {
    run_case(name, tree, args, inspect, true)
}

fn run_case(
    name: &str,
    tree: &Tree,
    args: &[&str],
    inspect: &dyn Fn(&Tree) -> TestResult,
    reset: bool,
) -> Result<Outcome, Box<dyn Error>> {
    let actual = tree.rust(args)?;
    inspect(tree)?;
    if let Some(python) = std::env::var_os(LEGACY_ENV).map(PathBuf::from) {
        if reset {
            tree.populate()?;
        }
        let legacy = tree.run(&python, &[repo_root().join(SCRIPT)], args)?;
        if std::env::var_os(CAPTURE_ENV).is_some() {
            let golden =
                json!({"code": legacy.code, "stdout": legacy.stdout, "stderr": legacy.stderr});
            fs::write(
                golden_path(name),
                serde_json::to_string_pretty(&golden)? + "\n",
            )?;
        }
        assert_eq!(
            actual,
            legacy,
            "{name}: port differs from legacy {}",
            python.display()
        );
    }
    let golden: Value = serde_json::from_slice(&fs::read(golden_path(name))?)?;
    let expected = Outcome {
        code: i32::try_from(golden["code"].as_i64().ok_or("golden code")?)?,
        stdout: golden["stdout"].as_str().ok_or("golden stdout")?.to_owned(),
        stderr: golden["stderr"].as_str().ok_or("golden stderr")?.to_owned(),
    };
    assert_eq!(
        actual, expected,
        "{name}: port differs from captured golden"
    );
    Ok(actual)
}

fn nothing(_: &Tree) -> TestResult {
    Ok(())
}

fn untouched(tree: &Tree) -> TestResult {
    assert!(tree.path("ws/target/debug/incremental/stale-1").is_dir());
    assert!(
        tree.path("ws/target/debug/deps/demo_pkg-abc.rlib")
            .is_file()
    );
    assert!(!tree.path("ws/clean.log").exists());
    assert!(tree.path("outside/keep/secret").is_file());
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_status_reports_defaults() -> TestResult {
    let tree = Tree::new("status", "")?;
    case("status_defaults", &tree, &["status"], &untouched)?;
    case(
        "status_limits",
        &tree,
        &["status", "--max-size", "10 KiB", "--max-age=max_age=3"],
        &untouched,
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_dry_run_reports_selection() -> TestResult {
    let tree = Tree::new("dry-run", "")?;
    case(
        "dry_run_text",
        &tree,
        &["prune", "--max-age", "30"],
        &untouched,
    )?;
    case(
        "dry_run_json",
        &tree,
        &["prune", "--json", "--max-size", "1kb", "--max-age", "30"],
        &untouched,
    )?;
    case(
        "dry_run_within_budget",
        &tree,
        &["prune", "--max-age", "100000", "--json"],
        &untouched,
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_execute_prunes_only_selected_targets() -> TestResult {
    let tree = Tree::new("execute", "")?;
    mutating(
        "execute_prune",
        &tree,
        &[
            "prune",
            "--execute",
            "--max-size",
            "13000",
            "--max-age",
            "30",
        ],
        &|tree| {
            assert!(!tree.path("ws/target/debug/incremental/stale-1").exists());
            assert!(
                !tree
                    .path("ws/target/aarch64-apple-darwin/release/incremental/cross-3")
                    .exists()
            );
            assert!(!tree.path("ws/target/debug/incremental/escape").exists());
            assert!(
                tree.path("outside/keep/secret").is_file(),
                "symlink target must survive"
            );
            assert!(
                tree.path("ws/target/debug/incremental/live-2/l").is_file(),
                "live output kept"
            );
            assert!(tree.path("ws/target/debug/deps/fresh_pkg-1.rlib").is_file());
            assert!(!tree.path("ws/target/debug/deps/demo_pkg-abc.rlib").exists());
            assert_eq!(
                fs::read_to_string(tree.path("ws/clean.log"))?,
                "clean demo-pkg\n"
            );
            Ok(())
        },
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_refuses_active_builds() -> TestResult {
    let tree = Tree::new("locked", "")?;
    fs::create_dir_all(tree.path("ws/target"))?;
    let lock = File::options()
        .create(true)
        .append(true)
        .open(tree.path("ws/target/.mesh-llm-cache-prune.lock"))?;
    lock.lock_shared()?;
    case("locked_execute", &tree, &["prune", "--execute"], &untouched)?;
    lock.unlock()?;
    lock.lock()?;
    case("locked_status", &tree, &["status"], &untouched)?;
    case("locked_dry_run", &tree, &["prune", "--json"], &untouched)?;
    drop(lock);
    let busy = Tree::new("compiler", "  4242 rustc rustc --crate-name demo\\n")?;
    case(
        "active_compiler",
        &busy,
        &["prune", "--execute", "--max-size", "1"],
        &untouched,
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_rejects_paths_outside_root() -> TestResult {
    let tree = Tree::new("outside", "")?;
    case(
        "outside_target",
        &tree,
        &["status", "--target-dir", "{root}/outside"],
        &untouched,
    )?;
    case(
        "workspace_as_target",
        &tree,
        &["prune", "--target-dir", "."],
        &untouched,
    )?;
    case(
        "parent_escape",
        &tree,
        &["prune", "--execute", "--target-dir", "target/../../outside"],
        &untouched,
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_rejects_symlink_escape() -> TestResult {
    let tree = Tree::new("symlink", "")?;
    std::os::unix::fs::symlink(tree.path("outside"), tree.path("ws/linked"))?;
    case(
        "symlink_target",
        &tree,
        &["prune", "--execute", "--target-dir", "linked"],
        &|tree| {
            untouched(tree)?;
            assert!(!tree.path("outside/.mesh-llm-cache-prune.lock").exists());
            Ok(())
        },
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_argparse_errors_exit_two() -> TestResult {
    let tree = Tree::new("argv", "")?;
    for (name, args) in [
        ("argv_missing_command", &[][..]),
        ("argv_invalid_choice", &["clean"][..]),
        ("argv_bad_size", &["status", "--max-size", "lots"][..]),
        ("argv_bad_age", &["prune", "--max-age", "1.5"][..]),
        ("argv_ambiguous", &["status", "--max", "1"][..]),
        ("argv_missing_value", &["prune", "--workspace"][..]),
        ("argv_flag_value", &["prune", "--execute=yes"][..]),
        (
            "argv_unrecognized",
            &["status", "--json", "extra", "-x"][..],
        ),
        ("argv_build_option", &["build", "--json"][..]),
    ] {
        let outcome = case(name, &tree, args, &untouched)?;
        assert_eq!(outcome.code, 2, "{name}");
    }
    let outcome = case(
        "negative_age",
        &tree,
        &["prune", "--max-age", "-1"],
        &untouched,
    )?;
    assert_eq!(outcome.code, 1);
    case("help_prune", &tree, &["prune", "-h"], &nothing)?;
    Ok(())
}

#[test]
fn migration_ci_operations_build_cache_build_runs_under_shared_lock() -> TestResult {
    let tree = Tree::new("build", "")?;
    let outcome = mutating(
        "build_status",
        &tree,
        &["build", "--", "sh", "-c", "echo built; exit 3"],
        &nothing,
    )?;
    assert_eq!(outcome.code, 3);
    case("build_missing_command", &tree, &["build", "--"], &nothing)?;
    Ok(())
}
