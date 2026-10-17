//! `repository capsules-plugin-gate GATE`: the capsules plugin's own gates
//! beyond fmt, Clippy and its tests. CI runs each gate as its own step, so a
//! failure names the gate. The plugin keeps the data (which mutant must fail
//! which test, the pinned vector digests); this module keeps the logic.

mod no_keys;

use super::{check_args::Grammar, check_report::CheckReport};
use crate::command::DynResult;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository capsules-plugin-gate {mutants|referee-mutants|unbounded-depth|no-keys|vectors|web-ui|package-twice} [--root <path>]",
    values: &["--root"],
    flags: &[],
};

const PLUGIN_DIR: &str = "mesh/crates/mesh-llm-capsules-plugin";
const PACKAGE: &str = "mesh-llm-capsules-plugin";
const BIN: &str = "capsules";
/// The plugin's mutant table: each feature, its tests, and what they print.
const MUTANTS: &str = "gates/mutants.json";
/// The referee's mutants, each with the corpus cases that must catch it.
const REFEREE_MUTANTS: &str = "vectors/parity/referee/mutants.json";
const REFEREE_RUN: &str = "referee::parity::the_referee_answers_the_corpus_as_expected";
const BUNDLE_ENTRY: &str = "register-mesh-plugin-ui.js";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gate {
    Mutants,
    RefereeMutants,
    UnboundedDepth,
    NoKeys,
    Vectors,
    WebUi,
    PackageTwice,
}

impl Gate {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "mutants" => Self::Mutants,
            "referee-mutants" => Self::RefereeMutants,
            "unbounded-depth" => Self::UnboundedDepth,
            "no-keys" => Self::NoKeys,
            "vectors" => Self::Vectors,
            "web-ui" => Self::WebUi,
            "package-twice" => Self::PackageTwice,
            _ => return None,
        })
    }
}

pub(super) fn run(
    args: &[String],
    default_root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    let gate = match parsed.positionals.as_slice() {
        [name] => match Gate::parse(name) {
            Some(gate) => gate,
            None => return GRAMMAR.error(&format!("unknown gate: {name}")).emit(),
        },
        _ => return GRAMMAR.error("exactly one gate is required").emit(),
    };
    let root = match parsed.last("--root") {
        Some(root) => super::RepositoryRoot::resolve(Some(Path::new(root)))?
            .as_path()
            .to_path_buf(),
        None => default_root()?,
    };
    let plugin = root.join(PLUGIN_DIR);
    let report = match gate {
        Gate::Mutants => mutants(&plugin)?,
        Gate::RefereeMutants => referee_mutants(&plugin)?,
        Gate::UnboundedDepth => unbounded_depth(&plugin)?,
        Gate::NoKeys => no_keys::check(&root, &plugin)?,
        Gate::Vectors => vectors(&plugin.join("vectors"))?,
        Gate::WebUi => web_ui(&plugin)?,
        Gate::PackageTwice => package_twice(&root, &plugin)?,
    };
    report.emit()
}

/// What a mutant's failing run must print: an exact line (a parity runner
/// prints each differing case on its own line), or a named test reported as
/// failed by the libtest harness.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum Expect {
    Line(String),
    FailedTest(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Mutant {
    feature: String,
    tests: String,
    must_print: Expect,
}

#[derive(Debug, Deserialize)]
struct MutantTable {
    mutants: Vec<Mutant>,
}

impl MutantTable {
    /// An empty table would pass with nothing checked.
    fn parse(text: &str) -> DynResult<Self> {
        let table: Self = serde_json::from_str(text)?;
        if table.mutants.is_empty() {
            return Err(format!("{MUTANTS} lists no mutant").into());
        }
        Ok(table)
    }
}

#[derive(Debug, Deserialize)]
struct RefereeMutant {
    feature: String,
    must_fail: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RefereeTable {
    mutants: Vec<RefereeMutant>,
}

impl RefereeTable {
    /// A mutant with no case listed would count as caught on any failure.
    fn parse(text: &str) -> DynResult<Self> {
        let table: Self = serde_json::from_str(text)?;
        if table.mutants.is_empty() {
            return Err(format!("{REFEREE_MUTANTS} lists no mutant").into());
        }
        if let Some(mutant) = table.mutants.iter().find(|m| m.must_fail.is_empty()) {
            return Err(format!("{} lists no case that must fail", mutant.feature).into());
        }
        Ok(table)
    }
}

fn printed(expect: &Expect, output: &str) -> bool {
    output.lines().any(|line| match expect {
        Expect::Line(wanted) => line == wanted,
        Expect::FailedTest(name) => line
            .strip_prefix("test ")
            .and_then(|rest| rest.strip_suffix(" ... FAILED"))
            .is_some_and(|path| path == name || path.ends_with(&format!("::{name}"))),
    })
}

/// The cases a referee parity run reports as differing (`differs: CASE`).
fn differing_cases(output: &str) -> BTreeSet<&str> {
    output
        .lines()
        .filter_map(|line| line.strip_prefix("differs: "))
        .filter(|case| !case.is_empty() && !case.contains(char::is_whitespace))
        .collect()
}

/// Builds the mutant first, so a compile error never passes as "caught",
/// then runs `filter` and returns whether it passed, and its output.
fn run_mutant(plugin: &Path, feature: &str, filter: &str) -> DynResult<(bool, String)> {
    let build = cargo(plugin)
        .args([
            "test",
            "--locked",
            "-p",
            PACKAGE,
            "--bin",
            BIN,
            "--features",
            feature,
            "--no-run",
        ])
        .status()?;
    if !build.success() {
        return Err(format!("mutant {feature} did not build").into());
    }
    let run = cargo(plugin)
        .args([
            "test",
            "--locked",
            "-p",
            PACKAGE,
            "--bin",
            BIN,
            "--features",
            feature,
            "--",
            filter,
        ])
        .stdin(Stdio::null())
        .output()?;
    let mut output = String::from_utf8_lossy(&run.stdout).into_owned();
    output.push('\n');
    output.push_str(&String::from_utf8_lossy(&run.stderr));
    Ok((run.status.success(), output))
}

fn mutants(plugin: &Path) -> DynResult<CheckReport> {
    let table = MutantTable::parse(&fs::read_to_string(plugin.join(MUTANTS))?)?;
    let mut caught = String::new();
    let mut errors = String::new();
    for mutant in &table.mutants {
        let (passed, output) = run_mutant(plugin, &mutant.feature, &mutant.tests)?;
        if passed {
            writeln!(errors, "the tests passed a build with {}", mutant.feature)?;
        } else if !printed(&mutant.must_print, &output) {
            writeln!(
                errors,
                "{output}\n{} failed, but not with {:?}",
                mutant.feature, mutant.must_print
            )?;
        } else {
            writeln!(caught, "caught: {}", mutant.feature)?;
        }
    }
    Ok(verdict(
        caught,
        errors,
        "every mutant failed its named tests",
    ))
}

fn referee_mutants(plugin: &Path) -> DynResult<CheckReport> {
    let table = RefereeTable::parse(&fs::read_to_string(plugin.join(REFEREE_MUTANTS))?)?;
    let mut caught = String::new();
    let mut errors = String::new();
    for mutant in &table.mutants {
        let (passed, output) = run_mutant(plugin, &mutant.feature, REFEREE_RUN)?;
        let reported = differing_cases(&output);
        let missing: Vec<_> = mutant
            .must_fail
            .iter()
            .filter(|case| !reported.contains(case.as_str()))
            .collect();
        if passed {
            writeln!(
                errors,
                "the referee parity run passed a build with {}",
                mutant.feature
            )?;
        } else if !missing.is_empty() {
            writeln!(
                errors,
                "{} failed the run, but not on: {missing:?}",
                mutant.feature
            )?;
        } else {
            writeln!(caught, "caught: {}", mutant.feature)?;
        }
    }
    Ok(verdict(
        caught,
        errors,
        "every referee mutant failed on its cases",
    ))
}

/// serde_json's `unbounded_depth` is for the parity runner's fixture reader
/// (a dev-dependency); the plugin binary parses a pushed body under the
/// default nesting limit.
fn unbounded_depth(plugin: &Path) -> DynResult<CheckReport> {
    let tree = cargo(plugin)
        .args([
            "tree",
            "--locked",
            "-p",
            PACKAGE,
            "-e",
            "normal,features",
            "-i",
            "serde_json",
        ])
        .output()?;
    if !tree.status.success() {
        return Err(format!(
            "cargo tree failed: {}",
            String::from_utf8_lossy(&tree.stderr)
        )
        .into());
    }
    Ok(unbounded_depth_verdict(&String::from_utf8_lossy(
        &tree.stdout,
    )))
}

fn unbounded_depth_verdict(tree: &str) -> CheckReport {
    if tree.contains("unbounded_depth") {
        CheckReport::failure(
            String::new(),
            "serde_json/unbounded_depth is enabled outside the tests\n".into(),
        )
    } else {
        CheckReport::success(
            "serde_json keeps its default nesting limit in the plugin binary\n".into(),
        )
    }
}

/// Every line of `SHA256SUMS` names a file under `dir` and its digest.
fn vectors(dir: &Path) -> DynResult<CheckReport> {
    let sums = fs::read_to_string(dir.join("SHA256SUMS"))?;
    let mut errors = String::new();
    let mut count = 0;
    for (number, line) in sums
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
    {
        let Some((digest, name)) = parse_sum_line(line) else {
            writeln!(errors, "SHA256SUMS:{}: not `<sha256>  <path>`", number + 1)?;
            continue;
        };
        count += 1;
        match fs::read(dir.join(name)) {
            Ok(bytes) if hex::encode(Sha256::digest(&bytes)) == digest => {}
            Ok(_) => writeln!(errors, "{name}: digest differs")?,
            Err(error) => writeln!(errors, "{name}: {error}")?,
        }
    }
    if count == 0 {
        writeln!(errors, "SHA256SUMS lists no vector")?;
    }
    Ok(verdict(
        String::new(),
        errors,
        &format!("{count} pinned vectors match SHA256SUMS"),
    ))
}

/// `<64 lowercase hex>  <path>` (or `*<path>`, binary mode), with a path
/// that stays inside the vectors directory.
fn parse_sum_line(line: &str) -> Option<(&str, &str)> {
    let (digest, rest) = line.split_once(' ')?;
    let name = rest.strip_prefix(' ').or_else(|| rest.strip_prefix('*'))?;
    let contained = !name.is_empty()
        && Path::new(name)
            .components()
            .all(|part| matches!(part, Component::Normal(_)));
    (digest.len() == 64
        && digest
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        && contained)
        .then_some((digest, name))
}

/// The Evidence page: install from the lockfile, typecheck, unit tests, and
/// a build that writes exactly one ES module.
fn web_ui(plugin: &Path) -> DynResult<CheckReport> {
    build_web_ui(plugin, &[&["typecheck"], &["test"]])?;
    Ok(CheckReport::success(format!(
        "the Evidence page typechecks, passes its tests and builds {BUNDLE_ENTRY}\n"
    )))
}

fn build_web_ui(plugin: &Path, checks: &[&[&str]]) -> DynResult<()> {
    let web_ui = plugin.join("web-ui");
    pnpm(&web_ui, &["install", "--frozen-lockfile"])?;
    for check in checks {
        pnpm(&web_ui, check)?;
    }
    pnpm(&web_ui, &["build"])?;
    let entries = sorted_entries(&plugin.join("bundle"))?;
    if entries != [BUNDLE_ENTRY] {
        return Err(format!("bundle/ must hold exactly {BUNDLE_ENTRY}, found {entries:?}").into());
    }
    Ok(())
}

/// The release packaging (`scripts/package.sh`) run twice on one debug build:
/// the two archives must be byte-identical.
fn package_twice(root: &Path, plugin: &Path) -> DynResult<CheckReport> {
    build_web_ui(plugin, &[])?;
    let build = cargo(plugin)
        .args(["build", "--locked", "-p", PACKAGE, "--bin", BIN])
        .status()?;
    if !build.success() {
        return Err("the plugin did not build".into());
    }
    let binary = target_directory(root)?.join("debug").join(BIN);
    let version = plugin_version(&fs::read_to_string(plugin.join("plugin.toml"))?)?;
    let host = host_triple()?;
    let out = tempfile::tempdir()?;
    let mut archives = Vec::new();
    for run in ["a", "b"] {
        let dir = out.path().join(run);
        let status = Command::new(plugin.join("scripts/package.sh"))
            .args([version.as_str(), host.as_str()])
            .arg(&binary)
            .arg(&dir)
            .env("SOURCE_DATE_EPOCH", "0")
            .status()?;
        if !status.success() {
            return Err(format!("scripts/package.sh failed (run {run})").into());
        }
        archives.push(fs::read(
            dir.join(format!("{BIN}-{version}-{host}.tar.gz")),
        )?);
    }
    let digests: Vec<_> = archives
        .iter()
        .map(|bytes| hex::encode(Sha256::digest(bytes)))
        .collect();
    if archives[0] != archives[1] {
        return Ok(CheckReport::failure(
            String::new(),
            format!(
                "the two packages differ: {} vs {}\n",
                digests[0], digests[1]
            ),
        ));
    }
    Ok(CheckReport::success(format!(
        "both packages are byte-identical, sha256 {}\n{}",
        digests[0],
        archive_listing(&archives[0])?
    )))
}

fn archive_listing(gz: &[u8]) -> DynResult<String> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(gz));
    let mut names = Vec::new();
    for entry in archive.entries()? {
        names.push(entry?.path()?.display().to_string());
    }
    names.sort();
    Ok(names.iter().map(|name| format!("{name}\n")).collect())
}

fn plugin_version(plugin_toml: &str) -> DynResult<String> {
    let marker: toml::Value = toml::from_str(plugin_toml)?;
    marker
        .get("version")
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "plugin.toml has no version".into())
}

fn host_triple() -> DynResult<String> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(rustc).arg("-vV").output()?;
    if !output.status.success() {
        return Err("rustc -vV failed".into());
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_owned)
        .ok_or_else(|| "rustc -vV printed no host".into())
}

fn target_directory(root: &Path) -> DynResult<PathBuf> {
    let output = cargo(root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    metadata["target_directory"]
        .as_str()
        .map(PathBuf::from)
        .ok_or_else(|| "cargo metadata gave no target_directory".into())
}

fn sorted_entries(dir: &Path) -> DynResult<Vec<String>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir)? {
        names.push(entry?.file_name().to_string_lossy().into_owned());
    }
    names.sort();
    Ok(names)
}

fn cargo(dir: &Path) -> Command {
    let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command.current_dir(dir);
    command
}

/// pnpm from PATH (pnpm 10 switches itself to the version web-ui's
/// `packageManager` names), else through Node's corepack.
fn pnpm(dir: &Path, args: &[&str]) -> DynResult<()> {
    let on_path = Command::new("pnpm")
        .arg("--version")
        .current_dir(dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    let mut command = if on_path {
        Command::new("pnpm")
    } else {
        let mut corepack = Command::new("corepack");
        corepack
            .arg("pnpm")
            .env("COREPACK_ENABLE_DOWNLOAD_PROMPT", "0");
        corepack
    };
    let status = command.args(args).current_dir(dir).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("pnpm {} failed", args.join(" ")).into())
    }
}

fn verdict(stdout: String, errors: String, passed: &str) -> CheckReport {
    if errors.is_empty() {
        CheckReport::success(format!("{stdout}{passed}\n"))
    } else {
        CheckReport::failure(stdout, errors)
    }
}

#[cfg(test)]
mod tests;
