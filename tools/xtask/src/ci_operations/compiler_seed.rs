//! Manual, disk-only paired compiler seed qualification. Never publishes a cache.
use super::{
    compiler_seed_evidence as evidence,
    compiler_seed_process::{Runner, environment, recorded_environment},
    runtime_seed_io::{environment as required, save},
    runtime_seed_types::{IMAGE, hexadecimal},
};
use crate::{command::DynResult, command_interrupt::Interrupt, process::Cancellation};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const USAGE: &str = "usage: ci-ops compiler-seed run --pair N --output DIR (N=1..3)";

pub(super) fn parse(args: &[String]) -> DynResult<(u8, PathBuf)> {
    let [operation, pair_flag, pair, output_flag, output] = args else {
        return Err(USAGE.into());
    };
    if operation != "run" || pair_flag != "--pair" || output_flag != "--output" || output.is_empty()
    {
        return Err(USAGE.into());
    }
    let pair: u8 = pair.parse()?;
    if !(1..=3).contains(&pair) {
        return Err(USAGE.into());
    }
    Ok((pair, output.into()))
}

pub(super) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let (pair, output) = parse(args)?;
    // Atomic creation admits ownership; a pre-existing path is never modified.
    fs::create_dir(&output)?;
    let output = output.canonicalize()?;
    let result = execute(root, pair, &output);
    save(
        &output.join("result.json"),
        &serde_json::json!({"schema":1,"pair":pair,"classification":if result.is_ok() {"measured"} else {"inconclusive"},"error":result.as_ref().err().map(ToString::to_string),"cache_published":false,"runtime_seed_restored":false}),
    )?;
    result
}

fn execute(root: &Path, pair: u8, output: &Path) -> DynResult<()> {
    let temporary = PathBuf::from(required("RUNNER_TEMP")?).canonicalize()?;
    if output.parent() != Some(temporary.as_path()) {
        return Err("output must be a fresh direct child of RUNNER_TEMP".into());
    }
    let work = temporary.join(format!("compiler-seed-work-pair-{pair}"));
    fs::create_dir(&work)?;
    fs::create_dir(work.join("tmp"))?;
    let interrupt = Interrupt::install()?;
    let runner = Runner {
        root: root.canonicalize()?,
        environment: environment(&work, &work.join("cache"), &work.join("target")),
        cancellation: interrupt.cancellation(),
        deadline: Instant::now() + Duration::from_secs(170 * 60),
    };
    let result = experiment(&runner, pair, output, &work);
    let finish = interrupt.finish();
    result?;
    finish?;
    Ok(())
}

fn experiment(runner: &Runner, pair: u8, output: &Path, work: &Path) -> DynResult<()> {
    preflight(runner, pair, output)?;
    runner.command(
        "just",
        &["with-lld", "cargo", "fetch", "--locked"],
        &output.join("dependency-fetch"),
    )?;
    let mut failures = Vec::new();
    for arm in order(pair) {
        let directory = output.join(arm);
        fs::create_dir(&directory)?;
        let owned = work.join(arm);
        fs::create_dir(&owned)?;
        let result = arm_run(runner, arm, &owned, &directory);
        save(
            &directory.join("result.json"),
            &serde_json::json!({"arm":arm,"classification":if result.is_ok(){"measured"}else{"inconclusive"},"error":result.as_ref().err().map(ToString::to_string)}),
        )?;
        if let Err(error) = result {
            failures.push(format!("{arm}: {error}"));
        }
    }
    if !failures.is_empty() {
        return Err(failures.join("; ").into());
    }
    Ok(())
}

fn preflight(runner: &Runner, pair: u8, output: &Path) -> DynResult<()> {
    for (name, value) in [
        ("GITHUB_EVENT_NAME", "workflow_dispatch"),
        ("RUNNER_ENVIRONMENT", "github-hosted"),
        ("RUNNER_OS", "Linux"),
        ("RUNNER_ARCH", "X64"),
        ("COMPILER_SEED_IMAGE", IMAGE),
    ] {
        if required(name)? != value {
            return Err(format!("environment mismatch: {name}").into());
        }
    }
    if required("PAIR")? != pair.to_string() {
        return Err("PAIR differs from --pair".into());
    }
    let source = required("GITHUB_SHA")?;
    let workflow = required("GITHUB_WORKFLOW_SHA")?;
    if !hexadecimal(&source, 40) || source != workflow {
        return Err("product and workflow SHA must be identical immutable commits".into());
    }
    let head = runner.command("git", &["rev-parse", "HEAD"], &output.join("source"))?;
    if String::from_utf8(head)?.trim() != source {
        return Err("checkout differs from workflow source".into());
    }
    let dirty = runner.command(
        "git",
        &["status", "--porcelain", "--untracked-files=no"],
        &output.join("tracked-status"),
    )?;
    if !dirty.is_empty() {
        return Err("tracked source has modifications".into());
    }
    let automation = PathBuf::from(required("MESH_LLM_AUTOMATION_BIN")?).canonicalize()?;
    if automation != std::env::current_exe()?.canonicalize()? {
        return Err("invoke the restored MESH_LLM_AUTOMATION_BIN directly".into());
    }
    for name in ["GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"] {
        if required(name)?.parse::<u64>()? == 0 {
            return Err("invalid run identity".into());
        }
    }
    evidence::host(output)?;
    runner.command("rustc", &["-vV"], &output.join("rustc-version"))?;
    runner.command("sccache", &["--version"], &output.join("sccache-version"))?;
    save(
        &output.join("context.json"),
        &serde_json::json!({
            "schema":1,"source_sha":source,"workflow_sha":workflow,"workflow_ref":required("GITHUB_WORKFLOW_REF")?,"run_id":required("GITHUB_RUN_ID")?,"run_attempt":required("GITHUB_RUN_ATTEMPT")?,"pair":pair,"order":order(pair),"image":IMAGE,
            "automation":automation,"root":runner.root,"environment":recorded_environment(&runner.environment)?,
            "commands":{"seed":"just ci-sccache-seed-build","additional_seed":"just skippy-cli-release-build","standalone":"just skippy-cli-release-build","mesh":"just release-host-build"},
            "features":{"standalone":"default + dynamic-native-runtime","mesh":"--no-default-features web-ui,dynamic-native-runtime","seed":"existing ci-sccache-seed-build recipe (Clippy all targets, release Mesh, CLI tests no-run, xtask Clippy)"},
            "target_policy":"same absolute arm target and build directory; extended seed appends without cleaning; fresh contents before each consumer; cache persists only within an arm","cache_limit_bytes":2147483648u64
        }),
    )
}

pub(super) fn order(pair: u8) -> [&'static str; 2] {
    if pair == 2 {
        ["extended", "current"]
    } else {
        ["current", "extended"]
    }
}

fn arm_run(parent: &Runner, arm: &str, work: &Path, output: &Path) -> DynResult<()> {
    let cache = work.join("cache");
    let target = work.join("target");
    for path in [&cache, &target, &work.join("tmp")] {
        fs::create_dir(path)?;
    }
    fs::write(
        work.join("sccache.toml"),
        "[cache.multilevel]\nchain = [\"disk\"]\nwrite_error_policy = \"all\"\n",
    )?;
    let runner = Runner {
        root: parent.root.clone(),
        environment: environment(work, &cache, &target),
        cancellation: parent.cancellation.clone(),
        deadline: parent.deadline,
    };
    save(
        &output.join("environment.json"),
        &recorded_environment(&runner.environment)?,
    )?;
    let result = phases(&runner, arm, work, output);
    // Cleanup cancellation is separate so an interrupted build still stops its private daemon.
    let cleanup = Runner {
        root: runner.root.clone(),
        environment: runner.environment.clone(),
        cancellation: Cancellation::default(),
        deadline: Instant::now() + Duration::from_secs(30),
    };
    let stopped = cleanup.command("sccache", &["--stop-server"], &output.join("stop-server"));
    let resources = evidence::host(output);
    let cleared = reset_target(work);
    result?;
    stopped?;
    cleared?;
    resources
}

fn phases(runner: &Runner, arm: &str, work: &Path, output: &Path) -> DynResult<()> {
    runner.command("sccache", &["--start-server"], &output.join("start-server"))?;
    let mut commands = vec![("seed", "ci-sccache-seed-build")];
    if arm == "extended" {
        commands.push(("additional-seed", "skippy-cli-release-build"));
    }
    commands.extend([
        ("consumer-standalone", "skippy-cli-release-build"),
        ("consumer-mesh", "release-host-build"),
    ]);
    let mut failures = Vec::new();
    for (phase, recipe) in commands {
        let directory = output.join(phase);
        fs::create_dir(&directory)?;
        if phase != "additional-seed" {
            reset_target(work)?;
        }
        let result = phase_run(runner, work, &directory, recipe);
        save(
            &directory.join("result.json"),
            &serde_json::json!({"phase":phase,"classification":if result.is_ok(){"measured"}else{"inconclusive"},"error":result.as_ref().err().map(ToString::to_string)}),
        )?;
        if let Err(error) = result {
            failures.push(format!("{phase}: {error}"));
        }
        if runner.cancellation.is_cancelled() {
            break;
        }
    }
    if !failures.is_empty() {
        return Err(failures.join("; ").into());
    }
    Ok(())
}

fn phase_run(runner: &Runner, work: &Path, output: &Path, recipe: &str) -> DynResult<()> {
    let cache = work.join("cache");
    let before = evidence::inventory(&cache)?;
    save(&output.join("objects-before.json"), &before)?;
    runner.command("sccache", &["--zero-stats"], &output.join("zero-stats"))?;
    let initial = runner.command(
        "sccache",
        &["--show-stats", "--stats-format=json"],
        &output.join("stats-before"),
    )?;
    if super::runtime_seed_stats::Snapshot::parse(&initial)?
        .stats
        .scalars["compile_requests"]
        != 0
    {
        return Err("counters did not reset".into());
    }
    let started = Instant::now();
    let build = runner.command("just", &[recipe], &output.join("build"));
    let seconds = started.elapsed().as_secs_f64();
    let stats = runner.command(
        "sccache",
        &["--show-stats", "--stats-format=json"],
        &output.join("stats-after"),
    );
    let after = evidence::inventory(&cache)?;
    save(&output.join("objects-after.json"), &after)?;
    save(
        &output.join("object-changes.json"),
        &evidence::change(&before, &after),
    )?;
    let measurement = stats.and_then(|bytes| evidence::measurement(&bytes));
    save(
        &output.join("phase.json"),
        &serde_json::json!({"command":["just",recipe],"seconds":seconds,"build_error":build.as_ref().err().map(ToString::to_string),"measurement":measurement.as_ref().ok(),"measurement_error":measurement.as_ref().err().map(ToString::to_string),"target":work.join("target")}),
    )?;
    if after.bytes > 2_147_483_648 {
        return Err("compiler cache exceeded the 2 GiB experiment bound".into());
    }
    build?;
    measurement?;
    Ok(())
}

pub(super) fn reset_target(work: &Path) -> DynResult<()> {
    let target = work.join("target");
    if fs::symlink_metadata(&target)?.file_type().is_symlink()
        || !target.is_dir()
        || work.canonicalize()? != work
    {
        return Err("owned target boundary changed".into());
    }
    // Only this command's atomically created work tree is eligible for removal.
    fs::remove_dir_all(&target)?;
    fs::create_dir(&target)?;
    Ok(())
}

#[cfg(test)]
mod tests;
