//! Own the Docker daemon's container as well as its attached client process.
use super::{Engine, Profile, command, write};
use crate::{command::DynResult, process};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

struct Launch {
    root: PathBuf,
    output: PathBuf,
    artifacts: PathBuf,
    hf_home: PathBuf,
    commit: String,
    name: String,
    seconds: u64,
}

pub(super) fn run(engine: Engine, model: &str, profile: Profile, output: PathBuf) -> DynResult<()> {
    let launch = prepare(output)?;
    let image = inspect(&launch)?;
    let args = create_args(&launch, &image, engine, model, profile)?;
    let spec = command::spec(&launch.root, "docker", &args)?;
    if let Err(error) = command::run(&spec, 60, &launch.output.join("container-create")) {
        // Creation can succeed at the daemon just as the attached client is cancelled.
        let _ = remove(&launch);
        return Err(error);
    }
    let interrupt = match crate::command_interrupt::Interrupt::install() {
        Ok(interrupt) => interrupt,
        Err(error) => {
            remove(&launch)?;
            return Err(error.into());
        }
    };
    let result = attached(&launch, &interrupt.cancellation());
    // A stopped container remains inspectable until explicitly removed. Cleanup
    // gets an independent cancellation token so SIGTERM cannot skip daemon cleanup.
    let cleanup = remove(&launch);
    let finished = interrupt.finish();
    write(
        &launch.artifacts.join("container-lifecycle.json"),
        &json!({"container":launch.name,"image":image,"source_commit":launch.commit,
            "benchmark_success":result.is_ok(),"removed":cleanup.is_ok()}),
    )?;
    cleanup?;
    finished?;
    result?;
    let result_path = std::env::var_os("JIANYANG_RESULT_PATH").ok_or("JY result path missing")?;
    std::fs::copy(launch.artifacts.join("container-result.json"), result_path)?;
    Ok(())
}

fn prepare(output: PathBuf) -> DynResult<Launch> {
    if std::env::consts::OS != "linux" || std::env::consts::ARCH != "aarch64" {
        return Err("AgentX Docker jobs must execute on the Linux ARM64 Spark".into());
    }
    let commit = std::env::var("JIANYANG_COMMIT")?;
    let run = std::env::var("JIANYANG_RUN_ID")?;
    validate_identity(&commit, &run)?;
    if !output.is_absolute() {
        return Err("container output must be absolute".into());
    }
    std::fs::create_dir(&output)?;
    let artifacts = PathBuf::from(
        std::env::var_os("JIANYANG_ARTIFACT_DIR").ok_or("JY artifact directory missing")?,
    )
    .canonicalize()?;
    let hf_home = PathBuf::from(std::env::var_os("HF_HOME").ok_or("host HF_HOME missing")?);
    if !hf_home.is_absolute() || !hf_home.is_dir() {
        return Err("host HF_HOME must be an existing absolute directory".into());
    }
    let seconds = remaining(std::env::var("JIANYANG_DEADLINE_EPOCH")?.parse()?)?;
    Ok(Launch {
        root: std::env::current_dir()?,
        output,
        artifacts,
        hf_home,
        commit,
        name: format!("agentx-{run}"),
        seconds,
    })
}

fn validate_identity(commit: &str, run: &str) -> DynResult<()> {
    if commit.len() != 40
        || !commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        || run.is_empty()
        || run.len() > 100
        || !run
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err("invalid JY source/run identity".into());
    }
    Ok(())
}

fn remaining(deadline: u64) -> DynResult<u64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    deadline
        .checked_sub(now + 60)
        .filter(|seconds| *seconds > 0)
        .ok_or_else(|| "JY deadline has no time left for benchmark and cleanup".into())
}

fn inspect(launch: &Launch) -> DynResult<String> {
    let spec = command::spec(
        &launch.root,
        "docker",
        &command::strings(&[
            "image",
            "inspect",
            &format!("mesh-agentx:{}", launch.commit),
        ]),
    )?;
    let log = launch.output.join("image-inspect");
    command::run(&spec, 30, &log)?;
    let bytes = std::fs::read(log.with_extension("stdout.log"))?;
    let images: Value = serde_json::from_slice(&bytes)?;
    let image = &images[0];
    if image["Architecture"] != "arm64"
        || image["Config"]["Labels"]["org.opencontainers.image.revision"] != launch.commit
    {
        return Err("container architecture/source label does not match JY job".into());
    }
    let id = image["Id"].as_str().ok_or("image ID missing")?;
    if !id.starts_with("sha256:") || id.len() != 71 {
        return Err("invalid immutable image ID".into());
    }
    std::fs::write(launch.artifacts.join("container-image.json"), bytes)?;
    Ok(id.into())
}

fn mount(source: &Path, destination: &str) -> DynResult<String> {
    let source = source.to_str().ok_or("non-Unicode mount path")?;
    if source.contains([',', '\n', '\r']) {
        return Err("unsupported Docker mount path".into());
    }
    Ok(format!("type=bind,source={source},target={destination}"))
}

fn create_args(
    launch: &Launch,
    image: &str,
    engine: Engine,
    model: &str,
    profile: Profile,
) -> DynResult<Vec<String>> {
    let mut args = command::strings(&[
        "create",
        "--name",
        &launch.name,
        "--gpus",
        "all",
        "--init",
        "--label",
        &format!("jianyang.run={}", launch.name),
        "--mount",
        &mount(&launch.output, "/output")?,
        "--mount",
        &mount(&launch.artifacts, "/artifacts")?,
        "--mount",
        &mount(
            &launch.hf_home,
            launch.hf_home.to_str().ok_or("HF_HOME path")?,
        )?,
        "--env",
        &format!("HF_HOME={}", launch.hf_home.display()),
        "--env",
        "JIANYANG_ARTIFACT_DIR=/artifacts",
        "--env",
        "JIANYANG_RESULT_PATH=/artifacts/container-result.json",
        "--env",
        &format!("JIANYANG_COMMIT={}", launch.commit),
        "--env",
        &format!("AGENTX_CONTAINER_IMAGE={image}"),
    ]);
    #[cfg(unix)]
    {
        // SAFETY: getuid/getgid only read the current process's numeric identity.
        let (uid, gid) = unsafe { (libc::getuid(), libc::getgid()) };
        args.extend(command::strings(&["--user", &format!("{uid}:{gid}")]));
    }
    args.extend(command::strings(&[
        image,
        &launch.seconds.to_string(),
        "/workspace/target/debug/xtask",
        "automation",
        "agentx",
        "run",
        "--engine",
        engine.label(),
        "--model",
        model,
        "--profile",
        profile.label(),
        "--output",
        "/output/campaign",
    ]));
    Ok(args)
}

fn attached(launch: &Launch, cancellation: &process::Cancellation) -> DynResult<()> {
    let spec = command::spec(
        &launch.root,
        "docker",
        &command::strings(&["start", "--attach", &launch.name]),
    )?;
    let report = process::supervise_inherited_closed_stdin(
        &spec,
        &docker_limits(launch.seconds + 30),
        cancellation,
    )?;
    if report.outcome != process::Outcome::Exited
        || !report.cleanup.complete
        || report.failure.is_some()
        || !report.status.is_some_and(|status| status.success())
    {
        return Err("AgentX container failed, timed out, or was cancelled".into());
    }
    Ok(())
}

fn remove(launch: &Launch) -> DynResult<()> {
    let spec = command::spec(
        &launch.root,
        "docker",
        &command::strings(&["container", "rm", "--force", &launch.name]),
    )?;
    let report = process::supervise_inherited_closed_stdin(
        &spec,
        &docker_limits(3),
        &process::Cancellation::default(),
    )?;
    if !report.cleanup.complete || !report.status.is_some_and(|status| status.success()) {
        return Err("Docker container removal failed".into());
    }
    Ok(())
}

fn docker_limits(seconds: u64) -> process::Limits {
    // JY escalates SIGTERM to SIGKILL after ten seconds. Leave time to remove
    // the daemon-owned container after terminating an unresponsive Docker client.
    let mut limits = command::limits(seconds);
    limits.graceful_shutdown = std::time::Duration::from_secs(1);
    limits.forced_shutdown = std::time::Duration::from_secs(1);
    limits
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_engine_reaches_container_with_same_host_cache_mount() {
        let launch = Launch {
            root: "/repo".into(),
            output: "/runs/one".into(),
            artifacts: "/artifacts/one".into(),
            hf_home: "/mnt/lab/huggingface".into(),
            commit: "a".repeat(40),
            name: "agentx-run_one".into(),
            seconds: 600,
        };
        for engine in [Engine::Llama, Engine::Skippy] {
            let args =
                create_args(&launch, "sha256:test", engine, "model", Profile::Smoke).unwrap();
            assert!(
                args.windows(2)
                    .any(|pair| pair == ["--engine", engine.label()])
            );
            assert!(args.windows(2).any(|pair| pair == ["--gpus", "all"]));
            assert!(
                args.iter().any(|arg| arg
                    == "type=bind,source=/mnt/lab/huggingface,target=/mnt/lab/huggingface")
            );
        }
    }
    #[test]
    fn rejects_ambiguous_identity_and_mounts() {
        assert!(validate_identity(&"a".repeat(40), "run_abc").is_ok());
        assert!(validate_identity("main", "run_abc").is_err());
        assert!(validate_identity(&"a".repeat(40), "../other").is_err());
        assert!(mount(Path::new("/cache,readonly"), "/cache").is_err());
        assert!(remaining(0).is_err());
    }
}
