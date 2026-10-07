use super::*;
use std::io::Write as _;
pub(super) fn prepare(args: EvalPrepareSweArgs) -> Result<()> {
    let root = absolute_path(cache_root(args.cache_root)?)?;
    if !cfg!(unix) {
        bail!("SWE preparation supports Unix only");
    }
    for path in [&root, &args.uv, &args.python] {
        if !path.is_absolute()
            || path
                .to_str()
                .is_none_or(|v| v.chars().any(char::is_control))
        {
            bail!("SWE preparation needs absolute control-free UTF-8 paths");
        }
    }
    let configuration = Configuration::new(args.deployment, args.index_url)?;
    if args.dry_run {
        println!(
            "SWE locked preparation at {} (pending; unqualified)",
            base(&root).display()
        );
        return Ok(());
    }
    harness_source::admit_run(&root, registry::definition(EvalId::SweBenchPro))?;
    custody::source_helpers(&original(&root))?;
    let uv = args.uv.canonicalize()?;
    let python = args.python.canonicalize()?;
    let tool_pins = sdk_environment::capture_tools(&uv, &python)?;
    let git_sha256 = sdk_environment::hash(Path::new("/usr/bin/git"))?;
    let destination = base(&root);
    fs::create_dir(&destination).context(
        "SWE preparation slot must be fresh; retain failed evidence or choose another cache root",
    )?;
    for name in [
        "home",
        "tmp",
        "cache",
        "logs",
        "probe",
        "project/source/swe-bench-pro",
    ] {
        fs::create_dir_all(destination.join(name))?;
    }
    let deadline = Instant::now() + Duration::from_secs(PREPARATION_SECONDS);
    versions(&root, &uv, &python, deadline)?;
    fs::write(project(&root).join("pyproject.toml"), PROJECT)?;
    fs::write(project(&root).join("uv.lock"), LOCK)?;
    if sdk_environment::hash_bytes(LOCK.as_bytes()) != LOCK_SHA {
        bail!("embedded SWE lock digest differs");
    }
    harness_source::clone_swe_agent_snapshot(
        &original(&root).join("SWE-agent"),
        &agent(&root),
        deadline - Duration::from_secs(6),
    )?;
    execute(
        &sync_spec(&root, &uv, &python),
        &root,
        "locked-sync",
        deadline,
    )?;
    let modules = probe(&root, deadline)?;
    custody::modules(&root, &modules)?;
    patch_sdk(&root, &configuration, &modules, deadline)?;
    let receipt = Receipt {
        schema_version: 1,
        parent: registry::SWE_BENCH_PRO_REF.into(),
        agent: registry::SWE_AGENT_REF.into(),
        lock_sha256: LOCK_SHA.into(),
        patch_profile: patch_profile(&configuration).into(),
        configuration,
        uv,
        python: python.clone(),
        git_sha256,
        tool_pins,
        environment_pins: sdk_environment::environment(
            &destination.join("environment"),
            &python,
            sdk_environment::PythonProfile::Swe311,
        )?,
        agent_package_pins: sdk_environment::package_roster(
            &agent(&root),
            &agent(&root).join("sweagent"),
        )?,
        modules,
        benchmark_qualified: false,
    };
    harness_source::admit_run(&root, registry::definition(EvalId::SweBenchPro))?;
    custody::admit(&root, &receipt)?;
    if Instant::now() >= deadline {
        bail!("SWE preparation terminal deadline");
    }
    publish(&destination, &receipt)?;
    println!(
        "SWE SDK prepared at {}; benchmark qualification remains false",
        destination.display()
    );
    Ok(())
}
fn publish(destination: &Path, receipt: &Receipt) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(receipt)?;
    if bytes.len() > 8 * 1048576 {
        bail!("SWE receipt exceeds finite bound");
    }
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join("receipt.pending"))?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    fs::rename(
        destination.join("receipt.pending"),
        destination.join("receipt.json"),
    )?;
    Ok(())
}
fn isolated(spec: CommandSpec, root: &Path) -> CommandSpec {
    spec.isolated()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", base(root).join("home").display().to_string())
        .env("TMPDIR", base(root).join("tmp").display().to_string())
        .env(
            "UV_CACHE_DIR",
            base(root).join("cache").display().to_string(),
        )
        .env(
            "UV_PROJECT_ENVIRONMENT",
            base(root).join("environment").display().to_string(),
        )
        .env("UV_PYTHON_DOWNLOADS", "never")
}
fn execute(spec: &CommandSpec, root: &Path, phase: &str, deadline: Instant) -> Result<String> {
    let execution = deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(6))
        .filter(|v| !v.is_zero())
        .context("SWE cleanup reserve unavailable")?;
    let stdout = base(root).join(format!("logs/{phase}.stdout"));
    let stderr = base(root).join(format!("logs/{phase}.stderr"));
    let result = sdk_environment::capture(spec, execution, &stdout, &stderr)?;
    if !result.success || result.timed_out || Instant::now() >= deadline {
        bail!("SWE preparation phase {phase} failed; evidence retained");
    }
    Ok(String::from_utf8(sdk_environment::read(&stdout, 1048576)?)?)
}
fn sync_spec(root: &Path, uv: &Path, python: &Path) -> CommandSpec {
    isolated(
        CommandSpec::new(uv.display().to_string()).args([
            "--no-config".into(),
            "sync".into(),
            "--locked".into(),
            "--no-build".into(),
            "--project".into(),
            project(root).display().to_string(),
            "--python".into(),
            python.display().to_string(),
            "--no-python-downloads".into(),
        ]),
        root,
    )
}
fn probe(root: &Path, deadline: Instant) -> Result<BTreeMap<String, PathBuf>> {
    probe_named(root, "sdk-import-only", deadline)
}
fn probe_named(root: &Path, phase: &str, deadline: Instant) -> Result<BTreeMap<String, PathBuf>> {
    let parent = serde_json::to_string(&original(root).display().to_string())?;
    // The fixed actual SDK leaves expose raw metadata only; Rust owns all admission decisions.
    let code = format!(
        "import sys,json,runpy,types,sweagent,swerex; import swerex.deployment.modal,swerex.runtime.remote; parent={parent}; helper=types.ModuleType('helper_code'); helper.__path__=[parent+'/helper_code']; sys.modules['helper_code']=helper; sys.path.insert(0,parent+'/helper_code'); runpy.run_path(parent+'/helper_code/generate_sweagent_instances.py',run_name='_swe_sdk_import_only'); runpy.run_path(parent+'/helper_code/gather_patches.py',run_name='_swe_sdk_import_only'); runpy.run_path(parent+'/swe_bench_pro_eval.py',run_name='_swe_sdk_import_only'); print(json.dumps({{'sweagent_version':sweagent.__version__,'swerex_version':swerex.__version__,'modules':{{'sweagent':sweagent.__file__,'swerex':swerex.__file__}}}}))"
    );
    let spec = isolated(
        CommandSpec::new(runtime_python(root).display().to_string()).args([
            "-I".into(),
            "-B".into(),
            "-c".into(),
            code,
        ]),
        root,
    )
    .cwd(base(root).join("probe"))
    .env("PYTHON_DOTENV_DISABLED", "1")
    .env("HF_HUB_OFFLINE", "1")
    .env("HF_DATASETS_OFFLINE", "1")
    .env("LITELLM_LOCAL_MODEL_COST_MAP", "True");
    let raw = execute(&spec, root, phase, deadline)?;
    let metadata: Value =
        serde_json::from_str(raw.lines().last().context("SWE import metadata missing")?)?;
    if metadata["sweagent_version"] != "1.1.0" || metadata["swerex_version"] != "1.4.0" {
        bail!("SWE SDK versions differ");
    }
    Ok(serde_json::from_value(metadata["modules"].clone())?)
}

fn versions(root: &Path, uv: &Path, python: &Path, deadline: Instant) -> Result<()> {
    let version = execute(
        &isolated(
            CommandSpec::new(python.display().to_string()).args(["-I", "-B", "--version"]),
            root,
        ),
        root,
        "python-version",
        deadline,
    )?;
    if version.trim() != PYTHON_VERSION {
        bail!("SWE profile requires existing CPython3.11.13");
    }
    execute(
        &isolated(
            CommandSpec::new(uv.display().to_string()).args(["--version"]),
            root,
        ),
        root,
        "uv-version",
        deadline,
    )?;
    Ok(())
}

fn patch_sdk(
    root: &Path,
    configuration: &Configuration,
    modules: &BTreeMap<String, PathBuf>,
    deadline: Instant,
) -> Result<()> {
    match configuration.deployment {
        SweDeployment::Docker => swerex_index::patch(
            &modules["swerex"]
                .parent()
                .context("SWE-ReX package parent missing")?
                .join("deployment/docker.py"),
            &base(root).join("environment"),
            &configuration.index_url,
        )?,
        SweDeployment::Modal => swerex_modal::patch(
            &modules["swerex"],
            &base(root).join("environment"),
            &agent(root).join("swerex_patches"),
        )?,
    }
    // Import the transformed installed SDK without constructors, listeners, images or task calls.
    let patched_modules = probe_named(root, "sdk-patched-import-only", deadline)?;
    if patched_modules != *modules {
        bail!("SWE patched import location changed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn locked_sync_argv_has_no_resolution_or_interpreter_acquisition() {
        let root = Path::new("/private/cache");
        let spec = sync_spec(root, Path::new("/tools/uv"), Path::new("/tools/python3.11"));
        assert!(spec.clear_environment);
        assert_eq!(
            spec.args,
            vec![
                "--no-config",
                "sync",
                "--locked",
                "--no-build",
                "--project",
                "/private/cache/swe-sdk-v1/project",
                "--python",
                "/tools/python3.11",
                "--no-python-downloads"
            ]
        );
        assert!(
            spec.envs
                .contains(&("UV_PYTHON_DOWNLOADS".into(), "never".into()))
        );
        assert!(!spec.envs.iter().any(|(name, _)| matches!(
            name.as_str(),
            "PYTHONPATH" | "PYTHONHOME" | "UV_INDEX_URL" | "PIP_INDEX_URL"
        )));
    }
}
