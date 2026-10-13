use super::*;
const PREPARATION_SECONDS: u64 = 600;
fn sync_spec(root: &Path, uv: &Path, python: &Path) -> CommandSpec {
    isolated(
        CommandSpec::new(uv.display().to_string()).args([
            "--no-config".to_owned(),
            "sync".to_owned(),
            "--locked".to_owned(),
            "--project".to_owned(),
            base(root).join("project").display().to_string(),
            "--python".to_owned(),
            python.display().to_string(),
            "--no-python-downloads".to_owned(),
        ]),
        root,
    )
}
fn execute(spec: &CommandSpec, root: &Path, phase: &str, deadline: Instant) -> Result<String> {
    let execution = deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(6))
        .context("SPEED preparation deadline expired")?;
    let stdout = base(root).join(format!("logs/{phase}.stdout"));
    let stderr = base(root).join(format!("logs/{phase}.stderr"));
    let outcome = sdk_environment::capture(spec, execution, &stdout, &stderr)?;
    if !outcome.success || outcome.timed_out || Instant::now() >= deadline {
        bail!("SPEED preparation phase {phase} failed; bounded logs retained");
    }
    Ok(String::from_utf8(sdk_environment::read(&stdout, 1048576)?)?)
}
fn acquire_dataset(root: &Path, deadline: Instant) -> Result<()> {
    let helper = external_sdk_source::leaf("speed-prepare-dataset.py")?;
    let hub = base(root).join("hub");
    let result = execute(
        &isolated(
            CommandSpec::new(runtime_python(root).display().to_string()).args([
                "-I".to_owned(),
                "-B".to_owned(),
                helper.display().to_string(),
                hub.display().to_string(),
            ]),
            root,
        ),
        root,
        "dataset",
        deadline,
    );
    external_sdk_source::leaf("speed-prepare-dataset.py")?;
    let text = result?;
    let path = PathBuf::from(text.trim());
    let actual = path.canonicalize()?;
    if !path.is_absolute() || text.lines().count() != 1 || !actual.starts_with(hub.canonicalize()?)
    {
        bail!("SPEED prepared dataset escaped private cache");
    }
    // HF snapshots legitimately link to blobs; admit the canonical immutable blob then copy regular bytes.
    let bytes = sdk_environment::read(&actual, DATASET_BYTES)?;
    validate_dataset(&bytes)?;
    fs::write(dataset(root), bytes)?;
    Ok(())
}
pub(super) fn prepare(args: EvalPrepareMcpArgs) -> Result<()> {
    let root = absolute_path(cache_root(args.cache_root)?)?;
    if !cfg!(unix) || !args.uv.is_absolute() || !args.python.is_absolute() {
        bail!("SPEED preparation needs Unix and explicit absolute existing uv/Python paths");
    }
    if [&root, &args.uv, &args.python]
        .iter()
        .any(|p| p.to_str().is_none_or(|s| s.chars().any(char::is_control)))
    {
        bail!("SPEED paths need control-free UTF-8");
    }
    if args.dry_run {
        println!(
            "{} (pending optional preparation; unqualified)",
            sync_spec(&root, &args.uv, &args.python).display()
        );
        return Ok(());
    }
    validate_source(&root)?;
    external_sdk_source::admit_run(EvalId::SpeedBench)?;
    let uv = args.uv.canonicalize()?;
    let python = args.python.canonicalize()?;
    let tool_pins = sdk_environment::capture_tools(&uv, &python)?;
    initialize(&root)?;
    let deadline = Instant::now() + Duration::from_secs(PREPARATION_SECONDS);
    let version = execute(
        &isolated(
            CommandSpec::new(python.display().to_string()).args(["-I", "-B", "--version"]),
            &root,
        ),
        &root,
        "python-version",
        deadline,
    )?;
    if !version.trim().starts_with("Python 3.12.") {
        bail!("SPEED preparation requires existing CPython3.12");
    }
    execute(
        &sync_spec(&root, &uv, &python),
        &root,
        "locked-sync",
        deadline,
    )?;
    acquire_dataset(&root, deadline)?;
    let receipt = Receipt {
        schema_version: 1,
        source: SOURCE.into(),
        dataset_revision: DATASET_REVISION.into(),
        dataset_sha256: DATASET_SHA.into(),
        python: python.clone(),
        tool_pins,
        environment_pins: sdk_environment::environment(
            &base(&root).join("environment"),
            &python,
            sdk_environment::PythonProfile::Mcp312,
        )?,
        benchmark_qualified: false,
    };
    validate_source(&root)?;
    validate_project(&root)?;
    validate_dataset(&sdk_environment::read(&dataset(&root), DATASET_BYTES)?)?;
    for (path, expected) in &receipt.tool_pins {
        if sdk_environment::hash(path)? != *expected {
            bail!("SPEED tool changed during preparation");
        }
    }
    external_sdk_source::admit_run(EvalId::SpeedBench)?;
    if Instant::now() >= deadline {
        bail!("SPEED preparation deadline expired before receipt publication");
    }
    fs::write(
        base(&root).join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("SPEED prepared; benchmark_qualified=false");
    Ok(())
}

fn initialize(root: &Path) -> Result<()> {
    fs::create_dir(base(root))
        .context("SPEED preparation destination must be fresh; preserve failed evidence")?;
    for name in [
        "home",
        "tmp",
        "cache",
        "logs",
        "project",
        "hub",
        "dataset-cache",
    ] {
        fs::create_dir_all(base(root).join(name))?;
    }
    for name in ["pyproject.toml", "uv.lock"] {
        fs::write(
            base(root).join("project").join(name),
            external_sdk_source::read(&format!("speed-project/{name}"))?,
        )?;
    }
    validate_project(root)?;
    Ok(())
}
