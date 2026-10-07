use super::*;
struct Fixture(PathBuf);
impl Fixture {
    fn new(name: &str) -> Self {
        let root = env::temp_dir().join(format!(
            "mcp-sdk-{name}-{}-{}",
            std::process::id(),
            unix_millis().unwrap()
        ));
        fs::create_dir(&root).unwrap();
        Self(root.canonicalize().unwrap())
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn missing_prepared_receipt_refuses_without_creating_sdk_or_output() {
    let f = Fixture::new("missing");
    let error = admit(&f.0).unwrap_err().to_string();
    assert!(error.contains("unprepared"));
    assert!(!base(&f.0).exists());
}
#[test]
fn preparation_has_exact_locked_args_and_clears_poisoned_ambient_environment() {
    let root = Path::new("/operator/cache");
    let spec = sync_spec(
        root,
        Path::new("/operator/uv"),
        Path::new("/operator/python"),
    );
    assert_eq!(spec.program, "/operator/uv");
    assert_eq!(
        spec.args,
        [
            "--no-config",
            "sync",
            "--locked",
            "--project",
            "/operator/cache/harnesses/mcp-atlas/services/mcp_eval",
            "--python",
            "/operator/python",
            "--no-python-downloads"
        ]
    );
    // Command introspection reports cleared inheritance as None. Explicit environment is finite.
    let command = spec.command();
    assert!(spec.clear_environment);
    let vars: BTreeMap<_, _> = command
        .get_envs()
        .map(|(k, v)| (k.to_owned(), v.map(std::ffi::OsStr::to_owned)))
        .collect();
    assert_eq!(
        vars.get(std::ffi::OsStr::new("UV_PYTHON_DOWNLOADS"))
            .unwrap()
            .as_deref(),
        Some(std::ffi::OsStr::new("never"))
    );
    assert!(!vars.contains_key(std::ffi::OsStr::new("PYTHONPATH")));
    assert!(!vars.contains_key(std::ffi::OsStr::new("PYTHONHOME")));
    assert!(spec.secret_envs.is_empty());
}
#[test]
fn source_fixture_detects_same_name_changes_and_untracked_import_or_bytecode() {
    let f = Fixture::new("source");
    fs::create_dir(f.0.join("mcp_completion")).unwrap();
    let file = f.0.join("mcp_completion/__init__.py");
    fs::write(&file, b"admitted source").unwrap();
    let digest = custody::hash(&file).unwrap();
    let pins = [("mcp_completion/__init__.py", digest.as_str())];
    custody::source_expected(&f.0, &pins).unwrap();
    fs::write(&file, b"replaced source").unwrap();
    assert!(custody::source_expected(&f.0, &pins).is_err());
    fs::write(&file, b"admitted source").unwrap();
    for name in ["poison.py", "poison.pyc"] {
        let extra = f.0.join("mcp_completion").join(name);
        fs::write(&extra, b"untracked import").unwrap();
        assert!(custody::source_expected(&f.0, &pins).is_err());
        fs::remove_file(extra).unwrap();
    }
}
#[cfg(unix)]
#[test]
fn environment_seal_detects_package_entry_and_interpreter_link_replacement() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let f = Fixture::new("seal");
    let python = f.0.join("python");
    fs::write(&python, b"python identity").unwrap();
    fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
    let environment = f.0.join("environment");
    fs::create_dir(&environment).unwrap();
    fs::create_dir(environment.join("bin")).unwrap();
    fs::write(environment.join("pyvenv.cfg"), b"private environment").unwrap();
    symlink(&python, environment.join("bin/python")).unwrap();
    let package = environment.join("package.py");
    fs::write(&package, b"package entry").unwrap();
    let before = custody::environment(&environment, &python).unwrap();
    fs::write(&package, b"changed entry").unwrap();
    assert_ne!(before, custody::environment(&environment, &python).unwrap());
    fs::remove_file(environment.join("bin/python")).unwrap();
    symlink("/unadmitted/python", environment.join("bin/python")).unwrap();
    assert!(custody::environment(&environment, &python).is_err());
}
#[cfg(unix)]
#[test]
fn tool_replacement_is_refused_before_any_sdk_child() {
    use std::os::unix::fs::PermissionsExt as _;
    let f = Fixture::new("tools");
    let uv = f.0.join("uv");
    let python = f.0.join("python");
    for path in [&uv, &python] {
        fs::write(path, path.file_name().unwrap().as_encoded_bytes()).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let pins = custody::capture_tools(&uv, &python).unwrap();
    let receipt = Receipt {
        schema_version: 1,
        source: registry::MCP_ATLAS_REF.into(),
        uv: uv.clone(),
        python,
        tool_pins: pins,
        environment_pins: BTreeMap::new(),
        probe_modules: BTreeMap::new(),
        benchmark_qualified: false,
    };
    custody::tools(&receipt).unwrap();
    fs::write(uv, b"changed tool").unwrap();
    assert!(custody::tools(&receipt).is_err());
}
#[cfg(unix)]
#[test]
fn actual_closed_child_ignores_poisoned_ambient_import_environment() {
    let executable = env::current_exe().unwrap();
    let output = Command::new(executable)
        .args([
            "--ignored",
            "--exact",
            "evals::mcp_environment::tests::poisoned_ambient_owned_child",
            "--nocapture",
        ])
        .env("PYTHONPATH", "/unadmitted/poison")
        .env("PYTHONHOME", "/unadmitted/home")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}
#[cfg(unix)]
#[test]
#[ignore = "owned isolated child invoked by causal parent test"]
fn poisoned_ambient_owned_child() {
    assert_eq!(env::var("PYTHONPATH").unwrap(), "/unadmitted/poison");
    assert_eq!(env::var("PYTHONHOME").unwrap(), "/unadmitted/home");
    let f = Fixture::new("ambient");
    fs::create_dir_all(base(&f.0).join("logs")).unwrap();
    let spec = isolated(
        CommandSpec::new("/bin/sh").args([
            "-c",
            "printf '%s:%s' \"${PYTHONPATH-unset}\" \"${PYTHONHOME-unset}\"",
        ]),
        &f.0,
    );
    let out = base(&f.0).join("logs/ambient.out");
    let err = base(&f.0).join("logs/ambient.err");
    let outcome = capture(&spec, Duration::from_secs(2), &out, &err).unwrap();
    assert!(outcome.success);
    assert_eq!(fs::read_to_string(out).unwrap(), "unset:unset");
}
#[cfg(unix)]
#[test]
fn actual_capture_overflow_refuses_and_retains_only_bounded_bytes() {
    let f = Fixture::new("capture");
    let out = f.0.join("out");
    let err = f.0.join("err");
    let spec = CommandSpec::new("/bin/sh").args(["-c", "head -c 2097152 /dev/zero"]);
    assert!(capture(&spec, Duration::from_secs(2), &out, &err).is_err());
    assert!(fs::metadata(out).unwrap().len() <= CAPTURE_LIMIT);
}
#[test]
fn runtime_template_has_three_isolated_leaves_and_no_package_acquisition() {
    let script = include_str!("../adapters/templates/mcp_atlas_run.sh");
    assert_eq!(script.matches("\"$SDK_PYTHON\" -I -B").count(), 3);
    for forbidden in ["uv run", "uv sync", "uv pip", "--with-requirements"] {
        assert!(!script.contains(forbidden));
    }
    for required in [
        "--no-filter",
        "--concurrency \"$COMPLETION_CONCURRENCY\"",
        "--evaluator-model \"$EVAL_MODEL\"",
        "--input_huggingface ScaleAI/MCP-Atlas",
    ] {
        assert!(script.contains(required));
    }
    assert!(!script.contains("HF_HUB_OFFLINE"));
    assert!(!script.contains("PYTHON_DOTENV_DISABLED"));
}
#[cfg(unix)]
#[test]
fn admitted_environment_boundary_refuses_stale_sdk_bytes_and_outside_import_location() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let f = Fixture::new("admission");
    let uv = f.0.join("uv");
    let python = f.0.join("python");
    for path in [&uv, &python] {
        fs::write(path, b"fixed executable").unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let environment = base(&f.0).join("environment");
    let site = environment.join("lib/python3.12/site-packages");
    fs::create_dir_all(&site).unwrap();
    fs::create_dir(environment.join("bin")).unwrap();
    fs::write(environment.join("pyvenv.cfg"), b"fixture environment").unwrap();
    symlink(&python, environment.join("bin/python")).unwrap();
    let project = project(&f.0);
    fs::create_dir_all(project.join("mcp_completion")).unwrap();
    let mut modules = BTreeMap::new();
    for name in [
        "mcp_completion",
        "mcp_completion.main",
        "pandas",
        "numpy",
        "aiohttp",
        "aiofiles",
        "aiocsv",
        "requests",
        "dotenv",
        "datasets",
        "litellm",
        "tenacity",
        "tqdm",
        "matplotlib",
        "nest_asyncio",
        "uvicorn",
        "fastapi",
        "pydantic",
        "httpx",
    ] {
        let path = if name == "mcp_completion" {
            project.join("mcp_completion/__init__.py")
        } else if name == "mcp_completion.main" {
            project.join("mcp_completion/main.py")
        } else {
            site.join(format!("{name}.py"))
        };
        fs::write(&path, name.as_bytes()).unwrap();
        modules.insert(name.into(), path);
    }
    let receipt = Receipt {
        schema_version: 1,
        source: registry::MCP_ATLAS_REF.into(),
        uv: uv.clone(),
        python: python.clone(),
        tool_pins: custody::capture_tools(&uv, &python).unwrap(),
        environment_pins: custody::environment(&environment, &python).unwrap(),
        probe_modules: modules,
        benchmark_qualified: false,
    };
    // Production admit invokes fixed source admission before this exact environment boundary.
    admit_environment(&f.0, &receipt).unwrap();
    fs::write(site.join("litellm.py"), b"replacement").unwrap();
    assert!(admit_environment(&f.0, &receipt).is_err());
    fs::write(site.join("litellm.py"), b"litellm").unwrap();
    admit_environment(&f.0, &receipt).unwrap();
    let mut receipt = receipt;
    receipt.probe_modules.insert("litellm".into(), uv);
    assert!(admit_environment(&f.0, &receipt).is_err());
}
#[cfg(unix)]
#[test]
fn descriptor_reader_refuses_fifo_and_symlink_without_a_writer_and_preserves_regular_cap() {
    use std::{ffi::CString, os::unix::fs::symlink};
    let f = Fixture::new("descriptor-read");
    let fifo = f.0.join("fifo");
    let name = CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: the CString lives through this call; this fresh private path has no reader/writer.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(custody::read(&fifo, 16).is_err());
    let regular = f.0.join("regular");
    fs::write(&regular, b"exact bounded bytes").unwrap();
    let link = f.0.join("link");
    symlink(&regular, &link).unwrap();
    assert!(custody::read(&link, 64).is_err());
    assert_eq!(custody::read(&regular, 19).unwrap(), b"exact bounded bytes");
    assert!(custody::read(&regular, 18).is_err());
}
