use super::*;
struct Fixture(PathBuf);
impl Fixture {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = env::temp_dir().join(format!(
            "swe-sdk-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path.canonicalize().unwrap())
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn absent_preparation_refuses_before_any_sdk_or_output_creation() {
    let f = Fixture::new("missing");
    assert!(admit(&f.0).unwrap_err().to_string().contains("unprepared"));
    assert!(!base(&f.0).exists());
}
#[test]
fn runtime_configuration_is_closed_and_receipt_config_is_exact() {
    assert!(Configuration::new(SweDeployment::Docker, "https://pypi.org/simple".into()).is_ok());
    for url in ["", "file:///tmp", "https://x/$(evil)", "https://x/\n"] {
        assert!(Configuration::new(SweDeployment::Docker, url.into()).is_err());
    }
    let docker =
        Configuration::new(SweDeployment::Docker, "https://pypi.org/simple".into()).unwrap();
    let modal = Configuration::new(SweDeployment::Modal, "https://pypi.org/simple".into()).unwrap();
    assert_ne!(docker, modal);
    assert_ne!(
        docker,
        Configuration::new(
            SweDeployment::Docker,
            "https://mirror.invalid/simple".into()
        )
        .unwrap()
    );
}
#[test]
fn locked_workspace_bytes_preserve_exact_source_path_and_intentional_override() {
    let lock_bytes = super::super::external_sdk_source::read("swe-project/uv.lock").unwrap();
    let project_bytes =
        super::super::external_sdk_source::read("swe-project/pyproject.toml").unwrap();
    let lock = std::str::from_utf8(&lock_bytes).unwrap();
    let project = std::str::from_utf8(&project_bytes).unwrap();
    assert_eq!(sdk_environment::hash_bytes(&lock_bytes), LOCK_SHA);
    assert_eq!(sdk_environment::hash_bytes(&project_bytes), PROJECT_SHA);
    assert!(project.contains("members = [\"source/swe-bench-pro/SWE-agent\"]"));
    assert!(project.contains("setuptools==84.0.0"));
    assert!(project.contains("override-dependencies = [\"swe-rex[modal]==1.4.0\"]"));
    assert!(lock.contains("version = \"0.13.3\""));
}
#[test]
fn template_uses_sealed_interpreter_and_preserves_task_interfaces_without_install_or_patch() {
    let source = include_str!("../adapters/templates/swe_bench_pro_run.sh");
    for operation in [
        "uv run",
        "uv venv",
        "uv pip",
        "pip install",
        "eval patch-swerex",
    ] {
        assert!(!source.contains(operation), "{operation}");
    }
    for preserved in [
        "SDK_GENERATE",
        "gather_patches.py",
        "sweap_eval_full_v2.jsonl",
        "-I -B -m sweagent.run.run run-batch",
        "--random_delay_multiplier 1",
        "--instances.shuffle=False",
        "--agent.model.total_cost_limit 0",
    ] {
        assert!(source.contains(preserved), "{preserved}");
    }
    let generator = super::super::external_sdk_source::read("swe-generate-instances.py").unwrap();
    assert!(
        std::str::from_utf8(&generator)
            .unwrap()
            .contains("generate_sweagent_instances.py")
    );
    assert!(include_str!("../adapters/swe_bench_pro.rs").contains("\"--use_local_docker\""));
    assert!(!source.contains("HF_HUB_OFFLINE"));
    assert!(!source.contains("PYTHON_DOTENV_DISABLED"));
}
#[test]
fn module_admission_refuses_same_named_package_outside_private_sdk() {
    let f = Fixture::new("module");
    let installed = base(&f.0).join("environment/lib/python3.11/site-packages/swerex");
    fs::create_dir_all(&installed).unwrap();
    fs::create_dir_all(agent(&f.0).join("sweagent")).unwrap();
    fs::write(installed.join("__init__.py"), b"installed").unwrap();
    fs::write(agent(&f.0).join("sweagent/__init__.py"), b"source").unwrap();
    let mut modules = BTreeMap::from([
        ("sweagent".into(), agent(&f.0).join("sweagent/__init__.py")),
        ("swerex".into(), installed.join("__init__.py")),
    ]);
    custody::modules(&f.0, &modules).unwrap();
    fs::write(f.0.join("decoy.py"), b"same name").unwrap();
    modules.insert("swerex".into(), f.0.join("decoy.py"));
    assert!(custody::modules(&f.0, &modules).is_err());
}
#[cfg(unix)]
#[test]
fn swe_environment_seal_detects_extra_import_module_and_wrong_python_link() {
    use std::os::unix::fs::symlink;
    let f = Fixture::new("seal");
    let python = f.0.join("python");
    fs::write(&python, b"interpreter").unwrap();
    let environment = f.0.join("environment");
    fs::create_dir_all(environment.join("bin")).unwrap();
    fs::write(environment.join("pyvenv.cfg"), b"private").unwrap();
    symlink(&python, environment.join("bin/python")).unwrap();
    let pins = sdk_environment::environment(
        &environment,
        &python,
        sdk_environment::PythonProfile::Swe311,
    )
    .unwrap();
    fs::write(environment.join("poison.py"), b"new importable code").unwrap();
    assert_ne!(
        pins,
        sdk_environment::environment(
            &environment,
            &python,
            sdk_environment::PythonProfile::Swe311
        )
        .unwrap()
    );
    symlink(&python, environment.join("bin/python3.12")).unwrap();
    assert!(
        sdk_environment::environment(
            &environment,
            &python,
            sdk_environment::PythonProfile::Swe311
        )
        .is_err()
    );
}
#[test]
fn descriptor_and_tool_mutations_refuse_before_sdk_execution() {
    let f = Fixture::new("tools");
    let uv = f.0.join("uv");
    let python = f.0.join("python");
    fs::write(&uv, b"uv tool").unwrap();
    fs::write(&python, b"python tool").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for p in [&uv, &python] {
            fs::set_permissions(p, fs::Permissions::from_mode(0o700)).unwrap();
        }
    }
    let pins = sdk_environment::capture_tools(&uv, &python).unwrap();
    fs::write(&uv, b"changed uv").unwrap();
    assert_ne!(pins, sdk_environment::capture_tools(&uv, &python).unwrap());
    assert!(!base(&f.0).exists());
}
#[test]
fn helper_namespace_refuses_untracked_imports_even_when_expected_script_bytes_match() {
    let f = Fixture::new("helpers");
    fs::create_dir(f.0.join("helper_code")).unwrap();
    // The native gate requires its exact pinned namespace, not only the entry script.
    fs::write(
        f.0.join("helper_code/generate_sweagent_instances.py"),
        b"same entry",
    )
    .unwrap();
    fs::write(
        f.0.join("helper_code/yaml.py"),
        b"shadow installed dependency",
    )
    .unwrap();
    assert!(custody::source_helpers(&f.0).is_err());
    assert!(!base(&f.0).exists());
}
#[test]
fn optional_dry_run_neither_requires_tools_nor_creates_prepared_slot() {
    let f = Fixture::new("dryrun");
    prepare(EvalPrepareSweArgs {
        cache_root: Some(f.0.clone()),
        uv: PathBuf::from("/missing/uv"),
        python: PathBuf::from("/missing/python3.11"),
        deployment: SweDeployment::Docker,
        index_url: "https://pypi.org/simple".into(),
        dry_run: true,
    })
    .unwrap();
    assert!(!base(&f.0).exists());
}
#[test]
fn prepared_index_refuses_plain_or_encoded_credentials_without_echoing_them() {
    for url in [
        "https://user:credential-marker@host/simple",
        "https://user%3Acredential-marker@host/simple",
        "https://host/simple?token=credential-marker",
    ] {
        let error = Configuration::new(SweDeployment::Docker, url.into())
            .unwrap_err()
            .to_string();
        assert!(!error.contains("credential-marker"));
    }
}
#[cfg(unix)]
#[test]
fn docker_read_only_profile_admission_refuses_changed_source_without_mutation() {
    let f = Fixture::new("docker-profile");
    let environment = f.0.join("environment");
    let module = environment.join("lib/python3.11/site-packages/swerex/deployment/docker.py");
    fs::create_dir_all(module.parent().unwrap()).unwrap();
    let official = include_str!("docker-original.txt");
    fs::write(&module, official).unwrap();
    super::super::swerex_index::patch(&module, &environment, "https://pypi.org/simple").unwrap();
    super::super::swerex_index::admit_current(&module, &environment, "https://pypi.org/simple")
        .unwrap();
    let mut drift = fs::read(&module).unwrap();
    drift.extend_from_slice(b"\n# changed sealed SDK\n");
    fs::write(&module, &drift).unwrap();
    assert!(
        super::super::swerex_index::admit_current(&module, &environment, "https://pypi.org/simple")
            .is_err()
    );
    assert_eq!(fs::read(&module).unwrap(), drift);
}
#[test]
fn correctly_sealed_older_modal_profile_refuses_before_filesystem_or_sdk_admission() {
    let configuration =
        Configuration::new(SweDeployment::Modal, "https://pypi.org/simple".into()).unwrap();
    let mut receipt:Receipt=serde_json::from_value(json!({
        "schema_version":1,"parent":registry::SWE_BENCH_PRO_REF,"agent":registry::SWE_AGENT_REF,
        "lock_sha256":LOCK_SHA,"patch_profile":patch_profile(&configuration),"configuration":configuration,
        "uv":"/not-executed/uv","python":"/not-executed/python","git_sha256":"recorded","tool_pins":{},"environment_pins":{},"agent_package_pins":{},"modules":{},"benchmark_qualified":false,
    })).unwrap();
    validate_receipt_contract(&receipt, &configuration).unwrap();
    receipt.patch_profile = "swerex-1.4.0-modal-modern-v2".into();
    assert!(validate_receipt_contract(&receipt, &configuration).is_err());
}
