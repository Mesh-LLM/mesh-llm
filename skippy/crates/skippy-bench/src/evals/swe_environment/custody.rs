use super::*;
pub(super) fn admit(root: &Path, receipt: &Receipt) -> Result<()> {
    source_helpers(&original(root))?;
    if sdk_environment::hash(Path::new("/usr/bin/git"))? != receipt.git_sha256 {
        bail!("SWE local Git identity changed");
    }
    if sdk_environment::capture_tools(&receipt.uv, &receipt.python)? != receipt.tool_pins {
        bail!("prepared SWE tool identity changed");
    }
    if sdk_environment::hash_bytes(&sdk_environment::read(
        &project(root).join("pyproject.toml"),
        1048576,
    )?) != PROJECT_SHA
        || sdk_environment::hash_bytes(&sdk_environment::read(
            &project(root).join("uv.lock"),
            1048576,
        )?) != LOCK_SHA
    {
        bail!("prepared SWE project/lock changed");
    }
    harness_source::admit_swe_agent_snapshot(&agent(root))?;
    let package = sdk_environment::package_roster(&agent(root), &agent(root).join("sweagent"))?;
    if package != receipt.agent_package_pins {
        bail!("prepared SWE editable package changed");
    }
    if sdk_environment::environment(
        &base(root).join("environment"),
        &receipt.python,
        sdk_environment::PythonProfile::Swe311,
    )? != receipt.environment_pins
    {
        bail!("prepared SWE environment changed");
    }
    modules(root, &receipt.modules)?;
    let environment = base(root).join("environment");
    match receipt.configuration.deployment {
        SweDeployment::Docker => swerex_index::admit_current(
            &environment.join("lib/python3.11/site-packages/swerex/deployment/docker.py"),
            &environment,
            &receipt.configuration.index_url,
        ),
        SweDeployment::Modal => swerex_modal::admit_patched(&environment),
    }
}
pub(super) fn modules(root: &Path, modules: &BTreeMap<String, PathBuf>) -> Result<()> {
    if modules.len() != 2 || !modules.contains_key("sweagent") || !modules.contains_key("swerex") {
        bail!("prepared SWE module roster differs");
    }
    let expected_agent = agent(root).join("sweagent/__init__.py").canonicalize()?;
    let expected_rex = base(root)
        .join("environment/lib/python3.11/site-packages/swerex/__init__.py")
        .canonicalize()?;
    if modules["sweagent"].canonicalize()? != expected_agent
        || modules["swerex"].canonicalize()? != expected_rex
    {
        bail!("prepared SWE SDK import escaped admitted source/environment");
    }
    Ok(())
}

const HELPERS: &[(&str, &str)] = &[
    (
        "helper_code/create_problem_statement.py",
        "cc8baae4f0be15629ae214d7b30e3faed45bf5d426942d0ac4bc286b1f6b4c88",
    ),
    (
        "helper_code/extract_gold_patches.py",
        "23838dfbdcc054af1d62f62531b952f85fc3e5a949d069e9b7556ca08d7ac4ba",
    ),
    (
        "helper_code/gather_patches.py",
        "a1d4628461fa2b7e374160cad92f0e65f6bcdc67b0fc762fd4cc61438f8ab364",
    ),
    (
        "helper_code/generate_sweagent_instances.py",
        "9d017c12e266a16b9e9f6fedf5216a5c81fe39fdc83c1e880fe834b45b4b8a1b",
    ),
    (
        "helper_code/image_uri.py",
        "d1a858866dd2622c0e37986dd7b86698e5ea53546f30901d1bf0d6ba1b97384f",
    ),
    (
        "helper_code/sweap_eval_full_v2.jsonl",
        "b5b2462bfbf5aeb2cb7ba7d215778a1768b85f9d7ad7f748546c7f80a0ad1510",
    ),
];
pub(super) fn source_helpers(parent: &Path) -> Result<()> {
    let expected: BTreeMap<PathBuf, String> = HELPERS
        .iter()
        .map(|(p, h)| (PathBuf::from(p), format!("file:{h}")))
        .collect();
    if sdk_environment::package_roster(parent, &parent.join("helper_code"))? != expected {
        bail!("SWE helper namespace has unadmitted source or bytecode");
    }
    Ok(())
}
