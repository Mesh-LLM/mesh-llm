//! Real shell/Just environment behavior and structured canary propagation.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use crate::workflow_yaml;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
use workflow_yaml::Node;

fn root() -> PathBuf {
    fs::canonicalize(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
}
fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").expect("canonical tool PATH"))
        .map(|path| path.join(name))
        .find(|path| path.is_file())
        .unwrap_or_else(|| panic!("required fixture tool {name} is unavailable"))
}
fn run(
    executable: PathBuf,
    arguments: Vec<String>,
    path: &str,
    override_target: Option<&str>,
) -> String {
    let mut environment = BTreeMap::from([("PATH".into(), Value::Public(path.into()))]);
    if let Some(home) = std::env::var_os("HOME") {
        environment.insert("HOME".into(), Value::Public(home));
    }
    if let Some(value) = override_target {
        environment.insert(
            "MACOSX_DEPLOYMENT_TARGET".into(),
            Value::Public(value.into()),
        );
    }
    let spec = ProcessSpec {
        executable,
        arguments: arguments
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
        cwd: root(),
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(10),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.cleanup.complete, "{report:?}");
    assert!(
        report.status.is_some_and(|status| status.success()),
        "{report:?}"
    );
    assert!(!report.stdout.truncated, "{report:?}");
    String::from_utf8(report.stdout.bytes_retained)
        .unwrap()
        .trim()
        .into()
}
fn default_target() -> String {
    fs::read_to_string(root().join("scripts/lib/macos-deployment-target.txt"))
        .unwrap()
        .trim()
        .into()
}
#[test]
fn actual_shell_helper_exports_default_override_and_retains_non_macos_scope() {
    let default = default_target();
    let bash = tool("bash");
    for (host, override_target, expected) in [
        ("Darwin", None, default.as_str()),
        ("Darwin", Some(""), default.as_str()),
        ("Darwin", Some("14.0"), "14.0"),
        ("Linux", None, "unset"),
        ("Linux", Some("14.0"), "14.0"),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let uname = temp.path().join("uname");
        fs::write(&uname, format!("#!/bin/sh\nprintf '%s\\n' '{host}'\n")).unwrap();
        fs::set_permissions(&uname, fs::Permissions::from_mode(0o755)).unwrap();
        let path = format!(
            "{}:{}",
            temp.path().display(),
            std::env::var("PATH").unwrap()
        );
        let args = vec![
            "-c".into(),
            "source \"$1\"; \"$2\" -c 'printf \"%s\\n\" \"${MACOSX_DEPLOYMENT_TARGET-unset}\"'"
                .into(),
            "fixture".into(),
            root()
                .join("scripts/lib/macos-deployment-target.sh")
                .to_str()
                .unwrap()
                .into(),
            bash.to_str().unwrap().into(),
        ];
        assert_eq!(
            run(bash.clone(), args, &path, override_target),
            expected,
            "{host}/{override_target:?}"
        );
    }
}
#[test]
fn actual_just_facade_exports_default_empty_and_explicit_override() {
    let just = tool("just");
    let bash = tool("bash");
    let default = default_target();
    for override_target in [None, Some(""), Some("14.0")] {
        let args = vec![
            "--command".into(),
            bash.to_str().unwrap().into(),
            "-c".into(),
            "printf '%s\\n' \"$MACOSX_DEPLOYMENT_TARGET\"".into(),
        ];
        assert_eq!(
            run(
                just.clone(),
                args,
                &std::env::var("PATH").unwrap(),
                override_target
            ),
            override_target
                .filter(|value| !value.is_empty())
                .unwrap_or(&default)
        );
    }
}
fn load(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(root().join(path)).unwrap()).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn steps(node: &Node) -> &[Node] {
    let Some(Node::Seq(steps)) = node.get("steps") else {
        panic!("actual steps required")
    };
    steps
}
fn propagation(action: &Node, workflow: &Node) -> Result<(), String> {
    let action_steps = steps(action.get("runs").ok_or("composite action missing")?);
    let setup = action_steps
        .iter()
        .position(|step| {
            text(step, "run")
                .lines()
                .any(|line| line.trim() == "source scripts/lib/macos-deployment-target.sh")
        })
        .ok_or("shared target source missing")?;
    let export = text(&action_steps[setup], "run");
    if !export.contains("MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET")
        || !export.contains("$GITHUB_ENV")
    {
        return Err("resolved target must reach later steps".into());
    }
    let cache = action_steps
        .iter()
        .position(|step| text(step, "run").contains("macos-deployment-target=%s"))
        .ok_or("target missing from compiler cache identity")?;
    if setup >= cache {
        return Err("cache identity precedes target admission".into());
    }
    let jobs = workflow.get("jobs").ok_or("jobs missing")?;
    let build = steps(jobs.get("build").ok_or("producer missing")?);
    let setup = build
        .iter()
        .position(|step| text(step, "uses") == "./.github/actions/setup-canary-runner")
        .ok_or("setup call missing")?;
    let compile = build
        .iter()
        .position(|step| text(step, "id") == "build")
        .ok_or("build call missing")?;
    if setup >= compile {
        return Err("producer compilation precedes setup".into());
    }
    for step in steps(jobs.get("family").ok_or("family missing")?) {
        if text(step, "run")
            .split([';', '|', '&', '(', ')', '`', '\'', '"'])
            .flat_map(str::split_whitespace)
            .any(|token| {
                token.trim_matches(['\'', '"']) == "cargo"
                    || token.trim_matches(['\'', '"']).ends_with("/cargo")
            })
        {
            return Err("family consumer compiles instead of consuming producer bytes".into());
        }
    }
    Ok(())
}
fn field_mut<'a>(node: &'a mut Node, key: &str) -> &'a mut Node {
    let Node::Map(fields) = node else {
        panic!("mapping required")
    };
    &mut fields.iter_mut().find(|(name, _)| name == key).unwrap().1
}
fn steps_mut(node: &mut Node) -> &mut Vec<Node> {
    let Node::Seq(steps) = field_mut(node, "steps") else {
        panic!("steps required")
    };
    steps
}
#[test]
fn actual_canary_setup_exports_target_before_cache_and_build_and_consumer_does_not_compile() {
    let action = load(".github/actions/setup-canary-runner/action.yml");
    let workflow = load(".github/workflows/llama-canary-family-pass.yml");
    propagation(&action, &workflow).unwrap();
    let mut missing_export = action.clone();
    let action_steps = steps_mut(field_mut(&mut missing_export, "runs"));
    let setup = action_steps
        .iter_mut()
        .find(|step| text(step, "run").contains("source scripts/lib/macos-deployment-target.sh"))
        .unwrap();
    *field_mut(setup, "run") = Node::Scalar("source scripts/lib/macos-deployment-target.sh".into());
    assert!(propagation(&missing_export, &workflow).is_err());
    let mut wrong_order = action.clone();
    let action_steps = steps_mut(field_mut(&mut wrong_order, "runs"));
    let setup = action_steps
        .iter()
        .position(|step| {
            text(step, "run").contains("source scripts/lib/macos-deployment-target.sh")
        })
        .unwrap();
    let cache = action_steps
        .iter()
        .position(|step| text(step, "run").contains("macos-deployment-target=%s"))
        .unwrap();
    action_steps.swap(setup, cache);
    assert!(propagation(&wrong_order, &workflow).is_err());
    let mut wrong_build = workflow.clone();
    let build = steps_mut(field_mut(field_mut(&mut wrong_build, "jobs"), "build"));
    let setup = build
        .iter()
        .position(|step| text(step, "uses") == "./.github/actions/setup-canary-runner")
        .unwrap();
    let compile = build
        .iter()
        .position(|step| text(step, "id") == "build")
        .unwrap();
    build.swap(setup, compile);
    assert!(propagation(&action, &wrong_build).is_err());
    for command in [
        "cargo build --locked",
        "just with-lld cargo build --locked",
        "env CARGO_BUILD_JOBS=2 cargo build --locked",
        "/usr/bin/cargo build --locked",
        "true;cargo build --locked",
        r#"echo "$(cargo build --locked)""#,
    ] {
        let mut compiling_consumer = workflow.clone();
        steps_mut(field_mut(
            field_mut(&mut compiling_consumer, "jobs"),
            "family",
        ))
        .push(Node::Map(vec![(
            "run".into(),
            Node::Scalar(command.into()),
        )]));
        assert!(
            propagation(&action, &compiling_consumer).is_err(),
            "{command}"
        );
    }
}
