use super::{
    Error, Options,
    boundary::{Canary, Roots},
};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Target {
    pub(crate) base: PathBuf,
    pub(crate) path: PathBuf,
}
impl Target {
    pub(crate) fn new(base: &Path, path: PathBuf) -> Self {
        Self {
            base: base.to_owned(),
            path,
        }
    }
}
pub(crate) struct Plan {
    pub(crate) targets: Vec<Target>,
    pub(crate) replay: Option<(PathBuf, PathBuf)>,
}
impl Plan {
    pub(crate) fn files(targets: Vec<Target>) -> Self {
        Self {
            targets,
            replay: None,
        }
    }
}

fn common(roots: &Roots, canary: &Canary) -> Vec<Target> {
    ["target/debug", ".deps/llama.cpp"]
        .map(|name| Target::new(&roots.workspace, canary.root.join(name)))
        .into()
}

pub(super) fn build(roots: &Roots, canary: &Canary, options: &Options) -> Plan {
    let mut targets = common(roots, canary);
    let native = format!(".deps/llama-{}-{}", canary.key, canary.pass);
    for suffix in [
        String::new(),
        "-workloads".into(),
        format!("-verification-{}", canary.key),
        format!("-verification-{}-workloads", canary.key),
    ] {
        targets.push(Target::new(
            &roots.workspace,
            roots.workspace.join(format!("{native}{suffix}")),
        ));
    }
    for name in [
        format!("canary-previous-{}", canary.pass),
        format!("canary-feedback-{}", canary.pass),
    ] {
        targets.push(Target::new(&roots.temporary, roots.temporary.join(name)));
    }
    if options.package_uploaded {
        targets.push(Target::new(
            &roots.temporary,
            roots
                .temporary
                .join(format!("canary-export-{}-{}", canary.key, canary.pass)),
        ));
    }
    if options.evidence_uploaded {
        targets.push(Target::new(
            &roots.workspace,
            canary.root.join(format!(
                ".deps/llama-canary-state-{}-{}",
                canary.key, canary.pass
            )),
        ));
    }
    Plan::files(targets)
}

pub(super) fn family(roots: &Roots, canary: &Canary, outcome: (&str, bool)) -> Plan {
    let (shard, uploaded) = outcome;
    let mut targets = common(roots, canary);
    targets.extend([
        Target::new(
            &roots.workspace,
            roots.workspace.join(format!(
                ".deps/canary-input-{}-{}-{shard}",
                canary.key, canary.pass
            )),
        ),
        Target::new(
            &roots.workspace,
            canary.root.join(".deps/canary-workload-oracles"),
        ),
        Target::new(
            &roots.workspace,
            roots.workspace.join("ci/canary-python/.venv"),
        ),
    ]);
    if uploaded {
        targets.push(Target::new(
            &roots.workspace,
            roots.workspace.join(format!(
                "target/canary-evidence-{}/{}-{shard}",
                canary.key, canary.pass
            )),
        ));
    }
    Plan::files(targets)
}

pub(super) fn replay(roots: &Roots, uploaded: bool) -> Plan {
    let root = roots.temporary.join("agentic-replay-worktrees");
    let mut targets = vec![
        Target::new(
            &roots.workspace,
            roots.workspace.join("ci/agentic-replay-nightly/.venv"),
        ),
        Target::new(
            &roots.temporary,
            roots.temporary.join("agentic-replay-history"),
        ),
        Target::new(&roots.temporary, root.clone()),
    ];
    if uploaded {
        targets.push(Target::new(
            &roots.temporary,
            roots.temporary.join("agentic-replay-artifacts"),
        ));
    }
    Plan {
        targets,
        replay: Some((roots.workspace.clone(), root)),
    }
}

pub(super) fn cuda(roots: &Roots, uploaded: bool) -> Plan {
    let mut targets = ["target", ".deps/llama.cpp", ".deps/llama-build"]
        .map(|name| Target::new(&roots.workspace, roots.workspace.join(name)))
        .to_vec();
    if uploaded {
        targets.push(Target::new(
            &roots.workspace,
            roots.workspace.join("dist/native-runtimes"),
        ));
    }
    Plan::files(targets)
}

pub(super) fn smoke_path(workspace: &Path, value: &Path, binary: bool) -> Result<PathBuf, Error> {
    if value
        .components()
        .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(Error::Input("parent traversal in smoke output"));
    }
    let path = if value.is_absolute() {
        value.to_owned()
    } else {
        workspace.join(value)
    };
    let relative = path
        .strip_prefix(workspace)
        .map_err(|_| Error::Escape(path.clone()))?;
    let mut parts = relative.components();
    if !matches!(parts.next(), Some(Component::Normal(name)) if name == "target" || name == "ci-artifacts")
    {
        return Err(Error::Input(
            "smoke output is not in a generated output tree",
        ));
    }
    if binary && parts.next().is_none() {
        return Err(Error::Input(
            "smoke binary must be below a generated output tree",
        ));
    }
    Ok(path)
}

pub(super) fn smoke(roots: &Roots, artifact: &Path, binary: &Path) -> Result<Plan, Error> {
    let artifact = smoke_path(&roots.workspace, artifact, false)?;
    let binary = smoke_path(&roots.workspace, binary, true)?;
    let parent = binary
        .parent()
        .ok_or(Error::Input("smoke binary has no parent"))?;
    let runtime = parent.join("native-runtimes");
    Ok(Plan::files(vec![
        Target::new(&roots.workspace, artifact),
        Target::new(&roots.workspace, binary),
        Target::new(&roots.workspace, runtime),
    ]))
}
