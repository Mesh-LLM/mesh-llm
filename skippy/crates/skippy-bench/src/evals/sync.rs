use super::{registry::selected_evals, *};

pub(super) fn sync_evals(args: EvalSyncArgs) -> Result<()> {
    let root = cache_root(args.cache_root)?;
    fs::create_dir_all(harness_root(&root)).with_context(|| {
        format!(
            "create eval harness cache {}",
            harness_root(&root).display()
        )
    })?;

    for definition in selected_evals(&args.evals, args.pack) {
        println!("sync {}", definition.id.as_str());
        sync_repo(definition, &root, args.dry_run)?;
        if definition.id == EvalId::SweBenchPro {
            run_step(
                &CommandSpec::new("git")
                    .args(["submodule", "update", "--init", "--recursive"])
                    .cwd(harness_dir(&root, definition)),
                args.dry_run,
            )?;
        }
        if !args.dry_run {
            super::harness_source::admit_run(&root, definition)?;
        }
        for step in sync_steps(definition, &root) {
            run_step(&step, args.dry_run)?;
        }
    }
    Ok(())
}

fn sync_repo(definition: EvalDefinition, root: &Path, dry_run: bool) -> Result<()> {
    let target = harness_dir(root, definition);
    if target.exists() {
        for step in existing_repo_sync_steps(&target, definition.repo_ref) {
            run_step(&step, dry_run)?;
        }
        return Ok(());
    }

    let steps = new_eval_repo_sync_steps(&target, definition);
    for step in steps {
        run_step(&step, dry_run)?;
    }
    Ok(())
}

pub(super) fn new_eval_repo_sync_steps(
    target: &Path,
    definition: EvalDefinition,
) -> [CommandSpec; 3] {
    let mut steps = new_repo_sync_steps(target, definition.repo_url, definition.repo_ref);
    if matches!(
        definition.id,
        EvalId::McpAtlas | EvalId::SweBenchPro | EvalId::SpeedBench
    ) {
        // Fixed checkout precedes submodule acquisition for these source capsules.
        steps[0].args.retain(|arg| arg != "--recurse-submodules");
    }
    steps
}

pub(super) fn new_repo_sync_steps(
    target: &Path,
    repo_url: &str,
    repo_ref: &str,
) -> [CommandSpec; 3] {
    let target = target.display().to_string();
    [
        CommandSpec::new("git").args(["clone", "--recurse-submodules", repo_url, &target]),
        CommandSpec::new("git").args(["-C", &target, "fetch", "--prune", "origin", repo_ref]),
        CommandSpec::new("git").args(["-C", &target, "checkout", "--detach", "FETCH_HEAD"]),
    ]
}

pub(super) fn existing_repo_sync_steps(target: &Path, repo_ref: &str) -> [CommandSpec; 2] {
    let target = target.display().to_string();
    [
        CommandSpec::new("git").args(["-C", &target, "fetch", "--prune", "origin", repo_ref]),
        CommandSpec::new("git").args(["-C", &target, "checkout", "--detach", "FETCH_HEAD"]),
    ]
}

fn sync_steps(definition: EvalDefinition, root: &Path) -> Vec<CommandSpec> {
    let harness = harness_dir(root, definition);
    match definition.id {
        EvalId::SpeedBench => Vec::new(),
        EvalId::TerminalBench | EvalId::SweGym => {
            vec![CommandSpec::new("uv").args(["sync"]).cwd(harness)]
        }
        EvalId::SweBenchPro => Vec::new(),
        EvalId::McpAtlas => {
            vec![CommandSpec::new("docker").args(["pull", super::registry::MCP_ATLAS_IMAGE])]
        }
    }
}

fn run_step(step: &CommandSpec, dry_run: bool) -> Result<()> {
    println!("{}", step.display());
    if dry_run {
        return Ok(());
    }

    let status = step
        .command()
        .status()
        .with_context(|| format!("start {}", step.program))?;
    if !status.success() {
        bail!("command failed with status {status}: {}", step.display());
    }
    Ok(())
}
