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

    run_step(
        &CommandSpec::new("git").args([
            "clone",
            "--recurse-submodules",
            "--branch",
            definition.repo_ref,
            definition.repo_url,
            &target.display().to_string(),
        ]),
        dry_run,
    )
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
        EvalId::TerminalBench => {
            vec![CommandSpec::new("uv").args([
                "tool",
                "install",
                "--python",
                "3.12",
                "terminal-bench",
            ])]
        }
        EvalId::SweBenchPro => vec![
            CommandSpec::new("git")
                .args(["submodule", "update", "--init", "--recursive"])
                .cwd(harness),
        ],
        EvalId::McpAtlas => {
            vec![CommandSpec::new("docker").args(["pull", "ghcr.io/scaleapi/mcp-atlas:1.2.5"])]
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
