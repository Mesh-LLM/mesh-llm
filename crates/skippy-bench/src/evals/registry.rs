use super::*;

pub(super) fn list_evals(args: EvalListArgs) -> Result<()> {
    let root = cache_root(args.cache_root.clone())?;
    let views = selected_evals(&[], EvalPack::Core)
        .into_iter()
        .map(|definition| eval_view(definition, &root))
        .collect::<Vec<_>>();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&views)?);
        return Ok(());
    }

    println!("SkippyBench external evals");
    println!("cache: {}", root.display());
    for view in views {
        let status = if view.installed {
            "installed"
        } else {
            "not installed"
        };
        println!(
            "  {:<16} {:<13} {}",
            view.id,
            format!("[{status}]"),
            view.description
        );
    }
    Ok(())
}

pub(super) fn info_eval(args: EvalInfoArgs) -> Result<()> {
    let root = cache_root(args.cache_root.clone())?;
    let definition = definition(args.eval);
    let view = eval_view(definition, &root);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&view)?);
        return Ok(());
    }

    println!("{} ({})", view.name, view.id);
    println!("description: {}", view.description);
    println!("repo: {} @ {}", view.repo_url, view.repo_ref);
    println!("pack: {}", view.pack);
    println!("disk: {}", view.disk_estimate);
    println!("installed: {}", view.installed);
    println!("harness: {}", view.harness_dir);
    println!("requires: {}", view.required_tools.join(", "));
    print_notes("sync", view.sync_notes);
    print_notes("run", view.run_notes);
    Ok(())
}

pub(super) fn definition(id: EvalId) -> EvalDefinition {
    match id {
        EvalId::SpeedBench => EvalDefinition {
            id,
            name: "llama.cpp SPEED-Bench",
            repo_url: "https://github.com/ggml-org/llama.cpp.git",
            repo_ref: "master",
            cache_name: "llama.cpp",
            description: "OpenAI-compatible serving latency and throughput benchmark.",
            disk_estimate: "1-3GB including clone and Python dataset cache",
            required_tools: &["git", "uv", "python3"],
            sync_notes: &["Clones llama.cpp; SPEED-Bench data is fetched by the Python runner."],
            run_notes: &[
                "Runs the upstream SPEED-Bench client with category=all and no sample limit.",
            ],
        },
        EvalId::TerminalBench => EvalDefinition {
            id,
            name: "Terminal-Bench",
            repo_url: "https://github.com/harbor-framework/terminal-bench.git",
            repo_ref: "main",
            cache_name: "terminal-bench",
            description: "Agent benchmark for real terminal tasks in Docker sandboxes.",
            disk_estimate: "5-20GB depending on Docker images",
            required_tools: &["git", "uv", "python3.12", "tb", "docker"],
            sync_notes: &["Clones task repo and installs the terminal-bench uv tool."],
            run_notes: &["Runs the Terminal-Bench CLI against the full selected dataset."],
        },
        EvalId::SweBenchPro => EvalDefinition {
            id,
            name: "SWE-Bench Pro",
            repo_url: "https://github.com/scaleapi/SWE-bench_Pro-os.git",
            repo_ref: "main",
            cache_name: "swe-bench-pro",
            description: "Long-horizon software-engineering patch benchmark.",
            disk_estimate: "10GB+ before task Docker images",
            required_tools: &["git", "uv", "python3", "docker"],
            sync_notes: &["Clones the official repo and initializes submodules."],
            run_notes: &[
                "Generates SWE-agent instances from the full SWE-Bench Pro test split.",
                "Runs the synced SWE-agent scaffold, gathers .pred patches, then invokes swe_bench_pro_eval.py.",
            ],
        },
        EvalId::McpAtlas => EvalDefinition {
            id,
            name: "MCP-Atlas",
            repo_url: "https://github.com/scaleapi/mcp-atlas.git",
            repo_ref: "main",
            cache_name: "mcp-atlas",
            description: "Tool-use benchmark over real MCP servers and tasks.",
            disk_estimate: "10GB+ including Docker image",
            required_tools: &["git", "uv", "python3", "docker", "make", "curl"],
            sync_notes: &["Clones repo and pulls the prebuilt MCP-Atlas Docker image."],
            run_notes: &[
                "Starts the MCP environment and completion service when they are not already running.",
                "Runs the MCP-Atlas completion and scoring scripts with --no-filter and without --num-tasks or tool_choice overrides.",
            ],
        },
    }
}

pub(super) fn selected_evals(requested: &[EvalId], pack: EvalPack) -> Vec<EvalDefinition> {
    let ids = if requested.is_empty() {
        match pack {
            EvalPack::Core => CORE_EVALS.to_vec(),
        }
    } else {
        requested.to_vec()
    };
    ids.into_iter().map(definition).collect()
}

fn eval_view(definition: EvalDefinition, root: &Path) -> EvalView {
    let harness_dir = harness_dir(root, definition);
    EvalView {
        id: definition.id.as_str(),
        name: definition.name,
        pack: "core",
        repo_url: definition.repo_url,
        repo_ref: definition.repo_ref,
        installed: harness_dir.exists(),
        harness_dir: harness_dir.display().to_string(),
        description: definition.description,
        disk_estimate: definition.disk_estimate,
        required_tools: definition.required_tools,
        sync_notes: definition.sync_notes,
        run_notes: definition.run_notes,
    }
}
