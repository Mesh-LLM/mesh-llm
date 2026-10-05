use super::super::{run::harbor_jobs_output_path, *};

pub(in crate::evals) fn harbor_command(
    definition: EvalDefinition,
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition);
    let session_id = args
        .session_id
        .clone()
        .or_else(|| args.run_id.clone().map(|run_id| format!("skippy-{run_id}")))
        .unwrap_or_else(|| "skippy-bench-session".to_string());
    let jobs_dir = harbor_jobs_output_path(run_dir);
    let task_root = run_dir.join("raw/harbor-tasks");
    let script_path = run_dir.join("raw/run-harbor.sh");
    let job_name = format!(
        "skippy-{}",
        harbor_job_slug(args.run_id.as_deref().unwrap_or("eval"))
    );
    let model = litellm_model_name(&args.model);
    let dataset = match definition.id {
        EvalId::TerminalBench => {
            if args.dataset != "lite" {
                args.dataset.clone()
            } else {
                "terminal-bench@2.0".to_string()
            }
        }
        EvalId::SweGym => match args.dataset.as_str() {
            "lite" | "full" => args.dataset.clone(),
            other => bail!("SWE-Gym dataset must be `lite` or `full`, got {other:?}"),
        },
        _ => bail!("{} is not a Harbor-backed eval", definition.id.as_str()),
    };

    let task_selection = match definition.id {
        EvalId::SweGym => {
            if let Some(task_id) = args.task_id.as_deref() {
                validate_task_id(task_id)?;
            }
            if args.task_id.is_none() {
                let harbor_dataset = if dataset == "lite" {
                    "swegym-lite"
                } else {
                    "swegym"
                };
                format!("set -- -d {}\n", shell_quote(harbor_dataset))
            } else {
                let instance = args
                    .task_id
                    .as_deref()
                    .map(|task_id| format!(" --instance-id {}", shell_quote(task_id)))
                    .unwrap_or_default();
                let task_path = args
                    .task_id
                    .as_deref()
                    .map(|task_id| shell_quote(&task_root.join(task_id).display().to_string()))
                    .unwrap_or_else(|| shell_quote(&task_root.display().to_string()));
                format!(
                    "set -- -p {}\nuv run --with swebench adapters/swegym/run_adapter.py --dataset {}{} --task-dir {}\n",
                    task_path,
                    shell_quote(&dataset),
                    instance,
                    shell_quote(&task_root.display().to_string())
                )
            }
        }
        EvalId::TerminalBench => {
            if args.task_id.is_some() {
                bail!("--task-id is currently supported only for swe-gym");
            }
            format!("set -- -d {}\n", shell_quote(&dataset))
        }
        _ => unreachable!(),
    };

    let task_container_endpoint = args
        .harbor_endpoint_url
        .as_ref()
        .map(|_| " --ae OPENAI_BASE_URL=\"$SKIPPY_BENCH_TASK_ENDPOINT_URL\"")
        .unwrap_or("");
    let script = format!(
        "#!/bin/sh\nset -eu\n{task_selection}exec uv run harbor jobs start \"$@\" -a {agent} -m {model} -o {jobs} --job-name {job} --n-concurrent {concurrency} --ak api_base=\"$SKIPPY_BENCH_BASE_URL\" --ak session_id=\"$SKIPPY_BENCH_SESSION_ID\"{task_container_endpoint}\n",
        task_selection = task_selection,
        agent = shell_quote(&args.agent),
        model = shell_quote(&model),
        jobs = shell_quote(&jobs_dir.display().to_string()),
        job = shell_quote(&job_name),
        concurrency = args.endpoint_concurrency,
        task_container_endpoint = task_container_endpoint,
    );
    fs::write(&script_path, script)
        .with_context(|| format!("write Harbor launcher {}", script_path.display()))?;

    let mut command = CommandSpec::new("sh")
        .args([script_path.display().to_string()])
        .cwd(harness)
        .env("SKIPPY_BENCH_BASE_URL", args.base_url.clone())
        .env("SKIPPY_BENCH_SESSION_ID", session_id.clone())
        .secret_env("OPENAI_API_KEY", args.api_key.clone());
    if let Some(endpoint) = &args.harbor_endpoint_url {
        command = command.env("SKIPPY_BENCH_TASK_ENDPOINT_URL", endpoint.clone());
    }
    Ok(command)
}

fn validate_task_id(task_id: &str) -> Result<()> {
    if task_id.is_empty()
        || Path::new(task_id).is_absolute()
        || task_id.contains('/')
        || task_id.contains('\\')
        || task_id.contains("..")
    {
        bail!(
            "SWE-Gym --task-id must be a single safe task name without separators, absolute paths, or '..': {task_id:?}"
        );
    }
    Ok(())
}

fn harbor_job_slug(run_id: &str) -> String {
    let mut slug = String::new();
    for character in run_id.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-').to_string();
    let slug = if slug.is_empty() {
        "eval".to_string()
    } else {
        slug
    };
    slug.chars().take(48).collect()
}
