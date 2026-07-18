use super::{registry::selected_evals, *};

pub(super) fn doctor_evals(args: EvalDoctorArgs) -> Result<()> {
    let root = cache_root(args.cache_root)?;
    let views = selected_evals(&args.evals, args.pack)
        .into_iter()
        .map(|definition| doctor_view(definition, &root))
        .collect::<Vec<_>>();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&views)?);
        return Ok(());
    }

    println!("SkippyBench eval doctor");
    println!("cache: {}", root.display());
    for view in views {
        println!();
        println!("{}:", view.eval_id);
        println!("  installed: {}", view.installed);
        println!("  harness: {}", view.harness_dir);
        for check in view.checks {
            let status = if check.ok { "ok" } else { "missing" };
            println!("  {status:<7} {} - {}", check.name, check.detail);
        }
    }
    Ok(())
}

pub(super) fn preflight_eval_run(definition: EvalDefinition) -> Result<()> {
    let failed = definition
        .required_tools
        .iter()
        .map(|tool| tool_check(tool))
        .filter(|check| !check.ok)
        .map(|check| format!("{} - {}", check.name, check.detail))
        .collect::<Vec<_>>();
    if failed.is_empty() {
        return Ok(());
    }
    bail!(
        "{} prerequisites failed; run `skippy-bench eval doctor {}` for details:\n  {}",
        definition.id.as_str(),
        definition.id.as_str(),
        failed.join("\n  ")
    )
}

fn doctor_view(definition: EvalDefinition, root: &Path) -> DoctorView {
    let harness = harness_dir(root, definition);
    let mut checks = definition
        .required_tools
        .iter()
        .map(|tool| tool_check(tool))
        .collect::<Vec<_>>();
    if definition.id == EvalId::McpAtlas {
        checks.push(port_check("mcp-atlas agent environment", 1984));
        checks.push(port_check("mcp-atlas completion service", 3000));
    }
    DoctorView {
        eval_id: definition.id.as_str(),
        installed: harness.exists(),
        harness_dir: harness.display().to_string(),
        checks,
    }
}

fn tool_check(tool: &str) -> DoctorCheck {
    if tool == "docker" {
        return docker_check();
    }
    let ok = command_exists(tool);
    DoctorCheck {
        name: tool.to_string(),
        ok,
        detail: if ok {
            "found on PATH".to_string()
        } else {
            "not found on PATH".to_string()
        },
    }
}

fn docker_check() -> DoctorCheck {
    let installed = command_exists("docker");
    if !installed {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail: "docker CLI not found on PATH".to_string(),
        };
    }

    let Some(info) = run_probe_command(
        CommandSpec::new("docker").args(["info"]),
        "docker-info",
        Duration::from_secs(10),
    ) else {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail: "docker CLI found, but daemon did not answer within 10s".to_string(),
        };
    };
    if !info.outcome.success {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail: format!(
                "docker CLI found, but daemon is not reachable{}",
                probe_detail_suffix(&info)
            ),
        };
    }

    let container_name = format!("skippybench-doctor-{}", unix_millis().unwrap_or_default());
    let start = run_probe_command(
        CommandSpec::new("docker").args([
            "run",
            "--rm",
            "--name",
            &container_name,
            "--pull=missing",
            "hello-world",
        ]),
        "docker-run",
        Duration::from_secs(60),
    );
    cleanup_docker_container(&container_name);
    let Some(start) = start else {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail:
                "docker daemon is reachable, but a hello-world container did not finish within 60s"
                    .to_string(),
        };
    };
    let ok = start.outcome.success;
    DoctorCheck {
        name: "docker".to_string(),
        ok,
        detail: if ok {
            "docker daemon is reachable and can start containers".to_string()
        } else {
            format!(
                "docker daemon is reachable, but hello-world container start failed{}",
                probe_detail_suffix(&start)
            )
        },
    }
}

struct ProbeCommandResult {
    outcome: CommandOutcome,
    stdout: String,
    stderr: String,
}

fn run_probe_command(
    spec: CommandSpec,
    label: &str,
    timeout: Duration,
) -> Option<ProbeCommandResult> {
    let run_dir = temp_probe_dir(label)?;
    let stdout_path = run_dir.join("stdout.log");
    let stderr_path = run_dir.join("stderr.log");
    let outcome =
        run_command_with_timeout(&spec, Some(timeout), &stdout_path, &stderr_path).ok()?;
    let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = fs::remove_dir_all(run_dir);
    Some(ProbeCommandResult {
        outcome,
        stdout,
        stderr,
    })
}

fn temp_probe_dir(label: &str) -> Option<PathBuf> {
    let millis = unix_millis().ok()?;
    let dir = env::temp_dir().join(format!(
        "skippy-bench-{label}-{}-{millis}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

fn cleanup_docker_container(name: &str) {
    let _ = Command::new("docker").args(["rm", "-f", name]).status();
}

fn probe_detail_suffix(result: &ProbeCommandResult) -> String {
    if result.outcome.timed_out {
        return " (timed out)".to_string();
    }
    let detail =
        first_nonempty_line(&result.stderr).or_else(|| first_nonempty_line(&result.stdout));
    detail.map(|line| format!(": {line}")).unwrap_or_default()
}

fn first_nonempty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}

fn port_check(name: &str, port: u16) -> DoctorCheck {
    let ok = localhost_port_ready(port);
    DoctorCheck {
        name: name.to_string(),
        ok,
        detail: if ok {
            format!("localhost:{port} is reachable")
        } else {
            format!("expected on localhost:{port} when running this eval")
        },
    }
}

fn localhost_port_ready(port: u16) -> bool {
    let timeout = Duration::from_millis(500);
    ("127.0.0.1", port)
        .to_socket_addrs()
        .ok()
        .into_iter()
        .flatten()
        .any(|addr| TcpStream::connect_timeout(&addr, timeout).is_ok())
}
