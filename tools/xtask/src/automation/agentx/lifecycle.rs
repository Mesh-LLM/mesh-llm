use super::{
    Prepared, Profile, cache_decisions::CacheDecisions, command, inputs, trace_context, write,
};
use crate::{
    command::DynResult,
    process::{
        self,
        retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
    },
};
use serde_json::json;
use std::time::Duration;

pub(super) const STARTUP_TIMEOUT_SECS: u64 = 900;
const LLAMA_CACHE_RAM_MIB_ENV: &str = "AGENTX_LLAMA_CACHE_RAM_MIB";

fn llama_comparison_context(concurrency: u32) -> DynResult<u32> {
    trace_context::LIMIT
        .checked_mul(u64::from(concurrency))
        .and_then(|tokens| u32::try_from(tokens).ok())
        .ok_or_else(|| "llama comparison context exceeds supported size".into())
}

fn parse_llama_cache_ram_mib(value: &str) -> DynResult<u32> {
    match value.parse::<u32>() {
        Ok(mib) if mib > 0 => Ok(mib),
        _ => Err(format!("{LLAMA_CACHE_RAM_MIB_ENV} must be a positive MiB count").into()),
    }
}

fn llama_cache_ram_mib() -> DynResult<Option<u32>> {
    match std::env::var(LLAMA_CACHE_RAM_MIB_ENV) {
        Ok(value) => parse_llama_cache_ram_mib(&value).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("{LLAMA_CACHE_RAM_MIB_ENV}: {error}").into()),
    }
}

fn llama_comparison_arguments(
    concurrency: u32,
    cache_ram_mib: Option<u32>,
) -> DynResult<Vec<String>> {
    let mut args = vec![
        "--ctx-size".into(),
        llama_comparison_context(concurrency)?.to_string(),
    ];
    if let Some(mib) = cache_ram_mib {
        args.extend(["--cache-ram".into(), mib.to_string()]);
    }
    Ok(args)
}

fn server(
    input: &Prepared,
    engine: &str,
    port: u16,
    concurrency: u32,
) -> DynResult<process::ProcessSpec> {
    // Skippy plans its own capacity. Preserve llama's explicit parallel setting:
    // its auto-parallel mode also changes the baseline's unified-KV policy.
    // For the fixed-work comparison, match the per-slot context ceiling to the
    // common trace budget; llama otherwise allocates the model's full native
    // context for every slot and pays for capacity the workload cannot use.
    let weights = input.weights.to_str().ok_or("weight path")?;
    let (binary, mut args) = if engine == "llama" {
        (
            input.root.join(".deps/agentx-llama-build/bin/llama-server"),
            command::strings(&[
                "--model",
                weights,
                "--alias",
                &input.model.key,
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--parallel",
                &concurrency.to_string(),
                "--batch-size",
                "2048",
                "--ubatch-size",
                "512",
                "--n-gpu-layers",
                "99",
                "--cache-type-k",
                "f16",
                "--cache-type-v",
                "f16",
                "--flash-attn",
                "on",
                "--jinja",
                "--chat-template-kwargs",
                r#"{"enable_thinking":false}"#,
            ]),
        )
    } else {
        (
            input.root.join("target/release/skippy"),
            command::strings(&[
                "serve",
                "--model-path",
                weights,
                "--model-id",
                &input.model.key,
                "--bind-addr",
                &format!("127.0.0.1:{port}"),
                "--startup-timeout-secs",
                &STARTUP_TIMEOUT_SECS.to_string(),
                "--batch-size",
                "2048",
                "--microbatch-size",
                "512",
                "--generation-queue-capacity",
                "256",
                "--generation-admission-timeout-secs",
                "1800",
                "--n-gpu-layers",
                "-1",
                "--kv-cache-type-k",
                "f16",
                "--kv-cache-type-v",
                "f16",
                "--flash-attention",
                "enabled",
                "--speculative-strategy",
                "disabled",
                "--prefix-cache",
                "on",
                "--prefix-cache-budget",
                "8GiB",
                "--kv-cache-disk",
                "off",
                "--chat-template-kwargs",
                r#"{"enable_thinking":false}"#,
            ]),
        )
    };
    if engine == "llama" && input.profile == Profile::Comparison {
        args.extend(llama_comparison_arguments(
            concurrency,
            llama_cache_ram_mib()?,
        )?);
    }
    let mut spec = command::spec(&input.root, binary.to_str().ok_or("server path")?, &args)?;
    if engine == "skippy" {
        spec.arguments.extend([
            process::Value::Public("--runtime-bundle".into()),
            process::Value::Public(input.root.join("dist/native-runtimes").into_os_string()),
        ]);
        if std::env::var("AGENTX_SKIPPY_DEBUG_TELEMETRY").as_deref() == Ok("1") {
            spec.arguments.extend([
                process::Value::Public("--telemetry-level".into()),
                process::Value::Public("debug".into()),
            ]);
            spec.environment.insert(
                "SKIPPY_TELEMETRY_STDERR".into(),
                process::Value::Public("1".into()),
            );
        }
        if std::env::var("AGENTX_SKIPPY_KV_SPAN_DEBUG").as_deref() == Ok("1") {
            enable_native_kv_span_debug(&mut spec);
        }
    }
    spec.environment.insert(
        "MESH_AGENT_SESSION_HEADER".into(),
        process::Value::Public("X-Session-ID".into()),
    );
    spec.environment.insert(
        "MESH_OPENAI_BACKEND_TIMEOUT_SECS".into(),
        process::Value::Public("1800".into()),
    );
    Ok(spec)
}

fn enable_native_kv_span_debug(spec: &mut process::ProcessSpec) {
    spec.arguments
        .push(process::Value::Public("--debug".into()));
    spec.environment.insert(
        "LLAMA_KV_CACHE_DEBUG".into(),
        process::Value::Public("1".into()),
    );
    spec.environment.insert(
        "GGML_LLAMA_LOG_LEVEL".into(),
        process::Value::Public("4".into()),
    );
}
pub(super) fn execute(input: &Prepared, engine: &str, concurrency: u32) -> DynResult<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    write(&input.output.join("worker.json"), input)?;
    let server = server(input, engine, port, concurrency)?;
    write(
        &input.output.join("launch.json"),
        &launch_identity(input, &server, engine, concurrency)?,
    )?;
    let worker = command::spec(
        &input.root,
        std::env::current_exe()?.to_str().ok_or("xtask path")?,
        &command::strings(&[
            "automation",
            "agentx",
            "worker",
            "--input",
            input
                .output
                .join("worker.json")
                .to_str()
                .ok_or("worker path")?,
            "--port",
            &port.to_string(),
            "--concurrency",
            &concurrency.to_string(),
        ]),
    )?;
    let seconds = if input.profile == Profile::Smoke {
        4800
    } else {
        10200
    };
    let memory = super::memory::Monitor::start(&input.output)?;
    let mut owner = Owner {
        memory_pid: std::sync::Arc::clone(&memory.pid),
        cache_decisions: (engine == "skippy"
            && std::env::var("AGENTX_SKIPPY_DEBUG_TELEMETRY").as_deref() == Ok("1"))
        .then(CacheDecisions::new),
        server: Some(launch(input, server, MemberId::Seed, "server", seconds)),
        worker: Some(launch(
            input,
            worker,
            MemberId::WorkerOne,
            "worker",
            seconds,
        )),
        stop: false,
        policy: ExpectedExit::new(&[0], Duration::from_secs(seconds))?,
    };
    drop(listener);
    let report = super::super::retained_session::run(&mut owner, &command::limits(seconds));
    if let Some(decisions) = owner.cache_decisions.as_ref() {
        write(&input.output.join("cache-decisions.json"), decisions)?;
    }
    let report = report?;
    write(
        &input.output.join("memory-summary.json"),
        &serde_json::to_value(memory.finish()?)?,
    )?;
    let clean = report.recovery_success()
        && report.members.len() == 2
        && report.members.iter().all(|member| {
            member.process.cleanup.complete
                && !member.process.cleanup.forced
                && member.process.failure.is_none()
        })
        && report.members.iter().any(|member| {
            member.member == MemberId::WorkerOne
                && member
                    .process
                    .status
                    .as_ref()
                    .and_then(std::process::ExitStatus::code)
                    == Some(0)
        });
    write(
        &input.output.join("lifecycle.json"),
        &json!({"clean":clean,"cleanup_complete":report.members.iter().all(|m| m.process.cleanup.complete),"outcome":format!("{:?}",report.outcome)}),
    )?;
    if !clean {
        return Err(format!("{engine} c{concurrency} failed; partial artifacts retained").into());
    }
    Ok(())
}
fn launch_identity(
    input: &Prepared,
    server: &process::ProcessSpec,
    engine: &str,
    concurrency: u32,
) -> DynResult<serde_json::Value> {
    let matched_llama_context = if engine == "llama" && input.profile == Profile::Comparison {
        Some(llama_comparison_context(concurrency)?)
    } else {
        None
    };
    let runtime = if engine == "skippy" {
        input.root.join("dist/native-runtimes")
    } else {
        input.root.join(".deps/agentx-llama-build/bin")
    };
    let arguments: Vec<_> = server
        .arguments
        .iter()
        .map(|value| match value {
            process::Value::Public(value) => value.to_string_lossy().into_owned(),
            process::Value::Secret(_) => "<redacted>".into(),
        })
        .collect();
    Ok(json!({
        "engine":engine,"command":arguments,
        "container_image":std::env::var("AGENTX_CONTAINER_IMAGE").ok(),
        "execution_mode":if cfg!(target_os = "macos") { "native_metal" } else { "gpu_container" },
        "stack_commit":std::env::var("JIANYANG_COMMIT")?,
        "binary_sha256":crate::product::digest::file_sha256(&server.executable).map_err(|error|error.error)?,
        "runtime_tree_sha256":crate::product::digest::tree_sha256(&runtime).map_err(|error|error.error)?,
        "harness_lock_sha256":crate::product::digest::file_sha256(&input.root.join("skippy/evals/agentx/uv.lock")).map_err(|error|error.error)?,
        "capacity_policy":super::capacity::policy(engine, input.profile),
        "context_tokens":matched_llama_context.map(|_| trace_context::LIMIT),
        "total_context_tokens":matched_llama_context,
        "generation_slots":null,"offered_sessions":concurrency,
        "native_kv_span_debug":engine == "skippy" && std::env::var("AGENTX_SKIPPY_KV_SPAN_DEBUG").as_deref() == Ok("1"),
        "batch_tokens":2048,"microbatch_tokens":512,
        "speculation":false,"dataset_sha256":inputs::DATA_SHA,
        "llama_upstream":std::fs::read_to_string(input.root.join("skippy/llama_cpp/upstream.txt"))?.trim()
    }))
}
fn launch(
    input: &Prepared,
    spec: process::ProcessSpec,
    member: MemberId,
    label: &str,
    seconds: u64,
) -> Launch {
    Launch {
        member,
        spec,
        files: process::OutputFiles {
            stdout: Some(input.output.join(format!("{label}.stdout.log"))),
            stderr: Some(input.output.join(format!("{label}.stderr.log"))),
        },
        readiness_deadline: Duration::from_secs(seconds),
    }
}
struct Owner {
    memory_pid: std::sync::Arc<std::sync::atomic::AtomicU32>,
    cache_decisions: Option<CacheDecisions>,
    server: Option<Launch>,
    worker: Option<Launch>,
    stop: bool,
    policy: ExpectedExit,
}

impl Coordinator for Owner {
    type Rejection = String;
    fn line(
        &mut self,
        member: MemberId,
        line: process::ObservedLine<'_>,
    ) -> process::ProbeDecision<String> {
        if let Some(decisions) = self.cache_decisions.as_mut() {
            decisions.observe(member, line);
        }
        process::ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(server) = context
            .members
            .iter()
            .find(|member| member.member == MemberId::Seed)
        {
            self.memory_pid
                .store(server.pid, std::sync::atomic::Ordering::Relaxed);
        }
        if self.stop {
            return Action::Complete;
        }
        if let Some(server) = self.server.take() {
            return Action::Start(server);
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::Seed && matches!(member.state, MemberState::Starting)
        }) {
            return Action::Admit(MemberId::Seed);
        }
        if let Some(worker) = self.worker.take() {
            return Action::StartExpected {
                launch: worker,
                policy: self.policy.clone(),
            };
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::WorkerOne
                && matches!(member.state, MemberState::ExpectedExit { .. })
        }) {
            self.stop = true;
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kv_span_diagnostic_enables_native_cell_span_logging() {
        let mut spec = command::spec(&std::env::temp_dir(), "sh", &[]).unwrap();
        enable_native_kv_span_debug(&mut spec);
        assert!(spec.arguments.iter().any(|argument| matches!(argument,
            process::Value::Public(value) if value == "--debug")));
        for (key, expected) in [("LLAMA_KV_CACHE_DEBUG", "1"), ("GGML_LLAMA_LOG_LEVEL", "4")] {
            assert!(
                matches!(spec.environment.get(std::ffi::OsStr::new(key)), Some(process::Value::Public(value)) if value == expected)
            );
        }
    }
    #[test]
    fn paired_engines_pin_identical_batch_capacity() {
        let models: Vec<super::super::Model> = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/agentx/models.json"
        ))
        .unwrap();
        let input = Prepared {
            root: "/repo".into(),
            output: "/output".into(),
            weights: "/cache/model.gguf".into(),
            dataset: "/traces".into(),
            model: models[1].clone(),
            profile: Profile::Smoke,
        };
        for (engine, microbatch) in [("llama", "--ubatch-size"), ("skippy", "--microbatch-size")] {
            let spec = server(&input, engine, 8080, 2).unwrap();
            let arguments: Vec<_> = spec
                .arguments
                .iter()
                .map(|value| match value {
                    process::Value::Public(value) => value.to_string_lossy().into_owned(),
                    process::Value::Secret(_) => panic!("unexpected secret in server arguments"),
                })
                .collect();
            for pair in [["--batch-size", "2048"], [microbatch, "512"]] {
                assert!(
                    arguments.windows(2).any(|args| args == pair),
                    "{engine}: {pair:?}"
                );
            }
            for flag in ["--generation-concurrency", "--ctx-size"] {
                assert!(!arguments.iter().any(|arg| arg == flag), "{engine}: {flag}");
            }
            if engine == "llama" {
                assert!(arguments.windows(2).any(|args| args == ["--parallel", "2"]));
            } else {
                assert!(!arguments.iter().any(|arg| arg == "--parallel"));
                assert!(arguments.windows(2).any(|args| {
                    args == ["--startup-timeout-secs", &STARTUP_TIMEOUT_SECS.to_string()]
                }));
            }
        }
    }

    #[test]
    fn comparison_matches_llama_slot_context_to_common_trace_budget() {
        let models: Vec<super::super::Model> = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/agentx/models.json"
        ))
        .unwrap();
        let input = Prepared {
            root: "/repo".into(),
            output: "/output".into(),
            weights: "/cache/model.gguf".into(),
            dataset: "/traces".into(),
            model: models[0].clone(),
            profile: Profile::Comparison,
        };
        for concurrency in [1, 2, 4] {
            let llama = server(&input, "llama", 8080, concurrency).unwrap();
            let arguments: Vec<_> = llama
                .arguments
                .iter()
                .map(|value| match value {
                    process::Value::Public(value) => value.to_string_lossy().into_owned(),
                    process::Value::Secret(_) => panic!("unexpected secret in server arguments"),
                })
                .collect();
            let expected_context = (trace_context::LIMIT * u64::from(concurrency)).to_string();
            assert!(
                arguments
                    .windows(2)
                    .any(|args| args == ["--ctx-size", &expected_context])
            );
            assert!(
                arguments
                    .windows(2)
                    .any(|args| args == ["--parallel", &concurrency.to_string()])
            );
            let skippy = server(&input, "skippy", 8080, concurrency).unwrap();
            assert!(!skippy.arguments.iter().any(|value| matches!(value,
                process::Value::Public(argument) if argument == "--ctx-size")));
        }
    }

    #[test]
    fn llama_cache_budget_control_is_explicit_and_rejects_invalid_values() {
        assert_eq!(parse_llama_cache_ram_mib("65536").unwrap(), 65536);
        for value in ["", "0", "-1", "8GiB", "4294967296"] {
            assert!(parse_llama_cache_ram_mib(value).is_err(), "{value}");
        }
        assert_eq!(
            llama_comparison_arguments(2, Some(65536)).unwrap(),
            ["--ctx-size", "262144", "--cache-ram", "65536"]
        );
        assert_eq!(
            llama_comparison_arguments(2, None).unwrap(),
            ["--ctx-size", "262144"]
        );
    }
}
