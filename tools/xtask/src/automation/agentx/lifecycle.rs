use super::{Prepared, Profile, command, inputs, write};
use crate::{
    command::DynResult,
    process::{
        self,
        retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
    },
};
use serde_json::json;
use std::time::Duration;

pub(super) fn hardware(input: &Prepared) -> DynResult<()> {
    if std::env::consts::OS != "linux" || std::env::consts::ARCH != "aarch64" {
        return Err("AgentX job requires a Linux aarch64 DGX Spark".into());
    }
    let spec = command::spec(
        &input.root,
        "nvidia-smi",
        &command::strings(&[
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader",
        ]),
    )?;
    let log = input.output.join("hardware");
    command::run(&spec, 30, &log)?;
    let name = std::fs::read_to_string(log.with_extension("stdout.log"))?;
    if !name.contains("GB10") {
        return Err("AgentX job requires an NVIDIA GB10".into());
    }
    Ok(())
}
fn server(
    input: &Prepared,
    engine: &str,
    port: u16,
    concurrency: u32,
) -> DynResult<process::ProcessSpec> {
    // llama divides its context pool among slots; Skippy takes a per-lane size.
    // Keep the complete 256k trace budget for every request on both engines.
    let total_context = 262144_u32
        .checked_mul(concurrency)
        .ok_or("context capacity overflow")?;
    let weights = input.weights.to_str().ok_or("weight path")?;
    let (binary, args) = if engine == "llama" {
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
                "--ctx-size",
                &total_context.to_string(),
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
                "--ctx-size",
                "262144",
                "--generation-concurrency",
                &concurrency.to_string(),
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
    let mut spec = command::spec(&input.root, binary.to_str().ok_or("server path")?, &args)?;
    if engine == "skippy" {
        spec.arguments.extend([
            process::Value::Public("--runtime-bundle".into()),
            process::Value::Public(input.root.join("dist/native-runtimes").into_os_string()),
        ]);
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
    let report = super::super::retained_session::run(&mut owner, &command::limits(seconds))?;
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
        "container_image":std::env::var("AGENTX_CONTAINER_IMAGE")?,
        "binary_sha256":crate::product::digest::file_sha256(&server.executable).map_err(|error|error.error)?,
        "runtime_tree_sha256":crate::product::digest::tree_sha256(&runtime).map_err(|error|error.error)?,
        "harness_lock_sha256":crate::product::digest::file_sha256(&input.root.join("skippy/evals/agentx/uv.lock")).map_err(|error|error.error)?,
        "context_tokens":262144,"total_context_tokens":262144_u64 * u64::from(concurrency),
        "generation_slots":concurrency,"offered_sessions":concurrency,
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
    server: Option<Launch>,
    worker: Option<Launch>,
    stop: bool,
    policy: ExpectedExit,
}

impl Coordinator for Owner {
    type Rejection = String;
    fn line(
        &mut self,
        _member: MemberId,
        _line: process::ObservedLine<'_>,
    ) -> process::ProbeDecision<String> {
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
            let (slots, context) = if engine == "llama" {
                ("--parallel", "524288")
            } else {
                ("--generation-concurrency", "262144")
            };
            assert!(arguments.windows(2).any(|args| args == [slots, "2"]));
            assert!(
                arguments
                    .windows(2)
                    .any(|args| args == ["--ctx-size", context])
            );
        }
    }
}
