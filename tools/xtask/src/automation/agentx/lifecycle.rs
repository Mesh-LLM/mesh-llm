use super::{Prepared, Profile, command, inputs, results, write};
use crate::{
    command::DynResult,
    process::{
        self,
        retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
    },
};
use serde_json::json;
use std::time::Duration;

pub(super) fn campaign(input: &Prepared) -> DynResult<()> {
    hardware(input)?;
    let mut cells = Vec::new();
    for &concurrency in input.profile.concurrencies() {
        for engine in ["llama", "skippy"] {
            inputs::recheck(input)?;
            let directory = input.output.join(format!("{engine}-c{concurrency}"));
            std::fs::create_dir(&directory)?;
            let mut cell: Prepared = serde_json::from_value(serde_json::to_value(input)?)?;
            cell.output = directory;
            execute(&cell, engine, concurrency)?;
            cells.push(results::read(&cell, engine, concurrency)?);
            write(
                &input.output.join("results.json"),
                &json!({"profile":input.profile,"model":input.model,"cells":cells,"promote_to_performance_history":false}),
            )?;
        }
    }
    if let Some(path) = std::env::var_os("JIANYANG_RESULT_PATH") {
        // No chart points: upstream local-corpus runs are explicitly not leaderboard submissions.
        use sha2::Digest;
        let model_digest = if input.model.files.len() == 1 {
            input.model.files[0].sha256.clone()
        } else {
            hex::encode(sha2::Sha256::digest(serde_json::to_vec(
                &input.model.files,
            )?))
        };
        write(
            std::path::Path::new(&path),
            &json!({"points":[],
                "custom":{"schema":"agentx","version":"1","data":{
                    "profile":input.profile,"model":input.model,"cells":cells,"promote_to_performance_history":false
                }},
                "provenance":{"stack_commit":std::env::var("JIANYANG_COMMIT")?,"model_package_sha256":model_digest}
            }),
        )?;
    }
    Ok(())
}
fn hardware(input: &Prepared) -> DynResult<()> {
    if std::env::consts::OS != "linux" || std::env::consts::ARCH != "aarch64" {
        return Err("AgentX job requires a Linux aarch64 DGX Spark".into());
    }
    let spec = command::spec(
        &input.root,
        "nvidia-smi",
        &command::strings(&["--query-gpu=name", "--format=csv,noheader"]),
    )?;
    let log = input.output.join("hardware");
    command::run(&spec, 30, &log)?;
    let name = std::fs::read_to_string(log.with_extension("stdout.log"))?;
    if !name.contains("GB10") {
        return Err("AgentX job requires an NVIDIA GB10".into());
    }
    Ok(())
}
fn server(input: &Prepared, engine: &str, port: u16) -> DynResult<process::ProcessSpec> {
    // One resident generation slot and one 256k pool on both engines. Offered
    // agent sessions (including subagents) are queued independently of this capacity.
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
                "262144",
                "--parallel",
                "1",
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
                "1",
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
fn execute(input: &Prepared, engine: &str, concurrency: u32) -> DynResult<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    write(&input.output.join("worker.json"), input)?;
    let server = server(input, engine, port)?;
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
    let mut owner = Owner {
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
        &json!({"clean":clean,"outcome":format!("{:?}",report.outcome)}),
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
        "binary_sha256":crate::product::digest::file_sha256(&server.executable).map_err(|error|error.error)?,
        "runtime_tree_sha256":crate::product::digest::tree_sha256(&runtime).map_err(|error|error.error)?,
        "harness_lock_sha256":crate::product::digest::file_sha256(&input.root.join("skippy/evals/agentx/uv.lock")).map_err(|error|error.error)?,
        "context_tokens":262144,"generation_slots":1,"offered_sessions":concurrency,
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
