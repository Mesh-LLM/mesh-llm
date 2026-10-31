//! One fresh product host with one deterministic cache client; both trees owned.
use super::super::{
    contract::{Backend, Cell, Model, Prepared},
    evidence::CacheEvidence,
    publish,
};
use super::{environment, limits};
use crate::{
    command::DynResult,
    process::{
        self, ObservedLine, OutputFiles, ProbeDecision, ProcessSpec, Stream, Value as Arg,
        retained::{
            self, Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
        },
    },
};
use serde_json::json;
use std::{ffi::OsString, path::Path, time::Duration};

struct Owner {
    server: Option<Launch>,
    client: Option<Launch>,
    ready: bool,
    stopping: bool,
    port: u16,
    cache_evidence: Option<CacheEvidence>,
    duplicate_evidence: bool,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn captured_line(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if line.bytes.len() > 16384 {
            return;
        }
        if member == MemberId::WorkerOne && line.stream == Stream::Stdout {
            // Only this declared cache receipt is retained before log sanitization.
            // Arbitrary stdout, environment and protocol headers are never copied.
            if let Ok(evidence) = serde_json::from_slice::<CacheEvidence>(line.bytes) {
                self.duplicate_evidence |= self.cache_evidence.is_some();
                self.cache_evidence = Some(evidence);
            }
            return;
        }
        if member != MemberId::Seed {
            return;
        }
        if let Ok(event) = serde_json::from_slice::<serde_json::Value>(line.bytes) {
            self.ready |= event["schema_version"] == 1
                && event["type"] == "status"
                && event["data"]["message"]
                    == format!("skippy-serving listening: openai=127.0.0.1:{}", self.port);
        }
    }
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if self.stopping {
            return Action::Complete;
        }
        if let Some(server) = self.server.take() {
            return Action::Start(server);
        }
        if self.ready
            && context
                .members
                .iter()
                .any(|m| m.member == MemberId::Seed && m.state == MemberState::Starting)
        {
            return Action::Admit(MemberId::Seed);
        }
        if context
            .members
            .iter()
            .any(|m| m.member == MemberId::Seed && matches!(m.state, MemberState::Ready { .. }))
            && let Some(client) = self.client.take()
        {
            return match ExpectedExit::new(&[0], context.remaining) {
                Ok(policy) => Action::StartExpected {
                    launch: client,
                    policy,
                },
                Err(e) => Action::Reject(e.to_string()),
            };
        }
        if context.members.iter().any(|m| {
            m.member == MemberId::WorkerOne
                && matches!(m.state, MemberState::ExpectedExit { status: 0, .. })
        }) {
            self.stopping = true;
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}
fn launch(
    binary: &Path,
    args: Vec<OsString>,
    output: &Path,
    label: &str,
    member: MemberId,
    duration: Duration,
) -> Launch {
    Launch {
        member,
        spec: ProcessSpec {
            executable: binary.into(),
            arguments: args.into_iter().map(Arg::Public).collect(),
            cwd: output.into(),
            environment: environment(&[]),
        },
        files: OutputFiles {
            stdout: Some(output.join(format!("{label}.stdout.log"))),
            stderr: Some(output.join(format!("{label}.stderr.log"))),
        },
        readiness_deadline: duration,
    }
}
pub(super) fn run(
    input: &Cell,
    prepared: &Prepared,
    model: &Model,
    output: &Path,
    duration: Duration,
    cancel: &process::Cancellation,
) -> DynResult<()> {
    let producer = &prepared.producer;
    let prompt = output.join("cache-prompt.txt");
    std::fs::write(
        &prompt,
        "Describe the red ball and its shape in a short answer.\n".repeat(16),
    )?;
    let args = vec![
        "--output".into(),
        "jsonl".into(),
        "--runtime-bundle".into(),
        producer.runtime_bundle.clone().into_os_string(),
        "--runtime-selection".into(),
        producer.backend.kind().into(),
        "serve".into(),
        "--model-path".into(),
        input.model.clone().into_os_string(),
        "--model-id".into(),
        model.key.clone().into(),
        "--bind-addr".into(),
        format!("127.0.0.1:{}", input.port).into(),
        "--ctx-size".into(),
        "2048".into(),
        "--generation-concurrency".into(),
        "1".into(),
        "--n-gpu-layers".into(),
        if producer.backend == Backend::Cpu {
            "0"
        } else {
            "99"
        }
        .into(),
        "--batch-size".into(),
        "128".into(),
        "--microbatch-size".into(),
        "64".into(),
        "--threads".into(),
        "4".into(),
        "--threads-batch".into(),
        "4".into(),
        "--flash-attention".into(),
        "enabled".into(),
        "--kv-cache-type-k".into(),
        "f16".into(),
        "--kv-cache-type-v".into(),
        "f16".into(),
        "--prefix-cache".into(),
        "on".into(),
        "--prefix-cache-ram".into(),
        "8GiB".into(),
        "--prefix-cache-min-tokens".into(),
        "1".into(),
        "--kv-cache-disk".into(),
        "off".into(),
        "--temperature".into(),
        "0".into(),
    ];
    publish(
        &output.join("cache-server-settings.json"),
        &json!({"arguments":args,"runtime_bundle":producer.runtime_bundle,"c1_only":true,"long_pressure_L3":false}),
    )?;
    let client_args = vec![
        "open-ai-cache-reuse".into(),
        "--base-url".into(),
        format!("http://127.0.0.1:{}/v1", input.port).into(),
        "--model".into(),
        model.key.clone().into(),
        "--prompt-file".into(),
        prompt.into_os_string(),
        "--max-tokens".into(),
        "8".into(),
        "--request-timeout-secs".into(),
        "120".into(),
    ];
    let mut owner = Owner {
        server: Some(launch(
            &producer.cli.path,
            args,
            output,
            "cache-server",
            MemberId::Seed,
            duration,
        )),
        client: Some(launch(
            &producer.correctness.path,
            client_args,
            output,
            "cache-client",
            MemberId::WorkerOne,
            duration,
        )),
        ready: false,
        stopping: false,
        port: input.port,
        cache_evidence: None,
        duplicate_evidence: false,
    };
    let report = retained::run(&mut owner, &limits(duration), cancel)?;
    publish(
        &output.join("cache-process.json"),
        &json!({"success":report.success(),"outcome":format!("{:?}",report.outcome),"rejection":report.rejection,"failure":report.failure.as_ref().map(ToString::to_string)}),
    )?;
    if !report.success() {
        return Err("fresh product cache correctness did not finish cleanly".into());
    }
    if owner.duplicate_evidence {
        return Err("generic cache client emitted duplicate evidence".into());
    }
    let evidence = owner
        .cache_evidence
        .ok_or("generic cache client evidence missing")?;
    evidence.verify()?;
    publish(&output.join("cache-evidence.json"), &evidence)
}
