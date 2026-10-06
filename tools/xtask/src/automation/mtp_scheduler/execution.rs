use super::Input;
use crate::{
    command::DynResult,
    process::{
        ObservedLine, ProbeDecision, ProcessSpec, Value,
        retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
    },
};
use std::{path::Path, time::Duration};
fn admit(input: &mut Input) -> DynResult<()> {
    super::profile(input)?;
    for path in [
        &mut input.old_bin,
        &mut input.new_bin,
        &mut input.client_bin,
    ] {
        *path = super::regular(path, 256 * 1024 * 1024)?;
    }
    for path in [&mut input.package, &mut input.native_build] {
        *path = path.canonicalize()?;
        if !path.is_dir() {
            return Err("package/native build must be existing directories".into());
        }
    }
    if !input.output_dir.is_absolute()
        || input.output_dir.exists()
        || std::fs::symlink_metadata(&input.output_dir).is_ok()
    {
        return Err("requires fresh absolute output directory".into());
    }
    let parent = input
        .output_dir
        .parent()
        .ok_or("output parent")?
        .canonicalize()?;
    input.output_dir = parent.join(input.output_dir.file_name().ok_or("output name")?);
    std::fs::create_dir(&input.output_dir)?;
    std::fs::create_dir(input.output_dir.join("private-home"))?;
    Ok(())
}
fn hashes(input: &Input) -> DynResult<serde_json::Value> {
    let mut hashes = serde_json::Map::new();
    for (key, path) in [
        ("old", &input.old_bin),
        ("new", &input.new_bin),
        ("client", &input.client_bin),
    ] {
        super::regular(path, 256 * 1024 * 1024)?;
        hashes.insert(
            key.into(),
            serde_json::json!({"path":path,"sha256":crate::product::digest::file_sha256(path).map_err(|failure| failure.error)?}),
        );
    }
    Ok(hashes.into())
}
pub(super) fn execute(mut input: Input) -> DynResult<()> {
    admit(&mut input)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let deadline = std::time::Instant::now() + arm_budget(&input)? * 2;
    let before = hashes(&input)?;
    crate::command::write_json_file(&input.output_dir.join("input.json"), &input)?;
    let mut comparison = serde_json::json!({"status":"running","model_id":input.model_id,"package":input.package,"layer_range":[input.layer_start,input.layer_end],"activation_width":input.activation_width,"supplied_binary_observations":before,"custody":"supplied binaries and local package/native-build; no independent build or package attestation","old":null,"new":null,"parity":null});
    let mut publication =
        super::final_publication::ReceiptFile::new(&input.output_dir.join("comparison.json"))?;
    publication.write(&comparison)?;
    let result = (|| -> DynResult<()> {
        for (label, binary) in [("old", &input.old_bin), ("new", &input.new_bin)] {
            if cancellation.is_cancelled() {
                return Err("MTP comparison cancelled".into());
            }
            let arm = arm(&input, label, binary, &cancellation)?;
            comparison[label] = arm;
            publication.write(&comparison)?;
        }
        if hashes(&input)? != before {
            return Err("supplied binaries changed during comparison".into());
        }
        comparison["parity"] = super::metrics::parity(&comparison["old"], &comparison["new"])?;
        Ok(())
    })();
    publication.finish(&mut comparison, result, interrupt, deadline, &mut |_| {
        println!("{}", input.output_dir.join("comparison.json").display());
        Ok(())
    })
}

fn launch(
    input: &Input,
    id: MemberId,
    executable: std::path::PathBuf,
    args: Vec<std::ffi::OsString>,
    file: &Path,
    budget: Duration,
) -> Launch {
    Launch {
        member: id,
        spec: ProcessSpec {
            executable,
            arguments: args.into_iter().map(Value::Public).collect(),
            cwd: input.output_dir.clone(),
            environment: super::environment(input),
        },
        files: crate::process::OutputFiles {
            stdout: Some(file.into()),
            stderr: Some(file.with_extension("stderr.log")),
        },
        readiness_deadline: budget,
    }
}
fn arm(
    input: &Input,
    label: &str,
    binary: &Path,
    cancellation: &crate::process::Cancellation,
) -> DynResult<serde_json::Value> {
    let reservation = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let address = reservation.local_addr()?.to_string();
    let config = input.output_dir.join(format!("{label}-stage.json"));
    crate::command::write_json_file(&config, &super::config(input, &address))?;
    let budget = arm_budget(input)?;
    let server = launch(
        input,
        MemberId::Seed,
        binary.into(),
        vec![
            "serve-binary".into(),
            "--config".into(),
            config.clone().into_os_string(),
            "--bind-addr".into(),
            address.clone().into(),
            "--max-inflight".into(),
            input
                .concurrency
                .iter()
                .max()
                .ok_or("empty sweep")?
                .to_string()
                .into(),
            "--telemetry-level".into(),
            "debug".into(),
        ],
        &input.output_dir.join(format!("{label}-server.log")),
        budget,
    );
    let worker = launch(
        input,
        MemberId::WorkerOne,
        std::env::current_exe()?,
        vec![
            "automation".into(),
            "mtp-scheduler-worker".into(),
            input.output_dir.join("input.json").into_os_string(),
            label.into(),
            address.into(),
        ],
        &input.output_dir.join(format!("{label}-worker.log")),
        budget,
    );
    let mut owner = Owner {
        server: Some(server),
        worker: Some(worker),
        policy: ExpectedExit::new(&[0], budget)?,
        stopping: false,
    };
    drop(reservation);
    let report = crate::process::retained::run(&mut owner, &super::limits(budget), cancellation);
    let evidence = match &report {
        Ok(report) => {
            serde_json::json!({"outcome":format!("{:?}",report.outcome),"members":report.members.iter().map(|member|serde_json::json!({"member":format!("{:?}",member.member),"disposition":member.disposition.label(),"exit_code":member.process.status.as_ref().and_then(std::process::ExitStatus::code),"unix_signal":signal(member.process.status.as_ref()),"cleanup_complete":member.process.cleanup.complete,"cleanup_forced":member.process.cleanup.forced,"graceful_signal_failed":member.process.cleanup.graceful_signal_failed,"stdout_complete":member.process.stdout.line_capture_complete,"stderr_complete":member.process.stderr.line_capture_complete})).collect::<Vec<_>>(),"shutdown_scope":"owned unforced intentional stop; signal/code observed, no graceful shutdown attestation"})
        }
        Err(error) => serde_json::json!({"failure":error.to_string()}),
    };
    crate::command::write_json_file(
        &input.output_dir.join(format!("{label}-lifecycle.json")),
        &evidence,
    )?;
    let report = report?;
    let worker_zero = report.members.iter().any(|member| {
        member.member == MemberId::WorkerOne
            && matches!(
                member.disposition,
                crate::process::retained::Disposition::ExpectedExit
            )
            && member
                .process
                .status
                .as_ref()
                .is_some_and(std::process::ExitStatus::success)
    });
    if !report.recovery_success()
        || !worker_zero
        || report.members.iter().any(|member| {
            member.process.cleanup.forced
                || !member.process.stdout.line_capture_complete
                || !member.process.stderr.line_capture_complete
        })
    {
        return Err(format!("MTP arm incomplete or unclean: {:?}", report.outcome).into());
    }
    let rows: serde_json::Value = serde_json::from_slice(&super::read(
        &input.output_dir.join(format!("{label}-result.json")),
    )?)?;
    if rows["status"] != "completed" {
        return Err("MTP worker result incomplete".into());
    }
    Ok(
        serde_json::json!({"label":label,"binary":binary,"config":config,"log":input.output_dir.join(format!("{label}-server.log")),"concurrency_sweep":rows["concurrency_sweep"],"stderr_log":input.output_dir.join(format!("{label}-server.stderr.log")),"diagnostic_capture":"bounded sanitized per-stream; stdout log and separate stderr log, no merged temporal ordering proof"}),
    )
}
struct Owner {
    server: Option<Launch>,
    worker: Option<Launch>,
    policy: ExpectedExit,
    stopping: bool,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if self.stopping {
            return Action::Complete;
        }
        if let Some(launch) = self.server.take() {
            return Action::Start(launch);
        }
        if context
            .members
            .iter()
            .any(|m| m.member == MemberId::Seed && matches!(m.state, MemberState::Starting))
        {
            return Action::Admit(MemberId::Seed);
        }
        if let Some(launch) = self.worker.take() {
            return Action::StartExpected {
                launch,
                policy: self.policy.clone(),
            };
        }
        if context.members.iter().any(|m| {
            m.member == MemberId::WorkerOne && matches!(m.state, MemberState::ExpectedExit { .. })
        }) {
            self.stopping = true;
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}

fn signal(status: Option<&std::process::ExitStatus>) -> Option<i32> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.and_then(std::process::ExitStatus::signal)
    }
    #[cfg(not(unix))]
    {
        let _ = status;
        None
    }
}

pub(super) fn arm_budget(input: &Input) -> DynResult<Duration> {
    let seconds = input.startup_seconds
        + input.client_seconds * input.concurrency.len() as u64
        + 3 * input.concurrency.len() as u64
        + 6;
    if seconds > 86400 {
        return Err("whole arm exceeds one-day supervisor limit".into());
    }
    Ok(Duration::from_secs(seconds))
}
