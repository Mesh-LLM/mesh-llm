use super::{
    l3_contract::{Config, Operation, Phase, Run, Status},
    l3_execution::{Disk, DiskRoots, Driver, Mode},
    l3_management::Client,
    l3_server::Session,
    manifest_preflight::Manifest,
    recorded_requests::Trajectory,
    run_workload::MeshBuild,
};
use crate::{
    automation::private_state::PrivateState,
    command::DynResult,
    process::{
        self, Value,
        retained::{Launch, MemberId},
    },
};
use std::{path::PathBuf, thread::Scope, time::Duration};
pub(super) struct Runtime<'scope, 'env> {
    pub scope: &'scope Scope<'scope, 'env>,
    pub config: Config,
    pub build: Box<MeshBuild>,
    pub manifest: Manifest,
    pub selected: Vec<Trajectory>,
    pub output: PathBuf,
    pub hf_home: Option<PathBuf>,
    pub startup: Duration,
    pub request_timeout: Duration,
    pub execution: Duration,
    pub cancellation: process::Cancellation,
    session: Option<Session<'scope>>,
    state: Option<PrivateState>,
    client: Option<Client>,
    base: String,
    model: String,
    index: usize,
    servers: Vec<serde_json::Value>,
}
impl<'scope, 'env> Runtime<'scope, 'env> {
    pub fn new(scope: &'scope Scope<'scope, 'env>, input: Input) -> Self {
        Self {
            scope,
            config: input.config,
            build: input.build,
            manifest: input.manifest,
            selected: input.selected,
            output: input.output,
            hf_home: input.hf_home,
            startup: input.startup,
            request_timeout: input.request_timeout,
            execution: input.execution,
            cancellation: input.cancellation,
            session: None,
            state: None,
            client: None,
            base: String::new(),
            model: String::new(),
            index: 0,
            servers: Vec::new(),
        }
    }
    fn client(&self) -> DynResult<&Client> {
        if self.cancellation.is_cancelled() || self.session.as_ref().is_none_or(Session::finished) {
            return Err("disk-L3 server exited or certification interrupted".into());
        }
        self.client
            .as_ref()
            .ok_or_else(|| "disk-L3 server not started".into())
    }
    async fn ready(&mut self) -> DynResult<()> {
        let deadline = tokio::time::Instant::now() + self.startup;
        loop {
            self.client()?;
            if tokio::time::Instant::now() >= deadline {
                return Err("disk-L3 model startup deadline exceeded".into());
            }
            if let Ok(Ok(bytes)) = tokio::time::timeout(
                Duration::from_secs(2),
                super::server_cell_worker::get(&format!("{}/models", self.base)),
            )
            .await
                && let Ok(document) = serde_json::from_slice::<serde_json::Value>(&bytes)
                && let Some(models) = document["data"].as_array()
                && models.len() == 1
                && let Some(model) = models[0]["id"].as_str()
                && !model.is_empty()
                && matches!(
                    tokio::time::timeout(Duration::from_secs(2), self.client()?.status()).await,
                    Ok(Ok(_))
                )
            {
                self.model = model.into();
                self.session.as_ref().ok_or("missing server owner")?.admit();
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
pub(super) struct Input {
    pub config: Config,
    pub build: Box<MeshBuild>,
    pub manifest: Manifest,
    pub selected: Vec<Trajectory>,
    pub output: PathBuf,
    pub hf_home: Option<PathBuf>,
    pub startup: Duration,
    pub request_timeout: Duration,
    pub execution: Duration,
    pub cancellation: process::Cancellation,
}
impl Driver for Runtime<'_, '_> {
    async fn start(&mut self, disk: Disk, roots: &mut DiskRoots) -> DynResult<()> {
        if self.session.is_some() {
            return Err("disk-L3 owned server already running".into());
        }
        super::run_transport::verify_mesh(&self.build)?;
        self.index += 1;
        let api = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let port = api.local_addr()?.port();
        let console = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let console_port = console.local_addr()?.port();
        let state = PrivateState::create(
            roots
                .cache()
                .parent()
                .ok_or("missing persistent root parent")?,
            "server",
        )?;
        state.prepare()?;
        let mut environment = state.environment(&self.build.runtime_root);
        environment.insert("SKIPPY_TELEMETRY_STDERR".into(), Value::Public("1".into()));
        if let Some(home) = &self.hf_home {
            environment.insert("HF_HOME".into(), Value::Public(home.as_os_str().into()));
        }
        let mut arguments = vec![
            "serve".to_owned(),
            "--model".into(),
            self.config.model.clone(),
            "--log-format".into(),
            "json".into(),
            "--port".into(),
            port.to_string(),
            "--console".into(),
            console_port.to_string(),
        ];
        if !matches!(disk, Disk::Off) {
            arguments.extend([
                "--kv-cache-disk".into(),
                if matches!(disk, Disk::LowSpace) {
                    self.config.low_space_disk_budget.clone()
                } else {
                    self.config.disk_budget.clone()
                },
                "--kv-cache-disk-dir".into(),
                roots
                    .cache()
                    .to_str()
                    .ok_or("non-Unicode disk root")?
                    .into(),
                "--kv-cache-min-free".into(),
                if matches!(disk, Disk::LowSpace) {
                    self.config.low_space_minimum_free.clone()
                } else {
                    self.config.minimum_free.clone()
                },
            ]);
        }
        let log = self.output.join(format!("logs/server-{}.log", self.index));
        std::fs::create_dir_all(log.parent().ok_or("missing log parent")?)?;
        let command = std::iter::once(self.build.binary.to_string_lossy().into_owned())
            .chain(arguments.iter().cloned())
            .collect::<Vec<_>>();
        let launch = Launch {
            member: MemberId::Seed,
            spec: process::ProcessSpec {
                executable: self.build.binary.clone(),
                arguments: arguments
                    .into_iter()
                    .map(|value| Value::Public(value.into()))
                    .collect(),
                cwd: self
                    .build
                    .worktree
                    .clone()
                    .ok_or("disk-L3 build missing worktree")?,
                environment,
            },
            files: process::OutputFiles {
                stdout: Some(log.clone()),
                stderr: Some(log.with_extension("stderr.log")),
            },
            readiness_deadline: self.startup,
        };
        self.base = format!("http://127.0.0.1:{port}/v1");
        self.client = Some(Client {
            base: format!("http://127.0.0.1:{console_port}"),
            timeout: self.request_timeout,
            cancellation: Some(self.cancellation.clone()),
            interval: Duration::from_millis(500),
        });
        self.state = Some(state);
        drop((api, console));
        self.session = Some(Session::start(
            self.scope,
            launch,
            self.execution,
            self.cancellation.clone(),
        ));
        self.ready().await?;
        self.servers.push(serde_json::json!({"index":self.index,"pid":self.session.as_ref().ok_or("missing server owner")?.pid(),"command":command,"log":log}));
        Ok(())
    }
    async fn stop(&mut self) -> DynResult<()> {
        let report = self.session.take().map(Session::finish).transpose();
        let retain = if let Some(state) = self.state.take() {
            let logs = state.retain_runtime_logs(
                &self
                    .output
                    .join(format!("native-runtime/server-{}", self.index)),
            );
            state
                .finish(logs)
                .map_err(|error| format!("disk-L3 state/log finalization: {error:?}"))
        } else {
            Ok(())
        };
        self.client = None;
        let report = report?;
        retain?;
        if let Some(report) = report {
            let clean = report.recovery_success()
                && report.members.len() == 1
                && report.members.iter().all(|member| {
                    !member.process.cleanup.forced && !member.process.cleanup.graceful_signal_failed
                });
            crate::command::write_json_file(
                &self
                    .output
                    .join(format!("sessions/server-{}/lifecycle.json", self.index)),
                &serde_json::json!({"infrastructure_clean":clean,"outcome":format!("{:?}",report.outcome),"status":report.members.first().and_then(|member|member.process.status.as_ref()).and_then(std::process::ExitStatus::code)}),
            )?;
            if !clean {
                return Err(format!("disk-L3 owned server failure: {:?}", report.outcome).into());
            }
        }
        Ok(())
    }
    async fn status(&mut self) -> DynResult<Status> {
        self.client()?
            .status()
            .await
            .map_err(|e| e.to_string().into())
    }
    async fn committed(&mut self, writes: u64) -> DynResult<Status> {
        self.client()?
            .committed(writes)
            .await
            .map_err(|e| e.to_string().into())
    }
    async fn empty(&mut self) -> DynResult<Operation> {
        self.client()?
            .empty()
            .await
            .map_err(|e| e.to_string().into())
    }
    async fn measure(&mut self, name: &str, mode: Mode) -> DynResult<Phase> {
        self.client()?;
        let trajectories = if let Mode::HighLoad(c) = mode {
            self.manifest
                .cohorts
                .get(&c.to_string())
                .ok_or("missing high-load cohort")?
        } else {
            &self.selected
        };
        let endpoint = super::l3_requests::Endpoint {
            base: &self.base,
            model: &self.model,
            timeout: self.request_timeout,
            cancellation: Some(self.cancellation.clone()),
        };
        let raw = self.output.join(format!("data/{name}.jsonl"));
        std::fs::create_dir_all(raw.parent().ok_or("missing data parent")?)?;
        super::l3_requests::measure(&endpoint, trajectories, &self.config, mode, name, &raw).await
    }
    async fn traffic(&mut self) -> DynResult<Phase> {
        let client = self.client()?;
        let endpoint = super::l3_requests::Endpoint {
            base: &self.base,
            model: &self.model,
            timeout: self.request_timeout,
            cancellation: Some(self.cancellation.clone()),
        };
        let raw = self.output.join("data/lifecycle_under_traffic.jsonl");
        let request = super::l3_requests::measure(
            &endpoint,
            &self.selected,
            &self.config,
            Mode::Identical(1),
            "lifecycle_under_traffic",
            &raw,
        );
        let operations = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let prune = client.prune().await?;
            let clear = client.clear().await?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((prune, clear))
        };
        let (phase, operations) = tokio::join!(request, operations);
        let mut phase = phase?;
        let (prune, clear) = operations.map_err(|e| e.to_string())?;
        phase.prune = Some(prune);
        phase.clear = Some(clear);
        phase.final_clear = Some(client.empty().await.map_err(|e| e.to_string())?);
        Ok(phase)
    }
    fn checkpoint(&mut self, run: &Run) -> DynResult<()> {
        let mut document = serde_json::to_value(run)?;
        document["servers"] = self.servers.clone().into();
        super::run_snapshot::write(&self.output.join("run.json"), &document)?;
        if let Some(phase) = run.phases.get("disk_off_cold") {
            use std::io::Write;
            let mut raw = std::fs::File::create(self.output.join("data/disk-off-cold.jsonl"))?;
            for request in &phase.requests {
                serde_json::to_writer(&mut raw, request)?;
                raw.write_all(b"\n")?;
            }
        }
        Ok(())
    }
}
impl Drop for Runtime<'_, '_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let _ = session.finish();
        }
    }
}
