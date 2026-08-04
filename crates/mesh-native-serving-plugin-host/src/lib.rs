//! Loads one native serving plugin and adapts its stable ABI to Skippy hooks.

use std::{
    ffi::{c_char, c_void},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use mesh_native_serving_plugin_api as abi;
use skippy_server::frontend::{
    GenerationAbort, GenerationCommit, GenerationLifecycleIngress, GenerationLifecycleObservation,
    GenerationReceipt, GenerationReceiptConfig, GenerationStart, LinearProposal,
    LinearProposalDiscardReason, LinearProposalDisposition, LinearProposalIngress,
    LinearProposalIngressConfig, LinearProposalQuery, LinearProposalReceipt,
    OpaqueProposalDecisionId,
};
use skippy_server::serving_hooks::{ModelServingHooks, ModelServingHooksFactory};
use skippy_server::tokenizer::TokenizerCapability;

const ERROR_BUFFER_BYTES: usize = 2_048;
const PLUGIN_COMMAND_CAPACITY: usize = 1_024;
const MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS: usize = 4_096;
const PROPOSAL_POLL_INTERVAL: Duration = Duration::from_micros(50);
const CLEAN_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Mesh-owned factory for one independently built native serving plugin.
#[derive(Clone)]
pub struct NativeServingPluginFactory {
    definition: Arc<LoadedDefinition>,
    config_path: PathBuf,
    state_directory: PathBuf,
    proposal_deadline: Duration,
}

impl NativeServingPluginFactory {
    /// Loads and validates a native plugin without activating model-specific state.
    pub fn load(
        library_path: &Path,
        config_path: PathBuf,
        state_directory: PathBuf,
        proposal_deadline: Duration,
    ) -> Result<Self> {
        validate_absolute_path("native serving plugin", library_path)?;
        validate_absolute_path("native serving plugin config", &config_path)?;
        validate_absolute_path("native serving plugin state", &state_directory)?;
        if !library_path.is_file() {
            bail!(
                "native serving plugin must be an existing file: {}",
                library_path.display()
            );
        }
        if !config_path.is_file() {
            bail!(
                "native serving plugin config must be an existing file: {}",
                config_path.display()
            );
        }
        if !state_directory.is_dir() {
            bail!(
                "native serving plugin state must be an existing directory: {}",
                state_directory.display()
            );
        }
        if proposal_deadline.is_zero() {
            bail!("native serving plugin proposal deadline must be greater than zero");
        }
        let definition = LoadedDefinition::load(library_path)?;
        Ok(Self {
            definition: Arc::new(definition),
            config_path,
            state_directory,
            proposal_deadline,
        })
    }
}

impl ModelServingHooksFactory for NativeServingPluginFactory {
    fn create(&self, tokenizer: TokenizerCapability) -> Result<ModelServingHooks> {
        let identity = tokenizer.identity();
        let inventory = ActivationInventory::from_inventory(
            tokenizer
                .inventory()
                .context("bound model does not expose a tokenizer inventory")?,
        )?;
        let inventory_view = inventory.view();
        let context = abi::ActivationContext {
            struct_size: size_of::<abi::ActivationContext>(),
            model_id: abi::ByteSlice::from_bytes(identity.model_id.as_bytes()),
            source_model_sha256: abi::ByteSlice::from_bytes(
                identity.source_model_sha256.as_bytes(),
            ),
            tokenizer_id: abi::ByteSlice::from_bytes(identity.tokenizer_id.as_bytes()),
            tokenizer_inventory: &raw const inventory_view,
            config_path: path_slice(&self.config_path),
            state_directory: path_slice(&self.state_directory),
            proposal_deadline_ns: u64::try_from(self.proposal_deadline.as_nanos())
                .unwrap_or(u64::MAX),
            host_clock_context: std::ptr::null_mut(),
            monotonic_now_ns,
        };
        let mut activation = abi::PluginActivation {
            instance: std::ptr::null_mut(),
        };
        let status = unsafe { (self.definition.api().activate)(&context, &raw mut activation) };
        if status != abi::PluginStatus::OK {
            return Err(self.definition.status_error(
                activation.instance,
                "activate native serving plugin",
                status,
            ));
        }
        let instance = NonNull::new(activation.instance)
            .context("native serving plugin returned a null active instance")?;
        let deadline = self.proposal_deadline;
        let active = ActivePlugin {
            definition: Arc::clone(&self.definition),
            instance: Some(instance),
            proposal_token_buffer: vec![0; MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS],
        };
        let driver = Arc::new(PluginDriver::spawn(active)?);
        let lifecycle: Arc<dyn GenerationLifecycleIngress> = Arc::new(NativeLifecycleIngress {
            driver: Arc::clone(&driver),
        });
        let proposals: Arc<dyn LinearProposalIngress> = Arc::new(NativeProposalIngress { driver });
        Ok(ModelServingHooks::new(
            GenerationReceiptConfig::from_lifecycle_ingress(lifecycle),
            LinearProposalIngressConfig::new(
                proposals,
                deadline,
                MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS,
            )?,
        ))
    }
}

struct ActivationInventory {
    entries: Vec<abi::TokenizerInventoryEntry>,
}

impl ActivationInventory {
    fn from_inventory(inventory: &abi::TokenizerInventory) -> Result<Self> {
        if inventory.schema_version != abi::TOKENIZER_INVENTORY_SCHEMA
            || inventory.tokens.is_empty()
        {
            bail!("bound model exposes an unsupported or empty tokenizer inventory");
        }
        let entries = inventory
            .tokens
            .iter()
            .map(|entry| {
                let (piece_kind, bytes) = match &entry.piece {
                    abi::TokenizerInventoryPiece::Bytes { bytes } => {
                        (abi::TokenizerPieceKind::BYTES, bytes.as_slice())
                    }
                    abi::TokenizerInventoryPiece::Control { identity } => {
                        (abi::TokenizerPieceKind::CONTROL, identity.as_bytes())
                    }
                };
                abi::TokenizerInventoryEntry {
                    id: entry.id,
                    piece_kind,
                    bytes: abi::ByteSlice::from_bytes(bytes),
                }
            })
            .collect();
        Ok(Self { entries })
    }

    fn view(&self) -> abi::TokenizerInventoryView {
        abi::TokenizerInventoryView {
            struct_size: size_of::<abi::TokenizerInventoryView>(),
            schema_version: abi::TOKENIZER_INVENTORY_SCHEMA,
            entries: self.entries.as_ptr(),
            entry_count: self.entries.len(),
        }
    }
}

struct LoadedDefinition {
    _library: Option<Library>,
    api: NonNull<abi::NativeServingPluginV1>,
    name: String,
}

// SAFETY: the ABI contract requires the static function table and loaded
// library to support concurrent host calls for the lifetime of the library.
unsafe impl Send for LoadedDefinition {}
// SAFETY: see the `Send` contract above; the table is immutable after load.
unsafe impl Sync for LoadedDefinition {}

impl LoadedDefinition {
    fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load native serving plugin {}", path.display()))?;
        let entry = unsafe {
            library.get::<abi::NativeServingPluginEntryV1>(abi::NATIVE_SERVING_PLUGIN_ENTRY_V1)
        }
        .with_context(|| {
            format!(
                "resolve native serving plugin entrypoint in {}",
                path.display()
            )
        })?;
        let api = NonNull::new(unsafe { entry() }.cast_mut())
            .context("native serving plugin entrypoint returned null")?;
        let name = validate_table(unsafe { api.as_ref() })?;
        Ok(Self {
            _library: Some(library),
            api,
            name,
        })
    }

    fn api(&self) -> &abi::NativeServingPluginV1 {
        unsafe { self.api.as_ref() }
    }

    fn status_error(
        &self,
        instance: abi::PluginInstance,
        action: &str,
        status: abi::PluginStatus,
    ) -> anyhow::Error {
        let detail = self.last_error(instance);
        anyhow!("{action} `{}` failed with {status:?}: {detail}", self.name)
    }

    fn last_error(&self, instance: abi::PluginInstance) -> String {
        let mut buffer = [0_u8; ERROR_BUFFER_BYTES];
        let written = unsafe {
            (self.api().last_error)(instance, buffer.as_mut_ptr().cast::<c_char>(), buffer.len())
        };
        let length = written.min(buffer.len());
        String::from_utf8_lossy(&buffer[..length]).into_owned()
    }
}

fn validate_table(table: &abi::NativeServingPluginV1) -> Result<String> {
    if table.abi_version != abi::NATIVE_SERVING_PLUGIN_ABI_V1 {
        bail!(
            "native serving plugin ABI {} is incompatible with host ABI {}",
            table.abi_version,
            abi::NATIVE_SERVING_PLUGIN_ABI_V1
        );
    }
    if table.struct_size != size_of::<abi::NativeServingPluginV1>() {
        bail!(
            "native serving plugin table size {} does not match host size {}",
            table.struct_size,
            size_of::<abi::NativeServingPluginV1>()
        );
    }
    let name = unsafe { read_utf8(table.plugin_name, "plugin name") }?;
    if name.trim().is_empty() {
        bail!("native serving plugin name must not be empty");
    }
    Ok(name)
}

struct ActivePlugin {
    definition: Arc<LoadedDefinition>,
    instance: Option<NonNull<c_void>>,
    proposal_token_buffer: Vec<i32>,
}

// SAFETY: activation succeeds only for plugins implementing the ABI's
// thread-safe instance contract. Mesh never mutates the opaque pointer.
unsafe impl Send for ActivePlugin {}
// SAFETY: see the `Send` contract above.
unsafe impl Sync for ActivePlugin {}

impl ActivePlugin {
    fn instance(&self) -> Result<abi::PluginInstance> {
        self.instance
            .map(NonNull::as_ptr)
            .context("native serving plugin is already shut down")
    }

    fn call_status(&self, action: &str, status: abi::PluginStatus) -> Result<()> {
        if status == abi::PluginStatus::OK {
            return Ok(());
        }
        Err(self.definition.status_error(
            self.instance.map_or(std::ptr::null_mut(), NonNull::as_ptr),
            action,
            status,
        ))
    }

    fn begin(&self, start: &GenerationStart) -> Result<()> {
        let agent_session_id = start.agent_session_id.as_deref().unwrap_or_default();
        let event = abi::GenerationStart {
            struct_size: size_of::<abi::GenerationStart>(),
            request_id: start.request_id,
            session_id: start.session_id,
            agent_session_id: abi::ByteSlice::from_bytes(agent_session_id.as_bytes()),
            prompt_token_ids: abi::TokenSlice::from_tokens(&start.prompt_token_ids),
        };
        let status = unsafe { (self.definition.api().begin_generation)(self.instance()?, &event) };
        self.call_status("begin generation", status)
    }

    fn committed(&self, commit: &GenerationCommit) -> Result<()> {
        let event = abi::GenerationCommit {
            struct_size: size_of::<abi::GenerationCommit>(),
            request_id: commit.request_id,
            session_id: commit.session_id,
            generated_token_count: u64::try_from(commit.generated_token_count)?,
            token_ids: abi::TokenSlice::from_tokens(&commit.token_ids),
        };
        let status = unsafe { (self.definition.api().commit_generation)(self.instance()?, &event) };
        self.call_status("commit generation", status)
    }

    fn abort(&self, abort: &GenerationAbort) -> Result<()> {
        let event = abi::GenerationAbort {
            struct_size: size_of::<abi::GenerationAbort>(),
            request_id: abort.request_id,
            session_id: abort.session_id,
        };
        let status = unsafe { (self.definition.api().abort_generation)(self.instance()?, &event) };
        self.call_status("abort generation", status)
    }

    fn finish(&self, receipt: &GenerationReceipt) -> Result<()> {
        let request_to_first_token_us = receipt.request_to_first_token_us.unwrap_or_default();
        let event = abi::GenerationFinish {
            struct_size: size_of::<abi::GenerationFinish>(),
            request_id: receipt.request_id,
            session_id: receipt.session_id,
            prompt_token_count: u64::try_from(receipt.prompt_token_count)?,
            prompt_token_digest: receipt.prompt_token_digest,
            prompt_token_ids: abi::TokenSlice::from_tokens(&receipt.prompt_token_ids),
            generated_token_ids: abi::TokenSlice::from_tokens(&receipt.generated_token_ids),
            final_session_position: receipt.final_session_position,
            termination: convert_termination(receipt.termination),
            model_generation_elapsed_us: receipt.model_generation_elapsed_us,
            has_request_to_first_token: receipt.request_to_first_token_us.is_some(),
            request_to_first_token_us,
            request_to_token_emission_us: abi::U64Slice::from_values(
                &receipt.request_to_token_emission_us,
            ),
        };
        let status = unsafe { (self.definition.api().finish_generation)(self.instance()?, &event) };
        self.call_status("finish generation", status)
    }

    fn propose(&mut self, query: LinearProposalQuery) -> Result<Option<LinearProposal>> {
        let event = abi::ProposalQuery {
            struct_size: size_of::<abi::ProposalQuery>(),
            request_id: query.request_id,
            session_id: query.session_id,
            prompt_token_count: u64::try_from(query.prompt_token_count)?,
            committed_token_count: u64::try_from(query.committed_token_count)?,
            decode_step: u64::try_from(query.decode_step)?,
            max_proposal_tokens: u64::try_from(query.max_proposal_tokens)?,
            absolute_deadline_ns: deadline_ns(query.deadline),
        };
        let mut operation = 0;
        let status = unsafe {
            (self.definition.api().start_proposal)(self.instance()?, &event, &raw mut operation)
        };
        self.call_status("start proposal", status)?;
        self.poll_until_deadline(operation, query.deadline, query.max_proposal_tokens)
    }

    fn poll_until_deadline(
        &mut self,
        operation: abi::ProposalOperation,
        deadline: Instant,
        max_proposal_tokens: usize,
    ) -> Result<Option<LinearProposal>> {
        let mut token_buffer = std::mem::take(&mut self.proposal_token_buffer);
        let token_capacity = max_proposal_tokens.min(token_buffer.len());
        let result =
            self.poll_with_buffer(operation, deadline, &mut token_buffer[..token_capacity]);
        self.proposal_token_buffer = token_buffer;
        result
    }

    fn poll_with_buffer(
        &self,
        operation: abi::ProposalOperation,
        deadline: Instant,
        token_ids: &mut [i32],
    ) -> Result<Option<LinearProposal>> {
        let instance = self.instance()?;
        let mut decision_id = [0_u8; abi::MAX_DECISION_ID_BYTES];
        while Instant::now() < deadline {
            let mut output = abi::ProposalOutput {
                struct_size: size_of::<abi::ProposalOutput>(),
                decision_id: decision_id.as_mut_ptr(),
                decision_id_capacity: decision_id.len(),
                decision_id_length: 0,
                token_ids: token_ids.as_mut_ptr(),
                token_capacity: token_ids.len(),
                token_length: 0,
            };
            let status = unsafe {
                (self.definition.api().poll_proposal)(instance, operation, &raw mut output)
            };
            match status {
                abi::ProposalPollStatus::PENDING => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if !remaining.is_zero() {
                        thread::sleep(PROPOSAL_POLL_INTERVAL.min(remaining));
                    }
                }
                abi::ProposalPollStatus::ABSTAIN => return Ok(None),
                abi::ProposalPollStatus::FAILED => {
                    let error = self.definition.status_error(
                        instance,
                        "poll proposal",
                        abi::PluginStatus::INTERNAL_ERROR,
                    );
                    unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
                    return Err(error);
                }
                abi::ProposalPollStatus::READY => {
                    let proposal = proposal_from_output(&decision_id, token_ids, &output);
                    if proposal.is_err() {
                        unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
                    }
                    return proposal.map(Some);
                }
                unknown => {
                    unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
                    bail!(
                        "native serving plugin returned unknown proposal poll status {}",
                        unknown.0
                    );
                }
            }
        }
        unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
        Ok(None)
    }

    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()> {
        let event = abi::ProposalOutcome {
            struct_size: size_of::<abi::ProposalOutcome>(),
            decision_id: abi::ByteSlice::from_bytes(receipt.decision_id.as_bytes()),
            disposition: convert_disposition(receipt.disposition),
            proposal_token_count: u64::try_from(receipt.proposal_token_count)?,
            verification_rows: u64::try_from(receipt.verification_rows)?,
            accepted_proposal_tokens: u64::try_from(receipt.accepted_proposal_tokens)?,
            committed_tokens: abi::TokenSlice::from_tokens(&receipt.committed_tokens),
            verification_row_predictions: abi::TokenSlice::from_tokens(
                &receipt.verification_row_predictions,
            ),
            canonical_prediction_count: u64::try_from(receipt.canonical_prediction_count)?,
            has_correction_or_boundary_token: receipt.correction_or_boundary_token.is_some(),
            correction_or_boundary_token: receipt.correction_or_boundary_token.unwrap_or_default(),
            base_position: receipt.base_position,
            position_after_verification: receipt.position_after_verification,
            canonical_position: receipt.canonical_position,
            trimmed_rows: u64::try_from(receipt.trimmed_rows)?,
        };
        let status = unsafe { (self.definition.api().report_proposal)(self.instance()?, &event) };
        self.call_status("report proposal", status)
    }

    fn discard(&self, decision_id: &[u8], reason: LinearProposalDiscardReason) -> Result<()> {
        let event = abi::ProposalDiscard {
            struct_size: size_of::<abi::ProposalDiscard>(),
            decision_id: abi::ByteSlice::from_bytes(decision_id),
            reason: convert_discard_reason(reason),
        };
        let status = unsafe { (self.definition.api().discard_proposal)(self.instance()?, &event) };
        self.call_status("discard proposal", status)
    }

    fn shutdown(&mut self) -> Result<()> {
        let Some(instance) = self.instance.take() else {
            return Ok(());
        };
        let status = unsafe { (self.definition.api().shutdown)(instance.as_ptr()) };
        self.call_status("shutdown", status)
    }
}

impl Drop for ActivePlugin {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("native serving plugin shutdown failed: {error:#}");
        }
    }
}

struct NativeLifecycleIngress {
    driver: Arc<PluginDriver>,
}

impl GenerationLifecycleIngress for NativeLifecycleIngress {
    fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
        let command = match observation {
            GenerationLifecycleObservation::Started(start) => PluginCommand::Begin(start),
            GenerationLifecycleObservation::Committed(commit) => PluginCommand::Committed(commit),
            GenerationLifecycleObservation::Aborted(abort) => PluginCommand::Abort(abort),
            GenerationLifecycleObservation::Completed(receipt) => PluginCommand::Finish(receipt),
            _ => return Ok(()),
        };
        self.driver.enqueue(command)
    }

    fn delivery_failures(&self) -> u64 {
        self.driver.lifecycle_delivery_failures()
    }
}

struct NativeProposalIngress {
    driver: Arc<PluginDriver>,
}

impl LinearProposalIngress for NativeProposalIngress {
    fn propose(&self, query: LinearProposalQuery) -> Result<Option<LinearProposal>> {
        self.driver.propose(query)
    }

    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()> {
        self.driver.enqueue(PluginCommand::Report(receipt.clone()))
    }

    fn discard(
        &self,
        decision_id: &OpaqueProposalDecisionId,
        reason: LinearProposalDiscardReason,
    ) -> Result<()> {
        self.driver.enqueue(PluginCommand::Discard(
            decision_id.as_bytes().to_vec(),
            reason,
        ))
    }
}

enum PluginCommand {
    Begin(GenerationStart),
    Committed(GenerationCommit),
    Abort(GenerationAbort),
    Finish(GenerationReceipt),
    Proposal(
        LinearProposalQuery,
        SyncSender<std::result::Result<Option<LinearProposal>, String>>,
    ),
    Report(LinearProposalReceipt),
    Discard(Vec<u8>, LinearProposalDiscardReason),
    Shutdown(SyncSender<std::result::Result<(), String>>),
}

struct PluginDriver {
    sender: SyncSender<PluginCommand>,
    fatal_error: Arc<Mutex<Option<String>>>,
    lifecycle_delivery_failures: Arc<AtomicU64>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl PluginDriver {
    fn spawn(active: ActivePlugin) -> Result<Self> {
        let (sender, receiver) = sync_channel(PLUGIN_COMMAND_CAPACITY);
        let fatal_error = Arc::new(Mutex::new(None));
        let worker_fatal_error = Arc::clone(&fatal_error);
        let lifecycle_delivery_failures = Arc::new(AtomicU64::new(0));
        let worker_lifecycle_delivery_failures = Arc::clone(&lifecycle_delivery_failures);
        let worker = thread::Builder::new()
            .name("mesh-native-serving-plugin".to_string())
            .spawn(move || {
                plugin_worker(
                    active,
                    receiver,
                    worker_fatal_error,
                    worker_lifecycle_delivery_failures,
                );
            })
            .context("spawn native serving plugin worker")?;
        Ok(Self {
            sender,
            fatal_error,
            lifecycle_delivery_failures,
            worker: Mutex::new(Some(worker)),
        })
    }

    fn ensure_healthy(&self) -> Result<()> {
        let error = self
            .fatal_error
            .lock()
            .map_err(|_| anyhow!("native serving plugin health lock poisoned"))?;
        if let Some(error) = error.as_deref() {
            bail!("native serving plugin worker failed: {error}");
        }
        Ok(())
    }

    fn enqueue(&self, command: PluginCommand) -> Result<()> {
        self.ensure_healthy()?;
        self.sender.try_send(command).map_err(|error| match error {
            TrySendError::Full(_) => anyhow!("native serving plugin command queue is full"),
            TrySendError::Disconnected(_) => anyhow!("native serving plugin worker stopped"),
        })
    }

    fn lifecycle_delivery_failures(&self) -> u64 {
        self.lifecycle_delivery_failures.load(Ordering::Relaxed)
    }

    fn propose(&self, query: LinearProposalQuery) -> Result<Option<LinearProposal>> {
        self.ensure_healthy()?;
        let deadline = query.deadline;
        let (reply, response) = sync_channel(1);
        match self.sender.try_send(PluginCommand::Proposal(query, reply)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Ok(None),
            Err(TrySendError::Disconnected(_)) => {
                bail!("native serving plugin worker stopped")
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        match response.recv_timeout(remaining) {
            Ok(result) => result.map_err(anyhow::Error::msg),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                bail!("native serving plugin worker stopped before replying")
            }
        }
    }
}

impl Drop for PluginDriver {
    fn drop(&mut self) {
        let (reply, response) = sync_channel(1);
        let queued = self.sender.try_send(PluginCommand::Shutdown(reply)).is_ok();
        if queued
            && response.recv_timeout(CLEAN_SHUTDOWN_TIMEOUT).is_ok()
            && let Ok(worker) = self.worker.get_mut()
            && let Some(worker) = worker.take()
        {
            let _ = worker.join();
        }
    }
}

fn plugin_worker(
    mut active: ActivePlugin,
    receiver: Receiver<PluginCommand>,
    fatal_error: Arc<Mutex<Option<String>>>,
    lifecycle_delivery_failures: Arc<AtomicU64>,
) {
    while let Ok(command) = receiver.recv() {
        let (result, terminal, lifecycle) = match command {
            PluginCommand::Begin(event) => (active.begin(&event), false, true),
            PluginCommand::Committed(event) => (active.committed(&event), false, true),
            PluginCommand::Abort(event) => (active.abort(&event), false, true),
            PluginCommand::Finish(event) => (active.finish(&event), false, true),
            PluginCommand::Proposal(query, reply) => {
                let result = active.propose(query);
                let _ = reply.send(
                    result
                        .as_ref()
                        .map(Clone::clone)
                        .map_err(ToString::to_string),
                );
                (result.map(|_| ()), false, false)
            }
            PluginCommand::Report(event) => (active.report(&event), false, false),
            PluginCommand::Discard(decision_id, reason) => {
                (active.discard(&decision_id, reason), false, false)
            }
            PluginCommand::Shutdown(reply) => {
                let result = active.shutdown();
                let _ = reply.send(result.as_ref().map(|_| ()).map_err(ToString::to_string));
                (result, true, false)
            }
        };
        if lifecycle && result.is_err() {
            lifecycle_delivery_failures.fetch_add(1, Ordering::Relaxed);
        }
        if terminal
            && let Err(error) = result
            && let Ok(mut fatal) = fatal_error.lock()
        {
            *fatal = Some(format!("{error:#}"));
        }
        if terminal {
            break;
        }
    }
}

fn proposal_from_output(
    decision_id: &[u8; abi::MAX_DECISION_ID_BYTES],
    token_ids: &[i32],
    output: &abi::ProposalOutput,
) -> Result<LinearProposal> {
    if output.decision_id_length == 0 || output.decision_id_length > decision_id.len() {
        bail!(
            "native serving plugin returned invalid decision ID length {}",
            output.decision_id_length
        );
    }
    if output.token_length == 0 || output.token_length > token_ids.len() {
        bail!(
            "native serving plugin returned invalid proposal length {}",
            output.token_length
        );
    }
    let decision =
        OpaqueProposalDecisionId::new(decision_id[..output.decision_id_length].to_vec())?;
    Ok(LinearProposal::new(
        decision,
        token_ids[..output.token_length].to_vec(),
    ))
}

fn convert_termination(
    value: skippy_server::frontend::GenerationTermination,
) -> abi::GenerationTermination {
    match value {
        skippy_server::frontend::GenerationTermination::CallbackStop => {
            abi::GenerationTermination::CALLBACK_STOP
        }
        skippy_server::frontend::GenerationTermination::MaxTokens => {
            abi::GenerationTermination::MAX_TOKENS
        }
        skippy_server::frontend::GenerationTermination::Cancelled => {
            abi::GenerationTermination::CANCELLED
        }
        _ => abi::GenerationTermination::CANCELLED,
    }
}

fn convert_disposition(value: LinearProposalDisposition) -> abi::ProposalDisposition {
    match value {
        LinearProposalDisposition::FullAccept => abi::ProposalDisposition::FULL_ACCEPT,
        LinearProposalDisposition::FirstMismatch => abi::ProposalDisposition::FIRST_MISMATCH,
        LinearProposalDisposition::Stopped => abi::ProposalDisposition::STOPPED,
        _ => abi::ProposalDisposition::STOPPED,
    }
}

fn convert_discard_reason(value: LinearProposalDiscardReason) -> abi::ProposalDiscardReason {
    match value {
        LinearProposalDiscardReason::DeadlineExceeded => {
            abi::ProposalDiscardReason::DEADLINE_EXCEEDED
        }
        LinearProposalDiscardReason::InvalidTokenCount => {
            abi::ProposalDiscardReason::INVALID_TOKEN_COUNT
        }
        LinearProposalDiscardReason::InvalidTokenId => abi::ProposalDiscardReason::INVALID_TOKEN_ID,
        LinearProposalDiscardReason::PositionMismatch => {
            abi::ProposalDiscardReason::POSITION_MISMATCH
        }
        LinearProposalDiscardReason::ExecutionFailed => {
            abi::ProposalDiscardReason::EXECUTION_FAILED
        }
        _ => abi::ProposalDiscardReason::EXECUTION_FAILED,
    }
}

fn deadline_ns(deadline: Instant) -> u64 {
    let remaining = deadline.saturating_duration_since(Instant::now());
    unsafe { monotonic_now_ns(std::ptr::null_mut()) }
        .saturating_add(u64::try_from(remaining.as_nanos()).unwrap_or(u64::MAX))
}

unsafe extern "C" fn monotonic_now_ns(_context: *mut c_void) -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let elapsed = ORIGIN.get_or_init(Instant::now).elapsed().as_nanos();
    u64::try_from(elapsed).unwrap_or(u64::MAX)
}

fn validate_absolute_path(label: &str, path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("{label} path must be absolute: {}", path.display());
    }
    Ok(())
}

fn path_slice(path: &Path) -> abi::ByteSlice {
    abi::ByteSlice::from_bytes(path.as_os_str().as_encoded_bytes())
}

unsafe fn read_utf8(slice: abi::ByteSlice, label: &str) -> Result<String> {
    if slice.pointer.is_null() && slice.length != 0 {
        bail!("native serving plugin {label} has a null pointer");
    }
    let bytes = if slice.length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(slice.pointer, slice.length) }
    };
    std::str::from_utf8(bytes)
        .with_context(|| format!("native serving plugin {label} is not UTF-8"))
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FAKE_NAME: &[u8] = b"test-serving-plugin";
    static CANCEL_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn tokenizer_inventory_view_borrows_host_owned_bytes_for_activation() {
        let inventory = abi::TokenizerInventory {
            schema_version: abi::TOKENIZER_INVENTORY_SCHEMA,
            model_id: "glm".to_string(),
            source_model_sha256: "a".repeat(64),
            tokenizer_id: "gguf-source-sha256:test".to_string(),
            tokens: vec![abi::TokenizerInventoryToken {
                id: 0,
                piece: abi::TokenizerInventoryPiece::Bytes {
                    bytes: b"hello".to_vec(),
                },
            }],
        };
        let activation = ActivationInventory::from_inventory(&inventory).unwrap();
        let view = activation.view();
        assert_eq!(view.schema_version, abi::TOKENIZER_INVENTORY_SCHEMA);
        assert_eq!(view.entry_count, 1);
        let entry = unsafe { &*view.entries };
        assert_eq!(entry.piece_kind, abi::TokenizerPieceKind::BYTES);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(entry.bytes.pointer, entry.bytes.length) },
            b"hello"
        );
    }

    struct FakeState {
        start_delay: Duration,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    unsafe extern "C" fn fake_activate(
        _context: *const abi::ActivationContext,
        _activation: *mut abi::PluginActivation,
    ) -> abi::PluginStatus {
        abi::PluginStatus::INTERNAL_ERROR
    }

    unsafe extern "C" fn fake_shutdown(instance: abi::PluginInstance) -> abi::PluginStatus {
        if !instance.is_null() {
            drop(unsafe { Box::from_raw(instance.cast::<FakeState>()) });
        }
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_begin(
        instance: abi::PluginInstance,
        _event: *const abi::GenerationStart,
    ) -> abi::PluginStatus {
        let state = unsafe { &*instance.cast::<FakeState>() };
        state.events.lock().unwrap().push("begin");
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_commit(
        instance: abi::PluginInstance,
        _event: *const abi::GenerationCommit,
    ) -> abi::PluginStatus {
        let state = unsafe { &*instance.cast::<FakeState>() };
        state.events.lock().unwrap().push("commit");
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_abort(
        _instance: abi::PluginInstance,
        _event: *const abi::GenerationAbort,
    ) -> abi::PluginStatus {
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_finish(
        _instance: abi::PluginInstance,
        _event: *const abi::GenerationFinish,
    ) -> abi::PluginStatus {
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_start_proposal(
        instance: abi::PluginInstance,
        _query: *const abi::ProposalQuery,
        operation: *mut abi::ProposalOperation,
    ) -> abi::PluginStatus {
        let state = unsafe { &*instance.cast::<FakeState>() };
        state.events.lock().unwrap().push("proposal");
        thread::sleep(state.start_delay);
        unsafe { *operation = 1 };
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_poll_proposal(
        _instance: abi::PluginInstance,
        _operation: abi::ProposalOperation,
        _output: *mut abi::ProposalOutput,
    ) -> abi::ProposalPollStatus {
        abi::ProposalPollStatus::ABSTAIN
    }

    unsafe extern "C" fn fake_cancel_proposal(
        _instance: abi::PluginInstance,
        _operation: abi::ProposalOperation,
    ) {
        CANCEL_COUNT.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn fake_report_proposal(
        _instance: abi::PluginInstance,
        _event: *const abi::ProposalOutcome,
    ) -> abi::PluginStatus {
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_discard_proposal(
        _instance: abi::PluginInstance,
        _event: *const abi::ProposalDiscard,
    ) -> abi::PluginStatus {
        abi::PluginStatus::OK
    }

    unsafe extern "C" fn fake_last_error(
        _instance: abi::PluginInstance,
        _output: *mut c_char,
        _capacity: usize,
    ) -> usize {
        0
    }

    fn fake_table() -> abi::NativeServingPluginV1 {
        abi::NativeServingPluginV1 {
            abi_version: abi::NATIVE_SERVING_PLUGIN_ABI_V1,
            struct_size: size_of::<abi::NativeServingPluginV1>(),
            plugin_name: abi::ByteSlice::from_bytes(FAKE_NAME),
            activate: fake_activate,
            shutdown: fake_shutdown,
            begin_generation: fake_begin,
            commit_generation: fake_commit,
            abort_generation: fake_abort,
            finish_generation: fake_finish,
            start_proposal: fake_start_proposal,
            poll_proposal: fake_poll_proposal,
            cancel_proposal: fake_cancel_proposal,
            report_proposal: fake_report_proposal,
            discard_proposal: fake_discard_proposal,
            last_error: fake_last_error,
        }
    }

    fn fake_active(start_delay: Duration) -> ActivePlugin {
        fake_active_with_events(start_delay).0
    }

    fn fake_active_with_events(
        start_delay: Duration,
    ) -> (ActivePlugin, Arc<Mutex<Vec<&'static str>>>) {
        let table = Box::leak(Box::new(fake_table()));
        let definition = Arc::new(LoadedDefinition {
            _library: None,
            api: NonNull::from(table),
            name: "test-serving-plugin".to_string(),
        });
        let events = Arc::new(Mutex::new(Vec::new()));
        let state = Box::new(FakeState {
            start_delay,
            events: Arc::clone(&events),
        });
        (
            ActivePlugin {
                definition,
                instance: NonNull::new(Box::into_raw(state).cast::<c_void>()),
                proposal_token_buffer: vec![0; MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS],
            },
            events,
        )
    }

    #[test]
    fn output_validation_is_fail_closed() {
        let decision = [1_u8; abi::MAX_DECISION_ID_BYTES];
        let tokens = [7_i32; 8_192];
        let mut output = abi::ProposalOutput {
            struct_size: size_of::<abi::ProposalOutput>(),
            decision_id: std::ptr::null_mut(),
            decision_id_capacity: decision.len(),
            decision_id_length: 1,
            token_ids: std::ptr::null_mut(),
            token_capacity: tokens.len(),
            token_length: 1,
        };
        assert!(proposal_from_output(&decision, &tokens, &output).is_ok());
        output.decision_id_length = decision.len() + 1;
        assert!(proposal_from_output(&decision, &tokens, &output).is_err());
        output.decision_id_length = 1;
        output.token_length = 0;
        assert!(proposal_from_output(&decision, &tokens, &output).is_err());
    }

    #[test]
    fn absolute_deadline_uses_the_host_clock_epoch() {
        let before = unsafe { monotonic_now_ns(std::ptr::null_mut()) };
        let deadline = deadline_ns(Instant::now() + Duration::from_millis(5));
        let after = unsafe { monotonic_now_ns(std::ptr::null_mut()) };
        assert!(deadline >= before.saturating_add(1_000_000));
        assert!(deadline <= after.saturating_add(10_000_000));
    }

    #[test]
    fn table_validation_rejects_version_and_layout_mismatches() {
        let mut table = fake_table();
        assert_eq!(validate_table(&table).unwrap(), "test-serving-plugin");

        table.abi_version += 1;
        assert!(
            validate_table(&table)
                .unwrap_err()
                .to_string()
                .contains("incompatible")
        );
        table.abi_version = abi::NATIVE_SERVING_PLUGIN_ABI_V1;
        table.struct_size -= 1;
        assert!(
            validate_table(&table)
                .unwrap_err()
                .to_string()
                .contains("table size")
        );
    }

    #[test]
    fn blocking_plugin_cannot_extend_the_decode_deadline() {
        CANCEL_COUNT.store(0, Ordering::SeqCst);
        let driver = PluginDriver::spawn(fake_active(Duration::from_millis(250))).unwrap();
        let started = Instant::now();
        let result = driver
            .propose(LinearProposalQuery::new(
                1,
                2,
                16,
                16,
                0,
                8_192,
                started + Duration::from_millis(5),
            ))
            .unwrap();
        let elapsed = started.elapsed();

        assert!(result.is_none());
        assert!(
            elapsed < Duration::from_millis(150),
            "decode waited {elapsed:?} for a blocking plugin"
        );
        drop(driver);
        assert_eq!(CANCEL_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn lifecycle_ingress_shares_plugin_queue_order_with_proposals() {
        let (active, events) = fake_active_with_events(Duration::ZERO);
        let driver = Arc::new(PluginDriver::spawn(active).unwrap());
        let ingress = NativeLifecycleIngress {
            driver: Arc::clone(&driver),
        };
        ingress
            .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
                request_id: 1,
                session_id: 2,
                agent_session_id: None,
                prompt_token_ids: Arc::from([3]),
            }))
            .unwrap();
        ingress
            .try_submit(GenerationLifecycleObservation::Committed(
                GenerationCommit {
                    request_id: 1,
                    session_id: 2,
                    generated_token_count: 1,
                    token_ids: vec![4].into_boxed_slice(),
                },
            ))
            .unwrap();
        driver
            .propose(LinearProposalQuery::new(
                1,
                2,
                1,
                1,
                0,
                8,
                Instant::now() + Duration::from_millis(100),
            ))
            .unwrap();

        assert_eq!(*events.lock().unwrap(), ["begin", "commit", "proposal"]);
    }
}
