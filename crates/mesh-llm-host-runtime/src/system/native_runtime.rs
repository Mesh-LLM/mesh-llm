#[cfg(feature = "dynamic-native-runtime")]
mod dynamic {
    use crate::runtime_events::engine::{RuntimeEventEngine, SyntheticTerminal};
    use crate::runtime_events::runtime_event_engine;
    use crate::system::native_runtime_install::{
        NativeRuntimeInstallOptions, NativeRuntimeInstallOutcome,
    };
    use anyhow::{Context, Result};
    use mesh_llm_native_runtime::{
        HostRuntimeProfile, InstalledNativeRuntime, NativeRuntimeArtifact, NativeRuntimeCache,
        NativeRuntimeLoadPlan, NativeRuntimeReleaseManifest, RuntimeSelection,
    };
    use mesh_llm_runtime_event_contracts::{
        BoundedNumericSummaries, DeliveryClass, DiagnosticEventKind, DiagnosticFact, FactData,
        HumanSummary, KvRuntimeStateEventKind, KvRuntimeStateFact, ModelLoadingEventKind,
        ModelLoadingFact, ModelUnloadingEventKind, ModelUnloadingFact, NumericSummary,
        NumericSummaryKey, NumericValue, OperationId, OperationScope, Outcome, ReasonCode,
        ResourceHealthEventKind, ResourceHealthFact, RuntimeEventIngress, RuntimeFact,
    };
    use skippy_runtime::{RuntimeEvent, RuntimeEventKind};
    use std::sync::Arc;
    use std::{future::Future, path::PathBuf};

    #[derive(Clone, Debug)]
    pub(crate) struct LoadedNativeRuntime {
        pub(crate) native_runtime_id: String,
        pub(crate) libraries: Vec<PathBuf>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) enum NativeRuntimePlanSource {
        CacheHit,
        LocalDiscovery,
        PostInstall,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct NativeRuntimeStartupLoadPlan {
        pub(crate) cache_mesh_version: String,
        pub(crate) native_runtime_id: String,
        pub(crate) root: PathBuf,
        pub(crate) selected_library_path: PathBuf,
        pub(crate) libraries: Vec<PathBuf>,
        pub(crate) source: NativeRuntimePlanSource,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct NativeRuntimeStartupSelection {
        pub(crate) mesh_version: String,
        pub(crate) skippy_abi: Option<String>,
        pub(crate) runtime_selection: RuntimeSelection,
    }

    impl NativeRuntimeStartupSelection {
        pub(crate) fn current() -> Self {
            Self {
                mesh_version: crate::RELEASE_VERSION.to_string(),
                skippy_abi: Some(
                    crate::system::native_runtime_install::current_skippy_abi_version(),
                ),
                runtime_selection: RuntimeSelection::Recommended,
            }
        }

        pub(crate) fn explicit(
            mesh_version: String,
            skippy_abi: Option<String>,
            runtime_selection: RuntimeSelection,
        ) -> Self {
            Self {
                mesh_version,
                skippy_abi,
                runtime_selection,
            }
        }
    }

    pub(crate) fn load_local_native_runtime_for_embedded_serving()
    -> Result<Option<LoadedNativeRuntime>> {
        if skippy_runtime::native_runtime_loaded() {
            return Ok(None);
        }
        // Reserved before any discovery/load work begins. A `?` failure
        // below drops the guard unresolved, which synthesizes
        // `RuntimeResolutionFailed`/`terminal_not_delivered` through the
        // engine's own Drop mechanism -- acceptable here since the actual
        // failure reason is already captured by the returned `anyhow::Error`
        // and the tracing/audit surfaces this function's callers already use.
        let resolution = crate::system::native_runtime_events::NativeRuntimeResolution::begin();
        let cache = default_native_runtime_cache()?;
        let local_runtimes =
            crate::system::native_runtime_install::discover_local_native_runtimes(&[], &cache)?;
        let Some(plan) = resolve_local_native_runtime_plan(
            &local_runtimes,
            &host_runtime_profile(),
            crate::BUILD_VERSION,
            crate::RELEASE_VERSION,
            Some(&crate::system::native_runtime_install::current_skippy_abi_version()),
            &RuntimeSelection::Recommended,
        )?
        else {
            // No compatible plan found at all -- a real
            // `NativeLibraryUnavailable`/`RuntimeResolutionFailed`, not a
            // no-op (the earlier `native_runtime_loaded()` check above is
            // the ONLY genuine "not needed" case in this function, and it
            // returns before `resolution` is even constructed).
            resolution.unavailable(mesh_llm_runtime_event_contracts::ReasonCode::MissingArtifact);
            return Ok(None);
        };
        unsafe { skippy_runtime::load_native_runtime_libraries(&plan.libraries) }
            .map_err(anyhow::Error::from)
            .with_context(|| {
                format!(
                    "load local native runtime {} from {} for embedded serving",
                    plan.native_runtime_id,
                    plan.root.display()
                )
            })?;
        resolution.library_loaded();
        install_runtime_scoped_event_reporter();
        resolution.initialized();
        resolution.completed();
        Ok(Some(LoadedNativeRuntime {
            native_runtime_id: plan.native_runtime_id,
            libraries: plan.libraries,
        }))
    }

    pub(crate) async fn try_load_installed_native_runtime(
        startup_selection: NativeRuntimeStartupSelection,
    ) -> Result<Option<LoadedNativeRuntime>> {
        let resolution = crate::system::native_runtime_events::NativeRuntimeResolution::begin();
        let outcome = try_load_installed_native_runtime_with(
            skippy_runtime::native_runtime_loaded,
            default_native_runtime_cache,
            host_runtime_profile,
            default_install_options,
            default_install_executor,
            startup_selection,
            |libraries| {
                let result = unsafe { skippy_runtime::load_native_runtime_libraries(libraries) }
                    .map_err(anyhow::Error::from);
                if result.is_ok() {
                    install_runtime_scoped_event_reporter();
                }
                result
            },
        )
        .await;
        match &outcome {
            Ok(Some(_)) => {
                resolution.initialized();
                resolution.completed();
            }
            Ok(None) => resolution.not_needed(),
            Err(_) => {
                resolution.failed(mesh_llm_runtime_event_contracts::ReasonCode::ArtifactIoFailure)
            }
        }
        outcome
    }

    /// Installs the process-global runtime-scoped event reporter right after
    /// a native runtime library loads. A no-op on a runtime that doesn't
    /// advertise the `runtime_event_reporter` family (probed by
    /// `skippy_runtime::probe_capabilities` internally) — older or
    /// differently-composed runtimes simply keep operating without this
    /// reporter, matching the model-open feature-probe fallback contract.
    fn install_runtime_scoped_event_reporter() {
        skippy_runtime::install_runtime_event_reporter(runtime_scoped_native_event_sink);
    }

    /// The installed callback (D7, `.omo/plans/event-system-fixes.md` task
    /// 10). Runs on the native worker thread that raised the event: maps it
    /// to a `RuntimeFact` (`native_family_fact`, a pure function -- no
    /// I/O, no logging, every byte it allocates becomes part of the
    /// returned fact) and submits it (`submit_native_family_fact`). Kind
    /// values 1-5 (`SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_*`) belong to the
    /// separate per-call model-open reporter and structurally never reach
    /// this process-global one (`events_internal.h`'s `dispatch()` seam is
    /// only called by `skippy_emit_{kv,device,diagnostic,unload}_event`/
    /// `skippy_emit_model_load_event_v2`); `native_family_fact` still
    /// returns `None` for them defensively rather than assuming that holds
    /// forever.
    fn runtime_scoped_native_event_sink(event: RuntimeEvent) {
        let Some(fact) = native_family_fact(&event) else {
            return;
        };
        let Some(engine) = runtime_event_engine() else {
            return;
        };
        submit_native_family_fact(&engine, fact);
    }

    /// One numeric correlation field carried from the native envelope. The
    /// key allocation becomes part of the `NumericSummary` stored on the
    /// returned fact -- not a transient/logging-only allocation.
    fn native_numeric_summary(key: &str, value: u64) -> Option<NumericSummary> {
        NumericSummaryKey::new(key)
            .ok()
            .map(|key| NumericSummary::new(key, NumericValue::Unsigned(value)))
    }

    /// `FactData` carrying only what the current `try_submit(RuntimeFact)`
    /// boundary can transport (README, `mesh-llm-runtime-event-contracts`:
    /// `NativeSourceEnvelope` lives on `RuntimeEventEnvelope`, which
    /// `try_submit` does not accept -- same limit `frames.rs::producer_str`
    /// documents). The native model/stage/session ids and numeric
    /// summaries are therefore threaded through as bounded
    /// `numeric_summaries` rather than dropped.
    fn native_fact_data(event: &RuntimeEvent) -> FactData {
        let summaries = [
            native_numeric_summary("native_sequence", event.sequence),
            native_numeric_summary("native_model_id", event.model_id),
            native_numeric_summary("native_stage_id", event.stage_id),
            native_numeric_summary("native_session_id", event.session_id),
            event
                .numeric_summary_0
                .and_then(|value| native_numeric_summary("native_numeric_summary_0", value)),
            event
                .numeric_summary_1
                .and_then(|value| native_numeric_summary("native_numeric_summary_1", value)),
            event
                .numeric_summary_2
                .and_then(|value| native_numeric_summary("native_numeric_summary_2", value)),
            event
                .numeric_summary_3
                .and_then(|value| native_numeric_summary("native_numeric_summary_3", value)),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let summary = (!event.detail_bytes.is_empty())
            .then(|| String::from_utf8_lossy(&event.detail_bytes).into_owned())
            .and_then(|text| HumanSummary::new(&text).ok());
        FactData {
            numeric_summaries: BoundedNumericSummaries::new(summaries).unwrap_or_default(),
            summary,
            ..FactData::default()
        }
    }

    fn native_terminal_failure_data(event: &RuntimeEvent, reason: ReasonCode) -> FactData {
        FactData {
            outcome: Some(Outcome::Failure),
            reason: Some(reason),
            ..native_fact_data(event)
        }
    }

    fn native_success_data(event: &RuntimeEvent) -> FactData {
        FactData {
            outcome: Some(Outcome::Success),
            ..native_fact_data(event)
        }
    }

    /// The `native_family_mappings` table (`inventory/runtime_events.toml`)
    /// realized as code: exactly the mapping the inventory contract test
    /// (`native_family_mappings.rs`) cross-checks against the native patch
    /// queue's kind literals. Pure: no I/O, no logging, no lock -- every
    /// allocation this function or its helpers perform ends up owned by
    /// the returned `RuntimeFact`.
    fn native_family_fact(event: &RuntimeEvent) -> Option<RuntimeFact> {
        let fact = match event.kind {
            RuntimeEventKind::ModelLoadPhaseChanged
            | RuntimeEventKind::ModelLoadTensorsOffloaded
            | RuntimeEventKind::ModelLoadTokenizerReady
            | RuntimeEventKind::ModelLoadAuxComponentReady => {
                RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                    ModelLoadingEventKind::ModelLoadPhaseChanged,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::ModelLoadMemoryAllocated => {
                RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                    ModelLoadingEventKind::ModelMemoryAllocationSummary,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvInitialized => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::KvCacheInitializationCompleted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvPressureCrossed => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::CachePressureCrossed,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvPressureCleared => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::CachePressureCleared,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvContextApproachingCapacity => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::ContextCapacityApproachingLimit,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvContextCapacityExhausted => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::ContextExhausted,
                    native_terminal_failure_data(event, ReasonCode::ContextExhausted),
                ))
            }
            RuntimeEventKind::DeviceBackendInitialized => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::BackendInitializationCompleted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceReady => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceReady,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceDegraded => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceDegraded,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceUnavailable => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceUnavailable,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceRecovered => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceRecovered,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceLost => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceLost,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceResourceAllocated => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::ResourceAllocationCompleted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceOutOfMemory => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::OutOfMemoryCondition,
                    native_terminal_failure_data(event, ReasonCode::OutOfMemory),
                ))
            }
            RuntimeEventKind::DeviceFallbackActivated => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::BackendFallbackActivated,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticWarningRaised => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::WarningRaised,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticWarningCleared => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::WarningCleared,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticRecoverableFailure => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::RecoverableNativeFailure,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticFatalFailure => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::FatalNativeFailure,
                    native_terminal_failure_data(event, ReasonCode::InternalRuntimeFailure),
                ))
            }
            RuntimeEventKind::DiagnosticInvariantViolation => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::InvariantProtocolViolation,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::UnloadStarted => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadStarted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::UnloadCompleted => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadCompleted,
                    native_success_data(event),
                ))
            }
            RuntimeEventKind::UnloadFailed => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadFailed,
                    native_terminal_failure_data(event, ReasonCode::InternalRuntimeFailure),
                ))
            }
            RuntimeEventKind::UnloadForced => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::ForcedUnload,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::UnloadSessionDraining => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::SessionDrainingStarted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::ModelOpenStarted
            | RuntimeEventKind::ModelOpenProgress
            | RuntimeEventKind::BackendDeviceSelected
            | RuntimeEventKind::ModelOpenFinished
            | RuntimeEventKind::ModelOpenFailedHandled
            | RuntimeEventKind::Unknown(_) => return None,
        };
        Some(fact)
    }

    fn native_family_terminal_not_delivered() -> FactData {
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        }
    }

    fn synthetic_kv_runtime_state_terminal() -> RuntimeFact {
        RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
            KvRuntimeStateEventKind::ContextExhausted,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_resource_health_terminal() -> RuntimeFact {
        RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
            ResourceHealthEventKind::OutOfMemoryCondition,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_diagnostic_terminal() -> RuntimeFact {
        RuntimeFact::Diagnostic(DiagnosticFact::with_data(
            DiagnosticEventKind::FatalNativeFailure,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_native_unloading_terminal() -> RuntimeFact {
        RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
            ModelUnloadingEventKind::UnloadFailed,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_terminal_for(fact: &RuntimeFact) -> SyntheticTerminal {
        match fact {
            RuntimeFact::KvRuntimeState(_) => synthetic_kv_runtime_state_terminal,
            RuntimeFact::ResourceHealth(_) => synthetic_resource_health_terminal,
            RuntimeFact::ModelUnloading(_) => synthetic_native_unloading_terminal,
            _ => synthetic_diagnostic_terminal,
        }
    }

    /// Submits one native-derived fact. `Terminal`-class facts reserve a
    /// fresh, throwaway root and submit through it -- the reservation
    /// settles in the same call (its `Drop` synthesizes nothing further
    /// once a terminal was written) so it never lives past this function.
    /// Every other class goes through `unreserved_ingress`
    /// (`engine/mod.rs`'s own documented boundary: an unreserved
    /// `Terminal`-class submission always reports `TerminalDeliveryFailed`,
    /// which is why that path is reserved for exactly the non-Terminal
    /// classes here), mirroring the existing
    /// `runtime::model_lifecycle::events::emit_available_model_set_changed`
    /// fire-and-forget pattern: a fresh `OperationId` per submission, no
    /// reservation-table pressure, bounded by the reducer's existing
    /// `UNRESERVED_OPERATION_BOUND`. Reservation exhaustion degrades to a
    /// silent no-op (engine health already counts it), matching every
    /// other producer in this codebase -- never blocks or fails primary
    /// native work.
    fn submit_native_family_fact(engine: &Arc<RuntimeEventEngine>, fact: RuntimeFact) {
        if fact.delivery_class() == DeliveryClass::Terminal {
            if let Some(reservation) =
                engine.reserve_root(OperationId::new(), synthetic_terminal_for(&fact))
            {
                let _ = reservation.ingress().try_submit(fact);
            }
        } else {
            let ingress = engine.unreserved_ingress(OperationScope::root_only(OperationId::new()));
            let _ = ingress.try_submit(fact);
        }
    }

    async fn try_load_installed_native_runtime_with<
        NativeRuntimeLoadedFn,
        CacheFn,
        ProfileFn,
        InstallOptionsFn,
        InstallExecutorFn,
        InstallFuture,
        LoadLibrariesFn,
    >(
        native_runtime_loaded: NativeRuntimeLoadedFn,
        cache: CacheFn,
        profile: ProfileFn,
        install_options: InstallOptionsFn,
        install_executor: InstallExecutorFn,
        startup_selection: NativeRuntimeStartupSelection,
        load_libraries: LoadLibrariesFn,
    ) -> Result<Option<LoadedNativeRuntime>>
    where
        NativeRuntimeLoadedFn: Fn() -> bool,
        CacheFn: Fn() -> Result<NativeRuntimeCache>,
        ProfileFn: Fn() -> HostRuntimeProfile,
        InstallOptionsFn: Fn() -> NativeRuntimeInstallOptions,
        InstallExecutorFn: Fn(NativeRuntimeInstallOptions) -> InstallFuture,
        InstallFuture: Future<Output = Result<NativeRuntimeInstallOutcome>>,
        LoadLibrariesFn: Fn(&[PathBuf]) -> Result<()>,
    {
        if native_runtime_loaded() {
            return Ok(None);
        }
        let Some(plan) = resolve_startup_native_runtime_plan_with(
            cache,
            profile,
            install_options,
            install_executor,
            startup_selection,
        )
        .await?
        else {
            return Ok(None);
        };
        load_libraries(&plan.libraries).with_context(|| {
            format!(
                "load native runtime {} from {}",
                plan.native_runtime_id,
                plan.root.display()
            )
        })?;
        Ok(Some(LoadedNativeRuntime {
            native_runtime_id: plan.native_runtime_id,
            libraries: plan.libraries,
        }))
    }

    async fn resolve_startup_native_runtime_plan_with<
        CacheFn,
        ProfileFn,
        InstallOptionsFn,
        InstallExecutorFn,
        InstallFuture,
    >(
        cache: CacheFn,
        profile: ProfileFn,
        install_options: InstallOptionsFn,
        install_executor: InstallExecutorFn,
        startup_selection: NativeRuntimeStartupSelection,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>>
    where
        CacheFn: Fn() -> Result<NativeRuntimeCache>,
        ProfileFn: Fn() -> HostRuntimeProfile,
        InstallOptionsFn: Fn() -> NativeRuntimeInstallOptions,
        InstallExecutorFn: Fn(NativeRuntimeInstallOptions) -> InstallFuture,
        InstallFuture: Future<Output = Result<NativeRuntimeInstallOutcome>>,
    {
        let cache = cache()?;
        let profile = profile();
        let mut options = install_options();
        options.mesh_version = startup_selection.mesh_version.clone();
        options.skippy_abi_version = startup_selection.skippy_abi.clone();
        options.selection = startup_selection.runtime_selection.clone();
        if options.cache_dir.is_none() {
            options.cache_dir = Some(cache.root().to_path_buf());
        }
        let discovered_bundle_dirs =
            crate::system::native_runtime_install::discover_native_runtime_bundle_dirs(
                &options.bundle_dirs,
            )?;
        let discovered_bundle_dirs_empty = discovered_bundle_dirs.is_empty();
        if discovered_bundle_dirs_empty {
            if let Some(plan) = resolve_installed_native_runtime_plan(
                &cache,
                &profile,
                crate::BUILD_VERSION,
                &startup_selection.mesh_version,
                startup_selection.skippy_abi.as_deref(),
                &startup_selection.runtime_selection,
            )? {
                return Ok(Some(plan));
            }
        } else {
            options.bundle_dirs = discovered_bundle_dirs;
        }

        tracing::info!(
            cache_root = %cache.root().display(),
            mesh_version = %options.mesh_version,
            "{}",
            startup_install_message(discovered_bundle_dirs_empty)
        );

        let install_result = install_executor(options.clone()).await;
        match install_result {
            Ok(outcome) => {
                let load_plan = outcome.runtime.load_plan()?;
                Ok(Some(startup_load_plan_from_installed(
                    outcome.runtime.mesh_version.clone(),
                    load_plan,
                    NativeRuntimePlanSource::PostInstall,
                )?))
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    cache_root = %cache.root().display(),
                    mesh_version = %options.mesh_version,
                    manifest_path = ?options.manifest_path,
                    manifest_url = ?options.manifest_url,
                    bundle_dirs = ?options.bundle_dirs,
                    allow_download = options.allow_download,
                    "Failed to install a compatible MeshLLM native runtime during startup; stopping before Skippy FFI load"
                );
                Err(err.context(startup_missing_native_runtime_guidance(&options)))
            }
        }
    }

    fn startup_missing_native_runtime_guidance(options: &NativeRuntimeInstallOptions) -> String {
        let abi = options
            .skippy_abi_version
            .as_deref()
            .unwrap_or("not configured");
        format!(
            "no compatible MeshLLM native runtime is installed or installable for MeshLLM {} / Skippy ABI {abi}; run `mesh-llm runtime install` or inspect available runtimes with `mesh-llm runtime list --available`",
            options.mesh_version
        )
    }

    const fn startup_install_message(discovered_bundle_dirs_empty: bool) -> &'static str {
        if discovered_bundle_dirs_empty {
            "No compatible installed MeshLLM native runtime found; attempting one-shot startup install"
        } else {
            "Discovered native runtime bundles take precedence over installed runtimes; attempting one-shot startup install"
        }
    }

    fn resolve_installed_native_runtime_plan(
        cache: &NativeRuntimeCache,
        profile: &HostRuntimeProfile,
        build_version: &str,
        target_mesh_version: &str,
        target_skippy_abi: Option<&str>,
        selection: &RuntimeSelection,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        let scan = cache.installed_lenient()?;
        for skipped in &scan.skipped {
            tracing::warn!(
                path = %skipped.path.display(),
                reason = %skipped.reason,
                "Skipping unusable native runtime cache entry during startup"
            );
        }
        let installed = scan.runtimes;
        if installed.is_empty() {
            return Ok(None);
        }
        let initial_cache_version =
            startup_native_runtime_cache_version(build_version, target_mesh_version);
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: initial_cache_version.to_string(),
            skippy_abi: target_skippy_abi.unwrap_or_default().to_string(),
            artifacts: installed
                .iter()
                .map(|runtime| runtime.manifest.runtime.clone())
                .collect(),
        };
        let Some(candidate) = mesh_llm_native_runtime::select_native_runtime_from_artifacts(
            &manifest.artifacts,
            profile,
            initial_cache_version,
            target_skippy_abi,
            selection,
        ) else {
            return Ok(None);
        };
        load_plan_from_candidate(cache, &manifest, candidate.artifact)
    }

    fn resolve_local_native_runtime_plan(
        runtimes: &[InstalledNativeRuntime],
        profile: &HostRuntimeProfile,
        build_version: &str,
        target_mesh_version: &str,
        target_skippy_abi: Option<&str>,
        selection: &RuntimeSelection,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        if runtimes.is_empty() {
            return Ok(None);
        }
        let cache_mesh_version =
            startup_native_runtime_cache_version(build_version, target_mesh_version);
        let artifacts = runtimes
            .iter()
            .map(|runtime| runtime.manifest.runtime.clone())
            .collect::<Vec<_>>();
        let Some(candidate) = mesh_llm_native_runtime::select_native_runtime_from_artifacts(
            &artifacts,
            profile,
            cache_mesh_version,
            target_skippy_abi,
            selection,
        ) else {
            return Ok(None);
        };
        let selected_mesh_version = candidate
            .artifact
            .mesh_version_or(cache_mesh_version)
            .to_string();
        let Some(runtime) = runtimes.iter().find(|runtime| {
            runtime.mesh_version == selected_mesh_version
                && runtime.native_runtime_id == candidate.artifact.native_runtime_id()
                && runtime.manifest.runtime.skippy_abi == candidate.artifact.skippy_abi
        }) else {
            return Ok(None);
        };
        Ok(Some(startup_load_plan_from_installed(
            selected_mesh_version,
            runtime.load_plan()?,
            NativeRuntimePlanSource::LocalDiscovery,
        )?))
    }

    fn startup_native_runtime_cache_version<'a>(
        _build_version: &'a str,
        release_version: &'a str,
    ) -> &'a str {
        release_version
    }

    fn load_plan_from_candidate(
        cache: &NativeRuntimeCache,
        manifest: &NativeRuntimeReleaseManifest,
        artifact: NativeRuntimeArtifact,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        let cache_mesh_version = artifact
            .mesh_version_or(manifest.mesh_version.as_str())
            .to_string();
        let Some(installed) =
            cache.find_installed(&cache_mesh_version, artifact.native_runtime_id())?
        else {
            return Ok(None);
        };
        let load_plan = installed.load_plan()?;
        Ok(Some(startup_load_plan_from_installed(
            cache_mesh_version,
            load_plan,
            NativeRuntimePlanSource::CacheHit,
        )?))
    }

    fn startup_load_plan_from_installed(
        cache_mesh_version: String,
        load_plan: NativeRuntimeLoadPlan,
        source: NativeRuntimePlanSource,
    ) -> Result<NativeRuntimeStartupLoadPlan> {
        let selected_library_path = load_plan
            .libraries
            .first()
            .cloned()
            .context("native runtime load plan did not include a library path")?;
        Ok(NativeRuntimeStartupLoadPlan {
            cache_mesh_version,
            native_runtime_id: load_plan.native_runtime_id,
            root: load_plan.root,
            selected_library_path,
            libraries: load_plan.libraries,
            source,
        })
    }

    fn default_native_runtime_cache() -> Result<NativeRuntimeCache> {
        crate::system::native_runtime_install::default_native_runtime_cache()
    }

    fn host_runtime_profile() -> HostRuntimeProfile {
        crate::system::native_runtime_install::host_runtime_profile()
    }

    fn default_install_options() -> NativeRuntimeInstallOptions {
        NativeRuntimeInstallOptions {
            mesh_version: crate::RELEASE_VERSION.to_string(),
            skippy_abi_version: Some(
                crate::system::native_runtime_install::current_skippy_abi_version(),
            ),
            selection: RuntimeSelection::Recommended,
            ..Default::default()
        }
    }

    async fn default_install_executor(
        options: NativeRuntimeInstallOptions,
    ) -> Result<NativeRuntimeInstallOutcome> {
        crate::system::native_runtime_install::install_native_runtime(options).await
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use mesh_llm_native_runtime::{
            NativeRuntimeBackend, NativeRuntimeManifest, NativeRuntimePlatform,
        };
        use std::{
            fs,
            path::Path,
            sync::{Arc, Mutex},
        };

        fn write_runtime(dir: &Path, version: &str, id: &str) {
            write_runtime_with_manifest_mesh_version(dir, Some(version), id);
        }

        fn write_runtime_without_mesh_version(dir: &Path, id: &str) {
            write_runtime_with_manifest_mesh_version(dir, None, id);
        }

        fn write_runtime_with_manifest_mesh_version(dir: &Path, version: Option<&str>, id: &str) {
            let library_rel_path = test_library_rel_path();
            fs::create_dir_all(dir.join(library_rel_path.parent().unwrap())).unwrap();
            fs::write(dir.join(&library_rel_path), b"native runtime").unwrap();
            let manifest = NativeRuntimeManifest {
                runtime: NativeRuntimeArtifact {
                    id: id.to_string(),
                    mesh_version: version.map(ToString::to_string),
                    skippy_abi: "0.1.25".to_string(),
                    platform: NativeRuntimePlatform {
                        os: std::env::consts::OS.to_string(),
                        arch: std::env::consts::ARCH.to_string(),
                        target: None,
                    },
                    backend: NativeRuntimeBackend::cpu(),
                    rank: 0,
                    libraries: vec![library_rel_path.to_string_lossy().to_string()],
                    files: Default::default(),
                    tools: Default::default(),
                    url: None,
                    sha256: None,
                    signature: None,
                },
            };
            manifest.write_to_dir(dir).unwrap();
        }

        fn test_library_rel_path() -> PathBuf {
            let file = if cfg!(target_os = "windows") {
                "meshllm_ffi.dll"
            } else if cfg!(target_os = "macos") {
                "libmeshllm_ffi.dylib"
            } else {
                "libmeshllm_ffi.so"
            };
            PathBuf::from("lib").join(file)
        }

        fn test_install_options() -> NativeRuntimeInstallOptions {
            NativeRuntimeInstallOptions {
                mesh_version: "0.68.0".to_string(),
                allow_download: false,
                ..Default::default()
            }
        }

        #[test]
        fn sha_build_uses_release_cache_identity_for_installed_runtime_lookup() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let sha_build_version = "0.68.0+gAB131C";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                sha_build_version,
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected cached runtime plan");

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.native_runtime_id, runtime_id);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
            assert_eq!(
                plan.selected_library_path,
                runtime_dir.join(test_library_rel_path())
            );
            assert_eq!(
                plan.libraries,
                vec![runtime_dir.join(test_library_rel_path())]
            );
        }

        #[test]
        fn stale_pre_checksum_cache_entry_does_not_block_startup_plan() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.75.0";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            // Simulate a cache entry written by a pre-0.75 loader: the
            // manifest has no per-file checksums (issue #1162).
            let legacy_dir = cache.runtime_dir("0.74.0", runtime_id);
            let library_rel_path = test_library_rel_path();
            fs::create_dir_all(legacy_dir.join(library_rel_path.parent().unwrap())).unwrap();
            fs::write(legacy_dir.join(&library_rel_path), b"legacy runtime").unwrap();
            fs::write(
                legacy_dir.join("manifest.json"),
                format!(
                    r#"{{
  "runtime": {{
    "id": "{runtime_id}",
    "mesh_version": "0.74.0",
    "skippy_abi": "0.1.25",
    "platform": {{"os": "{os}", "arch": "{arch}"}},
    "backend": {{"kind": "cpu"}},
    "libraries": ["{library}"]
  }}
}}"#,
                    os = std::env::consts::OS,
                    arch = std::env::consts::ARCH,
                    library = library_rel_path.to_string_lossy().replace('\\', "/"),
                ),
            )
            .unwrap();

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                release_version,
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("a stale pre-checksum cache entry must not block the valid runtime");

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.native_runtime_id, runtime_id);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
        }

        #[test]
        fn explicit_runtime_version_can_select_other_mesh_version() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let artifact_mesh_version = "0.69.0";
            let runtime_dir = cache.runtime_dir(artifact_mesh_version, runtime_id);
            write_runtime(&runtime_dir, artifact_mesh_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                "0.68.0+gAB131C.dirty",
                artifact_mesh_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected cached runtime plan");

            assert_eq!(plan.cache_mesh_version, artifact_mesh_version);
            assert_eq!(plan.root, runtime_dir);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
        }

        #[test]
        fn default_startup_plan_rejects_other_mesh_version() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let artifact_mesh_version = "0.69.0";
            let runtime_dir = cache.runtime_dir(artifact_mesh_version, runtime_id);
            write_runtime(&runtime_dir, artifact_mesh_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                "0.68.0+gAB131C.dirty",
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap();

            assert!(plan.is_none());
        }

        #[test]
        fn startup_plan_rejects_installed_runtime_without_mesh_version() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = cache.runtime_dir("unknown", runtime_id);
            write_runtime_without_mesh_version(&runtime_dir, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                "0.68.0+gAB131C.dirty",
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap();

            assert!(plan.is_none());
        }

        #[test]
        fn startup_plan_can_represent_post_install_source_without_loading() {
            let temp = tempfile::tempdir().unwrap();
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = temp.path().join(runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);
            let load_plan = NativeRuntimeLoadPlan {
                mesh_version: release_version.to_string(),
                native_runtime_id: runtime_id.to_string(),
                root: runtime_dir.clone(),
                libraries: vec![runtime_dir.join(test_library_rel_path())],
            };

            let plan = startup_load_plan_from_installed(
                release_version.to_string(),
                load_plan,
                NativeRuntimePlanSource::PostInstall,
            )
            .unwrap();

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.root, runtime_dir);
            assert_eq!(plan.source, NativeRuntimePlanSource::PostInstall);
        }

        #[test]
        fn local_discovery_prefers_bundle_over_identical_cached_runtime() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            write_runtime(
                &cache.runtime_dir(release_version, runtime_id),
                release_version,
                runtime_id,
            );
            let bundled_runtime_dir = temp
                .path()
                .join("product")
                .join("native-runtimes")
                .join(runtime_id);
            write_runtime(&bundled_runtime_dir, release_version, runtime_id);

            let local_runtimes =
                crate::system::native_runtime_install::discover_local_native_runtimes(
                    std::slice::from_ref(&bundled_runtime_dir),
                    &cache,
                )
                .unwrap();
            let plan = resolve_local_native_runtime_plan(
                &local_runtimes,
                &HostRuntimeProfile::current_without_gpu_probe(),
                release_version,
                release_version,
                Some("0.1.25"),
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected bundled runtime plan");

            assert_eq!(plan.native_runtime_id, runtime_id);
            assert_eq!(plan.source, NativeRuntimePlanSource::LocalDiscovery);
            assert_eq!(plan.root, bundled_runtime_dir.canonicalize().unwrap());
        }

        #[test]
        fn disappeared_cache_entry_is_treated_as_cache_miss() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let manifest = NativeRuntimeReleaseManifest {
                mesh_version: release_version.to_string(),
                skippy_abi: "0.1.25".to_string(),
                artifacts: Vec::new(),
            };
            let artifact = NativeRuntimeArtifact {
                id: runtime_id.to_string(),
                mesh_version: Some(release_version.to_string()),
                skippy_abi: "0.1.25".to_string(),
                platform: NativeRuntimePlatform {
                    os: std::env::consts::OS.to_string(),
                    arch: std::env::consts::ARCH.to_string(),
                    target: None,
                },
                backend: NativeRuntimeBackend::cpu(),
                rank: 0,
                libraries: vec![test_library_rel_path().to_string_lossy().to_string()],
                files: Default::default(),
                tools: Default::default(),
                url: None,
                sha256: None,
                signature: None,
            };

            let plan = load_plan_from_candidate(&cache, &manifest, artifact).unwrap();

            assert!(plan.is_none());
        }

        #[tokio::test]
        async fn cache_hit_skips_install_and_loads_cached_runtime_once() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(0_usize));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |_| {
                        let install_calls = Arc::clone(&install_calls);
                        async move {
                            *install_calls.lock().unwrap() += 1;
                            anyhow::bail!("install should not run on cache hit")
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    release_version.to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .await
            .unwrap()
            .expect("expected cached runtime to load");

            assert_eq!(*install_calls.lock().unwrap(), 0);
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![runtime_dir.join(test_library_rel_path())]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[tokio::test]
        async fn bundled_runtime_precedes_identical_cache_entry_at_startup() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let cached_runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&cached_runtime_dir, release_version, runtime_id);
            let product_root = temp.path().join("mesh-bundle");
            let bundled_runtime_dir = product_root.join("native-runtimes").join(runtime_id);
            write_runtime(&bundled_runtime_dir, release_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(0_usize));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));
            let options_product_root = product_root.clone();
            let options_cache_root = cache.root().to_path_buf();

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                move || NativeRuntimeInstallOptions {
                    mesh_version: release_version.to_string(),
                    skippy_abi_version: Some("0.1.25".to_string()),
                    bundle_dirs: vec![options_product_root.clone()],
                    cache_dir: Some(options_cache_root.clone()),
                    allow_download: false,
                    ..Default::default()
                },
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |options| {
                        let install_calls = Arc::clone(&install_calls);
                        async move {
                            *install_calls.lock().unwrap() += 1;
                            crate::system::native_runtime_install::install_native_runtime(options)
                                .await
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    release_version.to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .await
            .unwrap()
            .expect("expected bundled runtime to load");

            assert_eq!(*install_calls.lock().unwrap(), 1);
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![
                    bundled_runtime_dir
                        .canonicalize()
                        .unwrap()
                        .join(test_library_rel_path())
                ]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[tokio::test]
        async fn cache_miss_installs_once_and_loads_post_install_runtime() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let bundle_dir = temp.path().join("bundle");
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let manifest_mesh_version = "0.68.0";
            write_runtime(&bundle_dir, manifest_mesh_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(Vec::<NativeRuntimeInstallOptions>::new()));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    let bundle_dir = bundle_dir.clone();
                    let cache = cache.clone();
                    move |mut options| {
                        let install_calls = Arc::clone(&install_calls);
                        let bundle_dir = bundle_dir.clone();
                        let cache = cache.clone();
                        async move {
                            install_calls.lock().unwrap().push(options.clone());
                            let source = options.bundle_dirs.pop().unwrap_or(bundle_dir.clone());
                            let runtime = cache.install_from_dir(&source)?;
                            Ok(NativeRuntimeInstallOutcome {
                                status: crate::system::native_runtime_install::NativeRuntimeInstallStatus::Installed,
                                runtime,
                                resolution: mesh_llm_native_runtime::NativeRuntimeResolution {
                                    source: mesh_llm_native_runtime::NativeRuntimeSource::Bundle {
                                        path: source,
                                    },
                                    selected: NativeRuntimeManifest::read_from_dir(&bundle_dir)?
                                        .runtime,
                                    evaluated: Vec::new(),
                                },
                            })
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    "0.68.0".to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .await
            .unwrap()
            .expect("expected installed runtime to load");

            let recorded_options = install_calls.lock().unwrap();
            assert_eq!(recorded_options.len(), 1);
            assert_eq!(recorded_options[0].mesh_version, "0.68.0");
            assert_eq!(
                recorded_options[0].skippy_abi_version.as_deref(),
                Some("0.1.25")
            );
            assert_eq!(recorded_options[0].cache_dir.as_deref(), Some(cache.root()));
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![
                    cache
                        .runtime_dir(manifest_mesh_version, runtime_id)
                        .join(test_library_rel_path())
                ]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[tokio::test]
        async fn cache_miss_install_failure_stops_startup_before_ffi_load() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let install_calls = Arc::new(Mutex::new(Vec::<NativeRuntimeInstallOptions>::new()));
            let load_calls = Arc::new(Mutex::new(0_usize));

            let error = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |options| {
                        let install_calls = Arc::clone(&install_calls);
                        async move {
                            install_calls.lock().unwrap().push(options);
                            anyhow::bail!(
                                "no compatible native runtime found for Skippy ABI 0.1.25 on test/test"
                            )
                        }
                    }
                },
                NativeRuntimeStartupSelection::explicit(
                    "0.68.0".to_string(),
                    Some("0.1.25".to_string()),
                    RuntimeSelection::Recommended,
                ),
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |_| {
                        *load_calls.lock().unwrap() += 1;
                        Ok(())
                    }
                },
            )
            .await
            .expect_err("missing native runtime should stop startup");

            let message = error.to_string();
            assert!(message.contains("no compatible MeshLLM native runtime"));
            assert!(message.contains("mesh-llm runtime install"));
            assert!(message.contains("mesh-llm runtime list --available"));
            assert_eq!(install_calls.lock().unwrap().len(), 1);
            assert_eq!(*load_calls.lock().unwrap(), 0);
        }

        #[test]
        fn startup_install_message_distinguishes_empty_and_nonempty_discovery() {
            assert_eq!(
                startup_install_message(true),
                "No compatible installed MeshLLM native runtime found; attempting one-shot startup install"
            );
            assert_eq!(
                startup_install_message(false),
                "Discovered native runtime bundles take precedence over installed runtimes; attempting one-shot startup install"
            );
        }
    }

    /// Task 10 (D7, `.omo/plans/event-system-fixes.md`): the runtime-scoped
    /// native event reporter's decode-and-map sink. Named to satisfy the
    /// plan's own focused command
    /// (`cargo test -p mesh-llm-host-runtime system::native_runtime_events`);
    /// verified to actually match that filter as part of this task's QA
    /// rather than assumed.
    #[cfg(test)]
    mod native_runtime_events_tests {
        use super::*;
        use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
        use mesh_llm_runtime_event_contracts::ReasonCode;
        use skippy_runtime::{
            RuntimeEventCategory, RuntimeEventEmitterKind, RuntimeEventFailureCode,
            RuntimeEventProgressUnit, Status,
        };
        use std::thread;

        fn install_test_engine() -> Arc<RuntimeEventEngine> {
            clear_runtime_event_engine();
            let engine = RuntimeEventEngine::new();
            install_runtime_event_engine(engine.clone());
            engine
        }

        /// A well-formed decoded event for one native kind. `RuntimeEvent`
        /// is the SAFE, already-decoded type this sink actually receives
        /// (its `pub` fields are the crate's own contract; the raw FFI
        /// struct's `from_raw_ptr` decoder is `pub(crate)` to
        /// `skippy-runtime` and this crate never needs to duplicate it).
        fn event(kind: RuntimeEventKind, sequence: u64) -> RuntimeEvent {
            RuntimeEvent {
                abi_version: 1,
                category: RuntimeEventCategory::Unknown(0),
                kind,
                emitter: RuntimeEventEmitterKind::WorkerThread,
                sequence,
                timestamp_mono_ns: sequence,
                model_id: 7,
                stage_id: 0,
                session_id: 3,
                progress_current: 0,
                progress_total: 0,
                progress_unit: RuntimeEventProgressUnit::None,
                failure_code: RuntimeEventFailureCode::None,
                status: Status::Ok,
                detail_bytes: Vec::new(),
                numeric_summary_0: Some(sequence),
                numeric_summary_1: None,
                numeric_summary_2: None,
                numeric_summary_3: None,
            }
        }

        /// One row per `native_family_mappings` entry, the same 29 pairs
        /// the inventory contract test cross-checks, plus whether the
        /// target inventory id is Terminal-class.
        fn mapping_cases() -> Vec<(RuntimeEventKind, &'static str, bool)> {
            vec![
                (
                    RuntimeEventKind::ModelLoadPhaseChanged,
                    "model_load_phase_changed",
                    false,
                ),
                (
                    RuntimeEventKind::ModelLoadMemoryAllocated,
                    "model_memory_allocation_summary",
                    false,
                ),
                (
                    RuntimeEventKind::ModelLoadTensorsOffloaded,
                    "model_load_phase_changed",
                    false,
                ),
                (
                    RuntimeEventKind::ModelLoadTokenizerReady,
                    "model_load_phase_changed",
                    false,
                ),
                (
                    RuntimeEventKind::ModelLoadAuxComponentReady,
                    "model_load_phase_changed",
                    false,
                ),
                (
                    RuntimeEventKind::KvInitialized,
                    "kv_cache_initialization_completed",
                    false,
                ),
                (
                    RuntimeEventKind::KvPressureCrossed,
                    "cache_pressure_crossed",
                    false,
                ),
                (
                    RuntimeEventKind::KvPressureCleared,
                    "cache_pressure_cleared",
                    false,
                ),
                (
                    RuntimeEventKind::KvContextApproachingCapacity,
                    "context_capacity_approaching_limit",
                    false,
                ),
                (
                    RuntimeEventKind::KvContextCapacityExhausted,
                    "context_exhausted",
                    true,
                ),
                (
                    RuntimeEventKind::DeviceBackendInitialized,
                    "backend_initialization_completed",
                    false,
                ),
                (RuntimeEventKind::DeviceReady, "device_ready", false),
                (RuntimeEventKind::DeviceDegraded, "device_degraded", false),
                (
                    RuntimeEventKind::DeviceUnavailable,
                    "device_unavailable",
                    false,
                ),
                (RuntimeEventKind::DeviceRecovered, "device_recovered", false),
                (RuntimeEventKind::DeviceLost, "device_lost", false),
                (
                    RuntimeEventKind::DeviceResourceAllocated,
                    "resource_allocation_completed",
                    false,
                ),
                (
                    RuntimeEventKind::DeviceOutOfMemory,
                    "out_of_memory_condition",
                    true,
                ),
                (
                    RuntimeEventKind::DeviceFallbackActivated,
                    "backend_fallback_activated",
                    false,
                ),
                (
                    RuntimeEventKind::DiagnosticWarningRaised,
                    "warning_raised",
                    false,
                ),
                (
                    RuntimeEventKind::DiagnosticWarningCleared,
                    "warning_cleared",
                    false,
                ),
                (
                    RuntimeEventKind::DiagnosticRecoverableFailure,
                    "recoverable_native_failure",
                    false,
                ),
                (
                    RuntimeEventKind::DiagnosticFatalFailure,
                    "fatal_native_failure",
                    true,
                ),
                (
                    RuntimeEventKind::DiagnosticInvariantViolation,
                    "invariant_protocol_violation",
                    false,
                ),
                (RuntimeEventKind::UnloadStarted, "unload_started", false),
                (RuntimeEventKind::UnloadCompleted, "unload_completed", true),
                (RuntimeEventKind::UnloadFailed, "unload_failed", true),
                (RuntimeEventKind::UnloadForced, "forced_unload", false),
                (
                    RuntimeEventKind::UnloadSessionDraining,
                    "session_draining_started",
                    false,
                ),
            ]
        }

        #[test]
        fn every_mapped_kind_produces_the_expected_event_id_and_delivery_class() {
            for (kind, expected_id, expected_terminal) in mapping_cases() {
                let fact = native_family_fact(&event(kind, 1))
                    .unwrap_or_else(|| panic!("{kind:?} should map to a fact"));
                assert_eq!(fact.kind_id(), expected_id, "kind {kind:?}");
                assert_eq!(
                    fact.delivery_class() == DeliveryClass::Terminal,
                    expected_terminal,
                    "kind {kind:?} delivery class"
                );
            }
        }

        #[test]
        fn model_open_kinds_and_unknown_kinds_are_not_this_sinks_family() {
            for kind in [
                RuntimeEventKind::ModelOpenStarted,
                RuntimeEventKind::ModelOpenProgress,
                RuntimeEventKind::BackendDeviceSelected,
                RuntimeEventKind::ModelOpenFinished,
                RuntimeEventKind::ModelOpenFailedHandled,
                RuntimeEventKind::Unknown(9999),
            ] {
                assert!(
                    native_family_fact(&event(kind, 1)).is_none(),
                    "kind {kind:?}"
                );
            }
        }

        #[test]
        fn native_correlation_fields_land_on_the_fact_not_a_side_channel() {
            let fact = native_family_fact(&event(RuntimeEventKind::DeviceReady, 42))
                .expect("device_ready maps");
            let keys = fact
                .data()
                .numeric_summaries
                .as_slice()
                .iter()
                .map(|summary| summary.key.as_str())
                .collect::<Vec<_>>();
            assert!(keys.contains(&"native_sequence"));
            assert!(keys.contains(&"native_model_id"));
            assert!(keys.contains(&"native_session_id"));
        }

        #[test]
        #[serial_test::serial(runtime_event_engine_state)]
        fn non_terminal_facts_submit_unreserved_and_reach_the_state_lane() {
            let engine = install_test_engine();
            let fact = native_family_fact(&event(RuntimeEventKind::DeviceReady, 1))
                .expect("device_ready maps");
            submit_native_family_fact(&engine, fact);
            assert_eq!(
                engine.occupied_count(),
                0,
                "a StateTransition-class native fact must never consume a reservation slot"
            );
            assert!(engine.state_lane_kinds().contains(&"device_ready"));
            clear_runtime_event_engine();
        }

        #[test]
        #[serial_test::serial(runtime_event_engine_state)]
        fn terminal_facts_reserve_submit_and_settle_in_one_call() {
            let engine = install_test_engine();
            let fact = native_family_fact(&event(RuntimeEventKind::UnloadCompleted, 1))
                .expect("unload_completed maps");
            submit_native_family_fact(&engine, fact);
            engine.drain();
            assert_eq!(
                engine.occupied_count(),
                0,
                "the one-shot reservation must settle, not linger"
            );
            let delivered = engine.replay().snapshot().into_iter().any(|frame| {
                matches!(
                    frame.fact.as_ref(),
                    RuntimeFact::ModelUnloading(fact)
                        if *fact.kind() == ModelUnloadingEventKind::UnloadCompleted
                            && fact.data().outcome == Some(Outcome::Success)
                )
            });
            assert!(delivered, "unload_completed must actually reach replay");
            clear_runtime_event_engine();
        }

        #[test]
        #[serial_test::serial(runtime_event_engine_state)]
        fn a_kv_context_exhausted_terminal_carries_the_context_exhausted_reason() {
            let engine = install_test_engine();
            let fact = native_family_fact(&event(RuntimeEventKind::KvContextCapacityExhausted, 1))
                .expect("context_exhausted maps");
            submit_native_family_fact(&engine, fact);
            engine.drain();
            let reason = engine.replay().snapshot().into_iter().find_map(|frame| {
                let RuntimeFact::KvRuntimeState(fact) = frame.fact.as_ref() else {
                    return None;
                };
                (*fact.kind() == KvRuntimeStateEventKind::ContextExhausted)
                    .then(|| fact.data().reason.clone())
            });
            assert_eq!(reason, Some(Some(ReasonCode::ContextExhausted)));
            clear_runtime_event_engine();
        }

        /// Acceptance: concurrent callbacks from two threads submit in
        /// order. Each thread submits a run of Terminal-class native facts
        /// with strictly increasing `native_sequence` values in its own
        /// disjoint range; the replay stream (itself in ingress-sequence,
        /// i.e. real submission, order) must show each thread's own
        /// sequence values still strictly increasing -- proof that this
        /// sink never reorders or drops within one thread's callback
        /// stream under concurrent native worker-thread traffic.
        #[test]
        #[serial_test::serial(runtime_event_engine_state)]
        fn concurrent_two_thread_callbacks_submit_every_fact_in_order() {
            let engine = install_test_engine();
            const PER_THREAD: u64 = 200;
            let engine_a = engine.clone();
            let engine_b = engine.clone();
            let thread_a = thread::spawn(move || {
                for sequence in 0..PER_THREAD {
                    let fact = native_family_fact(&event(
                        RuntimeEventKind::DiagnosticFatalFailure,
                        sequence,
                    ))
                    .expect("fatal_native_failure maps");
                    submit_native_family_fact(&engine_a, fact);
                }
            });
            let thread_b = thread::spawn(move || {
                for sequence in 0..PER_THREAD {
                    let fact = native_family_fact(&event(
                        RuntimeEventKind::DiagnosticFatalFailure,
                        sequence + 1_000_000,
                    ))
                    .expect("fatal_native_failure maps");
                    submit_native_family_fact(&engine_b, fact);
                }
            });
            thread_a.join().expect("thread a");
            thread_b.join().expect("thread b");
            engine.drain();

            let sequences_in_replay_order = engine
                .replay()
                .snapshot()
                .into_iter()
                .filter_map(|frame| {
                    let RuntimeFact::Diagnostic(fact) = frame.fact.as_ref() else {
                        return None;
                    };
                    fact.data()
                        .numeric_summaries
                        .as_slice()
                        .iter()
                        .find(|summary| summary.key.as_str() == "native_sequence")
                        .and_then(|summary| match summary.value {
                            NumericValue::Unsigned(value) => Some(value),
                            _ => None,
                        })
                })
                .collect::<Vec<_>>();

            let thread_a_sequences = sequences_in_replay_order
                .iter()
                .copied()
                .filter(|&value| value < 1_000_000)
                .collect::<Vec<_>>();
            let thread_b_sequences = sequences_in_replay_order
                .iter()
                .copied()
                .filter(|&value| value >= 1_000_000)
                .collect::<Vec<_>>();
            assert_eq!(thread_a_sequences.len(), PER_THREAD as usize);
            assert_eq!(thread_b_sequences.len(), PER_THREAD as usize);
            assert!(
                thread_a_sequences.windows(2).all(|pair| pair[0] < pair[1]),
                "thread A's own submissions must stay in order: {thread_a_sequences:?}"
            );
            assert!(
                thread_b_sequences.windows(2).all(|pair| pair[0] < pair[1]),
                "thread B's own submissions must stay in order: {thread_b_sequences:?}"
            );
            clear_runtime_event_engine();
        }

        /// Acceptance: a callback after clear is impossible. This sink
        /// installs through the unmodified `skippy_runtime::install_
        /// runtime_event_reporter`/`clear_runtime_event_reporter` pair,
        /// whose native quiescence contract
        /// (`third_party/llama.cpp/patches/0043-*`'s `test_clear_is_
        /// quiescent`, and this crate's own
        /// `runtime_event_reporter::tests::clear_is_a_safe_no_op_when_
        /// nothing_was_installed`) already guarantees no callback fires
        /// after `clear_runtime_event_reporter()` returns; this sink adds
        /// no deferred work of its own (no spawned thread, no queue) that
        /// could outlive that guarantee, so the property carries over
        /// unmodified. (`clear_runtime_event_reporter()` itself is not
        /// re-exercised here: with `dynamic-native-runtime` active and no
        /// library loaded, its symbol lookup panics by design --
        /// `skippy_ffi::dynamic::symbols()` -- so only `skippy-runtime`'s
        /// own suite, which runs without that feature, calls it directly.)
        /// This test only proves this sink's install call, with no
        /// confirmed native family present, degrades without panicking --
        /// the same contract `install_returns_false_without_a_confirmed_
        /// family` pins in `runtime_event_reporter.rs`.
        #[test]
        fn install_never_panics_without_a_confirmed_native_family() {
            install_runtime_scoped_event_reporter();
        }

        /// Acceptance: "no I/O (no D7-era native-side log-note symbol
        /// reachable from it)". A lexical, mechanical proof: this sink's
        /// own source file no longer CALLS that symbol anywhere (the
        /// pre-task-10 log-note closure this file used to install is
        /// gone). Checks the call form specifically (name immediately
        /// followed by `(`), not mere mentions in prose -- this test's own
        /// doc comments legitimately name the removed symbol above.
        #[test]
        fn the_removed_native_log_note_call_is_not_reachable_from_this_file() {
            let source = include_str!("native_runtime.rs");
            let removed_call = ["write_native", "_log_note("].concat();
            assert!(
                !source.contains(removed_call.as_str()),
                "system::native_runtime must not call the D7-era log-note symbol"
            );
        }
    }
}

#[cfg(feature = "dynamic-native-runtime")]
pub(crate) use dynamic::*;

#[cfg(not(feature = "dynamic-native-runtime"))]
pub(crate) fn try_load_installed_native_runtime() -> anyhow::Result<Option<()>> {
    Ok(None)
}
