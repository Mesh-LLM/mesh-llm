//! Shared fake native plugin used by host and dispatch unit tests.

use std::{
    collections::HashMap,
    ffi::{c_char, c_void},
    ptr::NonNull,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use mesh_native_serving_plugin_api as abi;
use skippy_server::frontend::LinearProposalQuery;

use crate::{ActivePlugin, LoadedDefinition, MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS};

pub(crate) static FAKE_NAME: &[u8] = b"test-serving-plugin";

/// Deterministic callback gate for ordering tests. The callback announces
/// entry, then waits until the test explicitly releases it.
pub(crate) struct CallbackGate {
    state: Mutex<CallbackGateState>,
    signal: Condvar,
}

#[derive(Default)]
struct CallbackGateState {
    entered: bool,
    released: bool,
}

impl CallbackGate {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CallbackGateState::default()),
            signal: Condvar::new(),
        })
    }

    fn wait(&self) {
        let mut state = self.state.lock().unwrap();
        state.entered = true;
        self.signal.notify_all();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !state.released {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "timed out waiting for gated callback");
            let (next_state, timeout) = self.signal.wait_timeout(state, remaining).unwrap();
            state = next_state;
            assert!(
                !timeout.timed_out() || state.released,
                "timed out waiting for gated callback"
            );
        }
    }

    pub(crate) fn wait_until_entered(&self) {
        let mut state = self.state.lock().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !state.entered {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "timed out waiting for gated callback");
            let (next_state, timeout) = self.signal.wait_timeout(state, remaining).unwrap();
            state = next_state;
            assert!(
                !timeout.timed_out() || state.entered,
                "timed out waiting for gated callback"
            );
        }
    }

    pub(crate) fn release(&self) {
        let mut state = self.state.lock().unwrap();
        state.released = true;
        self.signal.notify_all();
    }

    pub(crate) fn release_on_drop(self: &Arc<Self>) -> CallbackGateRelease {
        CallbackGateRelease(Arc::clone(self))
    }
}

pub(crate) struct CallbackGateRelease(Arc<CallbackGate>);

impl Drop for CallbackGateRelease {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// Per-instance observations, so tests running in parallel cannot see each
/// other's callback counts.
pub(crate) struct FakeObservations {
    pub(crate) events: Arc<Mutex<Vec<&'static str>>>,
    pub(crate) discard_reasons: Arc<Mutex<Vec<abi::ProposalDiscardReason>>>,
    pub(crate) cancel_count: Arc<AtomicUsize>,
    pub(crate) shutdown_count: Arc<AtomicUsize>,
}

pub(crate) struct FakeState {
    start_delay: Duration,
    poll_delay: Duration,
    poll_returns_candidate: bool,
    poll_fails: bool,
    commit_delay: Duration,
    commit_gate: Option<Arc<CallbackGate>>,
    report_delay: Duration,
    report_gate: Option<Arc<CallbackGate>>,
    proposal_gate: Option<Arc<CallbackGate>>,
    begin_fails: bool,
    events: Arc<Mutex<Vec<&'static str>>>,
    discard_reasons: Arc<Mutex<Vec<abi::ProposalDiscardReason>>>,
    abort_count: Arc<AtomicUsize>,
    cancel_count: Arc<AtomicUsize>,
    shutdown_count: Arc<AtomicUsize>,
}

unsafe extern "C" fn fake_activate(
    _context: *const abi::ActivationContext,
    _activation: *mut abi::PluginActivation,
) -> abi::PluginStatus {
    abi::PluginStatus::INTERNAL_ERROR
}

unsafe extern "C" fn fake_shutdown(instance: abi::PluginInstance) -> abi::PluginStatus {
    if !instance.is_null() {
        let state = unsafe { Box::from_raw(instance.cast::<FakeState>()) };
        state.shutdown_count.fetch_add(1, Ordering::SeqCst);
    }
    abi::PluginStatus::OK
}

unsafe extern "C" fn fake_begin(
    instance: abi::PluginInstance,
    _event: *const abi::GenerationStart,
) -> abi::PluginStatus {
    let state = unsafe { &*instance.cast::<FakeState>() };
    state.events.lock().unwrap().push("begin");
    if state.begin_fails {
        abi::PluginStatus::INTERNAL_ERROR
    } else {
        abi::PluginStatus::OK
    }
}

unsafe extern "C" fn fake_commit(
    instance: abi::PluginInstance,
    _event: *const abi::GenerationCommit,
) -> abi::PluginStatus {
    let state = unsafe { &*instance.cast::<FakeState>() };
    state.events.lock().unwrap().push("commit");
    if let Some(gate) = &state.commit_gate {
        gate.wait();
    }
    thread::sleep(state.commit_delay);
    abi::PluginStatus::OK
}

unsafe extern "C" fn fake_abort(
    instance: abi::PluginInstance,
    _event: *const abi::GenerationAbort,
) -> abi::PluginStatus {
    let state = unsafe { &*instance.cast::<FakeState>() };
    state.abort_count.fetch_add(1, Ordering::SeqCst);
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
    if let Some(gate) = &state.proposal_gate {
        gate.wait();
    }
    thread::sleep(state.start_delay);
    unsafe { *operation = 1 };
    abi::PluginStatus::OK
}

unsafe extern "C" fn fake_poll_proposal(
    instance: abi::PluginInstance,
    _operation: abi::ProposalOperation,
    output: *mut abi::ProposalOutput,
) -> abi::ProposalPollStatus {
    let state = unsafe { &*instance.cast::<FakeState>() };
    thread::sleep(state.poll_delay);
    if state.poll_fails {
        return abi::ProposalPollStatus::FAILED;
    }
    if !state.poll_returns_candidate {
        return abi::ProposalPollStatus::ABSTAIN;
    }
    let output = unsafe { &mut *output };
    unsafe {
        *output.decision_id = 7;
        *output.token_ids = 42;
    }
    output.decision_id_length = 1;
    output.token_length = 1;
    abi::ProposalPollStatus::READY
}

unsafe extern "C" fn fake_cancel_proposal(
    instance: abi::PluginInstance,
    _operation: abi::ProposalOperation,
) {
    if !instance.is_null() {
        let state = unsafe { &*instance.cast::<FakeState>() };
        state.cancel_count.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn fake_report_proposal(
    instance: abi::PluginInstance,
    _event: *const abi::ProposalOutcome,
) -> abi::PluginStatus {
    let state = unsafe { &*instance.cast::<FakeState>() };
    state.events.lock().unwrap().push("report");
    if let Some(gate) = &state.report_gate {
        gate.wait();
        state.events.lock().unwrap().push("report_done");
    }
    thread::sleep(state.report_delay);
    state.events.lock().unwrap().push("report_complete");
    abi::PluginStatus::OK
}

unsafe extern "C" fn fake_discard_proposal(
    instance: abi::PluginInstance,
    event: *const abi::ProposalDiscard,
) -> abi::PluginStatus {
    let state = unsafe { &*instance.cast::<FakeState>() };
    state.events.lock().unwrap().push("discard");
    state
        .discard_reasons
        .lock()
        .unwrap()
        .push(unsafe { (*event).reason });
    thread::sleep(state.report_delay);
    abi::PluginStatus::OK
}

unsafe extern "C" fn fake_last_error(
    _instance: abi::PluginInstance,
    _output: *mut c_char,
    _capacity: usize,
) -> usize {
    0
}

pub(crate) fn fake_table() -> abi::NativeServingPluginV2 {
    abi::NativeServingPluginV2 {
        abi_version: abi::NATIVE_SERVING_PLUGIN_ABI_V2,
        struct_size: size_of::<abi::NativeServingPluginV2>(),
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

pub(crate) fn fake_active(start_delay: Duration) -> ActivePlugin {
    fake_active_with_events(start_delay).0
}

pub(crate) fn fake_active_with_events(
    start_delay: Duration,
) -> (ActivePlugin, Arc<Mutex<Vec<&'static str>>>) {
    let (active, events, _) = fake_active_with_observations(start_delay);
    (active, events)
}

pub(crate) fn fake_active_with_observations(
    start_delay: Duration,
) -> (
    ActivePlugin,
    Arc<Mutex<Vec<&'static str>>>,
    Arc<AtomicUsize>,
) {
    fake_active_with_options(start_delay, false)
}

pub(crate) fn fake_active_with_options(
    start_delay: Duration,
    begin_fails: bool,
) -> (
    ActivePlugin,
    Arc<Mutex<Vec<&'static str>>>,
    Arc<AtomicUsize>,
) {
    fake_active_with_timing(start_delay, Duration::ZERO, Duration::ZERO, begin_fails)
}

pub(crate) fn fake_active_with_timing(
    start_delay: Duration,
    commit_delay: Duration,
    report_delay: Duration,
    begin_fails: bool,
) -> (
    ActivePlugin,
    Arc<Mutex<Vec<&'static str>>>,
    Arc<AtomicUsize>,
) {
    fake_active_with_timing_and_gates(
        start_delay,
        commit_delay,
        report_delay,
        begin_fails,
        None,
        None,
        None,
    )
}

pub(crate) fn fake_active_with_timing_and_gates(
    start_delay: Duration,
    commit_delay: Duration,
    report_delay: Duration,
    begin_fails: bool,
    report_gate: Option<Arc<CallbackGate>>,
    proposal_gate: Option<Arc<CallbackGate>>,
    commit_gate: Option<Arc<CallbackGate>>,
) -> (
    ActivePlugin,
    Arc<Mutex<Vec<&'static str>>>,
    Arc<AtomicUsize>,
) {
    let table = Box::leak(Box::new(fake_table()));
    let definition = Arc::new(LoadedDefinition {
        _library: None,
        api: NonNull::from(table),
        name: "test-serving-plugin".to_string(),
    });
    let events = Arc::new(Mutex::new(Vec::new()));
    let discard_reasons = Arc::new(Mutex::new(Vec::new()));
    let abort_count = Arc::new(AtomicUsize::new(0));
    let cancel_count = Arc::new(AtomicUsize::new(0));
    let shutdown_count = Arc::new(AtomicUsize::new(0));
    let state = Box::new(FakeState {
        start_delay,
        poll_delay: Duration::ZERO,
        poll_returns_candidate: false,
        poll_fails: false,
        commit_delay,
        commit_gate,
        report_delay,
        report_gate,
        proposal_gate,
        begin_fails,
        events: Arc::clone(&events),
        discard_reasons,
        abort_count: Arc::clone(&abort_count),
        cancel_count: Arc::clone(&cancel_count),
        shutdown_count: Arc::clone(&shutdown_count),
    });
    (
        ActivePlugin {
            definition,
            instance: NonNull::new(Box::into_raw(state).cast::<c_void>()),
            _tokenizer_capability: None,
            proposal_token_buffer: Mutex::new(vec![0; MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS]),
            committed_generated_tokens: Mutex::new(HashMap::new()),
        },
        events,
        abort_count,
    )
}

pub(crate) fn fake_active_with_candidate_and_proposal_gate(
    proposal_gate: Arc<CallbackGate>,
) -> ActivePlugin {
    let (active, _, _) = fake_active_with_timing_and_gates(
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        false,
        None,
        Some(proposal_gate),
        None,
    );
    let instance = active.instance.unwrap().as_ptr().cast::<FakeState>();
    unsafe {
        (*instance).poll_returns_candidate = true;
    }
    active
}

pub(crate) fn fake_active_with_candidate_and_report_and_commit_gate(
    report_gate: Arc<CallbackGate>,
    commit_gate: Arc<CallbackGate>,
) -> (ActivePlugin, Arc<Mutex<Vec<&'static str>>>) {
    let (active, events, _) = fake_active_with_timing_and_gates(
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        false,
        Some(report_gate),
        None,
        Some(commit_gate),
    );
    let instance = active.instance.unwrap().as_ptr().cast::<FakeState>();
    unsafe {
        (*instance).poll_returns_candidate = true;
    }
    (active, events)
}

pub(crate) fn fake_active_with_late_candidate(poll_delay: Duration) -> ActivePlugin {
    let (active, _, _) =
        fake_active_with_timing(Duration::ZERO, Duration::ZERO, Duration::ZERO, false);
    let instance = active.instance.unwrap().as_ptr().cast::<FakeState>();
    unsafe {
        (*instance).poll_delay = poll_delay;
        (*instance).poll_returns_candidate = true;
    }
    active
}

/// Clones the per-instance observation handles out of a fake plugin.
///
/// The handles outlive the plugin, so a test can still assert on shutdown
/// after the driver has dropped it.
pub(crate) fn fake_observations(active: &ActivePlugin) -> FakeObservations {
    let state = unsafe { &*active.instance.unwrap().as_ptr().cast::<FakeState>() };
    FakeObservations {
        events: Arc::clone(&state.events),
        discard_reasons: Arc::clone(&state.discard_reasons),
        cancel_count: Arc::clone(&state.cancel_count),
        shutdown_count: Arc::clone(&state.shutdown_count),
    }
}

/// Builds a plugin whose proposal callback fails, exercising the fail-open path.
pub(crate) fn fake_active_with_failing_proposal() -> ActivePlugin {
    let (active, _, _) =
        fake_active_with_timing(Duration::ZERO, Duration::ZERO, Duration::ZERO, false);
    let instance = active.instance.unwrap().as_ptr().cast::<FakeState>();
    unsafe {
        (*instance).poll_fails = true;
    }
    active
}

pub(crate) fn wait_for_event(events: &Mutex<Vec<&'static str>>, event: &str) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while !events.lock().unwrap().contains(&event) {
        assert!(Instant::now() < deadline, "timed out waiting for {event}");
        thread::yield_now();
    }
}

pub(crate) fn proposal_query(deadline: Instant) -> LinearProposalQuery {
    LinearProposalQuery::new(1, 2, 1, 1, 0, 8, deadline)
}
