use super::{
    DEFAULT_PRETTY_DASHBOARD_EVENT_HISTORY_LIMIT, DashboardAcceptedRequestBucket,
    DashboardEndpointRow, DashboardLaunchPlan, DashboardModelRow, DashboardProcessRow,
    DashboardSnapshot, LlamaInstanceKind, ModelProgressStatus, OutputLevel,
    PRETTY_DASHBOARD_PANEL_COUNT, PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS,
    PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS, PRETTY_TUI_STARTUP_HISTORY_LIMIT, RuntimeStatus,
    format_invite_mesh_label,
};
use std::collections::{BTreeSet, VecDeque};
use tokio::time::Instant;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ModelProgressState {
    pub(super) label: String,
    pub(super) file: Option<String>,
    pub(super) downloaded_bytes: Option<u64>,
    pub(super) total_bytes: Option<u64>,
    pub(super) status: ModelProgressStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct StartupProgressState {
    pub(super) completed_steps: usize,
    pub(super) total_steps: usize,
    pub(super) detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StartupLifecyclePhase {
    Pending,
    Starting,
    Partial,
    Ready,
    Failed,
    ShuttingDown,
}

impl StartupLifecyclePhase {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            StartupLifecyclePhase::Pending => "pending",
            StartupLifecyclePhase::Starting => "starting",
            StartupLifecyclePhase::Partial => "partial",
            StartupLifecyclePhase::Ready => "ready",
            StartupLifecyclePhase::Failed => "failed",
            StartupLifecyclePhase::ShuttingDown => "shutting down",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StartupComponentState {
    pub phase: StartupLifecyclePhase,
    pub detail: Option<String>,
}

impl Default for StartupComponentState {
    fn default() -> Self {
        Self {
            phase: StartupLifecyclePhase::Pending,
            detail: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StartupLifecycleState {
    pub phase: StartupLifecyclePhase,
    pub mesh: StartupComponentState,
    pub api: StartupComponentState,
    pub console: StartupComponentState,
    pub llama_server: StartupComponentState,
    pub model_readiness: StartupComponentState,
    pub(super) boot_started: bool,
    pub(super) failure: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TruthfulStartupStatusKey {
    Console,
    Api,
    LlamaServer,
}

impl Default for StartupLifecycleState {
    fn default() -> Self {
        Self {
            phase: StartupLifecyclePhase::Pending,
            mesh: StartupComponentState::default(),
            api: StartupComponentState::default(),
            console: StartupComponentState::default(),
            llama_server: StartupComponentState::default(),
            model_readiness: StartupComponentState::default(),
            boot_started: false,
            failure: None,
        }
    }
}

impl StartupLifecycleState {
    pub(super) fn mark_boot_started(&mut self, detail: Option<String>) {
        self.boot_started = true;
        if self.mesh.detail.is_none() {
            self.mesh.detail = detail;
        }
        self.recompute_phase(false, false);
    }

    pub(super) fn update_component_starting(
        component: &mut StartupComponentState,
        detail: Option<String>,
    ) {
        component.phase = match component.phase {
            StartupLifecyclePhase::Ready => StartupLifecyclePhase::Partial,
            StartupLifecyclePhase::Failed => StartupLifecyclePhase::Failed,
            StartupLifecyclePhase::ShuttingDown => StartupLifecyclePhase::ShuttingDown,
            _ => StartupLifecyclePhase::Starting,
        };
        component.detail = detail.or_else(|| component.detail.clone());
    }

    pub(super) fn update_component_ready(
        component: &mut StartupComponentState,
        detail: Option<String>,
    ) {
        if !matches!(component.phase, StartupLifecyclePhase::Failed) {
            component.phase = StartupLifecyclePhase::Ready;
            component.detail = detail.or_else(|| component.detail.clone());
        }
    }

    pub(super) fn update_component_failed(
        component: &mut StartupComponentState,
        detail: Option<String>,
    ) {
        component.phase = StartupLifecyclePhase::Failed;
        component.detail = detail;
    }

    pub(super) fn update_component_shutting_down(component: &mut StartupComponentState) {
        if !matches!(component.phase, StartupLifecyclePhase::Pending) {
            component.phase = StartupLifecyclePhase::ShuttingDown;
        }
    }

    pub(super) fn finalize_for_runtime_ready(&mut self, api_url: &str, console_url: Option<&str>) {
        self.boot_started = true;
        let mesh_detail = self.mesh.detail.clone();
        let llama_detail = self.llama_server.detail.clone();
        let model_detail = self.model_readiness.detail.clone();
        Self::update_component_ready(&mut self.mesh, mesh_detail);
        Self::update_component_ready(&mut self.api, Some(format!("API ready at {api_url}")));
        if let Some(url) = console_url {
            Self::update_component_ready(
                &mut self.console,
                Some(format!("console ready at {url}")),
            );
        }
        if !matches!(
            self.llama_server.phase,
            StartupLifecyclePhase::Failed | StartupLifecyclePhase::ShuttingDown
        ) {
            Self::update_component_ready(
                &mut self.llama_server,
                llama_detail.or_else(|| Some("embedded runtime ready".to_string())),
            );
        }
        if matches!(
            self.model_readiness.phase,
            StartupLifecyclePhase::Starting | StartupLifecyclePhase::Partial
        ) {
            Self::update_component_ready(&mut self.model_readiness, model_detail);
        }
        self.recompute_phase(true, false);
    }

    pub(super) fn mark_failure(&mut self, detail: String) {
        self.boot_started = true;
        self.failure = Some(detail.clone());
        let target = if matches!(
            self.model_readiness.phase,
            StartupLifecyclePhase::Starting | StartupLifecyclePhase::Partial
        ) {
            &mut self.model_readiness
        } else if matches!(
            self.llama_server.phase,
            StartupLifecyclePhase::Starting | StartupLifecyclePhase::Partial
        ) {
            &mut self.llama_server
        } else if matches!(
            self.api.phase,
            StartupLifecyclePhase::Starting | StartupLifecyclePhase::Partial
        ) {
            &mut self.api
        } else if matches!(
            self.console.phase,
            StartupLifecyclePhase::Starting | StartupLifecyclePhase::Partial
        ) {
            &mut self.console
        } else {
            &mut self.mesh
        };
        Self::update_component_failed(target, Some(detail));
        self.recompute_phase(false, false);
    }

    pub(super) fn mark_shutting_down(&mut self) {
        self.boot_started = true;
        Self::update_component_shutting_down(&mut self.mesh);
        Self::update_component_shutting_down(&mut self.api);
        Self::update_component_shutting_down(&mut self.console);
        Self::update_component_shutting_down(&mut self.llama_server);
        Self::update_component_shutting_down(&mut self.model_readiness);
        self.recompute_phase(false, true);
    }

    pub(super) fn recompute_phase(&mut self, runtime_ready: bool, shutdown_in_progress: bool) {
        if shutdown_in_progress {
            self.phase = StartupLifecyclePhase::ShuttingDown;
            return;
        }
        if self.failure.is_some()
            || [
                &self.mesh,
                &self.api,
                &self.console,
                &self.llama_server,
                &self.model_readiness,
            ]
            .iter()
            .any(|component| matches!(component.phase, StartupLifecyclePhase::Failed))
        {
            self.phase = StartupLifecyclePhase::Failed;
            return;
        }
        if runtime_ready {
            self.phase = StartupLifecyclePhase::Ready;
            return;
        }
        if !self.boot_started {
            self.phase = StartupLifecyclePhase::Pending;
            return;
        }
        if [
            &self.mesh,
            &self.api,
            &self.console,
            &self.llama_server,
            &self.model_readiness,
        ]
        .iter()
        .any(|component| {
            matches!(
                component.phase,
                StartupLifecyclePhase::Ready | StartupLifecyclePhase::Partial
            )
        }) {
            self.phase = StartupLifecyclePhase::Partial;
        } else {
            self.phase = StartupLifecyclePhase::Starting;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LoadingProgressState {
    pub(super) ratio: f64,
    pub(super) detail: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaInstanceState {
    pub kind: LlamaInstanceKind,
    pub port: u16,
    pub status: RuntimeStatus,
    pub device: Option<String>,
    pub model: Option<String>,
    pub ctx_size: Option<u32>,
    pub log_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunningModelState {
    pub model: String,
    pub profile: String,
    pub status: RuntimeStatus,
    pub internal_port: Option<u16>,
    pub role: Option<String>,
    pub capacity_gb: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PassiveModeState {
    pub role: String,
    pub status: RuntimeStatus,
    pub capacity_gb: Option<f64>,
    pub models_on_disk: Vec<String>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiModelModeState {
    pub count: usize,
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointState {
    pub label: String,
    pub status: RuntimeStatus,
    pub url: String,
    pub details: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshEventState {
    pub timestamp: String,
    pub level: OutputLevel,
    pub summary: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DashboardPanel {
    JoinToken,
    Events,
    LlamaCpp,
    Webserver,
    Models,
    Requests,
}

impl DashboardPanel {
    pub(in crate::output) const ALL: [Self; PRETTY_DASHBOARD_PANEL_COUNT] = [
        Self::JoinToken,
        Self::Events,
        Self::LlamaCpp,
        Self::Webserver,
        Self::Models,
        Self::Requests,
    ];

    pub(in crate::output) const fn index(self) -> usize {
        match self {
            Self::JoinToken => 0,
            Self::Events => 1,
            Self::LlamaCpp => 2,
            Self::Webserver => 3,
            Self::Models => 4,
            Self::Requests => 5,
        }
    }

    pub(super) fn next(self) -> Self {
        Self::ALL[(self.index() + 1) % Self::ALL.len()]
    }

    pub(super) fn previous(self) -> Self {
        Self::ALL[(self.index() + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DashboardPanelViewState {
    pub(super) scroll_offset: usize,
    pub(super) selected_row: Option<usize>,
    pub(super) viewport_rows: usize,
}

impl Default for DashboardPanelViewState {
    fn default() -> Self {
        Self {
            scroll_offset: 0,
            selected_row: None,
            viewport_rows: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DashboardLayoutWidget {
    pub(super) rows: usize,
    pub(super) selectable: bool,
}

impl DashboardLayoutWidget {
    pub(super) fn new(rows: usize, selectable: bool) -> Self {
        Self {
            rows: rows.max(1),
            selectable,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DashboardLayoutState {
    pub(super) widgets: [DashboardLayoutWidget; PRETTY_DASHBOARD_PANEL_COUNT],
}

impl DashboardLayoutState {
    pub(in crate::output) fn new(
        events_rows: usize,
        llama_rows: usize,
        webserver_rows: usize,
        models_rows: usize,
        requests_rows: usize,
    ) -> Self {
        Self {
            widgets: [
                DashboardLayoutWidget::new(1, false),
                DashboardLayoutWidget::new(events_rows, true),
                DashboardLayoutWidget::new(llama_rows, true),
                DashboardLayoutWidget::new(webserver_rows, true),
                DashboardLayoutWidget::new(models_rows, false),
                DashboardLayoutWidget::new(requests_rows, false),
            ],
        }
    }

    pub(super) fn rows_for(self, panel: DashboardPanel) -> usize {
        self.widgets[panel.index()].rows.max(1)
    }

    pub(super) fn rows_are_selectable_for(self, panel: DashboardPanel) -> bool {
        self.widgets[panel.index()].selectable
    }
}

impl Default for DashboardLayoutState {
    fn default() -> Self {
        Self::new(1, 1, 1, 1, 1)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct DashboardEventsFilterState {
    pub(super) query: String,
    pub(super) editing: bool,
}

impl DashboardEventsFilterState {
    pub(super) fn is_active(&self) -> bool {
        !self.query.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DashboardJoinTokenState {
    pub(super) token: String,
    pub(super) mesh_id: String,
    pub(super) mesh_name: Option<String>,
    pub(super) copy_status: DashboardJoinTokenCopyStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum DashboardJoinTokenCopyStatus {
    Idle,
    Copied { at: Instant },
    Failed { message: String, at: Instant },
}

impl DashboardJoinTokenCopyStatus {
    pub(super) fn feedback_at(&self) -> Option<Instant> {
        match self {
            Self::Idle => None,
            Self::Copied { at } | Self::Failed { at, .. } => Some(*at),
        }
    }
}

impl DashboardJoinTokenState {
    pub(super) fn new(token: String, mesh_id: String, mesh_name: Option<String>) -> Self {
        Self {
            token,
            mesh_id,
            mesh_name,
            copy_status: DashboardJoinTokenCopyStatus::Idle,
        }
    }

    pub(super) fn mesh_label(&self) -> String {
        format_invite_mesh_label(self.mesh_name.as_deref(), &self.mesh_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DashboardRequestHistoryState {
    pub(super) current_inflight_requests: u64,
    pub(super) accepted_request_buckets: Vec<DashboardAcceptedRequestBucket>,
    pub(super) latency_samples_ms: Vec<u64>,
    pub(super) history_limit: usize,
}

impl Default for DashboardRequestHistoryState {
    fn default() -> Self {
        Self {
            current_inflight_requests: 0,
            accepted_request_buckets: (0..PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS)
                .map(|second_offset| DashboardAcceptedRequestBucket {
                    second_offset,
                    accepted_count: 0,
                })
                .collect(),
            latency_samples_ms: Vec::new(),
            history_limit: PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize,
        }
    }
}

impl DashboardRequestHistoryState {
    pub(super) fn from_snapshot(snapshot: &DashboardSnapshot) -> Self {
        Self {
            current_inflight_requests: snapshot.current_inflight_requests,
            accepted_request_buckets: crate::output::rendering::normalize_request_buckets(
                &snapshot.accepted_request_buckets,
            ),
            latency_samples_ms: snapshot.latency_samples_ms.clone(),
            history_limit: PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum DashboardRequestWindow {
    #[default]
    SixtySeconds,
    TenMinutes,
    SixtyMinutes,
    TwelveHours,
    TwentyFourHours,
}

impl DashboardRequestWindow {
    pub(in crate::output) const ALL: [Self; 5] = [
        Self::SixtySeconds,
        Self::TenMinutes,
        Self::SixtyMinutes,
        Self::TwelveHours,
        Self::TwentyFourHours,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::SixtySeconds => "60s",
            Self::TenMinutes => "10m",
            Self::SixtyMinutes => "60m",
            Self::TwelveHours => "12h",
            Self::TwentyFourHours => "24h",
        }
    }

    pub(super) fn bucket_label(self) -> &'static str {
        match self {
            Self::SixtySeconds => "2s buckets",
            Self::TenMinutes => "20s buckets",
            Self::SixtyMinutes => "2m buckets",
            Self::TwelveHours => "30m buckets",
            Self::TwentyFourHours => "60m buckets",
        }
    }

    pub(super) fn seconds(self) -> u32 {
        match self {
            Self::SixtySeconds => 60,
            Self::TenMinutes => 10 * 60,
            Self::SixtyMinutes => 60 * 60,
            Self::TwelveHours => 12 * 60 * 60,
            Self::TwentyFourHours => 24 * 60 * 60,
        }
    }

    pub(super) fn bucket_seconds(self) -> u32 {
        match self {
            Self::TwelveHours => 30 * 60,
            Self::TwentyFourHours => 60 * 60,
            _ => self.seconds() / PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS as u32,
        }
    }

    pub(super) fn bar_width_cap(self) -> Option<u16> {
        match self {
            Self::TwelveHours | Self::TwentyFourHours => Some(1),
            _ => None,
        }
    }

    pub(super) fn preferred_bar_gap(self) -> u16 {
        match self {
            Self::TwelveHours | Self::TwentyFourHours => 1,
            _ => 0,
        }
    }

    pub(super) fn previous(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|window| *window == self)
            .unwrap_or_default();
        Self::ALL[index.saturating_sub(1)]
    }

    pub(super) fn next(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|window| *window == self)
            .unwrap_or_default();
        Self::ALL[(index + 1).min(Self::ALL.len() - 1)]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DashboardState {
    pub(super) session_started_at: Instant,
    pub(super) version: Option<String>,
    pub(super) node_id: Option<String>,
    pub(super) mesh_id: Option<String>,
    pub(super) runtime_ready: bool,
    pub(super) peer_ids: BTreeSet<String>,
    pub(super) llama_instances: Vec<LlamaInstanceState>,
    pub(super) multi_model_mode: Option<MultiModelModeState>,
    pub(super) passive_mode: Option<PassiveModeState>,
    pub(super) running_models: Vec<RunningModelState>,
    pub(super) webserver: Option<EndpointState>,
    pub(super) api: Option<EndpointState>,
    pub(super) mesh_events: VecDeque<MeshEventState>,
    pub(super) mesh_event_limit: usize,
    pub(super) startup_history: VecDeque<MeshEventState>,
    pub(super) startup_history_limit: usize,
    pub(super) panel_focus: DashboardPanel,
    pub(super) full_screen_panel: Option<DashboardPanel>,
    pub(super) panel_layout: DashboardLayoutState,
    pub(super) panel_view_states: [DashboardPanelViewState; PRETTY_DASHBOARD_PANEL_COUNT],
    pub(super) events_follow: bool,
    pub(super) events_filter: DashboardEventsFilterState,
    pub(super) llama_process_rows: Vec<DashboardProcessRow>,
    pub(super) ready_llama_process_rows: BTreeSet<String>,
    pub(super) webserver_rows: Vec<DashboardEndpointRow>,
    pub(super) loaded_model_rows: Vec<DashboardModelRow>,
    pub(super) request_history: DashboardRequestHistoryState,
    pub(super) request_window: DashboardRequestWindow,
    pub(super) join_token: Option<DashboardJoinTokenState>,
    pub(super) terminal_size: Option<(u16, u16)>,
    pub(super) launch_plan: Option<DashboardLaunchPlan>,
    pub(super) model_progress: Option<ModelProgressState>,
    pub(super) startup_progress: Option<StartupProgressState>,
    pub(super) startup_milestones: BTreeSet<String>,
    pub(super) startup_lifecycle: StartupLifecycleState,
    pub(super) shutdown_in_progress: bool,
}

impl Default for DashboardState {
    fn default() -> Self {
        let panel_layout = DashboardLayoutState::default();
        let mut state = Self {
            session_started_at: Instant::now(),
            version: None,
            node_id: None,
            mesh_id: None,
            runtime_ready: false,
            peer_ids: BTreeSet::new(),
            llama_instances: Vec::new(),
            multi_model_mode: None,
            passive_mode: None,
            running_models: Vec::new(),
            webserver: None,
            api: None,
            mesh_events: VecDeque::new(),
            mesh_event_limit: DEFAULT_PRETTY_DASHBOARD_EVENT_HISTORY_LIMIT,
            startup_history: VecDeque::new(),
            startup_history_limit: PRETTY_TUI_STARTUP_HISTORY_LIMIT,
            panel_focus: DashboardPanel::Events,
            full_screen_panel: None,
            panel_layout,
            panel_view_states: [DashboardPanelViewState::default(); PRETTY_DASHBOARD_PANEL_COUNT],
            events_follow: true,
            events_filter: DashboardEventsFilterState::default(),
            llama_process_rows: Vec::new(),
            ready_llama_process_rows: BTreeSet::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: Vec::new(),
            request_history: DashboardRequestHistoryState::default(),
            request_window: DashboardRequestWindow::default(),
            join_token: None,
            terminal_size: None,
            launch_plan: None,
            model_progress: None,
            startup_progress: None,
            startup_milestones: BTreeSet::new(),
            startup_lifecycle: StartupLifecycleState::default(),
            shutdown_in_progress: false,
        };
        state.apply_layout(panel_layout);
        state
    }
}
