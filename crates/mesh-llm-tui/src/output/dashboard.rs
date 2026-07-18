use super::merging::*;
use super::rendering::*;
use super::state::*;
use super::{
    DashboardEndpointRow, DashboardModelRow, DashboardProcessRow, DashboardSnapshot,
    LlamaInstanceKind, OutputEvent, PRETTY_TUI_JOIN_TOKEN_COPY_STATUS_TTL,
    PRETTY_TUI_STARTUP_PROGRESS_MIN_STEPS, RuntimeStatus, TuiControlFlow, TuiEvent,
    TuiEventListRenderer, TuiKeyEvent, format_invite_mesh_label,
};
use crate::output::formatting::{OutputEventPresentation, dashboard_layout_for_terminal_size};
use chrono::Local;
use ratatui::layout::Rect;
use tokio::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum DashboardAction {
    OutputEvent(OutputEvent),
    SnapshotUpdated(DashboardSnapshot),
    FocusNextPanel,
    FocusPreviousPanel,
    EnterFullScreenPanel(DashboardPanel),
    ExitFullScreenPanel,
    ToggleFullScreenPanel,
    ToggleEventsFollow,
    StartEventsFilterEdit,
    InsertEventsFilterChar(char),
    BackspaceEventsFilter,
    ConfirmEventsFilter,
    CancelEventsFilter,
    SelectPreviousRequestWindow,
    SelectNextRequestWindow,
    SetJoinTokenCopyStatus(DashboardJoinTokenCopyStatus),
    #[cfg(test)]
    SetPanelScroll {
        panel: DashboardPanel,
        scroll_offset: usize,
    },
    #[cfg(test)]
    SetPanelSelection {
        panel: DashboardPanel,
        selected_row: Option<usize>,
    },
    Resize(DashboardLayoutState),
}

impl DashboardState {
    #[cfg(test)]
    pub(super) fn startup_lifecycle(&self) -> &StartupLifecycleState {
        &self.startup_lifecycle
    }

    pub(super) fn startup_mesh_component_active(&self) -> bool {
        !self.runtime_ready && !self.shutdown_in_progress
    }

    pub(super) fn update_startup_mesh_component_starting(&mut self, detail: Option<String>) {
        if !self.startup_mesh_component_active() {
            return;
        }
        StartupLifecycleState::update_component_starting(&mut self.startup_lifecycle.mesh, detail);
        self.startup_lifecycle
            .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
    }

    pub(super) fn update_startup_mesh_component_ready(&mut self, detail: Option<String>) {
        if !self.startup_mesh_component_active() {
            return;
        }
        StartupLifecycleState::update_component_ready(&mut self.startup_lifecycle.mesh, detail);
        self.startup_lifecycle
            .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
    }

    pub(super) fn mark_startup_mesh_component_failed(&mut self, detail: String) {
        if !self.startup_mesh_component_active() {
            return;
        }
        self.startup_lifecycle.failure = Some(detail.clone());
        StartupLifecycleState::update_component_failed(
            &mut self.startup_lifecycle.mesh,
            Some(detail),
        );
        self.startup_lifecycle
            .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
    }

    pub(in crate::output) fn startup_component_for_truthful_status(
        &self,
        key: TruthfulStartupStatusKey,
    ) -> StartupComponentState {
        match key {
            TruthfulStartupStatusKey::Console => self.startup_lifecycle.console.clone(),
            TruthfulStartupStatusKey::Api => self.startup_lifecycle.api.clone(),
            TruthfulStartupStatusKey::LlamaServer => self.startup_lifecycle.llama_server.clone(),
        }
    }

    pub(super) fn truthful_startup_key_for_process(name: &str) -> Option<TruthfulStartupStatusKey> {
        let normalized = name.to_ascii_lowercase();
        if normalized.contains("llama") {
            Some(TruthfulStartupStatusKey::LlamaServer)
        } else {
            None
        }
    }

    pub(super) fn truthful_startup_key_for_endpoint(
        label: &str,
    ) -> Option<TruthfulStartupStatusKey> {
        let normalized = label.to_ascii_lowercase();
        if normalized.contains("console") {
            Some(TruthfulStartupStatusKey::Console)
        } else if normalized == "api" || normalized.contains("openai-compatible api") {
            Some(TruthfulStartupStatusKey::Api)
        } else {
            None
        }
    }

    pub(in crate::output) fn truthful_runtime_status_for_component(
        component: &StartupComponentState,
        current: &RuntimeStatus,
    ) -> RuntimeStatus {
        match component.phase {
            StartupLifecyclePhase::Failed => match current {
                RuntimeStatus::Warning
                | RuntimeStatus::Error
                | RuntimeStatus::Exited
                | RuntimeStatus::Stopped
                | RuntimeStatus::ShuttingDown => current.clone(),
                _ => RuntimeStatus::Error,
            },
            StartupLifecyclePhase::ShuttingDown => RuntimeStatus::ShuttingDown,
            StartupLifecyclePhase::Ready => match current {
                RuntimeStatus::Warning
                | RuntimeStatus::Error
                | RuntimeStatus::Exited
                | RuntimeStatus::Stopped
                | RuntimeStatus::ShuttingDown => current.clone(),
                _ => RuntimeStatus::Ready,
            },
            StartupLifecyclePhase::Pending
            | StartupLifecyclePhase::Starting
            | StartupLifecyclePhase::Partial => match current {
                RuntimeStatus::NotReady => RuntimeStatus::NotReady,
                RuntimeStatus::Loading => RuntimeStatus::Loading,
                RuntimeStatus::Warning
                | RuntimeStatus::Error
                | RuntimeStatus::Exited
                | RuntimeStatus::Stopped
                | RuntimeStatus::ShuttingDown => current.clone(),
                _ => RuntimeStatus::Starting,
            },
        }
    }

    pub(in crate::output) fn truthful_runtime_status_for_process_component(
        component: &StartupComponentState,
        current: &RuntimeStatus,
        ready_event_seen: bool,
    ) -> RuntimeStatus {
        match component.phase {
            StartupLifecyclePhase::Pending
            | StartupLifecyclePhase::Starting
            | StartupLifecyclePhase::Partial
                if ready_event_seen
                    && matches!(
                        current,
                        RuntimeStatus::NotReady
                            | RuntimeStatus::Loading
                            | RuntimeStatus::Starting
                            | RuntimeStatus::Ready
                    ) =>
            {
                RuntimeStatus::Ready
            }
            _ => Self::truthful_runtime_status_for_component(component, current),
        }
    }

    pub(super) fn sync_truthful_startup_statuses(&mut self) {
        let console_component =
            self.startup_component_for_truthful_status(TruthfulStartupStatusKey::Console);
        let api_component =
            self.startup_component_for_truthful_status(TruthfulStartupStatusKey::Api);
        let llama_component =
            self.startup_component_for_truthful_status(TruthfulStartupStatusKey::LlamaServer);

        if let Some((webserver, key)) = self.webserver.as_mut().and_then(|webserver| {
            Self::truthful_startup_key_for_endpoint(&webserver.label).map(|key| (webserver, key))
        }) {
            webserver.status = Self::truthful_runtime_status_for_component(
                match key {
                    TruthfulStartupStatusKey::Console => &console_component,
                    TruthfulStartupStatusKey::Api => &api_component,
                    TruthfulStartupStatusKey::LlamaServer => &llama_component,
                },
                &webserver.status,
            );
        }
        if let Some((api, key)) = self.api.as_mut().and_then(|api| {
            Self::truthful_startup_key_for_endpoint(&api.label).map(|key| (api, key))
        }) {
            api.status = Self::truthful_runtime_status_for_component(
                match key {
                    TruthfulStartupStatusKey::Console => &console_component,
                    TruthfulStartupStatusKey::Api => &api_component,
                    TruthfulStartupStatusKey::LlamaServer => &llama_component,
                },
                &api.status,
            );
        }
        let ready_llama_process_rows = self.ready_llama_process_rows.clone();
        for row in &mut self.llama_process_rows {
            if let Some(key) = Self::truthful_startup_key_for_process(&row.name) {
                let ready_event_seen = ready_llama_process_rows
                    .iter()
                    .any(|ready_name| process_row_names_match(ready_name, &row.name));
                row.status = Self::truthful_runtime_status_for_process_component(
                    match key {
                        TruthfulStartupStatusKey::Console => &console_component,
                        TruthfulStartupStatusKey::Api => &api_component,
                        TruthfulStartupStatusKey::LlamaServer => &llama_component,
                    },
                    &row.status,
                    ready_event_seen,
                );
            }
        }
        for row in &mut self.webserver_rows {
            if let Some(key) = Self::truthful_startup_key_for_endpoint(&row.label) {
                row.status = Self::truthful_runtime_status_for_component(
                    match key {
                        TruthfulStartupStatusKey::Console => &console_component,
                        TruthfulStartupStatusKey::Api => &api_component,
                        TruthfulStartupStatusKey::LlamaServer => &llama_component,
                    },
                    &row.status,
                );
            }
        }
    }

    pub(in crate::output) fn reduce(&mut self, action: DashboardAction) {
        match action {
            DashboardAction::OutputEvent(event) => self.apply_output_event(&event),
            DashboardAction::SnapshotUpdated(snapshot) => self.apply_snapshot(&snapshot),
            DashboardAction::FocusNextPanel => {
                self.panel_focus = self.panel_focus.next();
                if self.full_screen_panel.is_some() {
                    self.full_screen_panel = Some(self.panel_focus);
                    self.sync_full_screen_panel_viewport();
                }
                if self.panel_focus != DashboardPanel::Events {
                    self.events_filter.editing = false;
                }
            }
            DashboardAction::FocusPreviousPanel => {
                self.panel_focus = self.panel_focus.previous();
                if self.full_screen_panel.is_some() {
                    self.full_screen_panel = Some(self.panel_focus);
                    self.sync_full_screen_panel_viewport();
                }
                if self.panel_focus != DashboardPanel::Events {
                    self.events_filter.editing = false;
                }
            }
            DashboardAction::EnterFullScreenPanel(panel) => {
                self.panel_focus = panel;
                self.full_screen_panel = Some(panel);
                self.sync_full_screen_panel_viewport();
                if self.panel_focus != DashboardPanel::Events {
                    self.events_filter.editing = false;
                }
            }
            DashboardAction::ExitFullScreenPanel => {
                self.full_screen_panel = None;
                self.apply_layout(self.panel_layout);
            }
            DashboardAction::ToggleFullScreenPanel => {
                if self.full_screen_panel.is_some() {
                    self.reduce(DashboardAction::ExitFullScreenPanel);
                } else {
                    self.reduce(DashboardAction::EnterFullScreenPanel(self.panel_focus));
                }
            }
            DashboardAction::ToggleEventsFollow => {
                self.events_follow = !self.events_follow;
                self.sync_events_panel();
            }
            DashboardAction::StartEventsFilterEdit => {
                self.panel_focus = DashboardPanel::Events;
                if self.full_screen_panel.is_some() {
                    self.full_screen_panel = Some(DashboardPanel::Events);
                    self.sync_full_screen_panel_viewport();
                }
                self.events_filter.editing = true;
                self.sync_events_panel();
            }
            DashboardAction::InsertEventsFilterChar(ch) => {
                self.panel_focus = DashboardPanel::Events;
                self.events_filter.editing = true;
                self.events_filter.query.push(ch);
                self.sync_events_panel();
            }
            DashboardAction::BackspaceEventsFilter => {
                self.panel_focus = DashboardPanel::Events;
                self.events_filter.editing = true;
                self.events_filter.query.pop();
                self.sync_events_panel();
            }
            DashboardAction::ConfirmEventsFilter => {
                self.events_filter.editing = false;
                self.sync_events_panel();
            }
            DashboardAction::CancelEventsFilter => {
                self.panel_focus = DashboardPanel::Events;
                self.events_filter.query.clear();
                self.events_filter.editing = false;
                self.sync_events_panel();
            }
            DashboardAction::SelectPreviousRequestWindow => {
                self.request_window = self.request_window.previous();
            }
            DashboardAction::SelectNextRequestWindow => {
                self.request_window = self.request_window.next();
            }
            DashboardAction::SetJoinTokenCopyStatus(copy_status) => {
                if let Some(join_token) = self.join_token.as_mut() {
                    join_token.copy_status = copy_status;
                }
            }
            #[cfg(test)]
            DashboardAction::SetPanelScroll {
                panel,
                scroll_offset,
            } => {
                self.panel_view_state_mut(panel).scroll_offset = scroll_offset;
                self.clamp_panel_view(panel);
            }
            #[cfg(test)]
            DashboardAction::SetPanelSelection {
                panel,
                selected_row,
            } => {
                self.panel_view_state_mut(panel).selected_row = selected_row;
                self.clamp_panel_view(panel);
            }
            DashboardAction::Resize(layout) => {
                self.apply_layout(layout);
            }
        }
    }

    pub(in crate::output) fn apply_layout(&mut self, layout: DashboardLayoutState) {
        self.panel_layout = layout;
        for panel in DashboardPanel::ALL {
            self.panel_view_state_mut(panel).viewport_rows = if panel == DashboardPanel::JoinToken {
                self.join_token_viewport_columns()
            } else {
                tui_panel_viewport_rows(panel, self.panel_layout.rows_for(panel))
            };
            self.clamp_panel_view(panel);
        }
        self.sync_full_screen_panel_viewport();
        self.sync_events_panel();
    }

    pub(super) fn sync_full_screen_panel_viewport(&mut self) {
        let Some(panel) = self.full_screen_panel else {
            return;
        };
        let viewport_rows = self.full_screen_panel_viewport_rows(panel);
        self.panel_view_state_mut(panel).viewport_rows = viewport_rows;
        self.clamp_panel_view(panel);
    }

    pub(super) fn full_screen_panel_viewport_rows(&self, panel: DashboardPanel) -> usize {
        let Some((columns, rows)) = self.terminal_size else {
            return self.panel_view_state(panel).viewport_rows.max(1);
        };
        let panel_area = Rect::new(0, 0, columns, rows);
        let inner_rows = usize::from(rows.saturating_sub(2)).max(1);
        match panel {
            DashboardPanel::JoinToken => usize::from(join_token_content_width(
                panel_area,
                tui_join_token_copy_button_area(panel_area),
            ))
            .max(1),
            DashboardPanel::LlamaCpp | DashboardPanel::Webserver => {
                inner_rows.saturating_sub(1).max(1)
            }
            DashboardPanel::Models => tui_panel_viewport_rows(DashboardPanel::Models, inner_rows),
            DashboardPanel::Events | DashboardPanel::Requests => inner_rows,
        }
    }

    pub(super) fn apply_snapshot(&mut self, snapshot: &DashboardSnapshot) {
        if self.shutdown_in_progress {
            self.merge_shutdown_process_snapshot(snapshot);
        } else if self.launch_plan_known() && !self.runtime_ready {
            self.merge_startup_process_snapshot(snapshot);
        } else {
            self.llama_process_rows = snapshot.llama_process_rows.clone();
            self.webserver_rows = snapshot.webserver_rows.clone();
            self.loaded_model_rows = merged_loaded_model_snapshot_rows(
                &self.loaded_model_rows,
                &snapshot.loaded_model_rows,
            );
        }
        self.sync_truthful_startup_statuses();
        self.request_history = DashboardRequestHistoryState::from_snapshot(snapshot);
        self.clamp_all_panel_views();
        self.sync_events_panel();
    }

    pub(super) fn merge_shutdown_process_snapshot(&mut self, snapshot: &DashboardSnapshot) {
        for snapshot_row in &snapshot.llama_process_rows {
            if let Some(existing) = self
                .llama_process_rows
                .iter_mut()
                .find(|row| row.name == snapshot_row.name)
            {
                *existing = snapshot_row.clone();
            } else {
                self.llama_process_rows.push(snapshot_row.clone());
            }
        }
        self.llama_process_rows
            .sort_by_key(|row| row.name.to_lowercase());

        for snapshot_row in &snapshot.loaded_model_rows {
            if let Some(existing) = self
                .loaded_model_rows
                .iter_mut()
                .find(|row| row.name == snapshot_row.name)
            {
                *existing = snapshot_row.clone();
            } else {
                self.loaded_model_rows.push(snapshot_row.clone());
            }
        }
        self.loaded_model_rows
            .sort_by(|left, right| left.name.cmp(&right.name));

        for snapshot_row in &snapshot.webserver_rows {
            if let Some(existing) = self
                .webserver_rows
                .iter_mut()
                .find(|row| row.label == snapshot_row.label && row.port == snapshot_row.port)
            {
                *existing = snapshot_row.clone();
            } else {
                self.webserver_rows.push(snapshot_row.clone());
            }
        }
        sort_dashboard_endpoint_rows(&mut self.webserver_rows);
    }

    pub(super) fn merge_startup_process_snapshot(&mut self, snapshot: &DashboardSnapshot) {
        for row in &snapshot.llama_process_rows {
            self.upsert_process_row(row.clone());
        }
        for row in &snapshot.webserver_rows {
            self.upsert_endpoint_row(row.clone());
        }
        for row in &snapshot.loaded_model_rows {
            self.upsert_loaded_model_row(row.clone());
        }

        if let Some(plan) = self.launch_plan.clone() {
            self.preseed_launch_plan_rows(&plan);
        }
    }

    pub(super) fn mark_runtime_shutting_down(&mut self) {
        self.shutdown_in_progress = true;
        self.runtime_ready = false;
        for instance in &mut self.llama_instances {
            instance.status = RuntimeStatus::ShuttingDown;
        }
        for model in &mut self.running_models {
            model.status = RuntimeStatus::ShuttingDown;
        }
        for row in &mut self.llama_process_rows {
            row.status = RuntimeStatus::ShuttingDown;
        }
        for row in &mut self.loaded_model_rows {
            row.status = RuntimeStatus::ShuttingDown;
        }
        for row in &mut self.webserver_rows {
            row.status = RuntimeStatus::ShuttingDown;
        }
        if let Some(webserver) = &mut self.webserver {
            webserver.status = RuntimeStatus::ShuttingDown;
        }
        if let Some(api) = &mut self.api {
            api.status = RuntimeStatus::ShuttingDown;
        }
    }

    pub(super) fn launch_plan_known(&self) -> bool {
        self.launch_plan.is_some()
    }

    pub(in crate::output) fn is_startup_loading(&self) -> bool {
        false
    }

    pub(in crate::output) fn active_loading_progress(&self) -> Option<LoadingProgressState> {
        if self.runtime_ready {
            return None;
        }

        if let Some((progress, ratio)) = self.model_progress.as_ref().and_then(|progress| {
            model_download_progress_ratio(progress).map(|ratio| (progress, ratio))
        }) {
            return Some(LoadingProgressState {
                ratio,
                detail: loading_progress_detail(model_progress_detail(progress), ratio, None),
            });
        }

        if let Some(progress) = self.startup_progress.as_ref() {
            let ratio = startup_progress_ratio(progress);
            return Some(LoadingProgressState {
                ratio,
                detail: loading_progress_detail(
                    progress.detail.clone(),
                    ratio,
                    Some((progress.completed_steps, progress.total_steps)),
                ),
            });
        }

        self.model_progress.as_ref().map(|progress| {
            let ratio = fallback_model_progress_ratio(progress);
            LoadingProgressState {
                ratio,
                detail: loading_progress_detail(model_progress_detail(progress), ratio, None),
            }
        })
    }

    pub(super) fn apply_startup_progress_event(&mut self, event: &OutputEvent) {
        if self.shutdown_in_progress && is_shutdown_suppressed_ready_event(event) {
            return;
        }

        if matches!(event, OutputEvent::Startup { .. }) {
            self.startup_milestones.clear();
            self.startup_progress = None;
        }

        let Some((milestone_key, detail)) = startup_progress_event(event) else {
            return;
        };

        if let Some(key) = milestone_key {
            self.startup_milestones.insert(key);
        }

        let completed_steps = self.startup_milestones.len();
        let total_steps = if matches!(event, OutputEvent::RuntimeReady { .. }) {
            completed_steps.max(1)
        } else {
            PRETTY_TUI_STARTUP_PROGRESS_MIN_STEPS.max(completed_steps.saturating_add(1))
        };

        self.startup_progress = Some(StartupProgressState {
            completed_steps,
            total_steps,
            detail,
        });
    }

    pub(super) fn apply_startup_lifecycle_event(&mut self, event: &OutputEvent) {
        match event {
            OutputEvent::Startup { version, .. } => {
                self.startup_lifecycle = StartupLifecycleState::default();
                self.startup_lifecycle
                    .mark_boot_started(Some(format!("starting mesh-llm {version}")));
            }
            OutputEvent::NodeIdentity { node_id, mesh_id } => {
                let detail = match mesh_id {
                    Some(mesh_id) => Some(format!("node {node_id} joined mesh {mesh_id}")),
                    None => Some(format!("node {node_id} initialized")),
                };
                self.update_startup_mesh_component_ready(detail);
            }
            OutputEvent::InviteToken {
                mesh_id, mesh_name, ..
            } => {
                self.update_startup_mesh_component_ready(Some(format!(
                    "invite ready for {}",
                    format_invite_mesh_label(mesh_name.as_deref(), mesh_id)
                )));
            }
            OutputEvent::DiscoveryStarting { source } => {
                self.update_startup_mesh_component_starting(Some(format!(
                    "discovering mesh via {source}"
                )));
            }
            OutputEvent::MeshFound { mesh, peers, .. } => {
                self.update_startup_mesh_component_starting(Some(format!(
                    "found mesh {mesh} with {peers} peer(s)"
                )));
            }
            OutputEvent::DiscoveryJoined { mesh } => {
                self.update_startup_mesh_component_ready(Some(format!("joined mesh {mesh}")));
            }
            OutputEvent::DiscoveryFailed { message, detail } => {
                let failure_detail = detail
                    .as_ref()
                    .map(|detail| format!("{message}: {detail}"))
                    .unwrap_or_else(|| message.clone());
                self.mark_startup_mesh_component_failed(failure_detail);
            }
            OutputEvent::WaitingForPeers { detail } => {
                self.update_startup_mesh_component_starting(
                    detail
                        .clone()
                        .or_else(|| Some("waiting for peers".to_string())),
                );
            }
            OutputEvent::PassiveMode { detail, .. } => {
                self.update_startup_mesh_component_ready(detail.clone());
            }
            OutputEvent::Info { message, .. }
                if message == "Connected to bootstrap peer; awaiting mesh admission" =>
            {
                self.update_startup_mesh_component_starting(Some(message.clone()));
            }
            OutputEvent::Warning { message, .. }
                if message == "Failed to join any peer — running standalone" =>
            {
                self.mark_startup_mesh_component_failed(message.clone());
            }
            OutputEvent::ModelQueued { model }
            | OutputEvent::ModelLoading { model, .. }
            | OutputEvent::ModelLoaded { model, .. }
            | OutputEvent::HostElected { model, .. } => {
                StartupLifecycleState::update_component_starting(
                    &mut self.startup_lifecycle.model_readiness,
                    Some(format!("preparing model {model}")),
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::LlamaStarting {
                model, http_port, ..
            } => {
                let detail = match model {
                    Some(model) => Some(format!("starting llama-server for {model}")),
                    None => Some(format!("starting llama-server on port {http_port}")),
                };
                StartupLifecycleState::update_component_starting(
                    &mut self.startup_lifecycle.llama_server,
                    detail,
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::LlamaReady { model, port, .. } => {
                let detail = match model {
                    Some(model) => Some(format!("llama-server ready for {model}")),
                    None => Some(format!("llama-server ready on port {port}")),
                };
                StartupLifecycleState::update_component_ready(
                    &mut self.startup_lifecycle.llama_server,
                    detail,
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::LlamaStartupFailed {
                model,
                http_port,
                detail,
                ..
            } => {
                self.startup_lifecycle.failure = Some(detail.clone());
                let llama_detail = match model {
                    Some(model) => {
                        format!("llama-server failed for {model} (port {http_port}): {detail}")
                    }
                    None => format!("llama-server failed on port {http_port}: {detail}"),
                };
                let model_detail = match model {
                    Some(model) => format!("model {model} failed during llama startup: {detail}"),
                    None => format!("model startup blocked by llama-server failure: {detail}"),
                };
                StartupLifecycleState::update_component_failed(
                    &mut self.startup_lifecycle.llama_server,
                    Some(llama_detail),
                );
                StartupLifecycleState::update_component_failed(
                    &mut self.startup_lifecycle.model_readiness,
                    Some(model_detail),
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::ModelReady { model, .. } => {
                StartupLifecycleState::update_component_ready(
                    &mut self.startup_lifecycle.model_readiness,
                    Some(format!("model {model} ready")),
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::WebserverStarting { url } => {
                StartupLifecycleState::update_component_starting(
                    &mut self.startup_lifecycle.console,
                    Some(format!("starting console at {url}")),
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::WebserverReady { url } => {
                StartupLifecycleState::update_component_ready(
                    &mut self.startup_lifecycle.console,
                    Some(format!("console ready at {url}")),
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::ApiStarting { url } => {
                StartupLifecycleState::update_component_starting(
                    &mut self.startup_lifecycle.api,
                    Some(format!("starting API at {url}")),
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::ApiReady { url } => {
                StartupLifecycleState::update_component_ready(
                    &mut self.startup_lifecycle.api,
                    Some(format!("API ready at {url}")),
                );
                self.startup_lifecycle
                    .recompute_phase(self.runtime_ready, self.shutdown_in_progress);
            }
            OutputEvent::RuntimeReady {
                api_url,
                console_url,
                ..
            } => {
                self.startup_lifecycle
                    .finalize_for_runtime_ready(api_url, console_url.as_deref());
            }
            OutputEvent::Error { message, context } | OutputEvent::Fatal { message, context } => {
                if self.runtime_ready || self.shutdown_in_progress {
                    return;
                }
                let detail = context
                    .as_ref()
                    .map(|context| format!("{context}: {message}"))
                    .unwrap_or_else(|| message.clone());
                self.startup_lifecycle.mark_failure(detail);
            }
            OutputEvent::ShutdownRequested { .. } | OutputEvent::Shutdown { .. } => {
                self.startup_lifecycle.mark_shutting_down();
            }
            _ => {}
        }
    }

    pub(super) fn mark_llama_process_row_pending(&mut self, name: &str) {
        self.ready_llama_process_rows
            .retain(|ready_name| !process_row_names_match(ready_name, name));
    }

    pub(super) fn mark_llama_process_row_ready(&mut self, name: String) {
        self.ready_llama_process_rows.insert(name);
    }

    pub(super) fn apply_model_queue_event(&mut self, model: &str) {
        self.upsert_model(
            model,
            String::new(),
            RuntimeStatus::Loading,
            None,
            None,
            None,
        );
        self.upsert_loading_model_row(model);
        self.upsert_loading_process_row(model);
    }

    pub(in crate::output) fn apply_model_ready_event(
        &mut self,
        model: &str,
        internal_port: Option<u16>,
        role: Option<String>,
    ) {
        self.upsert_model(
            model,
            String::new(),
            RuntimeStatus::Ready,
            internal_port,
            role.clone(),
            None,
        );
        self.upsert_loaded_model_row(DashboardModelRow {
            name: model.to_string(),
            role,
            status: RuntimeStatus::Ready,
            port: internal_port,
            device: None,
            slots: None,
            quantization: None,
            ctx_size: None,
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: None,
        });
    }

    pub(super) fn apply_model_event(&mut self, event: &OutputEvent) -> bool {
        match event {
            OutputEvent::ModelQueued { model }
            | OutputEvent::ModelLoading { model, .. }
            | OutputEvent::ModelLoaded { model, .. } => {
                self.apply_model_queue_event(model);
            }
            OutputEvent::ModelUnloading { model } | OutputEvent::ModelUnloaded { model } => {
                self.upsert_model(
                    model,
                    String::new(),
                    RuntimeStatus::Stopped,
                    None,
                    None,
                    None,
                );
            }
            OutputEvent::ModelReady {
                model,
                internal_port,
                role,
            } => self.apply_model_ready_event(model, *internal_port, role.clone()),
            OutputEvent::HostElected {
                model,
                role,
                capacity_gb,
                ..
            } => {
                self.upsert_model(
                    model,
                    String::new(),
                    RuntimeStatus::Starting,
                    None,
                    role.clone(),
                    *capacity_gb,
                );
            }
            _ => return false,
        }
        true
    }

    pub(in crate::output) fn apply_passive_mode_event(
        &mut self,
        role: &str,
        status: &RuntimeStatus,
        capacity_gb: Option<f64>,
        models_on_disk: Option<&Vec<String>>,
        detail: Option<&String>,
    ) {
        let next_models_on_disk = models_on_disk.cloned().unwrap_or_default();
        if let Some(existing) = self.passive_mode.as_mut() {
            existing.role = role.to_string();
            existing.status = status.clone();
            existing.capacity_gb = capacity_gb.or(existing.capacity_gb);
            if models_on_disk.is_some() {
                existing.models_on_disk = next_models_on_disk;
            }
            existing.detail = detail.cloned().or_else(|| existing.detail.clone());
        } else {
            self.passive_mode = Some(PassiveModeState {
                role: role.to_string(),
                status: status.clone(),
                capacity_gb,
                models_on_disk: next_models_on_disk,
                detail: detail.cloned(),
            });
        }
    }

    pub(super) fn apply_llama_event(&mut self, event: &OutputEvent) -> bool {
        match event {
            OutputEvent::LlamaStarting {
                model,
                http_port,
                ctx_size,
                log_path,
            } => {
                let process_name = llama_process_row_name(model.as_deref());
                self.mark_llama_process_row_pending(&process_name);
                self.upsert_llama_instance(LlamaInstanceState {
                    kind: LlamaInstanceKind::LlamaServer,
                    port: *http_port,
                    status: RuntimeStatus::Starting,
                    device: None,
                    model: model.clone(),
                    ctx_size: *ctx_size,
                    log_path: log_path.clone(),
                });
                self.upsert_process_row(DashboardProcessRow {
                    name: process_name,
                    backend: String::new(),
                    status: RuntimeStatus::Starting,
                    port: *http_port,
                    pid: 0,
                });
            }
            OutputEvent::LlamaReady {
                model,
                port,
                ctx_size,
                log_path,
            } => {
                let process_name = llama_process_row_name(model.as_deref());
                self.mark_llama_process_row_ready(process_name.clone());
                self.upsert_llama_instance(LlamaInstanceState {
                    kind: LlamaInstanceKind::LlamaServer,
                    port: *port,
                    status: RuntimeStatus::Ready,
                    device: None,
                    model: model.clone(),
                    ctx_size: *ctx_size,
                    log_path: log_path.clone(),
                });
                self.upsert_process_row(DashboardProcessRow {
                    name: process_name,
                    backend: String::new(),
                    status: RuntimeStatus::Ready,
                    port: *port,
                    pid: 0,
                });
            }
            OutputEvent::LlamaStartupFailed {
                model,
                http_port,
                ctx_size,
                log_path,
                ..
            } => {
                self.mark_llama_process_row_pending(&llama_process_row_name(model.as_deref()));
                self.upsert_llama_instance(LlamaInstanceState {
                    kind: LlamaInstanceKind::LlamaServer,
                    port: *http_port,
                    status: RuntimeStatus::Error,
                    device: None,
                    model: model.clone(),
                    ctx_size: *ctx_size,
                    log_path: log_path.clone(),
                });
                self.upsert_process_row(DashboardProcessRow {
                    name: llama_process_row_name(model.as_deref()),
                    backend: String::new(),
                    status: RuntimeStatus::Error,
                    port: *http_port,
                    pid: 0,
                });
                if let Some(model) = model {
                    self.upsert_model(
                        model,
                        String::new(),
                        RuntimeStatus::Error,
                        Some(*http_port),
                        None,
                        None,
                    );
                    self.upsert_loaded_model_row(DashboardModelRow {
                        name: model.clone(),
                        role: None,
                        status: RuntimeStatus::Error,
                        port: Some(*http_port),
                        device: None,
                        slots: None,
                        quantization: None,
                        ctx_size: *ctx_size,
                        ctx_used_tokens: None,
                        lanes: None,
                        file_size_gb: None,
                    });
                }
            }
            _ => return false,
        }
        true
    }

    pub(in crate::output) fn apply_endpoint_state(
        &mut self,
        label: &str,
        status: RuntimeStatus,
        url: &str,
        row_label: &str,
    ) {
        let state = EndpointState {
            label: label.to_string(),
            status: status.clone(),
            url: url.to_string(),
            details: Vec::new(),
        };
        let row = DashboardEndpointRow {
            label: row_label.to_string(),
            status,
            url: url.to_string(),
            port: dashboard_port_from_url(url),
            pid: None,
        };
        if row_label == "Console" {
            self.webserver = Some(state);
        } else {
            self.api = Some(state);
        }
        self.upsert_endpoint_row(row);
    }

    pub(in crate::output) fn apply_runtime_ready_event(
        &mut self,
        api_url: &str,
        console_url: Option<&String>,
        pi_command: Option<&String>,
        goose_command: Option<&String>,
    ) {
        self.runtime_ready = true;
        self.model_progress = None;
        if let Some(console_url) = console_url.cloned() {
            self.webserver = Some(EndpointState {
                label: "Console".to_string(),
                status: RuntimeStatus::Ready,
                url: console_url,
                details: Vec::new(),
            });
        }
        let mut details = Vec::new();
        if let Some(pi_command) = pi_command.cloned() {
            details.push(format!("pi:    {pi_command}"));
        }
        if let Some(goose_command) = goose_command.cloned() {
            details.push(format!("goose: {goose_command}"));
        }
        self.api = Some(EndpointState {
            label: "OpenAI-compatible API".to_string(),
            status: RuntimeStatus::Ready,
            url: api_url.to_string(),
            details,
        });
    }

    pub(super) fn apply_endpoint_event(&mut self, event: &OutputEvent) -> bool {
        match event {
            OutputEvent::WebserverStarting { url } => {
                self.apply_endpoint_state("Console", RuntimeStatus::Starting, url, "Console");
            }
            OutputEvent::WebserverReady { url } => {
                self.apply_endpoint_state("Console", RuntimeStatus::Ready, url, "Console");
            }
            OutputEvent::ApiStarting { url } => {
                self.apply_endpoint_state(
                    "OpenAI-compatible API",
                    RuntimeStatus::Starting,
                    url,
                    "API",
                );
            }
            OutputEvent::ApiReady { url } => {
                self.apply_endpoint_state(
                    "OpenAI-compatible API",
                    RuntimeStatus::Ready,
                    url,
                    "API",
                );
            }
            OutputEvent::RuntimeReady {
                api_url,
                console_url,
                pi_command,
                goose_command,
                ..
            } => self.apply_runtime_ready_event(
                api_url,
                console_url.as_ref(),
                pi_command.as_ref(),
                goose_command.as_ref(),
            ),
            _ => return false,
        }
        true
    }

    pub(super) fn apply_output_event(&mut self, event: &OutputEvent) {
        self.record_startup_history_event(event);

        if self.shutdown_in_progress && is_shutdown_suppressed_ready_event(event) {
            return;
        }

        match event {
            OutputEvent::Startup { version, .. } => {
                self.version = Some(version.clone());
                self.runtime_ready = false;
                self.launch_plan = None;
                self.ready_llama_process_rows.clear();
            }
            OutputEvent::LaunchPlan { plan } => {
                self.launch_plan = Some(plan.clone());
                self.preseed_launch_plan_rows(plan);
            }
            OutputEvent::NodeIdentity { node_id, mesh_id } => {
                self.node_id = Some(node_id.clone());
                self.mesh_id = mesh_id.clone();
            }
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => self.apply_passive_mode_event(
                role,
                status,
                *capacity_gb,
                models_on_disk.as_ref(),
                detail.as_ref(),
            ),
            OutputEvent::MultiModelMode { count, models } => {
                self.multi_model_mode = Some(MultiModelModeState {
                    count: *count,
                    models: models.clone(),
                });
            }
            OutputEvent::ModelDownloadProgress {
                label,
                file,
                downloaded_bytes,
                total_bytes,
                status,
            } => {
                self.model_progress = Some(ModelProgressState {
                    label: label.clone(),
                    file: file.clone(),
                    downloaded_bytes: *downloaded_bytes,
                    total_bytes: *total_bytes,
                    status: status.clone(),
                });
            }
            OutputEvent::ShutdownRequested { .. } | OutputEvent::Shutdown { .. } => {
                self.mark_runtime_shutting_down();
            }
            OutputEvent::Error { .. } => {}
            OutputEvent::InviteToken {
                token,
                mesh_id,
                mesh_name,
            } => {
                self.join_token = Some(DashboardJoinTokenState::new(
                    token.clone(),
                    mesh_id.clone(),
                    mesh_name.clone(),
                ));
                let join_token_view = self.panel_view_state_mut(DashboardPanel::JoinToken);
                join_token_view.scroll_offset = 0;
                join_token_view.selected_row = None;
            }
            OutputEvent::PeerJoined { peer_id, .. } => {
                self.peer_ids.insert(peer_id.clone());
            }
            OutputEvent::PeerLeft { peer_id, .. } => {
                self.peer_ids.remove(peer_id);
            }
            OutputEvent::Info { .. }
            | OutputEvent::Warning { .. }
            | OutputEvent::RpcServerStarting { .. }
            | OutputEvent::RpcReady { .. }
            | OutputEvent::RpcStartupFailed { .. }
            | OutputEvent::DiscoveryStarting { .. }
            | OutputEvent::MeshFound { .. }
            | OutputEvent::DiscoveryJoined { .. }
            | OutputEvent::DiscoveryFailed { .. }
            | OutputEvent::WaitingForPeers { .. }
            | OutputEvent::RequestRouted { .. }
            | OutputEvent::LlamaNativeLog { .. } => {}
            _ if self.apply_model_event(event)
                || self.apply_llama_event(event)
                || self.apply_endpoint_event(event) => {}
            _ => {}
        }

        self.apply_startup_lifecycle_event(event);
        self.sync_truthful_startup_statuses();
        self.apply_startup_progress_event(event);
        self.record_mesh_event(event);
        self.clamp_all_panel_views();
        self.sync_events_panel();
    }

    pub(in crate::output) fn panel_view_state(
        &self,
        panel: DashboardPanel,
    ) -> DashboardPanelViewState {
        self.panel_view_states[panel.index()]
    }

    pub(in crate::output) fn panel_view_state_mut(
        &mut self,
        panel: DashboardPanel,
    ) -> &mut DashboardPanelViewState {
        &mut self.panel_view_states[panel.index()]
    }

    pub(in crate::output) fn filtered_mesh_events(&self) -> Vec<&MeshEventState> {
        if !self.events_filter.is_active() {
            return self.mesh_events.iter().collect();
        }

        let needle = self.events_filter.query.to_lowercase();
        self.mesh_events
            .iter()
            .filter(|event| event_matches_filter(event, &needle))
            .collect()
    }

    pub(in crate::output) fn row_count_for_panel(&self, panel: DashboardPanel) -> usize {
        match panel {
            DashboardPanel::JoinToken => self
                .join_token
                .as_ref()
                .map(|join_token| join_token_char_count(&join_token.token))
                .unwrap_or(0),
            DashboardPanel::Events => self.filtered_mesh_events().len(),
            DashboardPanel::LlamaCpp => self.llama_process_rows.len(),
            DashboardPanel::Webserver => self.webserver_rows.len(),
            DashboardPanel::Models => self.loaded_model_rows.len(),
            DashboardPanel::Requests => {
                usize::from(!self.request_history.accepted_request_buckets.is_empty())
            }
        }
    }

    pub(super) fn rows_are_selectable_for_panel(&self, panel: DashboardPanel) -> bool {
        self.panel_layout.rows_are_selectable_for(panel)
    }

    pub(super) fn clamp_all_panel_views(&mut self) {
        for panel in DashboardPanel::ALL {
            self.clamp_panel_view(panel);
        }
    }

    pub(super) fn clamp_panel_view(&mut self, panel: DashboardPanel) {
        let row_count = self.row_count_for_panel(panel);
        let rows_are_selectable = self.rows_are_selectable_for_panel(panel);
        let panel_view = self.panel_view_state_mut(panel);
        let viewport_rows = panel_view.viewport_rows.max(1);

        if row_count == 0 {
            panel_view.scroll_offset = 0;
            panel_view.selected_row = None;
            return;
        }

        let max_scroll_offset = row_count.saturating_sub(viewport_rows);
        panel_view.scroll_offset = panel_view.scroll_offset.min(max_scroll_offset);
        if !rows_are_selectable {
            panel_view.selected_row = None;
            return;
        }
        panel_view.selected_row = panel_view
            .selected_row
            .map(|selected| selected.min(row_count - 1));

        if panel == DashboardPanel::Events
            && TuiEventListRenderer::ACTIVE == TuiEventListRenderer::Scrollbar
        {
            return;
        }

        if let Some(selected_row) = panel_view.selected_row {
            if selected_row < panel_view.scroll_offset {
                panel_view.scroll_offset = selected_row;
            }
            let visible_end = panel_view.scroll_offset + viewport_rows;
            if selected_row >= visible_end {
                panel_view.scroll_offset = selected_row + 1 - viewport_rows;
            }
            panel_view.scroll_offset = panel_view.scroll_offset.min(max_scroll_offset);
        }
    }

    pub(super) fn sync_events_panel(&mut self) {
        if !self.events_follow {
            self.clamp_panel_view(DashboardPanel::Events);
            return;
        }

        let row_count = self.filtered_mesh_events().len();
        let events_view = self.panel_view_state_mut(DashboardPanel::Events);
        if row_count == 0 {
            events_view.scroll_offset = 0;
            events_view.selected_row = None;
            return;
        }

        let viewport_rows = events_view.viewport_rows.max(1);
        events_view.selected_row = Some(row_count - 1);
        events_view.scroll_offset = row_count.saturating_sub(viewport_rows);
    }

    pub(super) fn event_scroll_bounds(&self) -> (usize, usize, usize) {
        let row_count = self.row_count_for_panel(DashboardPanel::Events);
        let viewport_rows = self
            .panel_view_state(DashboardPanel::Events)
            .viewport_rows
            .max(1);
        let max_scroll_offset = row_count.saturating_sub(viewport_rows);
        (row_count, viewport_rows, max_scroll_offset)
    }

    pub(super) fn scroll_events_by(&mut self, delta: isize) {
        let (row_count, _viewport_rows, max_scroll_offset) = self.event_scroll_bounds();
        let was_following = self.events_follow;
        let current_scroll = if was_following {
            max_scroll_offset
        } else {
            self.panel_view_state(DashboardPanel::Events)
                .scroll_offset
                .min(max_scroll_offset)
        };
        let events_view = self.panel_view_state_mut(DashboardPanel::Events);
        if row_count == 0 {
            events_view.scroll_offset = 0;
            events_view.selected_row = None;
            self.events_follow = true;
            return;
        }

        let next_scroll = current_scroll
            .saturating_add_signed(delta)
            .min(max_scroll_offset);
        events_view.scroll_offset = next_scroll;
        events_view.selected_row = row_count.checked_sub(1);
        self.events_follow = next_scroll == max_scroll_offset;
    }

    pub(super) fn page_events_by(&mut self, direction: isize) {
        let (_row_count, viewport_rows, _max_scroll_offset) = self.event_scroll_bounds();
        let step = viewport_rows.saturating_sub(1).max(1) as isize;
        self.scroll_events_by(direction.saturating_mul(step));
    }

    pub(super) fn jump_events_to_start(&mut self) {
        let (row_count, _viewport_rows, _max_scroll_offset) = self.event_scroll_bounds();
        let events_view = self.panel_view_state_mut(DashboardPanel::Events);
        if row_count == 0 {
            events_view.scroll_offset = 0;
            events_view.selected_row = None;
            self.events_follow = true;
        } else {
            events_view.scroll_offset = 0;
            events_view.selected_row = row_count.checked_sub(1);
            self.events_follow = false;
        }
    }

    pub(super) fn jump_events_to_end(&mut self) {
        self.events_follow = true;
        self.sync_events_panel();
    }

    pub(super) fn move_panel_selection(&mut self, panel: DashboardPanel, delta: isize) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }

        if !self.rows_are_selectable_for_panel(panel) {
            self.scroll_panel_rows_by(panel, delta);
            return;
        }

        let current = self
            .panel_view_state(panel)
            .selected_row
            .unwrap_or_else(|| {
                if delta.is_negative() || (panel == DashboardPanel::Events && self.events_follow) {
                    row_count - 1
                } else {
                    0
                }
            });

        let next = current.saturating_add_signed(delta).min(row_count - 1);
        self.panel_view_state_mut(panel).selected_row = Some(next);
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    pub(super) fn page_panel_selection(&mut self, panel: DashboardPanel, direction: isize) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }

        let current_view = self.panel_view_state(panel);
        let step = self
            .panel_view_state(panel)
            .viewport_rows
            .saturating_sub(1)
            .max(1) as isize;
        let delta = direction.saturating_mul(step);
        if !self.rows_are_selectable_for_panel(panel) {
            self.scroll_panel_rows_by(panel, delta);
            return;
        }
        let current_selection = current_view.selected_row.unwrap_or_else(|| {
            if direction.is_negative() || (panel == DashboardPanel::Events && self.events_follow) {
                row_count - 1
            } else {
                0
            }
        });
        let next_selection = current_selection
            .saturating_add_signed(delta)
            .min(row_count - 1);
        let next_scroll = current_view.scroll_offset.saturating_add_signed(delta);
        let panel_view = self.panel_view_state_mut(panel);
        panel_view.selected_row = Some(next_selection);
        panel_view.scroll_offset = next_scroll;
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    pub(super) fn jump_panel_selection_to_start(&mut self, panel: DashboardPanel) {
        if self.row_count_for_panel(panel) == 0 {
            return;
        }
        if !self.rows_are_selectable_for_panel(panel) {
            let panel_view = self.panel_view_state_mut(panel);
            panel_view.scroll_offset = 0;
            panel_view.selected_row = None;
            return;
        }
        self.panel_view_state_mut(panel).selected_row = Some(0);
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    pub(super) fn jump_panel_selection_to_end(&mut self, panel: DashboardPanel) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }
        if !self.rows_are_selectable_for_panel(panel) {
            let viewport_rows = self.panel_view_state(panel).viewport_rows.max(1);
            let panel_view = self.panel_view_state_mut(panel);
            panel_view.scroll_offset = row_count.saturating_sub(viewport_rows);
            panel_view.selected_row = None;
            return;
        }
        self.panel_view_state_mut(panel).selected_row = Some(row_count - 1);
        self.clamp_panel_view(panel);
        self.sync_follow_with_events_view(panel);
    }

    pub(super) fn scroll_panel_rows_by(&mut self, panel: DashboardPanel, delta: isize) {
        let row_count = self.row_count_for_panel(panel);
        if row_count == 0 {
            return;
        }
        let current_view = self.panel_view_state(panel);
        let max_scroll_offset = row_count.saturating_sub(current_view.viewport_rows.max(1));
        let next_scroll = current_view
            .scroll_offset
            .saturating_add_signed(delta)
            .min(max_scroll_offset);
        let panel_view = self.panel_view_state_mut(panel);
        panel_view.scroll_offset = next_scroll;
        panel_view.selected_row = None;
    }

    pub(super) fn join_token_viewport_columns(&self) -> usize {
        let Some((columns, rows)) = self.terminal_size else {
            return 1;
        };
        let areas = tui_layout(
            Rect {
                x: 0,
                y: 0,
                width: columns,
                height: rows,
            },
            self,
        );
        usize::from(join_token_content_width(
            areas.join_token_panel,
            areas.join_token_copy_button,
        ))
        .max(1)
    }

    pub(super) fn scroll_join_token_by(&mut self, delta: isize) {
        let row_count = self.row_count_for_panel(DashboardPanel::JoinToken);
        if row_count == 0 {
            return;
        }
        let viewport_columns = self.join_token_viewport_columns();
        let max_scroll_offset = row_count.saturating_sub(viewport_columns);
        let current = self
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset
            .min(max_scroll_offset);
        let next = current.saturating_add_signed(delta).min(max_scroll_offset);
        let join_token_view = self.panel_view_state_mut(DashboardPanel::JoinToken);
        join_token_view.viewport_rows = viewport_columns.max(1);
        join_token_view.scroll_offset = next;
        join_token_view.selected_row = None;
    }

    pub(super) fn jump_join_token_to_start(&mut self) {
        self.panel_view_state_mut(DashboardPanel::JoinToken)
            .scroll_offset = 0;
    }

    pub(super) fn jump_join_token_to_end(&mut self) {
        let row_count = self.row_count_for_panel(DashboardPanel::JoinToken);
        let viewport_columns = self.join_token_viewport_columns();
        let max_scroll_offset = row_count.saturating_sub(viewport_columns);
        let join_token_view = self.panel_view_state_mut(DashboardPanel::JoinToken);
        join_token_view.viewport_rows = viewport_columns.max(1);
        join_token_view.scroll_offset = max_scroll_offset;
        join_token_view.selected_row = None;
    }

    pub(super) fn sync_follow_with_events_view(&mut self, panel: DashboardPanel) {
        if panel != DashboardPanel::Events {
            return;
        }

        let row_count = self.row_count_for_panel(DashboardPanel::Events);
        if row_count == 0 {
            self.events_follow = true;
            return;
        }

        let view = self.panel_view_state(DashboardPanel::Events);
        let viewport_rows = view.viewport_rows.max(1);
        if row_count <= viewport_rows {
            if view.selected_row != Some(row_count - 1) {
                self.events_follow = false;
            }
            return;
        }

        let last_row = row_count - 1;
        let at_bottom =
            view.selected_row == Some(last_row) && view.scroll_offset + viewport_rows >= row_count;
        self.events_follow = at_bottom;
        if self.events_follow {
            self.sync_events_panel();
        }
    }

    pub(super) fn record_mesh_event(&mut self, event: &OutputEvent) {
        self.mesh_events.push_back(MeshEventState {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            level: event.level(),
            summary: event.summary_line(),
        });
        while self.mesh_events.len() > self.mesh_event_limit {
            self.mesh_events.pop_front();
        }
    }

    pub(super) fn record_startup_history_event(&mut self, event: &OutputEvent) {
        if self.shutdown_in_progress && is_shutdown_suppressed_ready_event(event) {
            return;
        }

        if matches!(event, OutputEvent::Startup { .. }) {
            self.startup_history.clear();
        }

        let Some(summary) = startup_history_summary(event) else {
            return;
        };

        self.startup_history.push_back(MeshEventState {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            level: event.level(),
            summary,
        });
        while self.startup_history.len() > self.startup_history_limit {
            self.startup_history.pop_front();
        }
    }

    pub(super) fn copy_join_token(&mut self) {
        let Some(token) = self
            .join_token
            .as_ref()
            .map(|join_token| join_token.token.clone())
        else {
            return;
        };
        let now = Instant::now();
        let copy_status = match copy_join_token_to_clipboard(&token) {
            Ok(()) => DashboardJoinTokenCopyStatus::Copied { at: now },
            Err(message) => DashboardJoinTokenCopyStatus::Failed { message, at: now },
        };
        self.reduce(DashboardAction::SetJoinTokenCopyStatus(copy_status));
    }

    pub(super) fn join_token_copy_shortcut_enabled(&self) -> bool {
        !self.events_filter.editing && self.join_token.is_some()
    }

    pub(in crate::output) fn clear_expired_join_token_copy_status(&mut self, now: Instant) -> bool {
        let Some(join_token) = self.join_token.as_mut() else {
            return false;
        };
        let Some(feedback_at) = join_token.copy_status.feedback_at() else {
            return false;
        };
        if now.saturating_duration_since(feedback_at) < PRETTY_TUI_JOIN_TOKEN_COPY_STATUS_TTL {
            return false;
        }
        join_token.copy_status = DashboardJoinTokenCopyStatus::Idle;
        true
    }

    pub(super) fn join_token_copy_button_contains(&self, column: u16, row: u16) -> bool {
        let Some((columns, rows)) = self.terminal_size else {
            return false;
        };
        if self.full_screen_panel == Some(DashboardPanel::JoinToken) {
            let panel_area = Rect::new(0, 0, columns, rows);
            return point_in_rect(column, row, tui_join_token_copy_button_area(panel_area));
        }
        let areas = tui_layout(
            Rect {
                x: 0,
                y: 0,
                width: columns,
                height: rows,
            },
            self,
        );
        point_in_rect(column, row, areas.join_token_copy_button)
    }

    pub(super) fn join_token_panel_contains(&self, column: u16, row: u16) -> bool {
        let Some((columns, rows)) = self.terminal_size else {
            return false;
        };
        if self.full_screen_panel == Some(DashboardPanel::JoinToken) {
            return point_in_rect(column, row, Rect::new(0, 0, columns, rows));
        }
        let areas = tui_layout(
            Rect {
                x: 0,
                y: 0,
                width: columns,
                height: rows,
            },
            self,
        );
        point_in_rect(column, row, areas.join_token_panel)
    }

    pub(in crate::output) fn apply_tui_event(&mut self, event: TuiEvent) -> TuiControlFlow {
        if let Some(flow) = self.apply_resize_tui_event(event) {
            return flow;
        }
        if let Some(flow) = self.apply_mouse_tui_event(event) {
            return flow;
        }
        if let Some(flow) = self.apply_global_tui_key_event(event) {
            return flow;
        }
        if let Some(flow) = self.apply_join_token_tui_key_event(event) {
            return flow;
        }
        if let Some(flow) = self.apply_requests_tui_key_event(event) {
            return flow;
        }
        if let Some(flow) = self.apply_events_scroll_tui_key_event(event) {
            return flow;
        }
        if let Some(flow) = self.apply_panel_navigation_tui_key_event(event) {
            return flow;
        }
        if let Some(flow) = self.apply_events_filter_tui_key_event(event) {
            return flow;
        }
        TuiControlFlow::Continue
    }

    pub(super) fn apply_resize_tui_event(&mut self, event: TuiEvent) -> Option<TuiControlFlow> {
        let TuiEvent::Resize { columns, rows } = event else {
            return None;
        };
        self.terminal_size = Some((columns, rows));
        self.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
            columns, rows,
        )));
        Some(TuiControlFlow::Continue)
    }

    pub(super) fn apply_mouse_tui_event(&mut self, event: TuiEvent) -> Option<TuiControlFlow> {
        let TuiEvent::MouseDown { column, row } = event else {
            return None;
        };
        if self.join_token_copy_button_contains(column, row) {
            self.panel_focus = DashboardPanel::JoinToken;
            self.copy_join_token();
            return Some(TuiControlFlow::Continue);
        }
        if self.join_token_panel_contains(column, row) {
            self.panel_focus = DashboardPanel::JoinToken;
            self.events_filter.editing = false;
            return Some(TuiControlFlow::Continue);
        }
        None
    }

    pub(super) fn apply_global_tui_key_event(&mut self, event: TuiEvent) -> Option<TuiControlFlow> {
        match event {
            TuiEvent::Key(TuiKeyEvent::Escape)
                if !self.events_filter.editing && self.full_screen_panel.is_some() =>
            {
                self.reduce(DashboardAction::ExitFullScreenPanel);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Interrupt) => {
                self.mark_runtime_shutting_down();
                Some(TuiControlFlow::Quit)
            }
            TuiEvent::Key(TuiKeyEvent::Char('q')) if !self.events_filter.editing => {
                self.mark_runtime_shutting_down();
                Some(TuiControlFlow::Quit)
            }
            TuiEvent::Key(TuiKeyEvent::Tab) => {
                self.reduce(DashboardAction::FocusNextPanel);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::BackTab) => {
                self.reduce(DashboardAction::FocusPreviousPanel);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Enter) | TuiEvent::Key(TuiKeyEvent::Char('z'))
                if !self.events_filter.editing =>
            {
                self.reduce(DashboardAction::ToggleFullScreenPanel);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('/')) if !self.events_filter.editing => {
                self.reduce(DashboardAction::StartEventsFilterEdit);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('f')) if !self.events_filter.editing => {
                self.reduce(DashboardAction::ToggleEventsFollow);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('c')) if self.join_token_copy_shortcut_enabled() => {
                self.copy_join_token();
                Some(TuiControlFlow::Continue)
            }
            _ => None,
        }
    }

    pub(super) fn apply_join_token_tui_key_event(
        &mut self,
        event: TuiEvent,
    ) -> Option<TuiControlFlow> {
        if self.events_filter.editing || self.panel_focus != DashboardPanel::JoinToken {
            return None;
        }
        match event {
            TuiEvent::Key(TuiKeyEvent::Left) | TuiEvent::Key(TuiKeyEvent::Char('h')) => {
                self.scroll_join_token_by(-1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Right) | TuiEvent::Key(TuiKeyEvent::Char('l')) => {
                self.scroll_join_token_by(1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('g')) => {
                self.jump_join_token_to_start();
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('G')) => {
                self.jump_join_token_to_end();
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Up)
            | TuiEvent::Key(TuiKeyEvent::Char('k'))
            | TuiEvent::Key(TuiKeyEvent::Down)
            | TuiEvent::Key(TuiKeyEvent::Char('j'))
            | TuiEvent::Key(TuiKeyEvent::PageUp)
            | TuiEvent::Key(TuiKeyEvent::PageDown) => Some(TuiControlFlow::Continue),
            _ => None,
        }
    }

    pub(super) fn apply_requests_tui_key_event(
        &mut self,
        event: TuiEvent,
    ) -> Option<TuiControlFlow> {
        if self.events_filter.editing || self.panel_focus != DashboardPanel::Requests {
            return None;
        }
        match event {
            TuiEvent::Key(TuiKeyEvent::Up) => {
                self.reduce(DashboardAction::SelectPreviousRequestWindow);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Down) => {
                self.reduce(DashboardAction::SelectNextRequestWindow);
                Some(TuiControlFlow::Continue)
            }
            _ => None,
        }
    }

    pub(super) fn apply_events_scroll_tui_key_event(
        &mut self,
        event: TuiEvent,
    ) -> Option<TuiControlFlow> {
        if self.events_filter.editing
            || self.panel_focus != DashboardPanel::Events
            || TuiEventListRenderer::ACTIVE != TuiEventListRenderer::Scrollbar
        {
            return None;
        }
        match event {
            TuiEvent::Key(TuiKeyEvent::Up) | TuiEvent::Key(TuiKeyEvent::Char('k')) => {
                self.scroll_events_by(-1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Down) | TuiEvent::Key(TuiKeyEvent::Char('j')) => {
                self.scroll_events_by(1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::PageUp) => {
                self.page_events_by(-1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::PageDown) => {
                self.page_events_by(1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('g')) => {
                self.jump_events_to_start();
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('G')) => {
                self.jump_events_to_end();
                Some(TuiControlFlow::Continue)
            }
            _ => None,
        }
    }

    pub(super) fn apply_panel_navigation_tui_key_event(
        &mut self,
        event: TuiEvent,
    ) -> Option<TuiControlFlow> {
        if self.events_filter.editing {
            return None;
        }
        match event {
            TuiEvent::Key(TuiKeyEvent::Left)
            | TuiEvent::Key(TuiKeyEvent::Char('h'))
            | TuiEvent::Key(TuiKeyEvent::Up)
            | TuiEvent::Key(TuiKeyEvent::Char('k')) => {
                self.move_panel_selection(self.panel_focus, -1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Right)
            | TuiEvent::Key(TuiKeyEvent::Char('l'))
            | TuiEvent::Key(TuiKeyEvent::Down)
            | TuiEvent::Key(TuiKeyEvent::Char('j')) => {
                self.move_panel_selection(self.panel_focus, 1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::PageUp) => {
                self.page_panel_selection(self.panel_focus, -1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::PageDown) => {
                self.page_panel_selection(self.panel_focus, 1);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('g')) => {
                self.jump_panel_selection_to_start(self.panel_focus);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char('G')) => {
                self.jump_panel_selection_to_end(self.panel_focus);
                Some(TuiControlFlow::Continue)
            }
            _ => None,
        }
    }

    pub(super) fn apply_events_filter_tui_key_event(
        &mut self,
        event: TuiEvent,
    ) -> Option<TuiControlFlow> {
        if !self.events_filter.editing {
            return None;
        }
        match event {
            TuiEvent::Key(TuiKeyEvent::Backspace) => {
                self.reduce(DashboardAction::BackspaceEventsFilter);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Enter) => {
                self.reduce(DashboardAction::ConfirmEventsFilter);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Escape) => {
                self.reduce(DashboardAction::CancelEventsFilter);
                Some(TuiControlFlow::Continue)
            }
            TuiEvent::Key(TuiKeyEvent::Char(ch)) if !ch.is_control() => {
                self.reduce(DashboardAction::InsertEventsFilterChar(ch));
                Some(TuiControlFlow::Continue)
            }
            _ => None,
        }
    }
}
