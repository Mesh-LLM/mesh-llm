use super::{
    Constraint, DashboardPanel, DashboardState, Direction, Flex, Layout,
    PRETTY_TUI_EVENTS_COLUMN_PERCENT, PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT,
    PRETTY_TUI_REMAINING_COLUMN_WEIGHT, Rect, tui_join_token_copy_button_area, tui_logo_line_width,
    tui_processes_block, tui_ready_logo_text,
};

#[derive(Clone, Copy)]
pub(in crate::output) struct TuiFrameAreas {
    pub(in crate::output) loading: Option<Rect>,
    pub(in crate::output) logo: Option<Rect>,
    pub(in crate::output) join_token_panel: Rect,
    pub(in crate::output) join_token_copy_button: Rect,
    pub(in crate::output) main_body: Rect,
    pub(in crate::output) requests: (Rect, Rect),
    pub(in crate::output) status_bar: Rect,
    pub(in crate::output) events: (Rect, Rect),
    pub(in crate::output) processes: Rect,
    pub(in crate::output) llama_processes: (Rect, Rect),
    pub(in crate::output) webserver_processes: (Rect, Rect),
    pub(in crate::output) models: (Rect, Rect),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::output) struct TuiBandHeights {
    pub(in crate::output) join_token: u16,
    pub(in crate::output) main_body: u16,
    pub(in crate::output) requests: u16,
    pub(in crate::output) status: u16,
}

pub(in crate::output) fn tui_layout(area: Rect, state: &DashboardState) -> TuiFrameAreas {
    let zero = Rect {
        x: area.x,
        y: area.y,
        width: 0,
        height: 0,
    };

    if state.is_startup_loading() {
        return TuiFrameAreas {
            loading: Some(area),
            logo: None,
            join_token_panel: zero,
            join_token_copy_button: zero,
            main_body: zero,
            requests: (zero, zero),
            status_bar: zero,
            events: (zero, zero),
            processes: zero,
            llama_processes: (zero, zero),
            webserver_processes: (zero, zero),
            models: (zero, zero),
        };
    }

    let band_heights = tui_band_heights(area, state);
    let content_height = band_heights
        .main_body
        .saturating_add(band_heights.join_token)
        .saturating_add(band_heights.requests);
    let dashboard_height = content_height
        .saturating_add(band_heights.status)
        .min(area.height);
    let [slack_area, dashboard_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(dashboard_height)])
        .areas(area);
    let loading = (slack_area.height > 0).then_some(slack_area);
    let logo = (state.runtime_ready && slack_area.height > 0)
        .then(|| tui_centered_logo_area(slack_area))
        .flatten();

    let [content_area, status_band] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(content_height),
            Constraint::Length(band_heights.status),
        ])
        .areas(dashboard_area);

    let [join_token_panel, main_body, requests_band] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(band_heights.join_token),
            Constraint::Length(band_heights.main_body),
            Constraint::Length(band_heights.requests),
        ])
        .areas(content_area);

    let join_token_copy_button = tui_join_token_copy_button_area(join_token_panel);

    let [events_column, processes_column, models_column] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(PRETTY_TUI_EVENTS_COLUMN_PERCENT),
            Constraint::Fill(PRETTY_TUI_REMAINING_COLUMN_WEIGHT),
            Constraint::Fill(PRETTY_TUI_REMAINING_COLUMN_WEIGHT),
        ])
        .areas(main_body);
    let [events_title, events_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(events_column);
    let [models_title, models_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(models_column);

    let processes_block = tui_processes_block(state);
    let processes_inner = processes_block.inner(processes_column);
    let (llama_panel_height, webserver_panel_height) = tui_process_panel_heights(
        processes_inner.height,
        state.panel_layout.rows_for(DashboardPanel::LlamaCpp),
        state.panel_layout.rows_for(DashboardPanel::Webserver),
    );
    let [llama_panel, webserver_panel] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(llama_panel_height),
            Constraint::Length(webserver_panel_height),
        ])
        .areas(processes_inner);
    let [llama_title, llama_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(llama_panel);
    let [webserver_title, webserver_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(webserver_panel);
    let [requests_title, requests_body] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(requests_band);
    TuiFrameAreas {
        loading,
        logo,
        join_token_panel,
        join_token_copy_button,
        main_body,
        requests: (requests_title, requests_body),
        status_bar: status_band,
        events: (events_title, events_body),
        processes: processes_column,
        llama_processes: (llama_title, llama_body),
        webserver_processes: (webserver_title, webserver_body),
        models: (models_title, models_body),
    }
}

pub(in crate::output) fn tui_ready_logo_height(area: Rect) -> u16 {
    if area.height == 0 {
        return 0;
    }
    let desired = tui_ready_logo_text()
        .map(|text| u16::try_from(text.lines.len()).unwrap_or(u16::MAX))
        .unwrap_or_else(|| (area.height / 4).max(3));
    desired.min(area.height)
}

pub(in crate::output) fn tui_ready_logo_width(area: Rect) -> u16 {
    if area.width == 0 {
        return 0;
    }
    tui_ready_logo_text()
        .map(|text| {
            text.lines
                .iter()
                .map(tui_logo_line_width)
                .max()
                .and_then(|width| u16::try_from(width).ok())
                .unwrap_or(area.width)
                .min(area.width)
        })
        .unwrap_or(area.width)
}

pub(in crate::output) fn tui_centered_logo_area(area: Rect) -> Option<Rect> {
    let logo_width = tui_ready_logo_width(area);
    let logo_height = tui_ready_logo_height(area);
    if logo_width == 0 || logo_height == 0 {
        return None;
    }

    let [vertical] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(logo_height)])
        .flex(Flex::Center)
        .areas(area);
    let [centered] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(logo_width)])
        .flex(Flex::Center)
        .areas(vertical);
    Some(centered)
}

pub(in crate::output) fn tui_desired_main_body_height(state: &DashboardState) -> u16 {
    u16::try_from(
        state
            .panel_layout
            .rows_for(DashboardPanel::Events)
            .saturating_add(2)
            .max(
                state
                    .panel_layout
                    .rows_for(DashboardPanel::Models)
                    .saturating_add(2),
            )
            .max(
                state
                    .panel_layout
                    .rows_for(DashboardPanel::LlamaCpp)
                    .saturating_add(state.panel_layout.rows_for(DashboardPanel::Webserver))
                    .saturating_add(5),
            ),
    )
    .unwrap_or(u16::MAX)
}

pub(in crate::output) fn tui_desired_requests_band_height(state: &DashboardState) -> u16 {
    u16::try_from(
        state
            .panel_layout
            .rows_for(DashboardPanel::Requests)
            .saturating_add(2),
    )
    .unwrap_or(u16::MAX)
}

pub(in crate::output) fn tui_band_heights(area: Rect, state: &DashboardState) -> TuiBandHeights {
    let status = area.height.min(1);
    let remaining_after_status = area.height.saturating_sub(status);
    let join_token = PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT.min(remaining_after_status);
    let remaining_after_join_token = remaining_after_status.saturating_sub(join_token);
    let main_body_desired = tui_desired_main_body_height(state);
    let requests_desired = tui_desired_requests_band_height(state);
    let requests_min = remaining_after_join_token.min(5);
    let requests = requests_desired
        .min(remaining_after_join_token)
        .max(requests_min);
    let main_body = remaining_after_join_token
        .saturating_sub(requests)
        .min(main_body_desired);

    TuiBandHeights {
        join_token,
        main_body,
        requests,
        status,
    }
}

pub(in crate::output) fn tui_process_panel_heights(
    available_height: u16,
    desired_llama_rows: usize,
    desired_webserver_rows: usize,
) -> (u16, u16) {
    if available_height == 0 {
        return (0, 0);
    }

    let desired_llama_block =
        u16::try_from(desired_llama_rows.saturating_add(2)).unwrap_or(u16::MAX);
    let desired_webserver_block =
        u16::try_from(desired_webserver_rows.saturating_add(2)).unwrap_or(u16::MAX);
    let desired_total = desired_llama_block.saturating_add(desired_webserver_block);

    if available_height == 1 {
        return (1, 0);
    }

    if desired_total == 0 {
        let llama_block = available_height / 2;
        return (llama_block, available_height.saturating_sub(llama_block));
    }

    let layout_height = available_height;
    let minimum_llama = 2.min(layout_height);
    let minimum_webserver = u16::from(layout_height > minimum_llama);
    let flexible_height = layout_height
        .saturating_sub(minimum_llama)
        .saturating_sub(minimum_webserver);
    let desired_flexible = desired_total
        .saturating_sub(minimum_llama)
        .saturating_sub(minimum_webserver);
    let llama_flexible = flexible_height
        .saturating_mul(desired_llama_block.saturating_sub(minimum_llama))
        .checked_div(desired_flexible)
        .unwrap_or(flexible_height / 2);
    let llama_block = minimum_llama + llama_flexible;
    let webserver_block = layout_height.saturating_sub(llama_block);

    (llama_block, webserver_block)
}
