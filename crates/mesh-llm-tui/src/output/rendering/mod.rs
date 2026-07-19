#[cfg(test)]
use super::PRETTY_TUI_LIST_HIGHLIGHT_SYMBOL_WIDTH;
use super::{
    DashboardEventsFilterState, DashboardPanel, DashboardState, MeshEventState,
    PRETTY_TUI_EVENT_LEVEL_WIDTH, PRETTY_TUI_EVENTS_COLUMN_PERCENT,
    PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL, PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING,
    PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT, PRETTY_TUI_MIN_DASHBOARD_WIDTH,
    PRETTY_TUI_MODEL_CARD_HEIGHT, PRETTY_TUI_MODEL_CARD_STRIDE, PRETTY_TUI_READY_LOGO_TEXT,
    PRETTY_TUI_REMAINING_COLUMN_WEIGHT, PRETTY_TUI_REQUEST_GRAPH_BASELINE_SYMBOL,
    PRETTY_TUI_REQUEST_GRAPH_GUIDE_SYMBOL, PRETTY_TUI_SPLASH_ANSI, PRETTY_TUI_SPLASH_TEXT,
    RuntimeStatus, StartupLifecycleState, TuiEventListRenderer, tui_theme,
};
use chrono::Local;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    buffer::Buffer,
    layout::{Alignment, Constraint, Direction, Flex, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, BorderType, Cell, Clear as RatatuiClear, HighlightSpacing, Padding, Paragraph, Row,
        Scrollbar, ScrollbarOrientation, ScrollbarState, StatefulWidget, Table, TableState, Widget,
    },
};
#[cfg(test)]
use std::fmt::Write as _;
use std::io;
use tokio::time::Duration;

mod events;
mod join_token;
mod layout;
mod logo;
mod models;
mod processes;
mod requests;
mod text;
mod tui;

pub(in crate::output) use events::*;
pub(in crate::output) use join_token::*;
pub(in crate::output) use layout::*;
pub(in crate::output) use logo::*;
pub(in crate::output) use models::*;
pub(in crate::output) use processes::*;
pub(in crate::output) use requests::*;
pub(in crate::output) use text::*;
pub(in crate::output) use tui::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::output) struct TuiListScrollbarLayout {
    pub(in crate::output) list_area: Rect,
    pub(in crate::output) scrollbar_area: Option<Rect>,
}

pub(in crate::output) fn tui_list_scrollbar_layout(
    inner_area: Rect,
    row_count: usize,
    viewport_rows: usize,
) -> TuiListScrollbarLayout {
    let show_scrollbar = row_count > viewport_rows && inner_area.width > 1;
    let list_area = if show_scrollbar {
        Rect {
            width: inner_area.width.saturating_sub(1),
            ..inner_area
        }
    } else {
        inner_area
    };
    let scrollbar_area = show_scrollbar.then_some(Rect {
        x: inner_area.right().saturating_sub(1),
        y: inner_area.y,
        width: 1,
        height: inner_area.height,
    });
    TuiListScrollbarLayout {
        list_area,
        scrollbar_area,
    }
}

pub(in crate::output) fn tui_list_scrollbar_state(
    row_count: usize,
    viewport_rows: usize,
    scroll_offset: usize,
) -> ScrollbarState {
    let visible_rows = viewport_rows.min(row_count);
    let max_scroll_offset = row_count.saturating_sub(visible_rows);
    let clamped_offset = scroll_offset.min(max_scroll_offset);
    let scrollbar_position = clamped_offset
        .saturating_mul(row_count.saturating_sub(1))
        .checked_div(max_scroll_offset)
        .unwrap_or(0);
    ScrollbarState::new(row_count)
        .position(scrollbar_position)
        .viewport_content_length(visible_rows)
}

#[cfg(test)]
pub(in crate::output) fn render_tui_events_snapshot(
    state: &DashboardState,
    columns: u16,
    rows: u16,
) -> String {
    let width = usize::from(columns.max(40));
    let max_lines = usize::from(rows.max(3));
    let mut output = String::new();
    let _ = writeln!(&mut output, "{}", truncate_with_ellipsis("mesh-llm", width));
    let _ = writeln!(
        &mut output,
        "{}",
        truncate_with_ellipsis(
            &spans_plain_text(&dashboard_status_line(state, columns).spans),
            width
        )
    );
    let _ = writeln!(
        &mut output,
        "{}",
        truncate_with_ellipsis(
            &format_tui_panel_title(state, DashboardPanel::Events),
            width,
        )
    );

    for row in visible_event_rows(state, state.panel_layout.rows_for(DashboardPanel::Events)) {
        match row {
            TuiEventRow::Event { event, .. } => {
                let _ = writeln!(&mut output, "{}", format_event_row(event, width));
            }
            TuiEventRow::Message(message) => {
                let _ = writeln!(&mut output, "{}", truncate_with_ellipsis(message, width));
            }
            TuiEventRow::Padding => {
                let _ = writeln!(&mut output);
            }
        }
    }

    let mut lines: Vec<&str> = output.lines().collect();
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        let mut truncated = lines.join("\n");
        truncated.push('\n');
        return truncated;
    }

    output
}

#[derive(Clone, Copy)]
#[cfg(test)]
pub(super) enum TuiEventRow<'a> {
    Event {
        absolute_index: usize,
        event: &'a MeshEventState,
    },
    Message(&'static str),
    Padding,
}

pub(in crate::output) type TuiTerminal = Terminal<CrosstermBackend<io::Stderr>>;
pub(in crate::output) fn dashboard_status_line(
    state: &DashboardState,
    width: u16,
) -> Line<'static> {
    let theme = tui_theme();
    let readiness = readiness_label(state);
    let mut left_spans = vec![Span::styled(
        readiness_badge(readiness),
        readiness_badge_style(readiness),
    )];
    left_spans.push(Span::raw(" "));
    push_status_key_hint(&mut left_spans, "Q", "Quit");
    push_status_key_hint(&mut left_spans, "Tab", "Next");
    push_status_key_hint(&mut left_spans, "Enter/Z", "Full");
    push_status_key_hint(&mut left_spans, "↑/↓", "Window");
    push_status_key_hint(&mut left_spans, "Shift-Tab", "Prev");
    push_status_key_hint(&mut left_spans, "/", "Filter");
    push_status_key_hint(&mut left_spans, "F", "Follow");
    push_status_key_hint(&mut left_spans, "R", "Refresh");

    let mut right_spans = Vec::new();
    push_status_metric(&mut right_spans, "peers", state.peer_ids.len().to_string());
    push_status_metric(
        &mut right_spans,
        "models",
        visible_model_count(state).to_string(),
    );
    push_status_metric(
        &mut right_spans,
        "processes",
        visible_process_count(state).to_string(),
    );
    push_status_metric(&mut right_spans, "uptime", dashboard_uptime_label(state));
    right_spans.push(status_separator_span());
    right_spans.push(Span::styled(
        Local::now().format("%H:%M:%S").to_string(),
        Style::default().fg(theme.muted),
    ));

    let mut spans = left_spans;
    let left_width = status_spans_width(&spans);
    let right_width = status_spans_width(&right_spans);
    let gap_width = usize::from(width)
        .saturating_sub(left_width)
        .saturating_sub(right_width)
        .max(1);
    spans.push(status_gap_span(gap_width));
    spans.extend(right_spans);

    Line::from(spans)
}

pub(in crate::output) fn status_spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|span| span.content.chars().count()).sum()
}

pub(in crate::output) fn status_gap_span(width: usize) -> Span<'static> {
    Span::raw(" ".repeat(width))
}

pub(in crate::output) fn push_status_metric(
    spans: &mut Vec<Span<'static>>,
    label: &'static str,
    value: String,
) {
    let theme = tui_theme();
    spans.push(status_separator_span());
    spans.push(Span::styled(
        format!("{label}: "),
        Style::default().fg(theme.dim),
    ));
    spans.push(Span::styled(value, Style::default().fg(theme.text)));
}

pub(in crate::output) fn status_separator_span() -> Span<'static> {
    Span::styled(" | ", Style::default().fg(tui_theme().dim))
}

pub(in crate::output) fn push_status_key_hint(
    spans: &mut Vec<Span<'static>>,
    key: &'static str,
    label: &'static str,
) {
    spans.push(key_hint_span(key));
    spans.push(Span::raw(" "));
    spans.push(hint_label_span(label));
    spans.push(Span::raw(" "));
}

pub(in crate::output) fn readiness_badge(readiness: &str) -> String {
    format!(" {} ", readiness.to_ascii_uppercase())
}

pub(in crate::output) fn readiness_badge_style(readiness: &str) -> Style {
    let theme = tui_theme();
    let color = match readiness {
        "ready" => theme.success,
        "degraded" => theme.warning,
        "starting" | "warming" => theme.accent_soft,
        "stopped" => theme.dim,
        _ => theme.muted,
    };
    Style::default()
        .fg(color)
        .bg(theme.surface)
        .add_modifier(Modifier::BOLD)
}

pub(in crate::output) fn dashboard_uptime_label(state: &DashboardState) -> String {
    format_duration_compact(state.session_started_at.elapsed())
}

pub(in crate::output) fn format_duration_compact(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

pub(in crate::output) fn key_hint_span(key: &'static str) -> Span<'static> {
    let theme = tui_theme();
    Span::styled(
        format!("[{key}]"),
        Style::default()
            .fg(theme.accent)
            .bg(theme.surface_raised)
            .add_modifier(Modifier::BOLD),
    )
}

pub(in crate::output) fn hint_label_span(label: &'static str) -> Span<'static> {
    Span::styled(label.to_string(), Style::default().fg(tui_theme().muted))
}

pub(in crate::output) fn format_tui_panel_title(
    state: &DashboardState,
    panel: DashboardPanel,
) -> String {
    let focus_marker = if state.panel_focus == panel {
        '▶'
    } else {
        ' '
    };
    let mut title = match panel {
        DashboardPanel::JoinToken => join_token_panel_left_title(state, focus_marker),
        DashboardPanel::Events => format!(
            "{focus_marker} Mesh Events  follow={}  filter={}",
            if state.events_follow { "ON" } else { "OFF" },
            events_filter_label(&state.events_filter)
        ),
        DashboardPanel::LlamaCpp => format!("{focus_marker} llama.cpp Processes"),
        DashboardPanel::Webserver => format!("{focus_marker} mesh-llm Processes"),
        DashboardPanel::Models => format!("{focus_marker} Loaded Models"),
        DashboardPanel::Requests => format!(
            "{focus_marker} Incoming Requests  {}  {}",
            state.request_window.label(),
            state.request_window.bucket_label()
        ),
    };
    if state.full_screen_panel == Some(panel) {
        title.push_str("  fullscreen  Esc=Back");
    }
    title
}

pub(in crate::output) fn panel_title_style(state: &DashboardState, panel: DashboardPanel) -> Style {
    let theme = tui_theme();
    if state.panel_focus == panel {
        Style::default()
            .fg(theme.accent)
            .bg(theme.surface_raised)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.dim).add_modifier(Modifier::DIM)
    }
}

pub(in crate::output) fn panel_border_style(
    state: &DashboardState,
    panel: DashboardPanel,
) -> Style {
    let theme = tui_theme();
    if state.panel_focus == panel {
        Style::default().fg(theme.accent)
    } else {
        Style::default().fg(theme.dim)
    }
}
pub(in crate::output) fn readiness_label(state: &DashboardState) -> &'static str {
    if state.runtime_ready {
        "ready"
    } else if state.llama_instances.iter().any(|instance| {
        matches!(
            instance.status,
            RuntimeStatus::Error | RuntimeStatus::Warning
        )
    }) || state
        .running_models
        .iter()
        .any(|model| matches!(model.status, RuntimeStatus::Error | RuntimeStatus::Warning))
        || state
            .loaded_model_rows
            .iter()
            .any(|row| matches!(row.status, RuntimeStatus::Error | RuntimeStatus::Warning))
        || state
            .webserver_rows
            .iter()
            .any(|row| matches!(row.status, RuntimeStatus::Error | RuntimeStatus::Warning))
    {
        "degraded"
    } else if state.llama_instances.iter().any(|instance| {
        matches!(
            instance.status,
            RuntimeStatus::Starting | RuntimeStatus::Loading
        )
    }) || state.running_models.iter().any(|model| {
        matches!(
            model.status,
            RuntimeStatus::Starting | RuntimeStatus::Loading
        )
    }) || state
        .loaded_model_rows
        .iter()
        .any(|row| matches!(row.status, RuntimeStatus::Starting | RuntimeStatus::Loading))
        || state
            .webserver_rows
            .iter()
            .any(|row| matches!(row.status, RuntimeStatus::Starting | RuntimeStatus::Loading))
    {
        "starting"
    } else if state
        .llama_instances
        .iter()
        .all(|instance| matches!(instance.status, RuntimeStatus::Stopped))
        && state
            .running_models
            .iter()
            .all(|model| matches!(model.status, RuntimeStatus::Stopped))
        && !matches!(
            state.webserver.as_ref().map(|endpoint| &endpoint.status),
            Some(RuntimeStatus::Ready)
        )
        && !matches!(
            state.api.as_ref().map(|endpoint| &endpoint.status),
            Some(RuntimeStatus::Ready)
        )
    {
        "stopped"
    } else {
        "warming"
    }
}

pub(in crate::output) fn visible_process_count(state: &DashboardState) -> usize {
    let snapshot_processes = state.llama_process_rows.len() + state.webserver_rows.len();
    if snapshot_processes > 0 {
        snapshot_processes
    } else {
        state.llama_instances.len()
            + usize::from(state.webserver.is_some())
            + usize::from(state.api.is_some())
    }
}

pub(in crate::output) fn visible_model_count(state: &DashboardState) -> usize {
    if !state.loaded_model_rows.is_empty() {
        state.loaded_model_rows.len()
    } else {
        state.running_models.len()
    }
}

pub(in crate::output) fn events_filter_label(filter: &DashboardEventsFilterState) -> String {
    if filter.editing {
        format!("/{query}_", query = filter.query)
    } else if filter.query.is_empty() {
        "(none)".to_string()
    } else {
        format!("/{query}", query = filter.query)
    }
}

pub(in crate::output) fn truncate_with_ellipsis(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let count = text.chars().count();
    if count <= width {
        return text.to_string();
    }
    if width == 1 {
        return "…".to_string();
    }
    text.chars().take(width - 1).collect::<String>() + "…"
}
