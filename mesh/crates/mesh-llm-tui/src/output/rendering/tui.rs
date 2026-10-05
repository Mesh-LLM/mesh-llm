use super::{
    Alignment, Block, BorderType, Constraint, DashboardPanel, DashboardState, Direction, Frame,
    Layout, Line, PRETTY_TUI_MIN_DASHBOARD_WIDTH, Paragraph, RatatuiClear, Rect, Span, Style,
    TuiTerminal, dashboard_status_line, render_events_panel, render_join_token_panel,
    render_model_progress_loader, render_models_panel, render_process_table,
    render_processes_panel, render_requests_panel, render_tui_logo, tui_layout, tui_theme,
};
use ratatui::backend::Backend;
use std::{fmt::Display, io};

pub(in crate::output) fn draw_tui_dashboard_with_terminal(
    terminal: &mut TuiTerminal,
    state: &DashboardState,
) -> io::Result<()> {
    draw_tui_dashboard_with_backend(terminal, state)
}

pub(in crate::output) fn draw_tui_dashboard_with_backend<B>(
    terminal: &mut ratatui::Terminal<B>,
    state: &DashboardState,
) -> io::Result<()>
where
    B: Backend,
    B::Error: Display,
{
    terminal
        .hide_cursor()
        .map_err(|error| io::Error::other(error.to_string()))?;
    terminal
        .set_cursor_position((0, 0))
        .map_err(|error| io::Error::other(error.to_string()))?;
    terminal
        .draw(|frame| render_tui_frame(frame, state))
        .map(|_| ())
        .map_err(|error| io::Error::other(error.to_string()))
}

/// Repair a physically desynchronized screen.
///
/// Ratatui diffs against its own idea of the screen, so once something else has
/// written to the terminal the damaged cells are never redrawn — they match the
/// buffer ratatui believes is displayed. Erasing the real screen and discarding
/// both internal buffers forces the next draw to emit every cell.
///
/// This runs only when the operator asks for it with `R`. Doing it on every
/// frame also works, but repaints the whole screen from blank ~2.6 times a
/// second on an idle dashboard, which reads as a black blink.
pub(in crate::output) fn repair_tui_terminal<B>(
    terminal: &mut ratatui::Terminal<B>,
) -> io::Result<()>
where
    B: Backend,
    B::Error: Display,
{
    terminal
        .backend_mut()
        .clear()
        .map_err(|error| io::Error::other(error.to_string()))?;
    // `swap_buffers` resets the inactive buffer before toggling, so two swaps
    // reset both buffers and leave the original active index in place.
    terminal.swap_buffers();
    terminal.swap_buffers();
    Ok(())
}

pub(in crate::output) fn render_tui_frame(frame: &mut Frame, state: &DashboardState) {
    frame.render_widget(RatatuiClear, frame.area());

    if frame.area().width < PRETTY_TUI_MIN_DASHBOARD_WIDTH {
        render_tui_too_narrow_message(frame, frame.area());
        return;
    }

    let areas = tui_layout(frame.area(), state);
    let _main_body = areas.main_body;
    let full_screen_loading = state.is_startup_loading();

    if let Some(loading_area) = areas.loading.filter(|_| full_screen_loading) {
        render_model_progress_loader(frame, state, loading_area);
        return;
    }

    if let Some(panel) = state.full_screen_panel {
        render_full_screen_panel(frame, state, panel);
        return;
    }

    if let Some(logo_area) = areas.logo {
        render_tui_logo(frame, logo_area, true);
    }

    render_join_token_panel(
        frame,
        state,
        areas.join_token_panel,
        areas.join_token_copy_button,
    );

    frame.render_widget(
        Paragraph::new(dashboard_status_line(state, areas.status_bar.width))
            .style(tui_theme().status_bar),
        areas.status_bar,
    );

    render_events_panel(frame, state, areas.events.0, areas.events.1);
    render_processes_panel(
        frame,
        state,
        areas.processes,
        areas.llama_processes,
        areas.webserver_processes,
    );
    render_models_panel(frame, state, areas.models.0, areas.models.1);
    render_requests_panel(frame, state, areas.requests.0, areas.requests.1);
}

pub(in crate::output) fn render_full_screen_panel(
    frame: &mut Frame,
    state: &DashboardState,
    panel: DashboardPanel,
) {
    let panel_area = frame.area();
    if panel_area.width == 0 || panel_area.height == 0 {
        return;
    }

    let [title_area, body_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(panel_area);

    match panel {
        DashboardPanel::JoinToken => {
            render_join_token_panel(frame, state, panel_area, Rect::default())
        }
        DashboardPanel::Events => render_events_panel(frame, state, title_area, body_area),
        DashboardPanel::LlamaCpp | DashboardPanel::Webserver => {
            render_process_table(frame, state, panel, title_area, body_area)
        }
        DashboardPanel::Models => render_models_panel(frame, state, title_area, body_area),
        DashboardPanel::Requests => render_requests_panel(frame, state, title_area, body_area),
    }
}

pub(in crate::output) fn render_tui_too_narrow_message(frame: &mut Frame, area: Rect) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let message = Line::from(vec![
        Span::styled(
            "mesh-llm dashboard needs ",
            Style::default().fg(tui_theme().muted),
        ),
        Span::styled(
            format!(">= {PRETTY_TUI_MIN_DASHBOARD_WIDTH} columns"),
            Style::default().fg(tui_theme().warning),
        ),
        Span::styled(
            ". Resize or use line-oriented pretty output.",
            Style::default().fg(tui_theme().muted),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(message)
            .alignment(Alignment::Center)
            .block(Block::bordered().border_type(BorderType::Rounded)),
        area,
    );
}
