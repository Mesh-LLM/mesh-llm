use super::super::{
    DashboardEndpointRow, DashboardModelRow, DashboardProcessRow,
    PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL, llama_process_model_name,
    model_name_without_variant_suffix, model_names_match,
};
use super::{
    Block, BorderType, Cell, Color, Constraint, DashboardPanel, DashboardState, Frame,
    HighlightSpacing, Line, Modifier, Paragraph, Rect, Row, RuntimeStatus, Style, Table,
    TableState, empty_panel_message, format_tui_panel_title, panel_border_style, panel_title_style,
    truncate_with_ellipsis, tui_theme,
};

pub(in crate::output) fn render_processes_panel(
    frame: &mut Frame,
    state: &DashboardState,
    processes_area: Rect,
    llama_processes: (Rect, Rect),
    webserver_processes: (Rect, Rect),
) {
    frame.render_widget(tui_processes_block(state), processes_area);
    render_process_table(
        frame,
        state,
        DashboardPanel::LlamaCpp,
        llama_processes.0,
        llama_processes.1,
    );
    render_process_table(
        frame,
        state,
        DashboardPanel::Webserver,
        webserver_processes.0,
        webserver_processes.1,
    );
}

pub(in crate::output) fn render_process_table(
    frame: &mut Frame,
    state: &DashboardState,
    panel: DashboardPanel,
    title_area: Rect,
    body_area: Rect,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, panel);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    let view = state.panel_view_state(panel);
    let is_focused = state.panel_focus == panel;
    match panel {
        DashboardPanel::LlamaCpp => {
            if state.llama_process_rows.is_empty() {
                frame.render_widget(
                    Paragraph::new(empty_panel_message(state, panel))
                        .style(Style::default().fg(Color::DarkGray)),
                    inner_area,
                );
                return;
            }

            let [model_width, pid_width, port_width, status_width] =
                llama_process_column_widths_for_rows(inner_area.width, &state.llama_process_rows);
            let available_rows = usize::from(inner_area.height.saturating_sub(1));
            let rows = state
                .llama_process_rows
                .iter()
                .enumerate()
                .skip(view.scroll_offset)
                .take(available_rows)
                .map(|(_, row)| {
                    let model = llama_process_model_metadata(row, &state.loaded_model_rows);
                    let model_name = model.map(|model| model.name.as_str()).unwrap_or(&row.name);
                    Row::new(vec![
                        Cell::from(truncate_with_ellipsis(
                            model_name_without_variant_suffix(model_name),
                            model_width,
                        )),
                        Cell::from(truncate_with_ellipsis(
                            &format_dashboard_pid((row.pid != 0).then_some(row.pid)),
                            pid_width,
                        )),
                        Cell::from(truncate_with_ellipsis(&row.port.to_string(), port_width)),
                        process_status_cell(&row.status, status_width),
                    ])
                })
                .collect::<Vec<_>>();
            let selected_local_index = view
                .selected_row
                .map(|selected| selected.saturating_sub(view.scroll_offset));
            let mut table_state = TableState::default();
            table_state.select(selected_local_index);
            let table = Table::new(
                rows,
                [
                    Constraint::Fill(1),
                    Constraint::Length(u16::try_from(pid_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(port_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(status_width).unwrap_or(u16::MAX)),
                ],
            )
            .header(process_table_header_row([
                "MODEL".to_string(),
                "PID".to_string(),
                "PORT".to_string(),
                right_align_text("STATE", status_width),
            ]))
            .column_spacing(1)
            .highlight_symbol(if is_focused { "› " } else { "  " })
            .highlight_spacing(HighlightSpacing::Always)
            .row_highlight_style(process_table_highlight_style(is_focused));
            frame.render_stateful_widget(table, inner_area, &mut table_state);
        }
        DashboardPanel::Webserver => {
            if state.webserver_rows.is_empty() {
                frame.render_widget(
                    Paragraph::new(empty_panel_message(state, panel))
                        .style(Style::default().fg(Color::DarkGray)),
                    inner_area,
                );
                return;
            }

            let [label_width, pid_width, port_width, status_width] =
                webserver_process_column_widths_for_rows(inner_area.width, &state.webserver_rows);
            let available_rows = usize::from(inner_area.height.saturating_sub(1));
            let rows = state
                .webserver_rows
                .iter()
                .enumerate()
                .skip(view.scroll_offset)
                .take(available_rows)
                .map(|(_, row)| {
                    Row::new(vec![
                        Cell::from(truncate_with_ellipsis(&row.label, label_width)),
                        Cell::from(truncate_with_ellipsis(
                            &format_dashboard_pid(row.pid),
                            pid_width,
                        )),
                        Cell::from(truncate_with_ellipsis(
                            &format_dashboard_port(row.port),
                            port_width,
                        )),
                        process_status_cell(&row.status, status_width),
                    ])
                })
                .collect::<Vec<_>>();
            let selected_local_index = view
                .selected_row
                .map(|selected| selected.saturating_sub(view.scroll_offset));
            let mut table_state = TableState::default();
            table_state.select(selected_local_index);
            let table = Table::new(
                rows,
                [
                    Constraint::Fill(1),
                    Constraint::Length(u16::try_from(pid_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(port_width).unwrap_or(u16::MAX)),
                    Constraint::Length(u16::try_from(status_width).unwrap_or(u16::MAX)),
                ],
            )
            .header(process_table_header_row([
                PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL.to_string(),
                "PID".to_string(),
                "PORT".to_string(),
                right_align_text("STATE", status_width),
            ]))
            .column_spacing(1)
            .highlight_symbol(if is_focused { "› " } else { "  " })
            .highlight_spacing(HighlightSpacing::Always)
            .row_highlight_style(process_table_highlight_style(is_focused));
            frame.render_stateful_widget(table, inner_area, &mut table_state);
        }
        _ => {}
    }
}

pub(in crate::output) fn combine_panel_rect(title_area: Rect, body_area: Rect) -> Rect {
    Rect {
        x: title_area.x,
        y: title_area.y,
        width: title_area.width.max(body_area.width),
        height: title_area.height.saturating_add(body_area.height),
    }
}

pub(in crate::output) fn tui_panel_block(
    state: &DashboardState,
    panel: DashboardPanel,
) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(panel_border_style(state, panel))
        .title(Line::styled(
            format_tui_panel_title(state, panel),
            panel_title_style(state, panel),
        ))
}

pub(in crate::output) fn tui_processes_block(state: &DashboardState) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(processes_border_style(state))
        .title(Line::styled(" Processes", processes_title_style(state)))
}

pub(in crate::output) fn processes_title_style(state: &DashboardState) -> Style {
    let theme = tui_theme();
    if matches!(
        state.panel_focus,
        DashboardPanel::LlamaCpp | DashboardPanel::Webserver
    ) {
        Style::default()
            .fg(theme.accent)
            .bg(theme.surface_raised)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.dim).add_modifier(Modifier::DIM)
    }
}

pub(in crate::output) fn processes_border_style(state: &DashboardState) -> Style {
    let theme = tui_theme();
    if matches!(
        state.panel_focus,
        DashboardPanel::LlamaCpp | DashboardPanel::Webserver
    ) {
        Style::default().fg(theme.accent)
    } else {
        Style::default().fg(theme.dim)
    }
}

pub(in crate::output) fn process_table_highlight_style(is_focused: bool) -> Style {
    let theme = tui_theme();
    if is_focused {
        Style::default()
            .bg(theme.selection_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    }
}

pub(in crate::output) fn process_table_header_row<const N: usize>(
    labels: [String; N],
) -> Row<'static> {
    let theme = tui_theme();
    Row::new(labels.into_iter().map(|label| {
        Cell::from(label).style(
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        )
    }))
    .style(Style::default().bg(theme.surface_raised))
}

pub(in crate::output) fn right_align_text(value: &str, width: usize) -> String {
    let value = truncate_with_ellipsis(value, width);
    format!("{value:>width$}")
}

pub(in crate::output) fn format_dashboard_pid(pid: Option<u32>) -> String {
    pid.map(|pid| pid.to_string())
        .unwrap_or_else(|| "-".to_string())
}

pub(in crate::output) fn format_dashboard_port(port: u16) -> String {
    if port == 0 {
        "-".to_string()
    } else {
        port.to_string()
    }
}

pub(in crate::output) fn dashboard_port_from_url(url: &str) -> u16 {
    url.rsplit(':')
        .next()
        .map(|tail| tail.trim_end_matches('/'))
        .and_then(|tail| tail.parse().ok())
        .unwrap_or(0)
}

pub(in crate::output) fn process_status_cell(
    status: &RuntimeStatus,
    width: usize,
) -> Cell<'static> {
    let theme = tui_theme();
    let style = match status {
        RuntimeStatus::NotReady => Style::default().fg(theme.muted),
        RuntimeStatus::Ready => Style::default().fg(theme.success),
        RuntimeStatus::Starting
        | RuntimeStatus::Loading
        | RuntimeStatus::ShuttingDown
        | RuntimeStatus::Warning => Style::default().fg(theme.warning),
        RuntimeStatus::Error => Style::default().fg(theme.error),
        RuntimeStatus::Stopped | RuntimeStatus::Exited => Style::default().fg(theme.dim),
    };
    Cell::from(right_align_text(status.as_str(), width)).style(style)
}

pub(in crate::output) fn llama_process_model_metadata<'a>(
    process: &DashboardProcessRow,
    models: &'a [DashboardModelRow],
) -> Option<&'a DashboardModelRow> {
    models
        .iter()
        .find(|model| model.port == Some(process.port))
        .or_else(|| {
            models.iter().find(|model| {
                llama_process_model_name(&process.name)
                    .map(|process_model| model_names_match(process_model, &model.name))
                    .unwrap_or(false)
            })
        })
}

#[cfg(test)]
pub(in crate::output) fn llama_process_column_widths(body_width: u16) -> [usize; 4] {
    process_column_widths(
        body_width,
        8,
        process_pid_width(std::iter::empty()),
        RuntimeStatus::NotReady.as_str().len(),
    )
}

pub(in crate::output) fn llama_process_column_widths_for_rows(
    body_width: u16,
    rows: &[DashboardProcessRow],
) -> [usize; 4] {
    process_column_widths(
        body_width,
        8,
        process_pid_width(rows.iter().map(|row| (row.pid != 0).then_some(row.pid))),
        process_status_width(rows.iter().map(|row| &row.status)),
    )
}

#[cfg(test)]
pub(in crate::output) fn webserver_process_column_widths(body_width: u16) -> [usize; 4] {
    process_column_widths(
        body_width,
        PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL.len(),
        process_pid_width(std::iter::empty()),
        RuntimeStatus::NotReady.as_str().len(),
    )
}

pub(in crate::output) fn webserver_process_column_widths_for_rows(
    body_width: u16,
    rows: &[DashboardEndpointRow],
) -> [usize; 4] {
    process_column_widths(
        body_width,
        PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL.len(),
        process_pid_width(rows.iter().map(|row| row.pid)),
        process_status_width(rows.iter().map(|row| &row.status)),
    )
}

pub(in crate::output) fn process_column_widths(
    body_width: u16,
    min_text_width: usize,
    pid_width: usize,
    status_width: usize,
) -> [usize; 4] {
    let port_width = 5usize;
    let reserved_width = pid_width + port_width + status_width + 3 + 2;
    let text_width = usize::from(body_width)
        .saturating_sub(reserved_width)
        .max(min_text_width);
    [text_width, pid_width, port_width, status_width]
}

pub(in crate::output) fn process_pid_width<I>(pids: I) -> usize
where
    I: IntoIterator<Item = Option<u32>>,
{
    pids.into_iter()
        .map(format_dashboard_pid)
        .map(|pid| pid.chars().count())
        .max()
        .unwrap_or(5)
        .max(5)
}

pub(in crate::output) fn process_status_width<'a, I>(statuses: I) -> usize
where
    I: IntoIterator<Item = &'a RuntimeStatus>,
{
    statuses
        .into_iter()
        .map(|status| status.as_str().chars().count())
        .max()
        .unwrap_or_else(|| RuntimeStatus::NotReady.as_str().len())
        .max("STATE".len())
}
