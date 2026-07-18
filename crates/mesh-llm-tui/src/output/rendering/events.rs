#[cfg(test)]
use super::spans_plain_text;
use super::{
    Buffer, Color, DashboardPanel, DashboardState, Frame, Line, MeshEventState, Modifier,
    PRETTY_TUI_EVENT_LEVEL_WIDTH, RatatuiClear, Rect, Scrollbar, ScrollbarOrientation, Span,
    StatefulWidget, Style, TuiEventListRenderer, Widget, combine_panel_rect,
    truncate_with_ellipsis, tui_list_scrollbar_layout, tui_list_scrollbar_state, tui_panel_block,
    tui_theme,
};
#[cfg(test)]
use super::{PRETTY_TUI_LIST_HIGHLIGHT_SYMBOL_WIDTH, TuiEventRow, process_table_highlight_style};
use mesh_llm_events::OutputLevel;

pub(in crate::output) fn render_events_panel(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
) {
    render_events_panel_with_renderer(
        frame,
        state,
        title_area,
        body_area,
        TuiEventListRenderer::ACTIVE,
    );
}

pub(in crate::output) fn render_events_panel_with_renderer(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
    renderer: TuiEventListRenderer,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, DashboardPanel::Events);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    match renderer {
        #[cfg(test)]
        TuiEventListRenderer::Legacy => render_legacy_events_list(frame, state, inner_area),
        TuiEventListRenderer::Scrollbar => render_scrollbar_events_list(frame, state, inner_area),
    }
}

#[cfg(test)]
pub(in crate::output) fn render_legacy_events_list(
    frame: &mut Frame,
    state: &DashboardState,
    inner_area: Rect,
) {
    let view = state.panel_view_state(DashboardPanel::Events);
    let row_count = state.row_count_for_panel(DashboardPanel::Events);
    let viewport_rows = usize::from(inner_area.height).max(1);
    let scroll_offset = effective_events_scroll_offset(state, row_count, viewport_rows);
    let layout = tui_list_scrollbar_layout(inner_area, row_count, viewport_rows);
    let content_width = usize::from(
        layout
            .list_area
            .width
            .saturating_sub(PRETTY_TUI_LIST_HIGHLIGHT_SYMBOL_WIDTH)
            .max(1),
    );
    let rows = visible_event_rows_from(state, viewport_rows, scroll_offset);
    let is_focused = state.panel_focus == DashboardPanel::Events;
    render_event_list_rows(
        frame,
        layout.list_area,
        &rows,
        view.selected_row,
        is_focused,
        content_width,
    );

    if let Some(scrollbar_area) = layout.scrollbar_area {
        let mut scrollbar_state = tui_list_scrollbar_state(row_count, viewport_rows, scroll_offset);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some("│"));
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

pub(in crate::output) fn render_scrollbar_events_list(
    frame: &mut Frame,
    state: &DashboardState,
    inner_area: Rect,
) {
    let row_count = state.row_count_for_panel(DashboardPanel::Events);
    let viewport_rows = usize::from(inner_area.height).max(1);
    let scroll_offset = effective_events_scroll_offset(state, row_count, viewport_rows);
    let events = state.filtered_mesh_events();
    frame.render_widget(
        TuiScrollbarEventList {
            events: &events,
            empty_message: empty_panel_message(state, DashboardPanel::Events),
            scroll_offset,
            wrap_lines: state.full_screen_panel == Some(DashboardPanel::Events),
        },
        inner_area,
    );
}

pub(in crate::output) struct TuiScrollbarEventList<'a> {
    pub(in crate::output) events: &'a [&'a MeshEventState],
    pub(in crate::output) empty_message: &'static str,
    pub(in crate::output) scroll_offset: usize,
    pub(in crate::output) wrap_lines: bool,
}

impl Widget for TuiScrollbarEventList<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        Widget::render(RatatuiClear, area, buf);
        if area.height == 0 {
            return;
        }

        let row_count = self.events.len();
        let viewport_rows = usize::from(area.height).max(1);
        let layout = tui_list_scrollbar_layout(area, row_count, viewport_rows);
        let content_width = usize::from(layout.list_area.width.max(1));

        if row_count == 0 {
            let line = Line::from(Span::styled(
                self.empty_message.to_string(),
                Style::default().fg(Color::DarkGray),
            ));
            Widget::render(line, single_line_rect(layout.list_area, 0), buf);
            return;
        }

        let scroll_offset = self
            .scroll_offset
            .min(row_count.saturating_sub(viewport_rows));
        if self.wrap_lines {
            let mut row_index = 0usize;
            for event in self.events.iter().skip(scroll_offset) {
                for line in wrapped_event_lines(event, content_width) {
                    if row_index >= viewport_rows {
                        break;
                    }
                    Widget::render(line, single_line_rect(layout.list_area, row_index), buf);
                    row_index = row_index.saturating_add(1);
                }
                if row_index >= viewport_rows {
                    break;
                }
            }
        } else {
            for (row_index, event) in self
                .events
                .iter()
                .skip(scroll_offset)
                .take(viewport_rows)
                .enumerate()
            {
                let row_area = single_line_rect(layout.list_area, row_index);
                if row_area.height == 0 {
                    break;
                }
                Widget::render(event_line(event, content_width), row_area, buf);
            }
        }

        if let Some(scrollbar_area) = layout.scrollbar_area {
            let mut scrollbar_state =
                tui_list_scrollbar_state(row_count, viewport_rows, scroll_offset);
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"));
            StatefulWidget::render(scrollbar, scrollbar_area, buf, &mut scrollbar_state);
        }
    }
}

pub(in crate::output) fn single_line_rect(area: Rect, row_index: usize) -> Rect {
    let y = area
        .y
        .saturating_add(u16::try_from(row_index).unwrap_or(u16::MAX));
    if y >= area.bottom() {
        return Rect { height: 0, ..area };
    }
    Rect {
        y,
        height: 1,
        ..area
    }
}

pub(in crate::output) fn effective_events_scroll_offset(
    state: &DashboardState,
    row_count: usize,
    viewport_rows: usize,
) -> usize {
    if row_count == 0 {
        return 0;
    }

    let max_scroll_offset = row_count.saturating_sub(viewport_rows);
    if state.events_follow {
        max_scroll_offset
    } else {
        state
            .panel_view_state(DashboardPanel::Events)
            .scroll_offset
            .min(max_scroll_offset)
    }
}

#[cfg(test)]
pub(in crate::output) fn render_event_list_rows(
    frame: &mut Frame,
    area: Rect,
    rows: &[TuiEventRow<'_>],
    selected_row: Option<usize>,
    is_focused: bool,
    content_width: usize,
) {
    frame.render_widget(RatatuiClear, area);

    let reserve_highlight_column = selected_row.is_some();
    let highlight_style = process_table_highlight_style(is_focused);
    for (row_index, row) in rows.iter().take(usize::from(area.height)).enumerate() {
        let y = area
            .y
            .saturating_add(u16::try_from(row_index).unwrap_or(u16::MAX));
        if y >= area.bottom() {
            break;
        }

        let row_area = Rect {
            y,
            height: 1,
            ..area
        };
        let selected = matches!(
            row,
            TuiEventRow::Event { absolute_index, .. }
                if Some(*absolute_index) == selected_row
        );
        let line = match row {
            TuiEventRow::Event { event, .. } => event_line(event, content_width),
            TuiEventRow::Message(message) => Line::from(Span::styled(
                (*message).to_string(),
                Style::default().fg(Color::DarkGray),
            )),
            TuiEventRow::Padding => Line::raw(""),
        };
        let line = event_list_line(line, reserve_highlight_column, selected, is_focused);
        Widget::render(line, row_area, frame.buffer_mut());
        if selected {
            frame.buffer_mut().set_style(row_area, highlight_style);
        }
    }
}

#[cfg(test)]
pub(in crate::output) fn event_list_line(
    mut line: Line<'static>,
    reserve_highlight_column: bool,
    selected: bool,
    is_focused: bool,
) -> Line<'static> {
    if reserve_highlight_column {
        let symbol = if selected && is_focused { "› " } else { "  " };
        line.spans.insert(0, Span::raw(symbol));
    }
    line
}

#[cfg(test)]
pub(in crate::output) fn visible_event_rows<'a>(
    state: &'a DashboardState,
    viewport_rows: usize,
) -> Vec<TuiEventRow<'a>> {
    let scroll_offset = state.panel_view_state(DashboardPanel::Events).scroll_offset;
    visible_event_rows_from(state, viewport_rows, scroll_offset)
}

#[cfg(test)]
pub(in crate::output) fn visible_event_rows_from<'a>(
    state: &'a DashboardState,
    viewport_rows: usize,
    scroll_offset: usize,
) -> Vec<TuiEventRow<'a>> {
    let row_count = state.row_count_for_panel(DashboardPanel::Events);
    let mut rows = if row_count == 0 {
        vec![TuiEventRow::Message(empty_panel_message(
            state,
            DashboardPanel::Events,
        ))]
    } else {
        state
            .filtered_mesh_events()
            .into_iter()
            .enumerate()
            .skip(scroll_offset)
            .take(viewport_rows)
            .map(|(absolute_index, event)| TuiEventRow::Event {
                absolute_index,
                event,
            })
            .collect::<Vec<_>>()
    };

    if state.events_follow && row_count > 0 {
        let padding = viewport_rows.saturating_sub(rows.len());
        if padding > 0 {
            let mut anchored_rows = Vec::with_capacity(viewport_rows);
            anchored_rows.extend((0..padding).map(|_| TuiEventRow::Padding));
            anchored_rows.extend(rows);
            rows = anchored_rows;
        }
    }

    while rows.len() < viewport_rows.max(1) {
        rows.push(TuiEventRow::Padding);
    }

    rows
}

pub(in crate::output) fn empty_panel_message(
    state: &DashboardState,
    panel: DashboardPanel,
) -> &'static str {
    match panel {
        DashboardPanel::JoinToken => "join token will appear here when the mesh invite is ready",
        DashboardPanel::Events if state.events_filter.is_active() => {
            "(no events match the current filter)"
        }
        DashboardPanel::Events => "(waiting for mesh events)",
        DashboardPanel::LlamaCpp => "(no llama.cpp processes yet)",
        DashboardPanel::Webserver => "(no webserver processes yet)",
        DashboardPanel::Models => "(no loaded models yet)",
        DashboardPanel::Requests => "(incoming request metrics will appear here)",
    }
}

pub(in crate::output) fn event_severity_badge(event: &MeshEventState) -> (&'static str, Style) {
    let theme = tui_theme();
    let summary_lower = event.summary.to_lowercase();
    if matches!(event.level, OutputLevel::Fatal) {
        (
            "FATAL",
            Style::default()
                .fg(theme.error)
                .add_modifier(Modifier::BOLD),
        )
    } else if matches!(event.level, OutputLevel::Error)
        || summary_lower.contains("err")
        || summary_lower.contains("failed")
    {
        (
            "ERR",
            Style::default()
                .fg(theme.error)
                .add_modifier(Modifier::BOLD),
        )
    } else if matches!(event.level, OutputLevel::Warn) || summary_lower.contains("warn") {
        (
            "WARN",
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        )
    } else if matches!(event.level, OutputLevel::Debug) {
        (
            "DBG",
            Style::default().fg(theme.dim).add_modifier(Modifier::BOLD),
        )
    } else if summary_lower.contains("ready")
        || summary_lower.contains("elected")
        || summary_lower.contains("joined")
        || summary_lower.contains("ok")
    {
        (
            "OK",
            Style::default()
                .fg(theme.success)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        (
            "INFO",
            Style::default()
                .fg(theme.accent_soft)
                .add_modifier(Modifier::BOLD),
        )
    }
}

pub(in crate::output) fn event_severity_badge_span(event: &MeshEventState) -> Span<'static> {
    let (badge_text, badge_style) = event_severity_badge(event);
    Span::styled(
        format!("{badge_text:<PRETTY_TUI_EVENT_LEVEL_WIDTH$}"),
        badge_style,
    )
}

pub(in crate::output) fn event_matches_filter(event: &MeshEventState, needle: &str) -> bool {
    let (badge_text, _) = event_severity_badge(event);
    let sanitized_message = sanitize_mesh_event_message(&event.summary);
    let rendered_search_text =
        format!("{} {} {}", event.timestamp, badge_text, sanitized_message).to_lowercase();
    rendered_search_text.contains(needle)
}

pub(in crate::output) fn event_line(event: &MeshEventState, width: usize) -> Line<'static> {
    let theme = tui_theme();
    let (badge_text, _) = event_severity_badge(event);
    let message = sanitize_mesh_event_message(&event.summary);
    let prefix = format!(
        "{} {:<PRETTY_TUI_EVENT_LEVEL_WIDTH$}",
        event.timestamp, badge_text
    );
    let prefix_len = prefix.chars().count();
    let remaining = width.saturating_sub(prefix_len);
    if remaining == 0 {
        return Line::from(vec![Span::styled(
            truncate_with_ellipsis(&prefix, width),
            Style::default().fg(theme.dim),
        )]);
    }

    Line::from(vec![
        Span::styled(event.timestamp.clone(), Style::default().fg(theme.dim)),
        Span::raw(" "),
        event_severity_badge_span(event),
        Span::styled(
            truncate_with_ellipsis(&message, remaining),
            Style::default().fg(theme.text),
        ),
    ])
}

pub(in crate::output) fn wrapped_event_lines(
    event: &MeshEventState,
    width: usize,
) -> Vec<Line<'static>> {
    let theme = tui_theme();
    let message = sanitize_mesh_event_message(&event.summary);
    let prefix_width = event
        .timestamp
        .chars()
        .count()
        .saturating_add(1)
        .saturating_add(PRETTY_TUI_EVENT_LEVEL_WIDTH);
    let message_width = width.saturating_sub(prefix_width);
    if message_width == 0 {
        return vec![event_line(event, width)];
    }

    let wrapped_message = wrap_plain_text(&message, message_width);
    let mut lines = Vec::with_capacity(wrapped_message.len().max(1));
    for (index, chunk) in wrapped_message.into_iter().enumerate() {
        if index == 0 {
            lines.push(Line::from(vec![
                Span::styled(event.timestamp.clone(), Style::default().fg(theme.dim)),
                Span::raw(" "),
                event_severity_badge_span(event),
                Span::styled(chunk, Style::default().fg(theme.text)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw(" ".repeat(prefix_width)),
                Span::styled(chunk, Style::default().fg(theme.text)),
            ]));
        }
    }

    lines
}

pub(in crate::output) fn wrap_plain_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let word_width = word.chars().count();
        if word_width > width {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
            }
            let mut chunk = String::new();
            for ch in word.chars() {
                if chunk.chars().count() == width {
                    lines.push(std::mem::take(&mut chunk));
                }
                chunk.push(ch);
            }
            if !chunk.is_empty() {
                current = chunk;
            }
        } else if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count().saturating_add(1 + word_width) <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }

    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

pub(in crate::output) fn sanitize_mesh_event_message(message: &str) -> String {
    let mut output = String::with_capacity(message.len());
    let mut last_was_space = false;
    for ch in message.chars().filter(|ch| !is_mesh_event_emoji(*ch)) {
        if ch.is_whitespace() {
            if !last_was_space {
                output.push(' ');
            }
            last_was_space = true;
        } else {
            output.push(ch);
            last_was_space = false;
        }
    }
    output.trim().to_string()
}

pub(in crate::output) fn is_mesh_event_emoji(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F300..=0x1FAFF | 0x2300..=0x23FF | 0x2600..=0x27BF | 0xFE0F
    )
}

#[cfg(test)]
pub(in crate::output) fn format_event_row(event: &MeshEventState, width: usize) -> String {
    spans_plain_text(&event_line(event, width).spans)
}
