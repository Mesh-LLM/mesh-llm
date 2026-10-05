use super::super::{DashboardModelLane, DashboardModelRow};
use super::{
    Alignment, Block, BorderType, Buffer, Color, Constraint, DashboardPanel, DashboardState,
    Direction, Frame, Layout, Line, Modifier, PRETTY_TUI_MODEL_CARD_HEIGHT,
    PRETTY_TUI_MODEL_CARD_STRIDE, Paragraph, Rect, RuntimeStatus, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Span, Style, Widget, combine_panel_rect, empty_panel_message,
    truncate_with_ellipsis, tui_panel_block, tui_theme,
};

pub(in crate::output) fn render_models_panel(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, DashboardPanel::Models);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    if state.loaded_model_rows.is_empty() {
        frame.render_widget(
            Paragraph::new(empty_panel_message(state, DashboardPanel::Models))
                .style(Style::default().fg(Color::DarkGray)),
            inner_area,
        );
        return;
    }

    let view = state.panel_view_state(DashboardPanel::Models);
    let is_focused = state.panel_focus == DashboardPanel::Models;
    let visible_height = usize::from(inner_area.height);
    let viewport_rows = tui_panel_viewport_rows(DashboardPanel::Models, visible_height);
    let row_count = state.row_count_for_panel(DashboardPanel::Models);
    let show_scrollbar = row_count > viewport_rows && inner_area.width > 1;
    let list_area = if show_scrollbar {
        Rect {
            width: inner_area.width.saturating_sub(1),
            ..inner_area
        }
    } else {
        inner_area
    };
    let content_width = usize::from(list_area.width.max(1));
    for (local_index, (row_index, row)) in state
        .loaded_model_rows
        .iter()
        .enumerate()
        .skip(view.scroll_offset)
        .take(viewport_rows)
        .enumerate()
    {
        let card_y = list_area.y.saturating_add(
            u16::try_from(local_index.saturating_mul(PRETTY_TUI_MODEL_CARD_STRIDE)).unwrap_or(0),
        );
        if card_y >= list_area.bottom() {
            break;
        }

        let row_area = Rect {
            x: list_area.x,
            y: card_y,
            width: list_area.width,
            height: PRETTY_TUI_MODEL_CARD_HEIGHT as u16,
        };
        let is_selected = view.selected_row == Some(row_index);

        frame.render_widget(
            TuiModelCardWidget {
                row,
                content_width,
                is_selected,
                is_focused,
            },
            row_area,
        );
    }

    if show_scrollbar {
        let scrollbar_area = Rect {
            x: inner_area.right().saturating_sub(1),
            y: inner_area.y,
            width: 1,
            height: inner_area.height,
        };
        let mut scrollbar_state = ScrollbarState::new(row_count)
            .position(view.scroll_offset)
            .viewport_content_length(viewport_rows.min(row_count));
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some("│"));
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

pub(in crate::output) fn tui_panel_viewport_rows(
    panel: DashboardPanel,
    visible_rows: usize,
) -> usize {
    match panel {
        DashboardPanel::Models => tui_models_viewport_rows(visible_rows as u16),
        _ => visible_rows.max(1),
    }
}

pub(in crate::output) fn tui_models_viewport_rows(visible_height: u16) -> usize {
    let visible_height = usize::from(visible_height);
    if visible_height == 0 {
        return 0;
    }
    (visible_height / PRETTY_TUI_MODEL_CARD_STRIDE).max(1)
}

pub(in crate::output) struct TuiModelCardWidget<'a> {
    pub(in crate::output) row: &'a DashboardModelRow,
    pub(in crate::output) content_width: usize,
    pub(in crate::output) is_selected: bool,
    pub(in crate::output) is_focused: bool,
}

impl Widget for TuiModelCardWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        let theme = tui_theme();
        let card_bg = if self.is_selected {
            theme.selection_bg
        } else {
            theme.surface_raised
        };
        let border_fg = if self.is_selected && self.is_focused {
            theme.accent
        } else {
            theme.dim
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .style(Style::default().bg(card_bg))
            .border_style(Style::default().fg(border_fg).bg(card_bg));
        let inner = block.inner(area);
        block.render(area, buf);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let [
            name_row,
            summary_top,
            summary_bottom,
            divider,
            ctx_row,
            slots_row,
        ] = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .areas(inner);

        render_tui_model_name_row(
            buf,
            name_row,
            card_bg,
            &self.row.name,
            self.content_width.saturating_sub(2).max(1),
        );

        render_tui_model_identity_cells(
            buf,
            summary_top,
            card_bg,
            self.row
                .port
                .map(|port| port.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.row.device.as_deref().unwrap_or("n/a").to_string(),
            self.row.status.as_str().to_string(),
            tui_model_status_style(&self.row.status).bg(card_bg),
        );

        render_tui_model_summary_cells(
            buf,
            summary_bottom,
            card_bg,
            vec![
                (
                    "SLOTS",
                    self.row
                        .slots
                        .map(|slots| slots.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                    Style::default().fg(theme.warning).bg(card_bg),
                ),
                (
                    "QUANT",
                    self.row
                        .quantization
                        .as_deref()
                        .unwrap_or("n/a")
                        .to_string(),
                    Style::default().fg(theme.text).bg(card_bg),
                ),
                (
                    "CTX",
                    self.row
                        .ctx_size
                        .map(|ctx_size| ctx_size.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                    Style::default().fg(theme.text).bg(card_bg),
                ),
            ],
        );

        Paragraph::new(tui_model_card_divider(usize::from(inner.width)))
            .style(Style::default().fg(theme.dim).bg(card_bg))
            .render(divider, buf);

        let ctx_value = self
            .row
            .ctx_used_tokens
            .map(|ctx_used_tokens| ctx_used_tokens.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        let ctx_max = self
            .row
            .ctx_size
            .map(|ctx_size| ctx_size.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        let ctx_label = format!("{ctx_value} / {ctx_max}");
        let slots_label = tui_model_slots_value_label(self.row);
        let metric_value_width =
            tui_model_metric_value_width([ctx_label.as_str(), slots_label.as_str()]);

        render_tui_model_metric_row(
            buf,
            ctx_row,
            card_bg,
            "CTX",
            ctx_label,
            metric_value_width,
            tui_model_gauge_ratio(
                self.row
                    .ctx_used_tokens
                    .map(|ctx_used_tokens| ctx_used_tokens as f64),
                self.row.ctx_size.map(f64::from).unwrap_or(0.0),
            ),
        );
        render_tui_model_slots_row(
            buf,
            slots_row,
            card_bg,
            slots_label,
            metric_value_width,
            self.row,
        );
    }
}

pub(in crate::output) fn tui_model_metric_value_width<'a>(
    labels: impl IntoIterator<Item = &'a str>,
) -> u16 {
    let width = labels
        .into_iter()
        .map(|label| label.chars().count())
        .max()
        .unwrap_or(8)
        .clamp(8, 20);
    u16::try_from(width).unwrap_or(20)
}

pub(in crate::output) fn render_tui_model_metric_row(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    label: &'static str,
    value_label: String,
    value_width: u16,
    ratio: f64,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let [label_area, bar_area, _, value_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(value_width),
        ])
        .areas(area);

    Paragraph::new(label)
        .style(
            Style::default()
                .fg(theme.muted)
                .bg(card_bg)
                .add_modifier(Modifier::BOLD),
        )
        .render(label_area, buf);
    render_tui_model_usage_bar(buf, bar_area, card_bg, ratio);
    Paragraph::new(truncate_with_ellipsis(
        &value_label,
        usize::from(value_area.width),
    ))
    .style(Style::default().fg(theme.text).bg(card_bg))
    .alignment(Alignment::Right)
    .render(value_area, buf);
}

pub(in crate::output) fn render_tui_model_slots_row(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    value_label: String,
    value_width: u16,
    row: &DashboardModelRow,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let [label_area, _, bar_area, _, value_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(value_width),
        ])
        .areas(area);

    Paragraph::new("SLOTS")
        .style(
            Style::default()
                .fg(theme.muted)
                .bg(card_bg)
                .add_modifier(Modifier::BOLD),
        )
        .render(label_area, buf);
    render_tui_model_slot_blocks(buf, bar_area, card_bg, row);
    Paragraph::new(truncate_with_ellipsis(
        &value_label,
        usize::from(value_area.width),
    ))
    .style(Style::default().fg(theme.text).bg(card_bg))
    .alignment(Alignment::Right)
    .render(value_area, buf);
}

pub(in crate::output) fn render_tui_model_slot_blocks(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    row: &DashboardModelRow,
) {
    let theme = tui_theme();
    let lanes = tui_model_slot_lanes(row);
    let max_width = usize::from(area.width);
    if max_width == 0 {
        return;
    }

    let spans = if lanes.is_empty() {
        vec![Span::styled(
            "n/a",
            Style::default().fg(theme.dim).bg(card_bg),
        )]
    } else {
        let visible_slots = lanes.len().min(max_width);
        let mut spans = Vec::with_capacity(visible_slots);
        for lane in lanes.into_iter().take(visible_slots) {
            spans.push(Span::styled(
                "◼",
                Style::default()
                    .fg(if lane.active {
                        theme.warning
                    } else {
                        theme.dim
                    })
                    .bg(card_bg),
            ));
        }
        spans
    };
    Paragraph::new(Line::from(spans))
        .style(Style::default().bg(card_bg))
        .render(area, buf);
}

pub(in crate::output) fn tui_model_slot_lanes(row: &DashboardModelRow) -> Vec<DashboardModelLane> {
    if let Some(lanes) = row.lanes.as_ref().filter(|lanes| !lanes.is_empty()) {
        let mut lanes = lanes.clone();
        lanes.sort_by_key(|lane| lane.index);
        return lanes;
    }

    let slot_count = row.slots.unwrap_or(0).min(usize::from(u16::MAX));
    (0..slot_count)
        .map(|index| DashboardModelLane {
            index,
            active: false,
        })
        .collect()
}

pub(in crate::output) fn tui_model_slots_value_label(row: &DashboardModelRow) -> String {
    let lanes = tui_model_slot_lanes(row);
    if lanes.is_empty() {
        return "n/a".to_string();
    }
    let active = lanes.iter().filter(|lane| lane.active).count();
    format!("{active} / {}", lanes.len())
}

pub(in crate::output) fn render_tui_model_identity_cells(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    port: String,
    device: String,
    status: String,
    status_style: Style,
) {
    render_tui_model_summary_cells(
        buf,
        area,
        card_bg,
        vec![
            (
                "PORT",
                port,
                Style::default().fg(tui_theme().text).bg(card_bg),
            ),
            ("STATUS", status, status_style),
            (
                "DEVICE",
                device,
                Style::default().fg(tui_theme().text).bg(card_bg),
            ),
        ],
    );
}

pub(in crate::output) fn render_tui_model_name_row(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    name: &str,
    max_width: usize,
) {
    if area.width == 0 {
        return;
    }

    Paragraph::new(truncate_with_ellipsis(
        name,
        usize::from(area.width).min(max_width),
    ))
    .style(
        Style::default()
            .fg(tui_theme().text)
            .bg(card_bg)
            .add_modifier(Modifier::BOLD),
    )
    .alignment(Alignment::Left)
    .render(area, buf);
}

pub(in crate::output) fn render_tui_model_summary_cell(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    label: &'static str,
    value: String,
    value_style: Style,
) {
    if area.width == 0 {
        return;
    }

    let label_text = format!("{label}: ");
    let label_width = label_text.chars().count();
    let value_width = usize::from(area.width).saturating_sub(label_width).max(1);
    let line = Line::from(vec![
        Span::styled(
            label_text,
            Style::default()
                .fg(tui_theme().dim)
                .bg(card_bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            truncate_with_ellipsis(&value, value_width),
            value_style.bg(card_bg),
        ),
    ]);
    Paragraph::new(line)
        .style(Style::default().bg(card_bg))
        .alignment(Alignment::Left)
        .render(area, buf);
}

pub(in crate::output) fn render_tui_model_summary_cells(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    entries: Vec<(&'static str, String, Style)>,
) {
    if area.width == 0 || area.height == 0 || entries.is_empty() {
        return;
    }

    let columns = entries.len();
    let gap_width = u16::from(columns > 1);
    let mut constraints = Vec::with_capacity(columns.saturating_mul(2).saturating_sub(1));
    for index in 0..columns {
        constraints.push(Constraint::Fill(1));
        if index + 1 < columns {
            constraints.push(Constraint::Length(gap_width));
        }
    }
    let cells = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    for (index, (label, value, value_style)) in entries.into_iter().enumerate() {
        let cell_index = index.saturating_mul(2);
        let Some(cell_area) = cells.get(cell_index).copied() else {
            continue;
        };
        if cell_area.width == 0 {
            continue;
        }

        render_tui_model_summary_cell(buf, cell_area, card_bg, label, value, value_style);
    }
}

pub(in crate::output) fn render_tui_model_usage_bar(
    buf: &mut Buffer,
    area: Rect,
    card_bg: Color,
    ratio: f64,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let ratio = ratio.clamp(0.0, 1.0);
    let filled_width = (ratio * f64::from(area.width)).round() as u16;
    let fill_color = tui_model_usage_color(ratio);
    let empty_style = Style::default().fg(theme.dim).bg(card_bg);
    let fill_style = Style::default().fg(fill_color).bg(card_bg);

    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let filled = x.saturating_sub(area.left()) < filled_width;
            buf[(x, y)]
                .set_symbol("█")
                .set_style(if filled { fill_style } else { empty_style });
        }
    }
}

pub(in crate::output) fn tui_model_usage_color(ratio: f64) -> Color {
    let theme = tui_theme();
    let ratio = ratio.clamp(0.0, 1.0);
    if ratio <= 0.5 {
        tui_lerp_rgb(theme.success, theme.warning, ratio / 0.5)
    } else {
        tui_lerp_rgb(theme.warning, theme.error, (ratio - 0.5) / 0.5)
    }
}

pub(in crate::output) fn tui_lerp_rgb(start: Color, end: Color, t: f64) -> Color {
    let Color::Rgb(start_r, start_g, start_b) = start else {
        return end;
    };
    let Color::Rgb(end_r, end_g, end_b) = end else {
        return start;
    };
    let t = t.clamp(0.0, 1.0);
    Color::Rgb(
        (f64::from(start_r) + (f64::from(end_r) - f64::from(start_r)) * t).round() as u8,
        (f64::from(start_g) + (f64::from(end_g) - f64::from(start_g)) * t).round() as u8,
        (f64::from(start_b) + (f64::from(end_b) - f64::from(start_b)) * t).round() as u8,
    )
}
pub(in crate::output) fn tui_model_card_divider(content_width: usize) -> Line<'static> {
    let theme = tui_theme();
    Line::from(Span::styled(
        "─".repeat(content_width),
        Style::default().fg(theme.dim).add_modifier(Modifier::DIM),
    ))
}

#[cfg(test)]
pub(in crate::output) fn spans_plain_text(spans: &[Span<'_>]) -> String {
    let mut text = String::new();
    for span in spans {
        text.push_str(span.content.as_ref());
    }
    text
}

pub(in crate::output) fn tui_model_gauge_ratio(value: Option<f64>, max_value: f64) -> f64 {
    let Some(value) = value.filter(|value| *value > 0.0) else {
        return 0.0;
    };
    if max_value <= 0.0 {
        return 0.0;
    }
    (value / max_value).clamp(0.0, 1.0)
}

pub(in crate::output) fn tui_model_status_style(status: &RuntimeStatus) -> Style {
    let theme = tui_theme();
    match status {
        RuntimeStatus::NotReady => Style::default().fg(theme.muted),
        RuntimeStatus::Starting | RuntimeStatus::Loading => Style::default().fg(theme.warning),
        RuntimeStatus::Ready => Style::default().fg(theme.success),
        RuntimeStatus::ShuttingDown => Style::default().fg(theme.warning),
        RuntimeStatus::Stopped => Style::default().fg(theme.dim),
        RuntimeStatus::Exited => Style::default().fg(theme.dim),
        RuntimeStatus::Warning => Style::default().fg(theme.warning),
        RuntimeStatus::Error => Style::default().fg(theme.error),
    }
}
