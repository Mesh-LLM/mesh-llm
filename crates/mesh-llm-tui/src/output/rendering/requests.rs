use super::super::{
    DashboardRequestHistoryState, DashboardRequestWindow, PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS,
    PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS,
};
use super::{
    Buffer, Color, Constraint, DashboardPanel, DashboardState, Direction, Frame, Layout, Line,
    Modifier, PRETTY_TUI_REQUEST_GRAPH_BASELINE_SYMBOL, PRETTY_TUI_REQUEST_GRAPH_GUIDE_SYMBOL,
    Paragraph, Rect, Span, Style, Widget, combine_panel_rect, tui_panel_block, tui_theme,
};
use mesh_llm_events::DashboardAcceptedRequestBucket;

pub(in crate::output) fn render_requests_panel(
    frame: &mut Frame,
    state: &DashboardState,
    title_area: Rect,
    body_area: Rect,
) {
    let panel_area = combine_panel_rect(title_area, body_area);
    let block = tui_panel_block(state, DashboardPanel::Requests);
    frame.render_widget(block.clone(), panel_area);
    let inner_area = block.inner(panel_area);
    if inner_area.height == 0 {
        return;
    }

    let is_focused = state.panel_focus == DashboardPanel::Requests;
    let [summary_area, graph_slot] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(inner_area);

    frame.render_widget(
        Paragraph::new(tui_requests_summary_line(
            &state.request_history,
            state.request_window,
        ))
        .style(if is_focused {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        }),
        summary_area,
    );

    if graph_slot.width == 0 || graph_slot.height == 0 {
        return;
    }

    let chart_spec = tui_request_chart_spec(
        &state.request_history,
        state.request_window,
        graph_slot.width,
    );
    frame.render_widget(
        TuiRequestChartWidget {
            chart_spec,
            is_focused,
        },
        graph_slot,
    );
}

pub(in crate::output) fn normalize_request_buckets(
    buckets: &[DashboardAcceptedRequestBucket],
) -> Vec<DashboardAcceptedRequestBucket> {
    let mut counts_by_offset = vec![0_u64; PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize];
    for bucket in buckets {
        let offset = bucket.second_offset as usize;
        if offset < counts_by_offset.len() {
            counts_by_offset[offset] = bucket.accepted_count;
        }
    }

    (0..PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize)
        .map(|index| {
            let second_offset =
                (PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize - 1 - index) as u32;
            DashboardAcceptedRequestBucket {
                second_offset,
                accepted_count: counts_by_offset[second_offset as usize],
            }
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::output) struct TuiRequestChartSpec {
    pub(in crate::output) bucket_values: Vec<u64>,
    pub(in crate::output) bar_width: u16,
    pub(in crate::output) bar_gap: u16,
    pub(in crate::output) visible_bucket_start: usize,
    pub(in crate::output) visible_bucket_count: usize,
    pub(in crate::output) scale_max: u64,
    pub(in crate::output) scale_width: u16,
}

pub(in crate::output) struct TuiRequestChartWidget {
    pub(in crate::output) chart_spec: TuiRequestChartSpec,
    pub(in crate::output) is_focused: bool,
}

impl Widget for TuiRequestChartWidget {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let (scale_area, plot_area) = tui_request_chart_areas(area, &self.chart_spec);
        tui_clear_request_chart_area(area, buf);
        tui_render_request_chart_guides(plot_area, buf, self.is_focused);
        tui_render_request_scale(scale_area, buf, &self.chart_spec, self.is_focused);
        tui_render_request_chart_braille(plot_area, buf, &self.chart_spec, self.is_focused);
    }
}

pub(in crate::output) fn tui_current_rps(history: &DashboardRequestHistoryState) -> u64 {
    history
        .accepted_request_buckets
        .last()
        .map(|bucket| bucket.accepted_count)
        .unwrap_or(0)
}

pub(in crate::output) fn tui_requests_summary_line(
    history: &DashboardRequestHistoryState,
    request_window: DashboardRequestWindow,
) -> Line<'static> {
    let label_style = Style::default().fg(Color::DarkGray);
    let value_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);
    let p50 = tui_p50_latency_ms(&history.latency_samples_ms)
        .map(|latency_ms| format!("{latency_ms}ms"))
        .unwrap_or_else(|| "n/a".to_string());

    Line::from(vec![
        Span::styled("RPS ", label_style),
        Span::styled(tui_current_rps(history).to_string(), value_style),
        Span::raw("  "),
        Span::styled("inflight ", label_style),
        Span::styled(history.current_inflight_requests.to_string(), value_style),
        Span::raw("  "),
        Span::styled("p50 ", label_style),
        Span::styled(p50, value_style),
        Span::raw("  "),
        Span::styled("window ", label_style),
        Span::styled(request_window.label(), value_style),
        Span::raw("  "),
        Span::styled(request_window.bucket_label(), label_style),
    ])
}

pub(in crate::output) fn tui_request_chart_spec(
    history: &DashboardRequestHistoryState,
    request_window: DashboardRequestWindow,
    graph_width: u16,
) -> TuiRequestChartSpec {
    let mut bucket_values = vec![0_u64; PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS];
    let bucket_seconds = request_window.bucket_seconds().max(1);
    let window_seconds = request_window.seconds();
    for bucket in &history.accepted_request_buckets {
        if bucket.second_offset >= window_seconds {
            continue;
        }
        let age_bucket = bucket.second_offset / bucket_seconds;
        let Some(visual_index) =
            PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS.checked_sub(1 + age_bucket as usize)
        else {
            continue;
        };
        if let Some(value) = bucket_values.get_mut(visual_index) {
            *value += bucket.accepted_count;
        }
    }
    let max_bucket_value = bucket_values.iter().copied().max().unwrap_or(0);
    let scale_max = tui_request_scale_ceiling(max_bucket_value);
    let scale_width = tui_request_scale_width(scale_max, graph_width);
    let plot_width = graph_width.saturating_sub(scale_width).max(1);
    let bucket_count = u16::try_from(bucket_values.len())
        .unwrap_or(u16::MAX)
        .max(1);
    let base_bar_width = if plot_width >= bucket_count {
        (plot_width / bucket_count).max(1)
    } else {
        1
    };
    let bar_width = request_window
        .bar_width_cap()
        .map(|cap| base_bar_width.min(cap))
        .unwrap_or(base_bar_width)
        .max(1);
    let remaining_width = plot_width.saturating_sub(bucket_count.saturating_mul(bar_width));
    let bar_gap = if bucket_count > 1 {
        request_window
            .preferred_bar_gap()
            .min(remaining_width / bucket_count.saturating_sub(1))
    } else {
        0
    };
    let slot_width = bar_width.saturating_add(bar_gap).max(1);
    let visible_bucket_count = usize::from(
        plot_width
            .saturating_add(bar_gap)
            .checked_div(slot_width)
            .unwrap_or(0)
            .max(1),
    )
    .min(bucket_values.len());
    TuiRequestChartSpec {
        bucket_values,
        bar_width,
        bar_gap,
        visible_bucket_start: PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS
            .saturating_sub(visible_bucket_count),
        visible_bucket_count,
        scale_max,
        scale_width,
    }
}

pub(in crate::output) fn tui_request_scale_ceiling(max_bucket_value: u64) -> u64 {
    let headroom = max_bucket_value / 5 + 1;
    tui_nice_request_scale(max_bucket_value.saturating_add(headroom))
}

pub(in crate::output) fn tui_nice_request_scale(value: u64) -> u64 {
    let value = value.max(1);
    let mut magnitude = 1_u64;
    while magnitude.saturating_mul(10) <= value {
        magnitude = magnitude.saturating_mul(10);
    }

    for multiplier in [1_u64, 2, 5, 10] {
        let candidate = magnitude.saturating_mul(multiplier);
        if candidate >= value {
            return candidate;
        }
    }
    magnitude.saturating_mul(10)
}

pub(in crate::output) fn tui_request_scale_width(scale_max: u64, graph_width: u16) -> u16 {
    if graph_width < 12 {
        return 0;
    }

    let label_width = u16::try_from(scale_max.to_string().chars().count())
        .unwrap_or(u16::MAX)
        .max(2);
    label_width
        .saturating_add(1)
        .min(graph_width.saturating_sub(1))
}

pub(in crate::output) fn tui_request_chart_areas(
    area: Rect,
    chart_spec: &TuiRequestChartSpec,
) -> (Rect, Rect) {
    let scale_width = chart_spec.scale_width.min(area.width.saturating_sub(1));
    let scale_area = Rect {
        width: scale_width,
        ..area
    };
    let plot_area = Rect {
        x: area.x.saturating_add(scale_width),
        width: area.width.saturating_sub(scale_width),
        ..area
    };
    (scale_area, plot_area)
}

pub(in crate::output) fn tui_clear_request_chart_area(area: Rect, buf: &mut Buffer) {
    let theme = tui_theme();
    let clear_style = Style::default().bg(theme.surface);
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            buf[(x, y)].set_symbol(" ").set_style(clear_style);
        }
    }
}

pub(in crate::output) fn tui_render_request_chart_braille(
    area: Rect,
    buf: &mut Buffer,
    chart_spec: &TuiRequestChartSpec,
    is_focused: bool,
) {
    if area.width == 0 || area.height == 0 || chart_spec.bucket_values.is_empty() {
        return;
    }

    let current_bar_style = Style::default().fg(if is_focused {
        Color::Cyan
    } else {
        Color::Rgb(70, 170, 220)
    });
    let history_bar_style = Style::default().fg(if is_focused {
        Color::Rgb(82, 150, 220)
    } else {
        Color::Rgb(70, 110, 170)
    });
    let vertical_units = u64::from(area.height.max(1)).saturating_mul(4);
    let visible_bucket_count = chart_spec
        .visible_bucket_count
        .min(chart_spec.bucket_values.len());
    let rendered_width = u16::try_from(visible_bucket_count)
        .unwrap_or(u16::MAX)
        .saturating_mul(chart_spec.bar_width)
        .saturating_add(
            u16::try_from(visible_bucket_count.saturating_sub(1))
                .unwrap_or(u16::MAX)
                .saturating_mul(chart_spec.bar_gap),
        );
    let x_origin = area.right().saturating_sub(rendered_width);

    for (visible_index, (index, value)) in chart_spec
        .bucket_values
        .iter()
        .enumerate()
        .skip(chart_spec.visible_bucket_start)
        .take(visible_bucket_count)
        .enumerate()
    {
        if *value == 0 {
            continue;
        }
        let filled_units = value
            .saturating_mul(vertical_units)
            .div_ceil(chart_spec.scale_max.max(1))
            .clamp(1, vertical_units);
        let Ok(visible_index_u16) = u16::try_from(visible_index) else {
            continue;
        };
        let x_start = x_origin.saturating_add(
            visible_index_u16
                .saturating_mul(chart_spec.bar_width.saturating_add(chart_spec.bar_gap)),
        );
        let style = if index + 1 == chart_spec.bucket_values.len() {
            current_bar_style
        } else {
            history_bar_style
        };

        for x_offset in 0..chart_spec.bar_width {
            let x = x_start.saturating_add(x_offset);
            if x >= area.right() {
                continue;
            }
            for row in 0..area.height {
                let y = area.bottom().saturating_sub(1 + row);
                if y < area.top() {
                    continue;
                }
                let cell_base_units = u64::from(row).saturating_mul(4);
                let filled_in_cell = filled_units.saturating_sub(cell_base_units).min(4) as u8;
                if filled_in_cell == 0 {
                    continue;
                }
                let symbol = tui_braille_bar_symbol(filled_in_cell, filled_in_cell);
                let symbol = symbol.to_string();
                buf[(x, y)].set_symbol(&symbol).set_style(style);
            }
        }
    }
}

pub(in crate::output) fn tui_render_request_scale(
    area: Rect,
    buf: &mut Buffer,
    chart_spec: &TuiRequestChartSpec,
    is_focused: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let style = Style::default()
        .fg(if is_focused { theme.muted } else { theme.dim })
        .add_modifier(Modifier::DIM);
    let labels = tui_request_scale_labels(area.height, chart_spec.scale_max);

    for (row, value) in labels {
        let y = area
            .y
            .saturating_add(row)
            .min(area.bottom().saturating_sub(1));
        let label = value.to_string();
        let label_width = u16::try_from(label.chars().count()).unwrap_or(u16::MAX);
        let x = area
            .right()
            .saturating_sub(1)
            .saturating_sub(label_width)
            .max(area.x);
        for (offset, ch) in label.chars().enumerate() {
            let Ok(offset) = u16::try_from(offset) else {
                continue;
            };
            let x = x.saturating_add(offset);
            if x >= area.right() {
                continue;
            }
            let symbol = ch.to_string();
            buf[(x, y)].set_symbol(&symbol).set_style(style);
        }
    }
}

pub(in crate::output) fn tui_request_scale_labels(height: u16, scale_max: u64) -> Vec<(u16, u64)> {
    if height == 0 {
        return Vec::new();
    }

    let mut labels = vec![(0_u16, scale_max)];
    if height > 2 && scale_max > 1 {
        labels.push((height / 2, scale_max / 2));
    }
    if height > 1 {
        labels.push((height.saturating_sub(1), 0));
    }
    labels
}

pub(in crate::output) fn tui_braille_bar_symbol(
    left_filled_dots: u8,
    right_filled_dots: u8,
) -> char {
    const LEFT_BOTTOM_TO_TOP: [u8; 4] = [0x40, 0x04, 0x02, 0x01];
    const RIGHT_BOTTOM_TO_TOP: [u8; 4] = [0x80, 0x20, 0x10, 0x08];

    let mut mask = 0_u32;
    for dot in LEFT_BOTTOM_TO_TOP
        .iter()
        .take(usize::from(left_filled_dots.min(4)))
    {
        mask |= u32::from(*dot);
    }
    for dot in RIGHT_BOTTOM_TO_TOP
        .iter()
        .take(usize::from(right_filled_dots.min(4)))
    {
        mask |= u32::from(*dot);
    }

    char::from_u32(0x2800 + mask).unwrap_or(' ')
}

pub(in crate::output) fn tui_render_request_chart_guides(
    area: Rect,
    buf: &mut Buffer,
    is_focused: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let guide_style = Style::default().fg(if is_focused {
        Color::Rgb(34, 38, 45)
    } else {
        Color::Rgb(26, 30, 36)
    });
    let baseline_style = Style::default().fg(if is_focused {
        Color::Rgb(42, 48, 56)
    } else {
        Color::Rgb(32, 36, 44)
    });

    for y in area.top()..area.bottom() {
        let is_baseline = y + 1 == area.bottom();
        for x in area.left()..area.right() {
            let cell = &mut buf[(x, y)];
            if cell.symbol() != " " {
                continue;
            }

            if is_baseline {
                cell.set_symbol(PRETTY_TUI_REQUEST_GRAPH_BASELINE_SYMBOL)
                    .set_style(baseline_style);
            } else if (x - area.left() + y - area.top()).is_multiple_of(4) {
                cell.set_symbol(PRETTY_TUI_REQUEST_GRAPH_GUIDE_SYMBOL)
                    .set_style(guide_style);
            }
        }
    }
}

pub(in crate::output) fn tui_p50_latency_ms(samples_ms: &[u64]) -> Option<u64> {
    if samples_ms.is_empty() {
        return None;
    }

    let mut sorted = samples_ms.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) / 2)
    }
}
