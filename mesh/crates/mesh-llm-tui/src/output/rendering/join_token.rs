use super::{
    DashboardPanel, DashboardState, Frame, Line, Modifier,
    PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING, Padding, Paragraph, Rect, Style, Text,
    format_tui_panel_title, single_line_status_text, truncate_with_ellipsis, tui_panel_block,
    tui_theme, wrap_plain_text,
};

pub(in crate::output) fn render_join_token_panel(
    frame: &mut Frame,
    state: &DashboardState,
    panel_area: Rect,
    _copy_button_area: Rect,
) {
    if panel_area.width == 0 || panel_area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let block = tui_panel_block(state, DashboardPanel::JoinToken).padding(Padding::horizontal(
        PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING,
    ));
    let inner_area = block.inner(panel_area);
    frame.render_widget(block, panel_area);
    render_join_token_title_status(frame, state, panel_area);

    if inner_area.height == 0 || inner_area.width == 0 {
        return;
    }

    let token_area = if state.full_screen_panel == Some(DashboardPanel::JoinToken) {
        join_token_full_screen_text_area(panel_area)
    } else {
        join_token_text_area(panel_area)
    };
    if token_area.width == 0 || token_area.height == 0 {
        return;
    }
    frame.render_widget(
        Paragraph::new(join_token_wrapped_text(
            state,
            usize::from(token_area.width),
        ))
        .style(Style::default().fg(theme.text)),
        token_area,
    );
}

pub(in crate::output) fn render_join_token_title_status(
    frame: &mut Frame,
    state: &DashboardState,
    panel_area: Rect,
) {
    if panel_area.width <= 4 || panel_area.height == 0 {
        return;
    }

    let theme = tui_theme();
    let left_title_width = format_tui_panel_title(state, DashboardPanel::JoinToken)
        .chars()
        .count();
    let max_status_width = usize::from(panel_area.width)
        .saturating_sub(left_title_width.saturating_add(5))
        .max(1);
    let status = truncate_with_ellipsis(&join_token_panel_right_title(state), max_status_width);
    let title = format!(" {status} ");
    let title_width = u16::try_from(title.chars().count())
        .unwrap_or(u16::MAX)
        .min(panel_area.width.saturating_sub(2));
    if title_width == 0 {
        return;
    }

    let title_area = Rect {
        x: panel_area
            .right()
            .saturating_sub(title_width)
            .saturating_sub(1),
        y: panel_area.y,
        width: title_width,
        height: 1,
    };
    frame.render_widget(
        Paragraph::new(Line::styled(
            title,
            Style::default()
                .fg(theme.muted)
                .bg(theme.surface_raised)
                .add_modifier(Modifier::BOLD),
        )),
        title_area,
    );
}

pub(in crate::output) fn join_token_panel_left_title(
    state: &DashboardState,
    focus_marker: char,
) -> String {
    let mut title = format!(
        "{focus_marker} Join Token  startup={}",
        state.startup_lifecycle.phase.as_str()
    );
    if let Some(join_token) = &state.join_token {
        title.push_str("  mesh=");
        title.push_str(&join_token.mesh_label());
    }
    title
}

pub(in crate::output) fn join_token_panel_right_title(state: &DashboardState) -> String {
    if let Some(failure) = state.startup_lifecycle.failure.as_ref() {
        return format!(
            "startup failed: {}",
            truncate_with_ellipsis(&single_line_status_text(failure), 40)
        );
    }
    if state.join_token.is_some() {
        "private mesh invite".to_string()
    } else {
        "waiting for cluster invite".to_string()
    }
}

pub(in crate::output) fn join_token_message(state: &DashboardState) -> String {
    match &state.join_token {
        Some(join_token) => format!(
            "Invite for mesh {} (keep private):\n{}",
            join_token.mesh_label(),
            join_token.token
        ),
        None => "Join metadata will appear here when the mesh invite is ready.".to_string(),
    }
}

pub(in crate::output) fn join_token_wrapped_text(
    state: &DashboardState,
    width: usize,
) -> Text<'static> {
    let theme = tui_theme();
    let lines = wrap_plain_text(&join_token_message(state), width.max(1))
        .into_iter()
        .map(|chunk| Line::styled(chunk, Style::default().fg(theme.muted)))
        .collect::<Vec<_>>();
    Text::from(lines)
}

pub(in crate::output) fn join_token_text_area(panel_area: Rect) -> Rect {
    if panel_area.width == 0 || panel_area.height < 4 {
        return Rect {
            x: panel_area.x,
            y: panel_area.y,
            width: 0,
            height: 0,
        };
    }

    let inner_x = panel_area
        .x
        .saturating_add(1)
        .saturating_add(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING);
    let inner_right = panel_area
        .right()
        .saturating_sub(1)
        .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING);
    Rect {
        x: inner_x,
        y: panel_area.y.saturating_add(1),
        width: inner_right.saturating_sub(inner_x),
        height: panel_area.height.saturating_sub(2),
    }
}

pub(in crate::output) fn join_token_full_screen_text_area(panel_area: Rect) -> Rect {
    if panel_area.width == 0 || panel_area.height < 4 {
        return Rect {
            x: panel_area.x,
            y: panel_area.y,
            width: 0,
            height: 0,
        };
    }

    let inner_x = panel_area
        .x
        .saturating_add(1)
        .saturating_add(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING);
    let inner_right = panel_area
        .right()
        .saturating_sub(1)
        .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING);
    Rect {
        x: inner_x,
        y: panel_area.y.saturating_add(2),
        width: inner_right.saturating_sub(inner_x),
        height: panel_area.height.saturating_sub(3),
    }
}

pub(in crate::output) fn point_in_rect(column: u16, row: u16, rect: Rect) -> bool {
    rect.width > 0
        && rect.height > 0
        && column >= rect.left()
        && column < rect.right()
        && row >= rect.top()
        && row < rect.bottom()
}
