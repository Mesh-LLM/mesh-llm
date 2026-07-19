use super::super::DashboardJoinTokenCopyStatus;
use super::{
    Alignment, DashboardPanel, DashboardState, Frame, Line, Modifier,
    PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL, PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING, Padding,
    Paragraph, Rect, Span, Style, Text, format_tui_panel_title, single_line_status_text,
    truncate_with_ellipsis, tui_panel_block, tui_theme, wrap_plain_text,
};

pub(in crate::output) fn render_join_token_panel(
    frame: &mut Frame,
    state: &DashboardState,
    panel_area: Rect,
    copy_button_area: Rect,
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

    if state.full_screen_panel == Some(DashboardPanel::JoinToken) {
        let token_area = join_token_full_screen_text_area(panel_area);
        if token_area.width > 0 && token_area.height > 0 {
            frame.render_widget(
                Paragraph::new(join_token_wrapped_text(
                    state,
                    usize::from(token_area.width),
                ))
                .style(Style::default().fg(theme.text)),
                token_area,
            );
        }
    } else {
        let token_area = join_token_text_area(panel_area, copy_button_area);

        let token_line = join_token_line(state, usize::from(token_area.width));
        frame.render_widget(
            Paragraph::new(token_line).style(Style::default().fg(theme.text)),
            token_area,
        );
    }

    if copy_button_area.width > 0 && copy_button_area.height > 0 {
        let copy_enabled = state.join_token.is_some();
        let (button_label, button_style) = match state
            .join_token
            .as_ref()
            .map(|join_token| &join_token.copy_status)
        {
            Some(DashboardJoinTokenCopyStatus::Copied { .. }) => (
                " Copied ",
                Style::default()
                    .fg(theme.surface)
                    .bg(theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
            Some(DashboardJoinTokenCopyStatus::Failed { .. }) => (
                " Failed ",
                Style::default()
                    .fg(theme.surface)
                    .bg(theme.error)
                    .add_modifier(Modifier::BOLD),
            ),
            _ if copy_enabled => (
                PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL,
                Style::default()
                    .fg(theme.surface)
                    .bg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            _ => (
                PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL,
                Style::default().fg(theme.dim).bg(theme.surface_raised),
            ),
        };
        frame.render_widget(
            Paragraph::new(button_label)
                .style(button_style)
                .alignment(Alignment::Center),
            copy_button_area,
        );
    }
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
    let Some(join_token) = &state.join_token else {
        return "waiting for cluster invite".to_string();
    };
    match &join_token.copy_status {
        DashboardJoinTokenCopyStatus::Idle => "press c to copy".to_string(),
        DashboardJoinTokenCopyStatus::Copied { .. } => "copied to clipboard".to_string(),
        DashboardJoinTokenCopyStatus::Failed { message, .. } => {
            format!("copy failed: {}", truncate_with_ellipsis(message, 40))
        }
    }
}

pub(in crate::output) fn join_token_line(state: &DashboardState, width: usize) -> Line<'static> {
    let theme = tui_theme();
    if let Some(join_token) = &state.join_token {
        let token_width = width.saturating_sub(6).max(1);
        let scroll_offset = state
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset;
        Line::from(vec![
            Span::styled("token ", Style::default().fg(theme.muted)),
            Span::styled(
                join_token_visible_slice(&join_token.token, scroll_offset, token_width),
                Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
            ),
        ])
    } else {
        Line::styled(
            "join token will appear here when the mesh invite is ready",
            Style::default().fg(theme.muted),
        )
    }
}

pub(in crate::output) fn join_token_wrapped_text(
    state: &DashboardState,
    width: usize,
) -> Text<'static> {
    let theme = tui_theme();
    if let Some(join_token) = &state.join_token {
        let token_width = width.saturating_sub(6).max(1);
        let wrapped = wrap_plain_text(&join_token.token, token_width);
        let lines = wrapped
            .into_iter()
            .enumerate()
            .map(|(index, chunk)| {
                let prefix = if index == 0 { "token " } else { "      " };
                Line::from(vec![
                    Span::styled(prefix, Style::default().fg(theme.muted)),
                    Span::styled(
                        chunk,
                        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
                    ),
                ])
            })
            .collect::<Vec<_>>();
        Text::from(lines)
    } else {
        Text::from(Line::styled(
            "join token will appear here when the mesh invite is ready",
            Style::default().fg(theme.muted),
        ))
    }
}

pub(in crate::output) fn join_token_text_area(panel_area: Rect, copy_button_area: Rect) -> Rect {
    if panel_area.width == 0 || panel_area.height < 3 {
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
    let inner_y = panel_area.y.saturating_add(panel_area.height / 2);
    let inner_right = panel_area
        .right()
        .saturating_sub(1)
        .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING);
    let token_right = if copy_button_area.width > 0 {
        copy_button_area.x.saturating_sub(1).min(inner_right)
    } else {
        inner_right
    };
    Rect {
        x: inner_x,
        y: inner_y,
        width: token_right.saturating_sub(inner_x),
        height: 1,
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

pub(in crate::output) fn join_token_content_width(panel_area: Rect, copy_button_area: Rect) -> u16 {
    join_token_text_area(panel_area, copy_button_area)
        .width
        .saturating_sub(6)
}

pub(in crate::output) fn join_token_char_count(token: &str) -> usize {
    token.chars().count()
}

pub(in crate::output) fn join_token_visible_slice(
    token: &str,
    scroll_offset: usize,
    width: usize,
) -> String {
    let token_len = join_token_char_count(token);
    if width == 0 || token_len == 0 {
        return String::new();
    }

    let hidden_left = scroll_offset > 0;
    let hidden_right = scroll_offset.saturating_add(width) < token_len;
    if !hidden_left && !hidden_right {
        return token.to_string();
    }
    if width == 1 {
        return "…".to_string();
    }

    let indicator_count = usize::from(hidden_left) + usize::from(hidden_right);
    let visible_width = width.saturating_sub(indicator_count);
    let mut visible = String::with_capacity(width);
    if hidden_left {
        visible.push('…');
    }
    visible.extend(token.chars().skip(scroll_offset).take(visible_width));
    if hidden_right {
        visible.push('…');
    }
    visible
}

pub(in crate::output) fn tui_join_token_copy_button_area(panel_area: Rect) -> Rect {
    if panel_area.width == 0 || panel_area.height < 3 {
        return Rect {
            x: panel_area.x,
            y: panel_area.y,
            width: 0,
            height: 0,
        };
    }
    let button_width = u16::try_from(PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL.chars().count())
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .min(panel_area.width.saturating_sub(2));
    Rect {
        x: panel_area
            .right()
            .saturating_sub(button_width)
            .saturating_sub(1)
            .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING),
        y: panel_area.y.saturating_add(panel_area.height / 2),
        width: button_width,
        height: 1,
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

pub(in crate::output) fn copy_join_token_to_clipboard(token: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|err| err.to_string())?;
    clipboard
        .set_text(token.to_string())
        .map_err(|err| err.to_string())
}
