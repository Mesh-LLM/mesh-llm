use super::super::{ModelProgressState, StartupProgressState};
use super::{
    Alignment, DashboardState, Frame, Line, Modifier, PRETTY_TUI_READY_LOGO_TEXT,
    PRETTY_TUI_SPLASH_ANSI, PRETTY_TUI_SPLASH_TEXT, Paragraph, Rect, Span, StartupLifecycleState,
    Style, Text, event_line, single_line_status_text, truncate_with_ellipsis, tui_theme,
};
use crate::output::formatting::{OutputEventPresentation, format_model_download_progress_message};
use ansi_to_tui::IntoText as _;
use mesh_llm_events::{ModelProgressStatus, OutputEvent};

pub(in crate::output) fn render_model_progress_loader(
    frame: &mut Frame,
    state: &DashboardState,
    area: Rect,
) {
    if area.height < 2 || area.width < 12 {
        return;
    }
    let progress = state.active_loading_progress();
    let logo_text = tui_logo_view(area, false);
    let raw_logo_height = logo_text
        .as_ref()
        .map(|text| u16::try_from(text.lines.len()).unwrap_or(u16::MAX))
        .unwrap_or(0)
        .min(area.height);
    let has_progress = progress.is_some();
    let bar_height = u16::from(has_progress);
    let detail_height = u16::from(has_progress);
    let desired_context_rows =
        u16::try_from((state.startup_history.len().saturating_add(1)).min(10)).unwrap_or(10);
    let max_logo_height = area
        .height
        .saturating_sub(u16::from(has_progress))
        .saturating_sub(bar_height)
        .saturating_sub(detail_height)
        .saturating_sub(desired_context_rows)
        .max(1);
    let logo_height = raw_logo_height.min(max_logo_height);
    let gap_height = u16::from(has_progress && logo_height > 0);
    let base_height = logo_height
        .saturating_add(gap_height)
        .saturating_add(bar_height)
        .saturating_add(detail_height)
        .max(logo_height.max(1));
    let context_lines =
        startup_loader_context_lines(state, area.width, area.height.saturating_sub(base_height));
    let context_height = u16::try_from(context_lines.len()).unwrap_or(u16::MAX);
    let loader_height = base_height.saturating_add(context_height).min(area.height);
    let loader_area = Rect {
        x: area.x,
        y: area.y + area.height.saturating_sub(loader_height) / 2,
        width: area.width,
        height: loader_height,
    };

    let theme = tui_theme();

    if let Some(logo_text) = logo_text {
        let logo_area = Rect {
            x: loader_area.x,
            y: loader_area.y,
            width: loader_area.width,
            height: logo_height,
        };
        frame.render_widget(
            Paragraph::new(logo_text).alignment(Alignment::Center),
            logo_area,
        );
    }

    if let Some(progress) = progress {
        let bar_y = loader_area.y + logo_height + gap_height;
        let bar_area = Rect {
            x: loader_area.x,
            y: bar_y,
            width: loader_area.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(
                loading_progress_bar(
                    progress.ratio,
                    usize::from(bar_area.width).saturating_sub(12),
                ),
                Style::default().fg(theme.accent),
            )]))
            .alignment(Alignment::Center),
            bar_area,
        );

        let detail_area = Rect {
            x: loader_area.x,
            y: bar_y.saturating_add(1),
            width: loader_area.width,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                progress.detail,
                Style::default().fg(theme.muted),
            )))
            .alignment(Alignment::Center),
            detail_area,
        );
    }

    if context_height > 0 {
        let context_y = loader_area.y + logo_height + gap_height + bar_height + detail_height;
        let context_area = Rect {
            x: loader_area.x,
            y: context_y,
            width: loader_area.width,
            height: context_height.min(loader_area.bottom().saturating_sub(context_y)),
        };
        if context_area.height > 0 {
            frame.render_widget(Paragraph::new(context_lines), context_area);
        }
    }
}

pub(in crate::output) fn startup_loader_context_lines(
    state: &DashboardState,
    width: u16,
    available_rows: u16,
) -> Vec<Line<'static>> {
    if available_rows == 0 {
        return Vec::new();
    }

    let content_width = usize::from(width.max(1));
    let mut lines = vec![startup_lifecycle_summary_line(
        &state.startup_lifecycle,
        content_width,
    )];
    lines.extend(
        state
            .startup_history
            .iter()
            .take(usize::from(available_rows).saturating_sub(lines.len()))
            .map(|event| event_line(event, content_width)),
    );
    lines.truncate(usize::from(available_rows));
    lines
}

pub(in crate::output) fn startup_lifecycle_summary_line(
    lifecycle: &StartupLifecycleState,
    width: usize,
) -> Line<'static> {
    let theme = tui_theme();
    let summary = format!(
        "startup={}{}  mesh={}  api={}  console={}  llama-server={}  model readiness={}",
        lifecycle.phase.as_str(),
        lifecycle
            .failure
            .as_ref()
            .map(|failure| format!("  failure={}", single_line_status_text(failure)))
            .unwrap_or_default(),
        lifecycle.mesh.phase.as_str(),
        lifecycle.api.phase.as_str(),
        lifecycle.console.phase.as_str(),
        lifecycle.llama_server.phase.as_str(),
        lifecycle.model_readiness.phase.as_str(),
    );
    Line::from(Span::styled(
        truncate_with_ellipsis(&summary, width),
        Style::default().fg(theme.dim),
    ))
}

pub(in crate::output) fn render_tui_logo(frame: &mut Frame, area: Rect, dimmed: bool) {
    let Some(logo_text) = tui_logo_view(area, dimmed) else {
        return;
    };
    let logo_height = u16::try_from(logo_text.lines.len())
        .unwrap_or(u16::MAX)
        .min(area.height);
    let logo_y = if dimmed {
        area.y
    } else {
        area.y + area.height.saturating_sub(logo_height) / 2
    };
    let logo_area = Rect {
        x: area.x,
        y: logo_y,
        width: area.width,
        height: logo_height,
    };
    frame.render_widget(
        Paragraph::new(logo_text).alignment(if dimmed {
            Alignment::Left
        } else {
            Alignment::Center
        }),
        logo_area,
    );
}

pub(in crate::output) fn tui_logo_view(area: Rect, dimmed: bool) -> Option<Text<'static>> {
    let source = if dimmed {
        tui_ready_logo_text()?
    } else {
        tui_logo_text()?
    };
    Some(tui_crop_logo_text(source, area, dimmed))
}

pub(in crate::output) fn tui_logo_text() -> Option<&'static Text<'static>> {
    PRETTY_TUI_SPLASH_TEXT
        .get_or_init(|| PRETTY_TUI_SPLASH_ANSI.into_text().ok().map(tui_static_text))
        .as_ref()
}

pub(in crate::output) fn tui_static_text(text: Text<'_>) -> Text<'static> {
    Text {
        alignment: text.alignment,
        style: text.style,
        lines: text
            .lines
            .into_iter()
            .map(|line| Line {
                alignment: line.alignment,
                style: line.style,
                spans: line
                    .spans
                    .into_iter()
                    .map(|span| Span {
                        content: span.content.into_owned().into(),
                        style: span.style,
                    })
                    .collect(),
            })
            .collect(),
    }
}

pub(in crate::output) fn tui_ready_logo_text() -> Option<&'static Text<'static>> {
    PRETTY_TUI_READY_LOGO_TEXT
        .get_or_init(|| tui_logo_text().map(tui_trim_logo_text))
        .as_ref()
}

pub(in crate::output) fn tui_trim_logo_text(source: &Text<'static>) -> Text<'static> {
    let first_visible = source
        .lines
        .iter()
        .position(tui_logo_line_has_visible_content)
        .unwrap_or(0);
    let last_visible = source
        .lines
        .iter()
        .rposition(tui_logo_line_has_visible_content)
        .map(|index| index + 1)
        .unwrap_or(source.lines.len());
    let visible_lines = &source.lines[first_visible..last_visible];
    let Some((first_column, last_column)) = tui_logo_visible_columns(visible_lines) else {
        return Text::from(visible_lines.to_vec());
    };
    Text::from(
        visible_lines
            .iter()
            .map(|line| tui_slice_logo_line(line, first_column, last_column))
            .collect::<Vec<_>>(),
    )
}

pub(in crate::output) fn tui_crop_logo_text(
    source: &Text<'static>,
    area: Rect,
    dimmed: bool,
) -> Text<'static> {
    if area.width == 0 || area.height == 0 {
        return Text::default();
    }

    let visible_height = source.lines.len().min(usize::from(area.height));
    let line_start = if dimmed {
        0
    } else {
        source.lines.len().saturating_sub(visible_height) / 2
    };
    let mut lines = Vec::with_capacity(visible_height);
    let dim_patch = dimmed.then(|| Style::default().add_modifier(Modifier::DIM));

    for line in source.lines.iter().skip(line_start).take(visible_height) {
        let mut cropped = tui_crop_logo_line(line, usize::from(area.width));
        if let Some(dim_patch) = dim_patch {
            for span in &mut cropped.spans {
                span.style = span.style.patch(dim_patch);
            }
        }
        lines.push(cropped);
    }

    Text::from(lines)
}

pub(in crate::output) fn tui_crop_logo_line(
    line: &Line<'static>,
    max_width: usize,
) -> Line<'static> {
    if max_width == 0 {
        return Line::default();
    }

    let line_width = tui_logo_line_width(line);
    if line_width <= max_width {
        return line.clone();
    }

    let crop_start = line_width.saturating_sub(max_width) / 2;
    let crop_end = crop_start + max_width;
    let mut spans = Vec::new();
    let mut offset = 0usize;

    for span in &line.spans {
        let span_width = span.content.chars().count();
        let span_start = offset;
        let span_end = offset + span_width;
        let take_start = crop_start.max(span_start);
        let take_end = crop_end.min(span_end);

        if take_start < take_end {
            let content: String = span
                .content
                .chars()
                .skip(take_start - span_start)
                .take(take_end - take_start)
                .collect();
            if !content.is_empty() {
                spans.push(Span::styled(content, span.style));
            }
        }

        offset = span_end;
        if offset >= crop_end {
            break;
        }
    }

    Line::from(spans)
}

pub(in crate::output) fn tui_slice_logo_line(
    line: &Line<'static>,
    start: usize,
    end: usize,
) -> Line<'static> {
    if start >= end {
        return Line::default();
    }

    let mut spans = Vec::new();
    let mut offset = 0usize;

    for span in &line.spans {
        let span_width = span.content.chars().count();
        let span_start = offset;
        let span_end = offset + span_width;
        let take_start = start.max(span_start);
        let take_end = end.min(span_end);

        if take_start < take_end {
            let content: String = span
                .content
                .chars()
                .skip(take_start - span_start)
                .take(take_end - take_start)
                .collect();
            if !content.is_empty() {
                spans.push(Span::styled(content, span.style));
            }
        }

        offset = span_end;
        if offset >= end {
            break;
        }
    }

    Line::from(spans)
}

pub(in crate::output) fn tui_logo_visible_columns(
    lines: &[Line<'static>],
) -> Option<(usize, usize)> {
    let mut first = usize::MAX;
    let mut last = 0usize;

    for line in lines {
        let mut offset = 0usize;
        for span in &line.spans {
            for ch in span.content.chars() {
                if !ch.is_whitespace() {
                    first = first.min(offset);
                    last = last.max(offset + 1);
                }
                offset += 1;
            }
        }
    }

    (first < last).then_some((first, last))
}

pub(in crate::output) fn tui_logo_line_width(line: &Line<'static>) -> usize {
    line.spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum()
}

pub(in crate::output) fn tui_logo_line_has_visible_content(line: &Line<'static>) -> bool {
    line.spans
        .iter()
        .any(|span| span.content.chars().any(|ch| !ch.is_whitespace()))
}

pub(in crate::output) fn loading_progress_bar(ratio: f64, width: usize) -> String {
    let width = width.clamp(8, 40);
    let ratio = ratio.clamp(0.0, 1.0);
    let filled = if ratio == 0.0 {
        0
    } else {
        (ratio * width as f64).round().clamp(1.0, width as f64) as usize
    };
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

pub(in crate::output) fn model_download_progress_ratio(
    progress: &ModelProgressState,
) -> Option<f64> {
    match (progress.downloaded_bytes, progress.total_bytes) {
        (Some(downloaded), Some(total))
            if total > 0 && matches!(progress.status, ModelProgressStatus::Downloading) =>
        {
            Some(downloaded.min(total) as f64 / total as f64)
        }
        _ => None,
    }
}

pub(in crate::output) fn fallback_model_progress_ratio(progress: &ModelProgressState) -> f64 {
    if let Some(ratio) = model_download_progress_ratio(progress) {
        return ratio;
    }

    match progress.status {
        ModelProgressStatus::Ready => 0.85,
        ModelProgressStatus::Downloading => 0.33,
        ModelProgressStatus::Ensuring => 0.20,
    }
}

pub(in crate::output) fn startup_progress_ratio(progress: &StartupProgressState) -> f64 {
    if progress.total_steps == 0 {
        return 0.0;
    }

    progress.completed_steps.min(progress.total_steps) as f64 / progress.total_steps as f64
}

pub(in crate::output) fn loading_progress_detail(
    detail: String,
    ratio: f64,
    steps: Option<(usize, usize)>,
) -> String {
    let percent = (ratio.clamp(0.0, 1.0) * 100.0).round() as usize;
    match steps {
        Some((completed, total)) => format!("{detail}  {percent}% ({completed}/{total})"),
        None => format!("{detail}  {percent}%"),
    }
}

pub(in crate::output) fn startup_progress_event(
    event: &OutputEvent,
) -> Option<(Option<String>, String)> {
    match event {
        OutputEvent::Startup { version, .. } => Some((
            Some("startup".to_string()),
            format!("starting mesh-llm {version}"),
        )),
        OutputEvent::DiscoveryStarting { source } => Some((
            Some("discovery_starting".to_string()),
            format!("discovering mesh via {source}"),
        )),
        OutputEvent::MeshFound { mesh, peers, .. } => Some((
            Some("mesh_found".to_string()),
            format!("found mesh {mesh} with {peers} peer(s)"),
        )),
        OutputEvent::DiscoveryJoined { mesh } => Some((
            Some("discovery_joined".to_string()),
            format!("joined mesh {mesh}"),
        )),
        OutputEvent::WaitingForPeers { detail } => Some((
            Some("waiting_for_peers".to_string()),
            detail
                .clone()
                .unwrap_or_else(|| "waiting for peers".to_string()),
        )),
        OutputEvent::ModelQueued { model } => Some((
            Some(format!("model_queued:{model}")),
            format!("queued model {model}"),
        )),
        OutputEvent::ModelLoading { model, .. } => Some((
            Some(format!("model_loading:{model}")),
            format!("loading model {model}"),
        )),
        OutputEvent::ModelLoaded { model, .. } => Some((
            Some(format!("model_loaded:{model}")),
            format!("loaded model {model}"),
        )),
        OutputEvent::ModelDownloadProgress {
            label,
            file,
            downloaded_bytes,
            total_bytes,
            status,
        } => {
            let progress = ModelProgressState {
                label: label.clone(),
                file: file.clone(),
                downloaded_bytes: *downloaded_bytes,
                total_bytes: *total_bytes,
                status: status.clone(),
            };
            let milestone_key = matches!(status, ModelProgressStatus::Ready)
                .then(|| format!("model_download_ready:{label}"));
            Some((milestone_key, model_progress_detail(&progress)))
        }
        OutputEvent::HostElected { model, host, .. } => Some((
            Some(format!("host_elected:{model}")),
            format!("elected {host} for {model}"),
        )),
        OutputEvent::LlamaStarting {
            model, http_port, ..
        } => Some((
            Some(format!("llama_starting:{}", model_key(model, *http_port))),
            model
                .as_ref()
                .map(|model| format!("starting llama-server for {model}"))
                .unwrap_or_else(|| format!("starting llama-server on port {http_port}")),
        )),
        OutputEvent::LlamaReady { model, port, .. } => Some((
            Some(format!("llama_ready:{}", model_key(model, *port))),
            model
                .as_ref()
                .map(|model| format!("llama-server ready for {model}"))
                .unwrap_or_else(|| format!("llama-server ready on port {port}")),
        )),
        OutputEvent::LlamaStartupFailed {
            model,
            http_port,
            detail,
            ..
        } => Some((
            Some(format!("llama_failed:{}", model_key(model, *http_port))),
            model
                .as_ref()
                .map(|model| format!("llama-server failed for {model}: {detail}"))
                .unwrap_or_else(|| format!("llama-server failed on port {http_port}: {detail}")),
        )),
        OutputEvent::ModelReady { model, .. } => Some((
            Some(format!("model_ready:{model}")),
            format!("model {model} ready"),
        )),
        OutputEvent::WebserverStarting { url } => Some((
            Some("webserver_starting".to_string()),
            format!("starting console at {url}"),
        )),
        OutputEvent::WebserverReady { url } => Some((
            Some("webserver_ready".to_string()),
            format!("console ready at {url}"),
        )),
        OutputEvent::ApiStarting { url } => Some((
            Some("api_starting".to_string()),
            format!("starting API at {url}"),
        )),
        OutputEvent::ApiReady { url } => {
            Some((Some("api_ready".to_string()), format!("API ready at {url}")))
        }
        OutputEvent::RuntimeReady { .. } => Some((
            Some("runtime_ready".to_string()),
            "mesh-llm runtime ready".to_string(),
        )),
        _ => None,
    }
}

pub(in crate::output) fn startup_history_summary(event: &OutputEvent) -> Option<String> {
    match event {
        OutputEvent::Startup { .. }
        | OutputEvent::LaunchPlan { .. }
        | OutputEvent::NodeIdentity { .. }
        | OutputEvent::InviteToken { .. }
        | OutputEvent::DiscoveryStarting { .. }
        | OutputEvent::MeshFound { .. }
        | OutputEvent::DiscoveryJoined { .. }
        | OutputEvent::DiscoveryFailed { .. }
        | OutputEvent::WaitingForPeers { .. }
        | OutputEvent::PassiveMode { .. }
        | OutputEvent::ModelQueued { .. }
        | OutputEvent::ModelLoading { .. }
        | OutputEvent::ModelLoaded { .. }
        | OutputEvent::HostElected { .. }
        | OutputEvent::LlamaStarting { .. }
        | OutputEvent::LlamaReady { .. }
        | OutputEvent::LlamaStartupFailed { .. }
        | OutputEvent::ModelReady { .. }
        | OutputEvent::WebserverStarting { .. }
        | OutputEvent::WebserverReady { .. }
        | OutputEvent::ApiStarting { .. }
        | OutputEvent::ApiReady { .. }
        | OutputEvent::RuntimeReady { .. }
        | OutputEvent::Error { .. }
        | OutputEvent::Warning { .. } => Some(event.summary_line()),
        OutputEvent::ModelDownloadProgress { status, .. } => {
            if matches!(status, ModelProgressStatus::Ready) {
                Some(event.summary_line())
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(in crate::output) fn is_shutdown_suppressed_ready_event(event: &OutputEvent) -> bool {
    matches!(
        event,
        OutputEvent::LlamaReady { .. }
            | OutputEvent::ModelReady { .. }
            | OutputEvent::WebserverReady { .. }
            | OutputEvent::ApiReady { .. }
            | OutputEvent::RuntimeReady { .. }
    )
}

pub(in crate::output) fn model_key(model: &Option<String>, port: u16) -> String {
    model
        .as_ref()
        .cloned()
        .unwrap_or_else(|| format!("port:{port}"))
}

pub(in crate::output) fn model_progress_detail(progress: &ModelProgressState) -> String {
    let target = progress.file.as_deref().unwrap_or(&progress.label);
    format_model_download_progress_message(
        &progress.label,
        Some(target),
        progress.downloaded_bytes,
        progress.total_bytes,
        &progress.status,
    )
}
