use super::*;
use crate::output::formatting::*;
use crate::output::rendering::*;
use ratatui::{
    Terminal,
    backend::TestBackend,
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    widgets::{Paragraph, Widget},
};
use serde_json::Value;
use std::{
    io::Write as _,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::time::Instant;

mod dashboard;
mod formatting;
mod merging;
mod native_visibility;
mod rendering;

struct StaticDashboardSnapshotProvider {
    snapshot: DashboardSnapshot,
}

impl DashboardSnapshotProvider for StaticDashboardSnapshotProvider {
    fn snapshot(&self) -> DashboardSnapshotFuture<'_> {
        let snapshot = self.snapshot.clone();
        Box::pin(async move { snapshot })
    }
}

#[derive(Default)]
struct DashboardReducerFixture {
    state: DashboardState,
}

impl DashboardReducerFixture {
    fn with_snapshot(mut self, snapshot: DashboardSnapshot) -> Self {
        self.state
            .reduce(DashboardAction::SnapshotUpdated(snapshot));
        self
    }

    fn with_events<I>(mut self, events: I) -> Self
    where
        I: IntoIterator<Item = OutputEvent>,
    {
        for event in events {
            self.state.reduce(DashboardAction::OutputEvent(event));
        }
        self
    }

    fn reduce(&mut self, action: DashboardAction) {
        self.state.reduce(action);
    }
}

fn sample_process_row(name: &str, port: u16) -> DashboardProcessRow {
    DashboardProcessRow {
        name: name.to_string(),
        backend: "metal".to_string(),
        status: RuntimeStatus::Ready,
        port,
        pid: u32::from(port) + 1000,
    }
}

fn sample_endpoint_row(label: &str, port: u16) -> DashboardEndpointRow {
    DashboardEndpointRow {
        label: label.to_string(),
        status: RuntimeStatus::Ready,
        url: format!("http://127.0.0.1:{port}"),
        port,
        pid: None,
    }
}

fn sample_model_row(name: &str, port: u16) -> DashboardModelRow {
    DashboardModelRow {
        name: name.to_string(),
        role: Some("host".to_string()),
        status: RuntimeStatus::Ready,
        port: Some(port),
        device: Some("GPU0".to_string()),
        slots: Some(4),
        quantization: Some("Q4_K_M".to_string()),
        ctx_size: Some(8192),
        ctx_used_tokens: Some(8192),
        lanes: Some(vec![
            DashboardModelLane {
                index: 0,
                active: true,
            },
            DashboardModelLane {
                index: 1,
                active: true,
            },
            DashboardModelLane {
                index: 2,
                active: false,
            },
            DashboardModelLane {
                index: 3,
                active: false,
            },
        ]),
        file_size_gb: Some(24.0),
    }
}

fn half_scale_model_row() -> DashboardModelRow {
    DashboardModelRow {
        name: "Half-Scale".to_string(),
        role: Some("host".to_string()),
        status: RuntimeStatus::Ready,
        port: Some(4002),
        device: Some("CUDA0".to_string()),
        slots: Some(8),
        quantization: Some("Q5_K_M".to_string()),
        ctx_size: Some(4096),
        ctx_used_tokens: Some(2048),
        lanes: Some(
            (0..8)
                .map(|index| DashboardModelLane {
                    index,
                    active: index == 0,
                })
                .collect(),
        ),
        file_size_gb: Some(12.0),
    }
}

fn line_x(line: &str, needle: &str, description: &str) -> usize {
    line.find(needle)
        .map(|index| line[..index].chars().count())
        .expect(description)
}

fn filled_gauge_bounds(line: &str, value_label: &str) -> (usize, usize, usize) {
    let gauge_byte = line.find('█').expect("expected gauge byte coordinate");
    let gauge_x = line[..gauge_byte].chars().count();
    let bar_end_x = gauge_x
        + line[gauge_byte..]
            .chars()
            .take_while(|ch| *ch == '█')
            .count();
    let value_x = line_x(line, value_label, "expected value label x coordinate");
    (gauge_x, bar_end_x, value_x)
}

fn first_block_x(line: &str, description: &str) -> usize {
    line.find('◼')
        .map(|index| line[..index].chars().count())
        .expect(description)
}

fn assert_segmented_model_card_layout(rendered: &str, buffer: &Buffer, theme: &TuiTheme) {
    let (full_title_y, full_title_line) = find_rendered_line(rendered, "Segmented-Model");
    let full_border_line = rendered
        .lines()
        .nth(full_title_y.saturating_sub(1))
        .expect("expected card border above model name");
    assert!(
        full_border_line.contains("│╭"),
        "expected model card to start flush against the panel content edge, without a highlight gutter, in {full_border_line}"
    );
    assert!(
        !full_title_line.contains("PORT:"),
        "model name should have its own interior row before metadata: {full_title_line}"
    );
    let (full_ctx_y, full_ctx_line) =
        find_rendered_line_after(rendered, full_title_y, "8192 / 8192");
    let (full_slots_y, full_slots_line) = find_rendered_line_after(rendered, full_ctx_y, "2 / 4");
    let (_, divider_line) = find_rendered_line_after(rendered, full_title_y, "──");
    assert!(
        !divider_line.contains('├') && !divider_line.contains('┤'),
        "expected subtle interior divider, not frame-joining divider, in {divider_line}"
    );
    assert!(
        full_ctx_line.contains("CTX") && full_ctx_line.contains("8192 / 8192"),
        "expected CTX row with right-aligned value label in {full_ctx_line}"
    );
    assert!(
        full_slots_line.contains("SLOTS") && full_slots_line.contains("2 / 4"),
        "expected SLOTS row with right-aligned value label in {full_slots_line}"
    );

    let (full_ctx_gauge_x, full_ctx_bar_end_x, full_ctx_value_x) =
        filled_gauge_bounds(full_ctx_line, "8192 / 8192");
    let full_slots_block_x = first_block_x(full_slots_line, "expected SLOTS block byte coordinate");
    let full_slots_value_x = line_x(
        full_slots_line,
        "2 / 4",
        "expected SLOTS value label x coordinate",
    );
    let full_slots_label_x = line_x(
        full_slots_line,
        "SLOTS",
        "expected SLOTS label x coordinate",
    );
    assert!(
        full_ctx_bar_end_x < full_ctx_value_x && full_slots_block_x < full_slots_value_x,
        "expected a visible gap between metric visuals and value labels: {full_ctx_line} / {full_slots_line}"
    );
    assert!(
        full_slots_block_x > full_slots_label_x + "SLOTS".chars().count(),
        "expected visible gap between SLOTS label and slot blocks: {full_slots_line}"
    );
    assert_eq!(
        buffer[(
            u16::try_from(full_slots_block_x + 1).unwrap(),
            u16::try_from(full_slots_y).unwrap()
        )]
            .symbol(),
        "◼",
        "expected adjacent visible slot blocks without separators"
    );
    assert_eq!(
        buffer[(
            u16::try_from(full_ctx_gauge_x).unwrap(),
            u16::try_from(full_ctx_y).unwrap()
        )]
            .style()
            .fg,
        Some(tui_model_usage_color(1.0))
    );
    assert_eq!(
        buffer[(
            u16::try_from(full_slots_block_x).unwrap(),
            u16::try_from(full_slots_y).unwrap()
        )]
            .style()
            .fg,
        Some(theme.warning)
    );
    assert_eq!(
        buffer[(
            u16::try_from(full_slots_block_x + 2).unwrap(),
            u16::try_from(full_slots_y).unwrap()
        )]
            .style()
            .fg,
        Some(theme.dim)
    );
}

fn assert_half_scale_model_card_segments(half_buffer: &Buffer, theme: &TuiTheme) {
    let half_rendered = buffer_to_rendered_string(half_buffer);
    let (half_title_y, _) = find_rendered_line(&half_rendered, "Half-Scale");
    let (half_ctx_y, half_ctx_line) =
        find_rendered_line_after(&half_rendered, half_title_y, "2048 / 4096");
    let (half_slots_y, half_slots_line) =
        find_rendered_line_after(&half_rendered, half_ctx_y, "1 / 8");
    let (half_ctx_gauge_x, _, ctx_value_x) = filled_gauge_bounds(half_ctx_line, "2048 / 4096");
    let half_slots_block_x = first_block_x(
        half_slots_line,
        "expected half-scale SLOTS block x coordinate",
    );
    let slots_value_x = line_x(
        half_slots_line,
        "1 / 8",
        "expected half SLOTS value label x coordinate",
    );
    assert_eq!(
        half_buffer[(
            u16::try_from(half_ctx_gauge_x).unwrap(),
            u16::try_from(half_ctx_y).unwrap()
        )]
            .style()
            .fg,
        Some(tui_model_usage_color(0.5))
    );
    assert_eq!(
        half_buffer[(
            u16::try_from(half_slots_block_x).unwrap(),
            u16::try_from(half_slots_y).unwrap()
        )]
            .style()
            .fg,
        Some(theme.warning)
    );
    assert!(
        ((half_ctx_gauge_x + 1)..ctx_value_x).any(|x| {
            half_buffer[(
                u16::try_from(x).unwrap(),
                u16::try_from(half_ctx_y).unwrap(),
            )]
                .style()
                .fg
                == Some(theme.dim)
        }),
        "expected CTX usage bar to show grey empty track after the fill"
    );
    assert!(
        ((half_slots_block_x + 1)..slots_value_x).any(|x| {
            half_buffer[(
                u16::try_from(x).unwrap(),
                u16::try_from(half_slots_y).unwrap(),
            )]
                .style()
                .fg
                == Some(theme.dim)
        }),
        "expected SLOTS row to show grey inactive blocks after the active lane"
    );
    assert!(
        half_slots_line.contains("◼◼") && !half_slots_line.contains("◼ ◼"),
        "expected slot blocks to render adjacently without separators: {half_slots_line}"
    );
}

fn sample_launch_plan() -> DashboardLaunchPlan {
    DashboardLaunchPlan {
        llama_process_rows: vec![DashboardProcessRow {
            name: "llama-server".to_string(),
            backend: String::new(),
            status: RuntimeStatus::Loading,
            port: 0,
            pid: 0,
        }],
        webserver_rows: vec![
            DashboardEndpointRow {
                label: "Console".to_string(),
                status: RuntimeStatus::NotReady,
                url: "http://localhost:3131".to_string(),
                port: 3131,
                pid: None,
            },
            DashboardEndpointRow {
                label: "API".to_string(),
                status: RuntimeStatus::NotReady,
                url: "http://localhost:9337".to_string(),
                port: 9337,
                pid: None,
            },
        ],
        loaded_model_rows: vec![DashboardModelRow {
            name: "Planned-Model".to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Loading,
            port: None,
            device: Some("GPU0".to_string()),
            slots: Some(4),
            quantization: Some("Q4_K_M".to_string()),
            ctx_size: Some(8192),
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: Some(7.5),
        }],
    }
}

fn port_zero_endpoint_launch_plan() -> DashboardLaunchPlan {
    DashboardLaunchPlan {
        llama_process_rows: Vec::new(),
        webserver_rows: vec![
            DashboardEndpointRow {
                label: "Plugin: alpha".to_string(),
                status: RuntimeStatus::Ready,
                url: "alpha-plugin".to_string(),
                port: 0,
                pid: Some(1000),
            },
            DashboardEndpointRow {
                label: "Plugin: beta".to_string(),
                status: RuntimeStatus::Ready,
                url: "beta-plugin".to_string(),
                port: 0,
                pid: Some(1002),
            },
            DashboardEndpointRow {
                label: "Plugin: zebra".to_string(),
                status: RuntimeStatus::Ready,
                url: "zebra-plugin".to_string(),
                port: 0,
                pid: Some(1001),
            },
        ],
        loaded_model_rows: Vec::new(),
    }
}

fn snapshot_fixture(model_rows: usize, request_buckets: usize) -> DashboardSnapshot {
    DashboardSnapshot {
        llama_process_rows: vec![sample_process_row("llama-server", 8001)],
        webserver_rows: vec![
            sample_endpoint_row("Console", 3131),
            sample_endpoint_row("API", 9337),
        ],
        loaded_model_rows: (0..model_rows)
            .map(|index| sample_model_row(&format!("Model-{index}"), 4000 + index as u16))
            .collect(),
        current_inflight_requests: 3,
        accepted_request_buckets: (0..request_buckets)
            .map(|second_offset| DashboardAcceptedRequestBucket {
                second_offset: second_offset as u32,
                accepted_count: second_offset as u64,
            })
            .collect(),
        latency_samples_ms: vec![11, 17, 19, 23],
    }
}

fn info_event(message: impl Into<String>) -> OutputEvent {
    OutputEvent::Info {
        message: message.into(),
        context: None,
    }
}

fn sample_events_covering_all_variants() -> Vec<OutputEvent> {
    vec![
            OutputEvent::Info {
                message: "mesh is private by default".to_string(),
                context: Some("publish=false".to_string()),
            },
            OutputEvent::Startup {
                version: "v0.64.0".to_string(),
                message: Some("mesh-llm starting".to_string()),
            },
            OutputEvent::LaunchPlan {
                plan: sample_launch_plan(),
            },
            OutputEvent::NodeIdentity {
                node_id: "node-123".to_string(),
                mesh_id: Some("mesh-abc".to_string()),
            },
            OutputEvent::InviteToken {
                token: "invite-token-123".to_string(),
                mesh_id: "mesh-abc".to_string(),
                mesh_name: None,
            },
            OutputEvent::DiscoveryStarting {
                source: "Nostr re-discovery".to_string(),
            },
            OutputEvent::MeshFound {
                mesh: "mesh-abc".to_string(),
                peers: 7,
                region: Some("us-west".to_string()),
            },
            OutputEvent::DiscoveryJoined {
                mesh: "mesh-abc".to_string(),
            },
            OutputEvent::DiscoveryFailed {
                message: "Could not re-join any mesh".to_string(),
                detail: Some("relay timeout".to_string()),
            },
            OutputEvent::WaitingForPeers {
                detail: Some("waiting for two more peers".to_string()),
            },
            OutputEvent::PassiveMode {
                role: "standby".to_string(),
                status: RuntimeStatus::Starting,
                capacity_gb: Some(24.0),
                models_on_disk: Some(vec!["Qwen2.5-32B".to_string(), "GLM-4.7-Flash".to_string()]),
                detail: Some("No matching model on disk — running as standby GPU node".to_string()),
            },
            OutputEvent::PeerJoined {
                peer_id: "peer-1".to_string(),
                label: Some("lab-gpu-1".to_string()),
            },
            OutputEvent::PeerLeft {
                peer_id: "peer-2".to_string(),
                reason: Some("shutdown".to_string()),
            },
            OutputEvent::ModelQueued {
                model: "Qwen3-32B".to_string(),
            },
            OutputEvent::ModelLoading {
                model: "Qwen3-32B".to_string(),
                source: Some("huggingface".to_string()),
            },
            OutputEvent::ModelLoaded {
                model: "Qwen3-32B".to_string(),
                bytes: Some(24_012_755_755),
            },
            OutputEvent::HostElected {
                model: "Qwen3-32B".to_string(),
                host: "node-7".to_string(),
                role: Some("host".to_string()),
                capacity_gb: Some(24.0),
            },
            OutputEvent::RpcServerStarting {
                port: 43683,
                device: "CUDA0".to_string(),
                log_path: Some("/tmp/rpc.log".to_string()),
            },
            OutputEvent::RpcReady {
                port: 43683,
                device: "CUDA0".to_string(),
                log_path: Some("/tmp/rpc.log".to_string()),
            },
            OutputEvent::LlamaStarting {
                model: Some("Qwen3-32B".to_string()),
                http_port: 8001,
                ctx_size: Some(8192),
                log_path: Some("/tmp/llama.log".to_string()),
            },
            OutputEvent::LlamaReady {
                model: Some("Qwen3-32B".to_string()),
                port: 8001,
                ctx_size: Some(8192),
                log_path: Some("/tmp/llama.log".to_string()),
            },
            OutputEvent::ModelReady {
                model: "Qwen3-32B".to_string(),
                internal_port: Some(38373),
                role: Some("host".to_string()),
            },
            OutputEvent::MultiModelMode {
                count: 2,
                models: vec!["Qwen3-32B".to_string(), "GLM-4.7-Flash".to_string()],
            },
            OutputEvent::WebserverStarting {
                url: "http://localhost:3131".to_string(),
            },
            OutputEvent::WebserverReady {
                url: "http://localhost:3131".to_string(),
            },
            OutputEvent::ApiStarting {
                url: "http://localhost:9337".to_string(),
            },
            OutputEvent::ApiReady {
                url: "http://localhost:9337".to_string(),
            },
            OutputEvent::RuntimeReady {
                api_url: "http://localhost:9337".to_string(),
                console_url: Some("http://localhost:3131".to_string()),
                api_port: 9337,
                console_port: Some(3131),
                models_count: Some(2),
                pi_command: Some("mesh-llm pi --host 127.0.0.1:9337 --model 'Qwen3-32B'".to_string()),
                goose_command: Some("GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen3-32B goose session".to_string()),
            },
            OutputEvent::ModelDownloadProgress {
                label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
                file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
                downloaded_bytes: Some(245_500_000),
                total_bytes: Some(491_000_000),
                status: ModelProgressStatus::Downloading,
            },
            OutputEvent::RequestRouted {
                model: "Qwen3-32B".to_string(),
                target: "peer-7".to_string(),
            },
            OutputEvent::Warning {
                message: "⚠️ legacy warning prefix still present".to_string(),
                context: Some("model=Qwen3-32B".to_string()),
            },
            OutputEvent::Error {
                message: "❌ llama-server exited".to_string(),
                context: Some("model=Qwen3-32B port=9337".to_string()),
            },
            OutputEvent::Fatal {
                message: "panic occurred".to_string(),
                context: Some("panic at crates/mesh-llm/src/lib.rs:42".to_string()),
            },
            OutputEvent::Shutdown {
                reason: Some("user requested shutdown".to_string()),
            },
        ]
}

fn sample_mesh_event_states(count: usize) -> Vec<MeshEventState> {
    (0..count)
        .map(|index| MeshEventState {
            timestamp: format!("12:34:{index:02}"),
            level: OutputLevel::Info,
            summary: format!("event-{index:02} tdd-scroll-marker"),
        })
        .collect()
}

fn render_scrollbar_event_list_widget_snapshot(
    events: &[MeshEventState],
    scroll_offset: usize,
    width: u16,
    height: u16,
) -> String {
    let event_refs = events.iter().collect::<Vec<_>>();
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            frame.render_widget(
                TuiScrollbarEventList {
                    events: &event_refs,
                    empty_message: "(waiting for mesh events)",
                    scroll_offset,
                    wrap_lines: false,
                },
                frame.area(),
            );
        })
        .unwrap();
    test_buffer_to_string(terminal.backend().buffer(), width, height)
}

fn render_events_panel_with_renderer_snapshot(
    state: &DashboardState,
    renderer: TuiEventListRenderer,
    width: u16,
    height: u16,
) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            let title_area = Rect {
                x: 0,
                y: 0,
                width,
                height: 1,
            };
            let body_area = Rect {
                x: 0,
                y: 1,
                width,
                height: height.saturating_sub(1),
            };
            render_events_panel_with_renderer(frame, state, title_area, body_area, renderer);
        })
        .unwrap();
    test_buffer_to_string(terminal.backend().buffer(), width, height)
}

fn test_buffer_to_string(buffer: &ratatui::buffer::Buffer, width: u16, height: u16) -> String {
    let mut lines = Vec::with_capacity(usize::from(height));
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            line.push_str(buffer[(x, y)].symbol());
        }
        lines.push(line.trim_end().to_string());
    }
    lines.join("\n")
}

fn assert_hh_mm_ss(text: &str) {
    assert_eq!(text.len(), 8, "timestamp should be HH:MM:SS, got {text}");
    for (index, ch) in text.chars().enumerate() {
        match index {
            2 | 5 => assert_eq!(ch, ':', "timestamp should use colon separators: {text}"),
            _ => assert!(
                ch.is_ascii_digit(),
                "timestamp should contain digits: {text}"
            ),
        }
    }
}

fn render_tui_frame_snapshot(state: &DashboardState, width: u16, height: u16) -> String {
    let backend = ratatui::backend::TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| render_tui_frame(frame, state))
        .expect("frame render should succeed");
    let buffer = terminal.backend().buffer();
    let mut lines = Vec::with_capacity(usize::from(height));
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            line.push_str(buffer[(x, y)].symbol());
        }
        lines.push(line.trim_end().to_string());
    }
    lines.join("\n")
}

fn render_tui_frame_snapshot_with_buffer(
    state: &DashboardState,
    width: u16,
    height: u16,
) -> (String, ratatui::buffer::Buffer) {
    let backend = ratatui::backend::TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| render_tui_frame(frame, state))
        .expect("frame render should succeed");
    let buffer = terminal.backend().buffer().clone();
    let mut lines = Vec::with_capacity(usize::from(height));
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            line.push_str(buffer[(x, y)].symbol());
        }
        lines.push(line.trim_end().to_string());
    }
    (lines.join("\n"), buffer)
}

fn buffer_to_rendered_string(buffer: &ratatui::buffer::Buffer) -> String {
    let area = buffer.area;
    let mut lines = Vec::with_capacity(usize::from(area.height));
    for y in area.y..area.bottom() {
        let mut line = String::new();
        for x in area.x..area.right() {
            line.push_str(buffer[(x, y)].symbol());
        }
        lines.push(line.trim_end().to_string());
    }
    lines.join("\n")
}

fn find_rendered_line<'a>(rendered: &'a str, needle: &str) -> (usize, &'a str) {
    rendered
        .lines()
        .enumerate()
        .find(|(_, line)| line.contains(needle))
        .unwrap_or_else(|| panic!("expected rendered line containing {needle:?}\n{rendered}"))
}

fn find_rendered_line_after<'a>(
    rendered: &'a str,
    start_index: usize,
    needle: &str,
) -> (usize, &'a str) {
    rendered
        .lines()
        .enumerate()
        .skip(start_index.saturating_add(1))
        .find(|(_, line)| line.contains(needle))
        .unwrap_or_else(|| {
            panic!(
                "expected rendered line containing {needle:?} after index {start_index}\n{rendered}"
            )
        })
}

fn requests_inner_area(state: &DashboardState, width: u16, height: u16) -> Rect {
    let areas = tui_layout(Rect::new(0, 0, width, height), state);
    tui_panel_block(state, DashboardPanel::Requests)
        .inner(combine_panel_rect(areas.requests.0, areas.requests.1))
}

fn request_graph_visible_row_count(buffer: &ratatui::buffer::Buffer, area: Rect) -> usize {
    (area.y.saturating_add(1)..area.bottom())
        .filter(|&y| {
            (area.x..area.right()).any(|x| {
                let symbol = buffer[(x, y)].symbol().chars().next();
                matches!(symbol, Some('·' | '─')) || symbol.is_some_and(is_braille_bar_symbol)
            })
        })
        .count()
}

fn request_graph_contains_bars(buffer: &ratatui::buffer::Buffer, area: Rect) -> bool {
    (area.y.saturating_add(1)..area.bottom()).any(|y| {
        (area.x..area.right()).any(|x| {
            buffer[(x, y)]
                .symbol()
                .chars()
                .next()
                .is_some_and(is_braille_bar_symbol)
        })
    })
}

fn is_braille_bar_symbol(ch: char) -> bool {
    matches!(ch as u32, 0x2801..=0x28ff)
}

fn request_graph_contains_guides(buffer: &ratatui::buffer::Buffer, area: Rect) -> bool {
    (area.y.saturating_add(1)..area.bottom()).any(|y| {
        (area.x..area.right())
            .any(|x| matches!(buffer[(x, y)].symbol().chars().next(), Some('·' | '─')))
    })
}

fn assert_join_token_layout(state: &DashboardState, areas: &TuiFrameAreas) {
    assert_eq!(
        areas.join_token_panel.y,
        areas.loading.map_or(0, |area| area.bottom())
    );
    assert_eq!(areas.join_token_panel.width, 120);
    assert_eq!(
        areas.join_token_panel.height,
        PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT
    );
    assert!(areas.join_token_copy_button.x > areas.join_token_panel.x);
    assert_eq!(areas.join_token_copy_button.y, areas.join_token_panel.y + 2);
    assert_eq!(
        areas.join_token_copy_button.right(),
        areas
            .join_token_panel
            .right()
            .saturating_sub(1)
            .saturating_sub(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING)
    );
    assert_eq!(
        join_token_text_area(areas.join_token_panel, areas.join_token_copy_button).x,
        areas
            .join_token_panel
            .x
            .saturating_add(1)
            .saturating_add(PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING)
    );
    assert_eq!(
        areas.main_body.y,
        areas.join_token_panel.y + areas.join_token_panel.height
    );
    assert_eq!(
        areas.requests.0.y,
        areas.main_body.y + areas.main_body.height
    );
    assert_eq!(areas.events.0.y, areas.main_body.y);
    assert!(areas.processes.x > areas.events.0.x);
    assert!(areas.models.0.x > areas.processes.x);

    let requests_inner = tui_panel_block(state, DashboardPanel::Requests)
        .inner(combine_panel_rect(areas.requests.0, areas.requests.1));
    assert_eq!(
        requests_inner.height as usize,
        state.panel_layout.rows_for(DashboardPanel::Requests)
    );
}

fn assert_process_table_layout(state: &DashboardState, areas: &TuiFrameAreas) {
    let events_inner = tui_panel_block(state, DashboardPanel::Events)
        .inner(combine_panel_rect(areas.events.0, areas.events.1));
    let models_inner = tui_panel_block(state, DashboardPanel::Models)
        .inner(combine_panel_rect(areas.models.0, areas.models.1));
    let llama_inner = tui_panel_block(state, DashboardPanel::LlamaCpp).inner(combine_panel_rect(
        areas.llama_processes.0,
        areas.llama_processes.1,
    ));
    let webserver_inner = tui_panel_block(state, DashboardPanel::Webserver).inner(
        combine_panel_rect(areas.webserver_processes.0, areas.webserver_processes.1),
    );

    assert_eq!(
        areas.requests.1.y,
        areas.requests.0.y + areas.requests.0.height
    );
    assert_eq!(
        areas.status_bar.y,
        areas.requests.1.y + areas.requests.1.height
    );
    assert_eq!(areas.status_bar.height, 1);
    assert_eq!(
        events_inner.height as usize,
        state.panel_layout.rows_for(DashboardPanel::Events)
    );
    assert_eq!(
        models_inner.height as usize,
        state.panel_layout.rows_for(DashboardPanel::Models)
    );
    assert_eq!(
        areas.llama_processes.0.y,
        tui_processes_block(state).inner(areas.processes).y
    );
    assert_eq!(
        areas.llama_processes.1.y,
        areas.llama_processes.0.y + areas.llama_processes.0.height
    );
    assert_eq!(
        areas.webserver_processes.0.y,
        combine_panel_rect(areas.llama_processes.0, areas.llama_processes.1).bottom()
    );
    assert_eq!(
        areas.webserver_processes.1.y,
        areas.webserver_processes.0.y + areas.webserver_processes.0.height
    );
    assert_eq!(
        llama_inner.height as usize,
        state.panel_layout.rows_for(DashboardPanel::LlamaCpp)
    );
    assert_eq!(
        webserver_inner.height as usize,
        state.panel_layout.rows_for(DashboardPanel::Webserver)
    );
    assert_eq!(state.panel_layout.rows_for(DashboardPanel::LlamaCpp), 1);
    assert_eq!(state.panel_layout.rows_for(DashboardPanel::Webserver), 2);
}

pub fn assert_tui_model_progress_renders_dashboard_without_loading_screen() {
    tui_model_progress_renders_dashboard_without_loading_screen();
}

pub fn assert_tui_startup_progress_continues_in_dashboard_after_model_download_ready() {
    tui_startup_progress_continues_in_dashboard_after_model_download_ready();
}

pub fn assert_startup_lifecycle_transitions_pending_partial_ready_failed() {
    dashboard::startup_lifecycle_transitions_pending_partial_ready_failed();
}

pub fn assert_startup_lifecycle_keeps_runtime_ready_as_final_edge() {
    dashboard::startup_lifecycle_keeps_runtime_ready_as_final_edge();
}

pub fn assert_startup_failures_surface_in_tui_events_and_status() {
    dashboard::startup_failures_surface_in_tui_events_and_status();
}

pub fn assert_startup_failure_summary_sanitizes_multiline_detail() {
    formatting::startup_failure_summary_sanitizes_multiline_detail();
}

pub fn assert_rpc_and_llama_startup_failures_mark_components_failed() {
    dashboard::llama_startup_failures_mark_components_failed();
}

pub fn assert_discovery_and_join_failures_mark_startup_mesh_component_failed() {
    dashboard::discovery_and_join_failures_mark_startup_mesh_component_failed();
}

pub fn assert_post_ready_peer_churn_does_not_reopen_startup_failure() {
    dashboard::post_ready_peer_churn_does_not_reopen_startup_failure();
}

pub fn assert_startup_history_is_visible_after_late_tui_attach() {
    dashboard::startup_history_is_visible_after_late_tui_attach();
}

pub fn assert_startup_history_keeps_order_when_tui_attaches_late() {
    dashboard::startup_history_keeps_order_when_tui_attaches_late();
}

pub fn assert_endpoint_rows_remain_starting_until_ready_events() {
    dashboard::endpoint_rows_remain_starting_until_ready_events();
}

pub fn assert_startup_launch_plan_renders_not_ready_rows_before_actions() {
    dashboard::startup_launch_plan_renders_not_ready_rows_before_actions();
}

pub fn assert_startup_progress_after_launch_plan_shows_dashboard_not_loader() {
    dashboard::startup_progress_after_launch_plan_shows_dashboard_not_loader();
}

pub fn assert_planned_rows_transition_from_not_ready_to_ready_events() {
    rendering::planned_rows_transition_from_not_ready_to_ready_events();
}

pub fn assert_launch_plan_rows_survive_empty_startup_snapshot() {
    merging::launch_plan_rows_survive_empty_startup_snapshot();
}

pub fn assert_launch_plan_preserves_distinct_port_zero_endpoint_rows() {
    merging::launch_plan_preserves_distinct_port_zero_endpoint_rows();
}

pub fn assert_snapshot_upsert_preserves_distinct_port_zero_endpoint_rows() {
    merging::snapshot_upsert_preserves_distinct_port_zero_endpoint_rows();
}

pub fn assert_planned_port_zero_process_rows_bind_to_concrete_startup_events() {
    merging::planned_port_zero_process_rows_bind_to_concrete_startup_events();
}

pub fn assert_fallback_mode_surfaces_startup_failures_without_tui() {
    formatting::fallback_mode_surfaces_startup_failures_without_tui();
}

pub fn assert_shutdown_suppresses_late_ready_render() {
    dashboard::shutdown_suppresses_late_ready_render();
}

pub fn assert_interactive_post_terminal_exit_resumes_plain_event_output() {
    formatting::interactive_post_terminal_exit_resumes_plain_event_output();
}

pub fn assert_tui_model_card_separates_name_from_metadata_columns() {
    tui_model_card_separates_name_from_metadata_columns();
}

fn parse_json_line(rendered: &str) -> Value {
    assert!(
        rendered.ends_with('\n'),
        "json formatter should emit newline-delimited output"
    );
    serde_json::from_str(rendered.trim_end()).expect("line should parse as json")
}

fn format_json_event(formatter: &mut JsonFormatter, event: OutputEvent) -> Value {
    parse_json_line(
        &formatter
            .format(&event)
            .expect("json formatter should preserve representative metadata"),
    )
}

fn assert_dashboard_snapshot_shell(rendered: &str) {
    for expected in [
        "Mesh Events",
        "Processes",
        "llama.cpp",
        "mesh-llm Processes",
        "Loaded Models",
        "Incoming Requests",
        "RPS ",
        "READY",
        "[Tab] Next",
        "[Enter/Z] Full",
        "[Shift-Tab] Prev",
        "q",
    ] {
        assert!(rendered.contains(expected));
    }

    for ch in ['📋', '⚙', '🔧', '📊', '📈'] {
        assert!(!rendered.contains(ch));
    }

    assert!(rendered.contains('─'));
    assert!(rendered.contains('│'));
    assert!(!rendered.contains("Running llama.cpp instances"));
    assert!(!rendered.contains("Running models"));
}

fn assert_dashboard_panel_borders(buffer: &ratatui::buffer::Buffer, areas: &TuiFrameAreas) {
    for panel_area in [
        combine_panel_rect(areas.events.0, areas.events.1),
        combine_panel_rect(areas.llama_processes.0, areas.llama_processes.1),
        combine_panel_rect(areas.webserver_processes.0, areas.webserver_processes.1),
        combine_panel_rect(areas.models.0, areas.models.1),
        combine_panel_rect(areas.requests.0, areas.requests.1),
    ] {
        assert_eq!(buffer[(panel_area.x, panel_area.y)].symbol(), "╭");
        assert_eq!(
            buffer[(panel_area.right().saturating_sub(1), panel_area.y)].symbol(),
            "╮"
        );
    }
}

fn assert_model_ready_metadata(model_ready: &Value) {
    assert_eq!(model_ready["model"], "Qwen3-32B");
    assert_eq!(model_ready["port"], 38373);
    assert_eq!(model_ready["internal_port"], 38373);
    assert_eq!(model_ready["role"], "host");
}

fn assert_rpc_starting_metadata(rpc_starting: &Value) {
    assert_eq!(rpc_starting["port"], 43683);
    assert_eq!(rpc_starting["device"], "CUDA0");
    assert_eq!(rpc_starting["log_path"], "/tmp/rpc.log");
}

fn assert_llama_starting_metadata(llama_starting: &Value) {
    assert_eq!(llama_starting["model"], "Qwen3-32B");
    assert_eq!(llama_starting["http_port"], 8001);
    assert_eq!(llama_starting["ctx_size"], 8192);
    assert_eq!(llama_starting["log_path"], "/tmp/llama.log");
}

fn assert_runtime_ready_metadata(runtime_ready: &Value) {
    assert_eq!(runtime_ready["api_port"], 9337);
    assert_eq!(runtime_ready["console_port"], 3131);
    assert_eq!(runtime_ready["console_url"], "http://localhost:3131");
    assert_eq!(runtime_ready["models_count"], 2);
    assert_eq!(
        runtime_ready["pi_command"],
        "mesh-llm pi --host 127.0.0.1:9337 --model 'Qwen3-32B'"
    );
    assert_eq!(runtime_ready["goose_command"], "goose session");
}

fn assert_required_json_envelope(value: &Value, event: &OutputEvent) {
    let timestamp = value
        .get("timestamp")
        .and_then(Value::as_str)
        .expect("json output should include string timestamp");
    assert!(
        timestamp.ends_with('Z') && timestamp.contains('T'),
        "timestamp should be RFC3339 UTC, got {timestamp}"
    );
    assert_eq!(
        value.get("level").and_then(Value::as_str),
        Some(event.level().as_str()),
        "json output should include level for {event:?}"
    );
    assert_eq!(
        value.get("event").and_then(Value::as_str),
        Some(event.event_name()),
        "json output should include event name for {event:?}"
    );
    assert_eq!(
        value.get("message").and_then(Value::as_str),
        Some(event.message().as_str()),
        "json output should include message for {event:?}"
    );
}

pub fn assert_interactive_preterminal_render_uses_plain_event_output() {
    formatting::interactive_preterminal_render_uses_plain_event_output();
}
#[test]
pub(super) fn tui_model_progress_renders_dashboard_without_loading_screen() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        120, 24,
    )));
    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::ModelDownloadProgress {
            label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
            file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
            downloaded_bytes: Some(245_500_000),
            total_bytes: Some(491_000_000),
            status: ModelProgressStatus::Downloading,
        },
    ));

    let rendered = render_tui_frame_snapshot(&state, 120, 48);

    assert!(
        rendered.contains("Mesh Events"),
        "startup progress should render inside the dashboard, not a loading screen: {rendered}"
    );
    assert!(
        !rendered.contains('█'),
        "startup progress should not render the old progress bar: {rendered}"
    );
}

#[test]
pub(super) fn tui_startup_progress_continues_in_dashboard_after_model_download_ready() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        120, 24,
    )));
    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::ModelDownloadProgress {
            label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
            file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
            downloaded_bytes: Some(491_000_000),
            total_bytes: Some(491_000_000),
            status: ModelProgressStatus::Ready,
        },
    ));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Qwen2.5-0.5B-Instruct-Q4_K_M".to_string()),
        http_port: 9338,
        ctx_size: Some(4096),
        log_path: None,
    }));

    let progress = state
        .active_loading_progress()
        .expect("startup loading progress should remain active before runtime ready");
    let rendered = render_tui_frame_snapshot(&state, 120, 48);

    assert!(
        progress.ratio < 1.0,
        "startup progress must not jump to 100%"
    );
    assert!(
        progress
            .detail
            .contains("starting llama-server for Qwen2.5")
    );
    assert!(
        rendered.contains("Mesh Events"),
        "startup progress should stay in the dashboard instead of taking over the frame: {rendered}"
    );
    assert!(!rendered.contains('█'));
}

#[test]
pub(super) fn tui_startup_progress_advances_with_startup_milestones() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::ModelDownloadProgress {
            label: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
            file: Some("qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()),
            downloaded_bytes: Some(491_000_000),
            total_bytes: Some(491_000_000),
            status: ModelProgressStatus::Ready,
        },
    ));
    let after_download = state
        .active_loading_progress()
        .expect("download-ready progress should seed startup progress")
        .ratio;
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Qwen2.5-0.5B-Instruct-Q4_K_M".to_string()),
        http_port: 9338,
        ctx_size: Some(4096),
        log_path: None,
    }));
    let after_llama_start = state
        .active_loading_progress()
        .expect("llama startup should advance startup progress")
        .ratio;

    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));
    let after_model_ready = state
        .active_loading_progress()
        .expect("model ready should advance startup progress")
        .ratio;

    assert!(after_llama_start > after_download);
    assert!(after_model_ready > after_llama_start);
}

#[test]
pub(super) fn tui_runtime_ready_keeps_dimmed_logo_above_dashboard() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 48,
    )));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(0),
        pi_command: None,
        goose_command: None,
    }));

    let area = Rect::new(0, 0, 160, 48);
    let areas = tui_layout(area, &state);
    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 48);
    let slack_area = areas
        .loading
        .expect("runtime-ready layout should expose slack above dashboard");
    let logo_area = areas
        .logo
        .expect("runtime-ready layout should center a logo in the slack area");
    let ready_logo_height = u16::try_from(
        tui_ready_logo_text()
            .expect("ready logo text should be available")
            .lines
            .len(),
    )
    .unwrap_or(u16::MAX);
    let ready_logo_width = tui_ready_logo_text()
        .expect("ready logo text should be available")
        .lines
        .iter()
        .map(tui_logo_line_width)
        .max()
        .and_then(|width| u16::try_from(width).ok())
        .unwrap_or(logo_area.width);
    let first_visible_logo_row = (logo_area.y..logo_area.bottom())
        .find(|&y| {
            (logo_area.x..logo_area.right()).any(|x| {
                let cell = &buffer[(x, y)];
                cell.symbol() != " " && cell.style().add_modifier.contains(Modifier::DIM)
            })
        })
        .expect("expected dimmed ANSI logo content in the centered slack area");

    assert!(rendered.contains("Mesh Events"));
    assert!(rendered.contains("READY"));
    assert!(
        logo_area.height > 0 && logo_area.bottom() <= areas.main_body.y,
        "expected centered logo area above dashboard"
    );
    assert_eq!(logo_area.height, ready_logo_height.min(slack_area.height));
    assert_eq!(logo_area.width, ready_logo_width.min(slack_area.width));
    assert_eq!(
        logo_area.y,
        slack_area.y + (slack_area.height - logo_area.height) / 2
    );
    assert_eq!(
        logo_area.x,
        slack_area.x + (slack_area.width - logo_area.width) / 2
    );
    assert_eq!(first_visible_logo_row, logo_area.y);
    assert!(
        (logo_area.y..logo_area.bottom()).any(|y| {
            (logo_area.x..logo_area.right()).any(|x| {
                let cell = &buffer[(x, y)];
                cell.symbol() != " " && cell.style().add_modifier.contains(Modifier::DIM)
            })
        }),
        "expected dimmed ANSI logo content in the centered slack area\n{rendered}"
    );
}

#[test]
pub(super) fn tui_snapshot_renders_full_dashboard_spec() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        260, 32,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(snapshot_fixture(2, 30)));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.64.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::NodeIdentity {
        node_id: "node-7".to_string(),
        mesh_id: Some("poker-night".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::PeerJoined {
        peer_id: "peer-1".to_string(),
        label: Some("alice".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(2),
        pi_command: None,
        goose_command: None,
    }));
    state.reduce(DashboardAction::OutputEvent(info_event(
        "mesh named poker-night is private by default",
    )));

    let areas = tui_layout(Rect::new(0, 0, 220, 24), &state);
    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 220, 24);

    assert_dashboard_snapshot_shell(&rendered);
    assert_dashboard_panel_borders(&buffer, &areas);
}

#[test]
pub(super) fn tui_narrow_terminal_renders_resize_guidance_instead_of_dashboard() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        PRETTY_TUI_MIN_DASHBOARD_WIDTH - 1,
        24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(snapshot_fixture(2, 30)));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(2),
        pi_command: None,
        goose_command: None,
    }));

    let rendered = render_tui_frame_snapshot(&state, PRETTY_TUI_MIN_DASHBOARD_WIDTH - 1, 12);

    assert!(rendered.contains(">= 60 columns"));
    assert!(rendered.contains("Resize"));
    assert!(!rendered.contains("Mesh Events"));
}

#[test]
pub(super) fn tui_survives_rapid_event_bursts_without_scroll_jump() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 140,
        rows: 18,
    });

    for index in 0..40 {
        let _ = formatter.handle_output_event(&info_event(format!("seed event {index}")));
    }

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
    let before = formatter.state.panel_view_state(DashboardPanel::Events);
    assert!(
        !formatter.state.events_follow,
        "manual scroll should disable follow"
    );

    for index in 0..200 {
        let _ = formatter.handle_output_event(&info_event(format!("burst event {index}")));
    }

    let after = formatter.state.panel_view_state(DashboardPanel::Events);
    assert_eq!(after.scroll_offset, before.scroll_offset);
    assert_eq!(after.selected_row, before.selected_row);
    assert!(!formatter.state.events_follow);
    let rendered = render_tui_frame_snapshot(&formatter.state, 140, 18);
    assert!(rendered.contains("seed event"));
}

#[test]
pub(super) fn tui_models_render_ten_cell_ctx_and_cap_segments() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        260, 32,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        loaded_model_rows: vec![
            sample_model_row("Segmented-Model", 4001),
            half_scale_model_row(),
        ],
        ..snapshot_fixture(0, 30)
    }));

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 260, 32);
    let theme = tui_theme();
    assert_segmented_model_card_layout(&rendered, &buffer, &theme);

    let half_row = half_scale_model_row();
    let mut half_buffer = Buffer::empty(Rect::new(0, 0, 80, PRETTY_TUI_MODEL_CARD_HEIGHT as u16));
    TuiModelCardWidget {
        row: &half_row,
        content_width: 78,
        is_selected: false,
        is_focused: false,
    }
    .render(half_buffer.area, &mut half_buffer);
    assert_half_scale_model_card_segments(&half_buffer, &theme);
}
#[test]
pub(super) fn tui_models_panel_renders_two_loaded_model_cards_in_compact_dashboard() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        260, 33,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        loaded_model_rows: vec![
            sample_model_row("unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL", 37615),
            sample_model_row("unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL", 34097),
        ],
        ..snapshot_fixture(0, 30)
    }));

    let rendered = render_tui_frame_snapshot(&state, 260, 33);

    assert!(
        rendered.contains("unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"),
        "expected first loaded model card in compact dashboard: {rendered}"
    );
    assert!(
        rendered.contains("unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL"),
        "expected second loaded model card in compact dashboard: {rendered}"
    );
    let (first_y, _) = find_rendered_line(&rendered, "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL");
    let (second_y, _) = find_rendered_line(&rendered, "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL");
    assert!(
        second_y.saturating_sub(first_y) >= PRETTY_TUI_MODEL_CARD_HEIGHT,
        "expected the first card to keep its full height before the second card: {rendered}"
    );
}
#[test]
pub(super) fn tui_models_snapshot_includes_quant_slots_and_status() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        260, 24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        loaded_model_rows: vec![DashboardModelRow {
            name: "Metadata-Model".to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Warning,
            port: Some(4011),
            device: Some("CUDA0".to_string()),
            slots: Some(8),
            quantization: Some("Q8_0".to_string()),
            ctx_size: Some(8192),
            ctx_used_tokens: Some(8192),
            lanes: Some(vec![
                DashboardModelLane {
                    index: 0,
                    active: true,
                },
                DashboardModelLane {
                    index: 1,
                    active: true,
                },
                DashboardModelLane {
                    index: 2,
                    active: true,
                },
                DashboardModelLane {
                    index: 3,
                    active: false,
                },
                DashboardModelLane {
                    index: 4,
                    active: false,
                },
                DashboardModelLane {
                    index: 5,
                    active: false,
                },
                DashboardModelLane {
                    index: 6,
                    active: false,
                },
                DashboardModelLane {
                    index: 7,
                    active: false,
                },
            ]),
            file_size_gb: Some(24.0),
        }],
        ..snapshot_fixture(0, 30)
    }));

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 260, 24);
    let (title_y, title_line) = find_rendered_line(&rendered, "Metadata-Model");
    assert!(
        !title_line.contains("PORT:"),
        "model name should be separated from metadata: {title_line}"
    );
    let (meta_y, meta_line) = find_rendered_line_after(&rendered, title_y, "STATUS");
    let (_, detail_line) = find_rendered_line_after(&rendered, title_y, "QUANT");
    assert!(
        meta_line.contains("STATUS: warning"),
        "expected warning status in {meta_line}"
    );
    assert!(
        meta_line.contains("PORT: 4011"),
        "expected port in {meta_line}"
    );
    assert!(
        meta_line.contains("DEVICE: CUDA0"),
        "expected device in {meta_line}"
    );
    assert!(
        !meta_line.contains("DEV:"),
        "expected full DEVICE label rather than DEV in {meta_line}"
    );
    let areas = tui_layout(Rect::new(0, 0, 260, 24), &state);
    let models_area = combine_panel_rect(areas.models.0, areas.models.1);
    let models_meta_line = (models_area.x..models_area.right())
        .map(|x| buffer[(x, meta_y as u16)].symbol())
        .collect::<String>();
    let port_byte = models_meta_line
        .find("PORT:")
        .expect("expected PORT label x coordinate");
    let status_byte = models_meta_line
        .find("STATUS:")
        .expect("expected STATUS label x coordinate");
    let device_byte = models_meta_line
        .find("DEVICE:")
        .expect("expected DEVICE label x coordinate");
    let port_x = models_meta_line[..port_byte].chars().count();
    let status_x = models_meta_line[..status_byte].chars().count();
    let device_x = models_meta_line[..device_byte].chars().count();
    assert!(
        port_x < status_x && status_x < device_x,
        "expected PORT, STATUS, and DEVICE to stay ordered in {models_meta_line}"
    );
    assert!(
        detail_line.contains("SLOTS: 8"),
        "expected slots in {detail_line}"
    );
    assert!(
        detail_line.contains("Q8_0"),
        "expected quantization in {detail_line}"
    );
    assert!(
        detail_line.contains("CTX: 8192"),
        "expected runtime context size in {detail_line}"
    );
    assert!(
        !detail_line.contains("ROLE:"),
        "role should not render in model details: {detail_line}"
    );
    let (ctx_y, ctx_line) = find_rendered_line_after(&rendered, title_y, "8192 / 8192");
    let (_, divider_line) = find_rendered_line_after(&rendered, title_y, "──");
    let (slots_y, slots_line) = find_rendered_line_after(&rendered, title_y, "3 / 8");
    assert!(
        !divider_line.contains('├') && !divider_line.contains('┤'),
        "expected subtle interior divider, not frame-joining divider, in {divider_line}"
    );
    assert!(
        ctx_line.contains("CTX") && ctx_line.contains("8192 / 8192"),
        "expected visible ctx stat with right label in {ctx_line}"
    );
    assert!(
        slots_line.contains("SLOTS") && slots_line.contains("3 / 8"),
        "expected visible slot stat with right label in {slots_line}"
    );
    let ctx_gauge_x = ctx_line
        .find('█')
        .map(|index| ctx_line[..index].chars().count())
        .expect("expected CTX usage bar x coordinate");
    let slots_block_x = slots_line
        .find('◼')
        .map(|index| slots_line[..index].chars().count())
        .expect("expected SLOTS block x coordinate");
    assert_eq!(
        buffer[(
            u16::try_from(ctx_gauge_x).unwrap(),
            u16::try_from(ctx_y).unwrap()
        )]
            .style()
            .fg,
        Some(tui_model_usage_color(1.0))
    );
    assert_eq!(
        buffer[(
            u16::try_from(slots_block_x).unwrap(),
            u16::try_from(slots_y).unwrap()
        )]
            .style()
            .fg,
        Some(tui_theme().warning)
    );
}
#[test]
pub(super) fn tui_model_card_separates_name_from_metadata_columns() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        120, 24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        loaded_model_rows: vec![DashboardModelRow {
            name: "qwen2.5-0.5b-instruct-q4_k_m".to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Ready,
            port: Some(49201),
            device: Some("GPU0".to_string()),
            slots: Some(4),
            quantization: Some("Q4_K_M".to_string()),
            ctx_size: Some(8192),
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: Some(0.5),
        }],
        ..snapshot_fixture(0, 30)
    }));

    let rendered = render_tui_frame_snapshot(&state, 120, 24);
    let (name_y, name_line) = find_rendered_line(&rendered, "qwen2.5-0.5b");
    let (meta_y, meta_line) = find_rendered_line_after(&rendered, name_y, "STATUS:");
    let (_, detail_line) = find_rendered_line_after(&rendered, name_y, "QUANT:");

    assert!(
        !name_line.contains("PORT:")
            && !name_line.contains("DEVICE:")
            && !name_line.contains("STATUS:"),
        "model name row should not share space with metadata columns: {name_line}"
    );
    assert!(
        meta_y > name_y,
        "metadata should render on a row after the model name"
    );
    assert!(
        !meta_line.contains("qwen2.5"),
        "metadata row should not include the model name: {meta_line}"
    );
    assert!(
        meta_line.contains("PORT:")
            && meta_line.contains("STATUS:")
            && meta_line.contains("DEVICE:"),
        "top metadata row should expose PORT, STATUS, and DEVICE: {meta_line}"
    );
    assert!(
        detail_line.contains("SLOTS:")
            && detail_line.contains("QUANT:")
            && detail_line.contains("CTX:"),
        "bottom metadata row should expose SLOTS, QUANT, and CTX: {detail_line}"
    );
}
#[test]
pub(super) fn tui_models_truncate_long_names_without_wrapping() {
    let long_name = "Extremely-Verbose-Model-Name-That-Should-Never-Wrap-Onto-A-Second-Line";
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        220, 24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        loaded_model_rows: vec![DashboardModelRow {
            name: long_name.to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Ready,
            port: Some(4022),
            device: Some("GPU0".to_string()),
            slots: Some(4),
            quantization: Some("Q4_K_M".to_string()),
            ctx_size: Some(8192),
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: Some(24.0),
        }],
        ..snapshot_fixture(0, 30)
    }));

    let rendered = render_tui_frame_snapshot(&state, 220, 24);
    let (title_y, title_line) = rendered
        .lines()
        .enumerate()
        .find(|(_, line)| line.contains('…'))
        .expect("expected truncated model name line");
    let (meta_y, meta_line) = find_rendered_line_after(&rendered, title_y, "DEVICE");
    let (_, detail_line) = find_rendered_line_after(&rendered, title_y, "Q4_K_M");
    assert!(
        title_line.contains('…'),
        "expected ellipsis in truncated model title: {title_line}"
    );
    assert!(
        detail_line.contains("Q4_K_M"),
        "expected quantization to remain visible: {detail_line}"
    );
    assert!(
        meta_line.contains("DEVICE: GPU0"),
        "expected readable device column: {meta_line}"
    );
    assert!(
        meta_line.contains("PORT:") && meta_line.contains("STATUS:"),
        "top metadata row should keep three columns visible: {meta_line}"
    );
    assert!(meta_y > title_y, "expected metadata on a later card row");
    assert!(
        !rendered.contains(long_name),
        "full long model name should not survive truncation"
    );
}
#[test]
pub(super) fn tui_models_cards_scroll_without_selecting_inner_cards() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_snapshot(DashboardSnapshot {
        loaded_model_rows: (0..5)
            .map(|index| sample_model_row(&format!("Model-{index}"), 4000 + index as u16))
            .collect(),
        ..snapshot_fixture(0, 30)
    });
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 180,
        rows: 24,
    });
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));

    let initial_view = formatter.state.panel_view_state(DashboardPanel::Models);
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
    assert_eq!(initial_view.viewport_rows, 1);

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));

    let after = formatter.state.panel_view_state(DashboardPanel::Models);
    assert_eq!(after.selected_row, None);
    assert_eq!(after.scroll_offset, 3);

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&formatter.state, 180, 24);
    assert!(
        rendered.contains("▶ Loaded Models"),
        "expected the outer models pane to remain focused in {rendered}"
    );
    assert!(
        rendered.contains("Model-3"),
        "expected first visible card in {rendered}"
    );
    let (model_y, _) = find_rendered_line(&rendered, "Model-3");
    let areas = tui_layout(Rect::new(0, 0, 180, 24), &formatter.state);
    let models_area = combine_panel_rect(areas.models.0, areas.models.1);
    let model_x = (models_area.x..models_area.right())
        .find(|&x| buffer[(x, model_y as u16)].symbol() == "M")
        .expect("model name should have an x coordinate inside the models panel");
    let theme = tui_theme();
    assert_ne!(
        buffer[(model_x, model_y as u16)].style().bg,
        Some(theme.selection_bg),
        "model card content should not use the selected-row background"
    );
    assert!(
        !rendered.contains("Model-2"),
        "expected previous card to be scrolled off in {rendered}"
    );
    assert!(
        !rendered.contains("Model-0"),
        "expected scrolled-off card to disappear"
    );
}
