use super::super::EndpointState;
use super::{
    DashboardState, PRETTY_TUI_EVENT_LEVEL_WIDTH, StartupLifecycleState, event_severity_badge,
    sanitize_mesh_event_message,
};
use std::fmt::Write as FmtWrite;

pub(in crate::output) fn single_line_status_text(message: &str) -> String {
    message.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(in crate::output) fn render_dashboard_text(state: &DashboardState) -> String {
    let mut output = String::new();
    let mut header = String::from("mesh-llm");
    if let Some(version) = &state.version {
        header.push(' ');
        header.push_str(version);
    }
    if let Some(node_id) = &state.node_id {
        header.push_str(&format!("  node={node_id}"));
    }
    if let Some(mesh_id) = &state.mesh_id {
        header.push_str(&format!("  mesh={mesh_id}"));
    }
    let _ = writeln!(&mut output, "{header}");
    let _ = writeln!(&mut output);

    write_dashboard_section(
        &mut output,
        "Startup status",
        &render_startup_summary(state),
    );
    let _ = writeln!(&mut output);
    write_dashboard_section(
        &mut output,
        "Running llama.cpp instances",
        &render_llama_instances(state),
    );
    let _ = writeln!(&mut output);
    write_dashboard_section(&mut output, "Running models", &render_models(state));
    let _ = writeln!(&mut output);
    write_dashboard_section(&mut output, "Running webserver", &render_webserver(state));
    let _ = writeln!(&mut output);
    write_dashboard_section(&mut output, "Running API", &render_api(state));
    let _ = writeln!(&mut output);
    write_dashboard_section(
        &mut output,
        &format!("Mesh events (latest {})", state.mesh_event_limit),
        &render_mesh_events(state),
    );
    output
}

pub(in crate::output) fn write_dashboard_section(
    output: &mut String,
    title: &str,
    lines: &[String],
) {
    let _ = writeln!(
        output,
        "┌ {title} ────────────────────────────────────────────────────────────"
    );
    if lines.is_empty() {
        let _ = writeln!(output, "│ (none)");
    } else {
        for line in lines {
            let _ = writeln!(output, "│ {line}");
        }
    }
    let _ = writeln!(
        output,
        "└────────────────────────────────────────────────────────────────────"
    );
}

pub(in crate::output) fn render_startup_summary(state: &DashboardState) -> Vec<String> {
    let lifecycle = &state.startup_lifecycle;
    let mut lines = vec![format!(
        "startup={}{}",
        lifecycle.phase.as_str(),
        lifecycle
            .failure
            .as_ref()
            .map(|failure| format!("  failure={}", single_line_status_text(failure)))
            .unwrap_or_default()
    )];
    lines.extend(startup_component_summary_lines(lifecycle));
    lines
}

pub(in crate::output) fn startup_component_summary_lines(
    lifecycle: &StartupLifecycleState,
) -> Vec<String> {
    vec![
        format!(
            "mesh={}  api={}  console={}",
            lifecycle.mesh.phase.as_str(),
            lifecycle.api.phase.as_str(),
            lifecycle.console.phase.as_str(),
        ),
        format!(
            "llama-server={}  model readiness={}",
            lifecycle.llama_server.phase.as_str(),
            lifecycle.model_readiness.phase.as_str(),
        ),
    ]
}

pub(in crate::output) fn render_llama_instances(state: &DashboardState) -> Vec<String> {
    let mut lines = Vec::new();
    for instance in &state.llama_instances {
        let mut line = format!(
            "{}   {}   port={} ",
            instance.kind.as_str(),
            instance.status.as_str(),
            instance.port
        );
        if let Some(device) = &instance.device {
            line.push_str(&format!("  device={device}"));
        }
        if let Some(model) = &instance.model {
            line.push_str(&format!("  model={model}"));
        }
        if let Some(ctx_size) = instance.ctx_size {
            line.push_str(&format!("  ctx={ctx_size}"));
        }
        lines.push(line.trim_end().to_string());
        if let Some(log_path) = &instance.log_path {
            lines.push(format!("             logs={log_path}"));
        }
    }
    lines
}

pub(in crate::output) fn render_models(state: &DashboardState) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(passive_mode) = &state.passive_mode {
        let mut line = format!("{}   {}", passive_mode.role, passive_mode.status.as_str());
        if let Some(capacity_gb) = passive_mode.capacity_gb {
            line.push_str(&format!("   capacity={capacity_gb:.1}GB"));
        }
        if !passive_mode.models_on_disk.is_empty() {
            line.push_str(&format!(
                "   models={}",
                passive_mode.models_on_disk.join(", ")
            ));
        }
        if let Some(detail) = &passive_mode.detail {
            line.push_str(&format!("   {detail}"));
        }
        lines.push(line);
    }
    if let Some(multi_model_mode) = &state.multi_model_mode {
        let models = if multi_model_mode.models.is_empty() {
            "(none)".to_string()
        } else {
            multi_model_mode.models.join(", ")
        };
        lines.push(format!(
            "multi-model mode   {} model(s)   models={models}",
            multi_model_mode.count
        ));
    }

    lines.extend(state.running_models.iter().map(|model| {
        let mut line = if model.profile.is_empty() {
            format!("{}   {}", model.model, model.status.as_str())
        } else {
            format!(
                "{} [{}]   {}",
                model.model,
                model.profile,
                model.status.as_str()
            )
        };
        if let Some(port) = model.internal_port {
            line.push_str(&format!("   port={port}"));
        }
        if let Some(role) = &model.role {
            line.push_str(&format!("   role={role}"));
        }
        if let Some(capacity_gb) = model.capacity_gb {
            line.push_str(&format!("   capacity={capacity_gb:.1}GB"));
        }
        line
    }));

    lines
}

pub(in crate::output) fn render_webserver(state: &DashboardState) -> Vec<String> {
    render_endpoint(&state.webserver)
}

pub(in crate::output) fn render_api(state: &DashboardState) -> Vec<String> {
    render_endpoint(&state.api)
}

pub(in crate::output) fn render_endpoint(endpoint: &Option<EndpointState>) -> Vec<String> {
    endpoint
        .iter()
        .flat_map(|endpoint| {
            let mut lines = vec![format!(
                "{}   {}   {}",
                endpoint.label,
                endpoint.status.as_str(),
                endpoint.url
            )];
            lines.extend(
                endpoint
                    .details
                    .iter()
                    .map(|detail| format!("    {detail}")),
            );
            lines
        })
        .collect()
}

pub(in crate::output) fn render_mesh_events(state: &DashboardState) -> Vec<String> {
    state
        .mesh_events
        .iter()
        .map(|event| {
            let (badge_text, _) = event_severity_badge(event);
            format!(
                "{} {:<PRETTY_TUI_EVENT_LEVEL_WIDTH$}{}",
                event.timestamp,
                badge_text,
                sanitize_mesh_event_message(&event.summary)
            )
        })
        .collect()
}
