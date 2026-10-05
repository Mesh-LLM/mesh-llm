use super::{DEFAULT_LOCAL_PORT, SummaryAssembly};

pub(super) fn format_local_model(name: &str, port: u16, assembly: &mut SummaryAssembly) {
    assembly.command.push(' ');
    assembly.command.push_str(name);
    assembly.port(port, DEFAULT_LOCAL_PORT);
    assembly.redact("name", true);
}

pub(super) fn format_runtime(
    command: Option<&mesh_llm_cli::runtime::RuntimeCommand>,
    assembly: &mut SummaryAssembly,
) {
    use mesh_llm_cli::runtime::RuntimeCommand;
    assembly.command.push_str(" runtime");
    let Some(command) = command else { return };
    match command {
        RuntimeCommand::Status { port } => {
            assembly.command.push_str(" status");
            assembly.port(*port, DEFAULT_LOCAL_PORT);
        }
        RuntimeCommand::Load { port, .. } => format_local_model("load", *port, assembly),
        RuntimeCommand::Unload { port, .. } => format_local_model("unload", *port, assembly),
        RuntimeCommand::Guardrails { mode, port, json } => {
            assembly.command.push_str(" guardrails --mode ");
            assembly.command.push_str(mode.as_str());
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("json", *json);
        }
        RuntimeCommand::Bootstrap { port, json } => {
            assembly.command.push_str(" bootstrap");
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("json", *json);
        }
        RuntimeCommand::List {
            available,
            installed,
            manifest,
            bundle_dirs,
            cache_dir,
            json,
        } => {
            assembly.command.push_str(" list");
            assembly.flag("available", *available);
            assembly.flag("installed", *installed);
            assembly.redact("--manifest", manifest.is_some());
            assembly.redact("--bundle-dir", !bundle_dirs.is_empty());
            assembly.redact("--cache-dir", cache_dir.is_some());
            assembly.flag("json", *json);
        }
        RuntimeCommand::Install {
            runtime,
            manifest,
            bundle_dirs,
            cache_dir,
            json,
        } => {
            assembly.command.push_str(" install");
            assembly.redact("runtime_ref", runtime.is_some());
            assembly.redact("--manifest", manifest.is_some());
            assembly.redact("--bundle-dir", !bundle_dirs.is_empty());
            assembly.redact("--cache-dir", cache_dir.is_some());
            assembly.flag("json", *json);
        }
        RuntimeCommand::Remove {
            mesh_version,
            cache_dir,
            json,
            ..
        } => {
            assembly.command.push_str(" remove");
            assembly.redact("native_runtime_id", true);
            assembly.redact("--mesh-version", mesh_version.is_some());
            assembly.redact("--cache-dir", cache_dir.is_some());
            assembly.flag("json", *json);
        }
        RuntimeCommand::Prune {
            active_only,
            mesh_version,
            cache_dir,
            json,
        } => {
            assembly.command.push_str(" prune");
            assembly.flag("active-only", *active_only);
            assembly.redact("--mesh-version", mesh_version.is_some());
            assembly.redact("--cache-dir", cache_dir.is_some());
            assembly.flag("json", *json);
        }
        RuntimeCommand::GetConfig { port, json, .. }
        | RuntimeCommand::ScanRefresh { port, json, .. }
        | RuntimeCommand::RefreshInventory { port, json, .. } => {
            assembly.command.push_str(" remote");
            assembly.redact("--endpoint", true);
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("json", *json);
        }
        RuntimeCommand::LoadModel {
            profile,
            port,
            json,
            ..
        }
        | RuntimeCommand::EnsureModel {
            profile,
            port,
            json,
            ..
        } => {
            assembly.command.push_str(" remote-model");
            assembly.redact("--endpoint", true);
            assembly.redact("--model", true);
            assembly.redact("--profile", profile.is_some());
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("json", *json);
        }
        RuntimeCommand::UnloadModel {
            model,
            instance_id,
            port,
            json,
            ..
        }
        | RuntimeCommand::DrainModel {
            model,
            instance_id,
            port,
            json,
            ..
        } => {
            assembly.command.push_str(" remote-model");
            assembly.redact("--endpoint", true);
            assembly.redact("--model", model.is_some());
            assembly.redact("--instance-id", instance_id.is_some());
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("json", *json);
        }
        RuntimeCommand::ApplyConfig { port, json, .. } => {
            assembly.command.push_str(" apply-config");
            assembly.redact("--endpoint", true);
            assembly.redact("--expected-revision", true);
            assembly.redact("--config", true);
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("json", *json);
        }
    }
}
