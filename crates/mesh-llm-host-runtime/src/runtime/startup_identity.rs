use super::RuntimeOptions;
use crate::mesh;
use crate::network::discovery as mesh_discovery;
use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event};

pub(super) fn emit_private_mesh_name_warning(options: &RuntimeOptions) {
    let Some(mesh_name) = options
        .mesh_name
        .as_ref()
        .filter(|_| !options.publish && !options.auto && options.discover.is_none())
    else {
        return;
    };

    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Mesh named '{}' — private by default. Add --publish to make it publicly discoverable.",
            mesh_name
        ),
        context: None,
    });
}

pub(super) fn handle_public_identity_transition(options: &RuntimeOptions) -> Result<()> {
    let is_public = options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
        && (options.auto || options.publish || options.discover.is_some());
    if is_public {
        mesh::mark_was_public()?;
        return Ok(());
    }

    if mesh::was_previously_public() {
        let _ = emit_event(OutputEvent::Info {
            message: "Previous run was public — rotating identity for private mesh".to_string(),
            context: None,
        });
        mesh::clear_public_identity()?;
    }
    Ok(())
}
