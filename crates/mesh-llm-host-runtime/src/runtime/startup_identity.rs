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

/// An explicit bootstrap token pins all discovery phases, not only the first probe.
pub(super) fn pin_explicit_join(options: &mut RuntimeOptions) {
    if !options.join.is_empty() {
        options.auto = false;
        options.discover = None;
        options.nostr_discovery = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_join_disables_public_fallback_but_preserves_admission_and_publish() {
        let mut options = RuntimeOptions {
            join: vec!["private-token".into()],
            auto: true,
            discover: Some("public".into()),
            nostr_discovery: true,
            owner_required: true,
            publish: true,
            ..RuntimeOptions::default()
        };
        pin_explicit_join(&mut options);
        assert!(!options.auto);
        assert!(options.discover.is_none());
        assert!(!options.nostr_discovery);
        assert!(options.owner_required);
        assert!(options.publish);
        assert_eq!(options.join, ["private-token"]);
    }

    #[test]
    fn discovery_without_explicit_join_is_unchanged() {
        let mut options = RuntimeOptions {
            auto: true,
            ..RuntimeOptions::default()
        };
        pin_explicit_join(&mut options);
        assert!(options.auto);
    }
}
