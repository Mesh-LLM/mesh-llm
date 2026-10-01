use super::builder::Builder;
use crate::command::DynResult;

pub(super) fn append(builder: &mut Builder<'_>) -> DynResult<()> {
    if !builder.options.cargo {
        return Ok(());
    }
    let cargo = std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join("cargo"))
        .find(|path| path.is_file())
        .ok_or("mixed-version Cargo command missing")?
        .canonicalize()?;
    for (name, arguments) in [
        (
            "config-missing-endpoint-required",
            vec![
                "test",
                "-p",
                "mesh-llm",
                "--test",
                "protocol_compat_v0_client",
                "missing_control_endpoint_rejects_config_bootstrap",
            ],
        ),
        (
            "config-new-client-owner-control",
            vec![
                "test",
                "-p",
                "mesh-llm",
                "--test",
                "protocol_compat_v0_client",
                "explicit_control_endpoint_selects_owner_control",
            ],
        ),
        (
            "config-control-rejects-legacy-frames",
            vec![
                "test",
                "-p",
                "mesh-llm-client",
                "--test",
                "protocol_wire",
                "owner_control_legacy_json_rejects_with_structured_error",
            ],
        ),
        (
            "config-owner-control-protocol-contract",
            vec!["test", "-p", "mesh-llm-protocol", "--lib", "owner_control"],
        ),
        (
            "config-owner-control-client-hardening",
            vec![
                "test",
                "-p",
                "mesh-llm-client",
                "--test",
                "control_plane_client",
            ],
        ),
        (
            "config-owner-control-host-hardening",
            vec![
                "test",
                "-p",
                "mesh-llm-host-runtime",
                "--lib",
                "owner_control",
            ],
        ),
        (
            "config-owned-node-cli",
            vec![
                "test",
                "-p",
                "mesh-llm",
                "--test",
                "owned_node_commands_cli",
            ],
        ),
    ] {
        builder.command(
            name,
            &cargo,
            arguments.into_iter().map(str::to_owned).collect(),
            false,
        )?;
    }
    Ok(())
}
