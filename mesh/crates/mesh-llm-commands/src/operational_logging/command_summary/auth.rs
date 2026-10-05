use super::SummaryAssembly;

pub(super) fn format_auth(command: &mesh_llm_cli::AuthCommand, assembly: &mut SummaryAssembly) {
    use mesh_llm_cli::{AuthCommand, TrustCommand};
    match command {
        AuthCommand::Init {
            owner_key,
            force,
            no_passphrase,
            keychain,
        } => {
            assembly.command.push_str(" auth init");
            assembly.redact("--owner-key", owner_key.is_some());
            assembly.flag("force", *force);
            assembly.flag("no-passphrase", *no_passphrase);
            assembly.flag("keychain", *keychain);
        }
        AuthCommand::Status {
            owner_key,
            node_key,
            node_ownership,
            trust_store,
        } => {
            assembly.command.push_str(" auth status");
            assembly.redact("--owner-key", owner_key.is_some());
            assembly.redact("--node-key", node_key.is_some());
            assembly.redact("--node-ownership", node_ownership.is_some());
            assembly.redact("--trust-store", trust_store.is_some());
        }
        AuthCommand::SignNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
        }
        | AuthCommand::RenewNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
        } => {
            assembly
                .command
                .push_str(if matches!(command, AuthCommand::SignNode { .. }) {
                    " auth sign-node"
                } else {
                    " auth renew-node"
                });
            assembly.redact("--owner-key", owner_key.is_some());
            assembly.redact("--node-key", node_key.is_some());
            assembly.redact("--out", out.is_some());
            assembly.redact("--hostname-hint", hostname_hint.is_some());
            assembly.redact("--node-label", node_label.is_some());
            assembly.redact("--expires-in-hours", *expires_in_hours != 168);
        }
        AuthCommand::VerifyNode {
            file,
            node_id,
            trust_store,
            trust_policy,
        } => {
            assembly.command.push_str(" auth verify-node");
            assembly.redact("--file", file.is_some());
            assembly.redact("--node-id", node_id.is_some());
            assembly.redact("--trust-store", trust_store.is_some());
            assembly.redact("--verify-trust-policy", trust_policy.is_some());
        }
        AuthCommand::RotateNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
            revoke_current,
            reason,
            trust_store,
        } => {
            assembly.command.push_str(" auth rotate-node");
            assembly.redact("--owner-key", owner_key.is_some());
            assembly.redact("--node-key", node_key.is_some());
            assembly.redact("--out", out.is_some());
            assembly.redact("--hostname-hint", hostname_hint.is_some());
            assembly.redact("--node-label", node_label.is_some());
            assembly.redact("--expires-in-hours", *expires_in_hours != 168);
            assembly.flag("revoke-current", *revoke_current);
            assembly.redact("--reason", reason.is_some());
            assembly.redact("--trust-store", trust_store.is_some());
        }
        AuthCommand::RevokeOwner {
            reason,
            trust_store,
            ..
        } => {
            assembly.command.push_str(" auth revoke-owner");
            assembly.redact("owner_id", true);
            assembly.redact("--reason", reason.is_some());
            assembly.redact("--trust-store", trust_store.is_some());
        }
        AuthCommand::RevokeNode {
            cert_id,
            node_id,
            reason,
            trust_store,
        } => {
            assembly.command.push_str(" auth revoke-node");
            assembly.redact("--cert-id", cert_id.is_some());
            assembly.redact("--node-id", node_id.is_some());
            assembly.redact("--reason", reason.is_some());
            assembly.redact("--trust-store", trust_store.is_some());
        }
        AuthCommand::RotateOwner {
            owner_key,
            no_passphrase,
            force,
        } => {
            assembly.command.push_str(" auth rotate-owner");
            assembly.redact("--owner-key", owner_key.is_some());
            assembly.flag("no-passphrase", *no_passphrase);
            assembly.flag("force", *force);
        }
        AuthCommand::Trust { command } => match command {
            TrustCommand::Add {
                label, trust_store, ..
            } => {
                assembly.command.push_str(" auth trust add");
                assembly.redact("owner_id", true);
                assembly.redact("--label", label.is_some());
                assembly.redact("--trust-store", trust_store.is_some());
            }
            TrustCommand::Remove { trust_store, .. } => {
                assembly.command.push_str(" auth trust remove");
                assembly.redact("owner_id", true);
                assembly.redact("--trust-store", trust_store.is_some());
            }
            TrustCommand::List { trust_store } => {
                assembly.command.push_str(" auth trust list");
                assembly.redact("--trust-store", trust_store.is_some());
            }
        },
    }
}
