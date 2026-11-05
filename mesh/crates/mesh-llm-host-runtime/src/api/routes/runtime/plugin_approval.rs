//! Parsing and forwarding explicit plugin invocation approvals.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct RawPluginCommandApproval {
    plugin: String,
    command: String,
    invocation: Option<RawPluginArguments>,
}

#[derive(Debug, Deserialize)]
struct RawPluginArguments {
    args: Vec<String>,
}

impl RawPluginCommandApproval {
    pub(super) fn to_proto(&self) -> mesh_client::proto::node::OwnerControlPluginCommandApproval {
        mesh_client::proto::node::OwnerControlPluginCommandApproval {
            plugin: self.plugin.clone(),
            command: self.command.clone(),
            invocation: self.invocation.as_ref().map(|invocation| {
                mesh_client::proto::node::OwnerControlPluginArguments {
                    args: invocation.args.clone(),
                }
            }),
        }
    }
}
