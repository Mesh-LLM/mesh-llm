use crate::mesh;
use crate::network::discovery as mesh_discovery;
use crate::runtime::RuntimeOptions;
use std::net::IpAddr;
use tokio::task::JoinHandle;

pub(crate) fn effective_quic_bind_ip(options: &RuntimeOptions) -> Option<IpAddr> {
    // Explicit operator override always wins.
    if let Some(ip) = options.bind_ip {
        return Some(ip);
    }

    // Default: bind iroh's wildcard sockets (0.0.0.0 + [::]) and let iroh own
    // candidate discovery. We used to auto-pin the detected primary LAN IPv4
    // here, which created a SECOND local UDP socket alongside iroh's wildcard
    // socket on one endpoint id. iroh's docs are explicit that manual binding is
    // advanced/usually-unnecessary and replaces the wildcard bind; pinning one
    // family left a stray socket whose 4-tuple diverged from the path iroh
    // validated, contributing to direct paths idling out. With QAD supplying the
    // public reflexive candidate and iroh's own NAT traversal, LAN pinning is no
    // longer needed. Override with --bind-ip for genuinely special cases.
    // See mesh-llm issue #1065.
    None
}

/// Background tasks spawned by [`spawn_mdns_reverse_dial`] for relay-less LAN
/// direct-path bootstrap. Dropping this guard aborts those loops.
#[derive(Default)]
pub(crate) struct LanBootstrapTasks {
    handles: Vec<JoinHandle<()>>,
}

impl LanBootstrapTasks {
    pub(crate) fn abort(&self) {
        for handle in &self.handles {
            handle.abort();
        }
    }
}

impl Drop for LanBootstrapTasks {
    fn drop(&mut self) {
        self.abort();
    }
}

pub(crate) fn spawn_mdns_reverse_dial(
    options: &RuntimeOptions,
    node: &mesh::Node,
) -> LanBootstrapTasks {
    if options.mesh_discovery_mode != mesh_discovery::MeshDiscoveryMode::Mdns {
        return LanBootstrapTasks::default();
    }

    let mut handles = Vec::new();

    if !options.publish {
        handles.push(tokio::spawn(Box::pin(mesh_discovery::publish_lan_loop(
            node.clone(),
            mesh_discovery::LanPublishConfig {
                name: options.mesh_name.clone(),
                region: options.region.clone(),
                max_clients: options.max_clients,
                api_port: options.console,
                details_reachable: options.listen_all,
                interval_secs: 30,
                status_tx: None,
            },
        ))));
    }

    handles.push(tokio::spawn(Box::pin(
        crate::network::mdns_reverse_dial::run_loop(
            node.clone(),
            options.mesh_name.clone(),
            options.region.clone(),
        ),
    )));
    handles.push(crate::network::lan_beacon::spawn(node.clone()));

    LanBootstrapTasks { handles }
}
