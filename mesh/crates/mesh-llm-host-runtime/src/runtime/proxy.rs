use crate::inference::election;
use crate::mesh;
use crate::network::affinity;
use crate::network::openai::accept::IngressListener;

pub(super) async fn api_proxy(
    node: mesh::Node,
    port: u16,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    existing_listener: Option<IngressListener>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    crate::network::openai::ingress::api_proxy(
        node,
        port,
        target_rx,
        existing_listener,
        listen_all,
        affinity,
    )
    .await;
}

pub(super) async fn bootstrap_proxy(
    node: mesh::Node,
    port: u16,
    stop_rx: tokio::sync::mpsc::Receiver<tokio::sync::oneshot::Sender<IngressListener>>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    crate::network::openai::ingress::bootstrap_proxy(node, port, stop_rx, listen_all, affinity)
        .await;
}

#[cfg(test)]
pub(super) fn callable_models(targets: &election::ModelTargets) -> Vec<String> {
    crate::network::openai::ingress::callable_models(targets)
}

#[cfg(test)]
pub(crate) mod tests;
