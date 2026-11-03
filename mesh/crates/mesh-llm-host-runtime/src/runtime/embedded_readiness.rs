/// In-process evidence that the embedded runtime worker itself bound its
/// required listeners, emitted by the worker after a successful bind so
/// startup readiness cannot be spoofed by an unaffiliated HTTP service that
/// already answers `/api/status` on the console port (#2147).
#[derive(Debug)]
pub(crate) struct EmbeddedRuntimeReadiness {
    pub(crate) invite_token: Option<String>,
}

pub(crate) type EmbeddedReadinessTx =
    std::sync::mpsc::Sender<Result<EmbeddedRuntimeReadiness, String>>;
