//! A client builder for a node's own plain-HTTP endpoints on 127.0.0.1.

/// A builder for clients that only call this node's own plain-HTTP endpoints
/// on 127.0.0.1 (API, console, planner). It trusts no roots: such a client
/// never talks TLS, so building it doesn't load the system's CA certificates,
/// which fails on systems without a CA bundle. Those endpoints never redirect,
/// so it doesn't follow redirects either, and it ignores proxy settings:
/// requests stay on the address they were sent to. Callers add their own
/// timeout.
pub fn local_http_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .tls_certs_only([])
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
}
