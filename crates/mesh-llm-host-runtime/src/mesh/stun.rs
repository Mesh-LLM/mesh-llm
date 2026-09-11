use super::*;
use iroh::Watcher;

/// Provenance of a public IPv4 candidate advertised in the invite token.
///
/// The #1300 defect was exactly that these two were indistinguishable: a
/// locally enumerated interface address carries the *bound* port, which a
/// port-remapping host never verifies externally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublicAddrSource {
    /// Observed by a relay probe (QAD) through the endpoint's own socket, so
    /// the tuple includes the NAT-mapped port by construction.
    Observed,
    /// Enumerated from local interfaces. The port is the bound port and was
    /// never verified externally; only fills a gap when nothing better exists.
    LocallyEnumerated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PublicAddr {
    pub(crate) addr: std::net::SocketAddr,
    pub(crate) source: PublicAddrSource,
}

fn locally_enumerated_public_ipv4(
    endpoint_addr: &iroh::EndpointAddr,
) -> Option<std::net::SocketAddr> {
    endpoint_addr
        .ip_addrs()
        .copied()
        .find(is_public_ipv4_candidate)
}

fn observed_public_ipv4(
    report: &iroh::unstable_net_report::NetReport,
) -> Option<std::net::SocketAddr> {
    if report.mapping_varies_by_dest_ipv4 == Some(true) {
        tracing::warn!(
            "QUIC endpoint public address varies by probe destination \
             (address-dependent mapping); direct UDP unlikely, not advertising it"
        );
        return None;
    }
    report.global_v4.map(std::net::SocketAddr::V4)
}

/// Wait (bounded by `iroh::NET_REPORT_TIMEOUT`) for a net report yielding a
/// usable observed public IPv4.
async fn wait_for_observed_public_ipv4(endpoint: &iroh::Endpoint) -> Option<std::net::SocketAddr> {
    let mut net_report = endpoint.net_report();
    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(iroh::NET_REPORT_TIMEOUT);

    loop {
        if let Some(report) = net_report.get() {
            return observed_public_ipv4(&report);
        }

        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .unwrap_or_default();
        match tokio::time::timeout(remaining, net_report.updated()).await {
            Ok(Ok(_)) => {}
            Ok(Err(_)) | Err(_) => {
                tracing::warn!(
                    "QUIC endpoint could not discover a public address \
                     (net report unavailable within timeout)"
                );
                return None;
            }
        }
    }
}

/// The address to advertise for direct connectivity, with its provenance.
///
/// Prefers the externally observed tuple from the endpoint's net report: a QAD
/// probe reply carries the NAT-mapped port, which local interface enumeration
/// cannot know. Falls back to a locally enumerated public candidate only when
/// no net report can complete (`relays_configured = false`) or one completes
/// without a usable observation — preserving relay-less hosts, whose port is
/// logged as unverified.
pub(crate) async fn stun_public_addr(
    endpoint: &iroh::Endpoint,
    relays_configured: bool,
) -> Option<PublicAddr> {
    if relays_configured && let Some(addr) = wait_for_observed_public_ipv4(endpoint).await {
        tracing::info!(%addr, "QUIC endpoint discovered public address (observed by relay probe)");
        return Some(PublicAddr {
            addr,
            source: PublicAddrSource::Observed,
        });
    }

    let addr = locally_enumerated_public_ipv4(&endpoint.addr());
    if let Some(addr) = addr {
        tracing::warn!(
            %addr,
            "advertising locally enumerated public address; port not externally verified"
        );
    }
    addr.map(|addr| PublicAddr {
        addr,
        source: PublicAddrSource::LocallyEnumerated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed_report(
        global_v4: Option<std::net::SocketAddrV4>,
        mapping_varies: Option<bool>,
    ) -> iroh::unstable_net_report::NetReport {
        let mut report = iroh::unstable_net_report::NetReport::default();
        report.udp_v4 = true;
        report.global_v4 = global_v4;
        report.mapping_varies_by_dest_ipv4 = mapping_varies;
        report
    }

    #[test]
    fn observed_tuple_preserves_nat_mapped_port() {
        // The #1300 shape: container binds 41842, NAT maps it to 23555. The
        // relay-observed tuple carries the mapped port; the locally enumerated
        // candidate (same IP, port 41842) must never win over it.
        let mapped = std::net::SocketAddrV4::new("213.5.72.196".parse().unwrap(), 23_555);
        let report = observed_report(Some(mapped), Some(false));

        assert_eq!(
            observed_public_ipv4(&report),
            Some(std::net::SocketAddr::from(mapped))
        );
    }

    #[test]
    fn observed_tuple_rejected_when_mapping_varies_by_destination() {
        let mapped = std::net::SocketAddrV4::new("213.5.72.196".parse().unwrap(), 23_555);
        let report = observed_report(Some(mapped), Some(true));

        assert_eq!(observed_public_ipv4(&report), None);
    }

    #[test]
    fn unmeasured_mapping_variance_is_still_accepted() {
        // Only one relay probed: variance is None, which is not evidence of
        // an address-dependent mapping.
        let mapped = std::net::SocketAddrV4::new("213.5.72.196".parse().unwrap(), 23_555);
        let report = observed_report(Some(mapped), None);

        assert_eq!(
            observed_public_ipv4(&report),
            Some(std::net::SocketAddr::from(mapped))
        );
    }

    #[test]
    fn no_observation_yields_none() {
        assert_eq!(observed_public_ipv4(&observed_report(None, None)), None);
    }

    #[test]
    fn enumeration_fallback_finds_public_ipv4() {
        let endpoint_id = iroh::SecretKey::generate().public();
        let addr = iroh::EndpointAddr::new(endpoint_id)
            .with_ip_addr(std::net::SocketAddr::from(([9, 9, 9, 9], 45_678)));

        assert_eq!(
            locally_enumerated_public_ipv4(&addr),
            Some(std::net::SocketAddr::from(([9, 9, 9, 9], 45_678)))
        );
    }

    #[test]
    fn enumeration_ignores_private_addrs() {
        let endpoint_id = iroh::SecretKey::generate().public();
        let addr = iroh::EndpointAddr::new(endpoint_id)
            .with_ip_addr(std::net::SocketAddr::from(([192, 168, 1, 8], 45_678)));

        assert_eq!(locally_enumerated_public_ipv4(&addr), None);
    }

    #[test]
    fn merge_observed_replaces_enumerated_candidate_on_same_ip() {
        use crate::mesh::node_identity::merge_public_addr_into_advertisement;

        // Exact #1300 field shape: the container's local interface holds the
        // public IP with the bound port (41842); the relay observed the same
        // IP behind the NAT-mapped port (23555). The observed tuple must
        // replace the enumerated one — advertising both leaves the dead
        // candidate first in the token's BTreeSet.
        let endpoint_id = iroh::SecretKey::generate().public();
        let mut addr = iroh::EndpointAddr::new(endpoint_id)
            .with_ip_addr(std::net::SocketAddr::from(([213, 5, 72, 196], 41_842)));
        let observed = PublicAddr {
            addr: std::net::SocketAddr::from(([213, 5, 72, 196], 23_555)),
            source: PublicAddrSource::Observed,
        };

        merge_public_addr_into_advertisement(&mut addr, &observed);

        let ip_addrs: Vec<_> = addr.ip_addrs().copied().collect();
        assert_eq!(
            ip_addrs,
            vec![std::net::SocketAddr::from(([213, 5, 72, 196], 23_555))]
        );
    }

    #[test]
    fn merge_enumerated_only_fills_a_gap() {
        use crate::mesh::node_identity::merge_public_addr_into_advertisement;

        // An enumerated candidate must not displace an existing public one.
        let endpoint_id = iroh::SecretKey::generate().public();
        let existing = std::net::SocketAddr::from(([93, 184, 216, 34], 9));
        let mut addr = iroh::EndpointAddr::new(endpoint_id).with_ip_addr(existing);
        let enumerated = PublicAddr {
            addr: std::net::SocketAddr::from(([213, 5, 72, 211], 41_842)),
            source: PublicAddrSource::LocallyEnumerated,
        };

        merge_public_addr_into_advertisement(&mut addr, &enumerated);

        assert!(addr.ip_addrs().copied().any(|a| a == existing));
        assert!(!addr.ip_addrs().copied().any(|a| a == enumerated.addr));
    }

    #[test]
    fn merge_enumerated_inserts_when_no_public_candidate_exists() {
        use crate::mesh::node_identity::merge_public_addr_into_advertisement;

        // Relay-less host: enumeration is the only source, and it must still
        // fill a completely empty advertisement.
        let endpoint_id = iroh::SecretKey::generate().public();
        let mut addr = iroh::EndpointAddr::new(endpoint_id)
            .with_ip_addr(std::net::SocketAddr::from(([192, 168, 1, 8], 45_678)));
        let enumerated = PublicAddr {
            addr: std::net::SocketAddr::from(([213, 5, 72, 211], 41_842)),
            source: PublicAddrSource::LocallyEnumerated,
        };

        merge_public_addr_into_advertisement(&mut addr, &enumerated);

        assert!(addr.ip_addrs().copied().any(|a| a == enumerated.addr));
    }
}
