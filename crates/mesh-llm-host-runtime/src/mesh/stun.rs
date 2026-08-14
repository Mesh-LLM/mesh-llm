use super::*;
use iroh::Watcher;

/// How a public IPv4 candidate for our direct path was obtained.
///
/// The two carry different trust: an observed address was reported back to us by
/// a remote probe server, so it carries the NAT-mapped port. An enumerated one
/// came from our own interfaces and carries whatever port we bound locally, which
/// is wrong on any host that remaps ports.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum PublicAddrSource {
    /// Reported by a remote probe server via the endpoint's net report.
    Observed,
    /// Read from our own interfaces; the port is unverified.
    LocallyEnumerated,
}

/// A public IPv4 candidate together with how much we trust its port.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct PublicAddr {
    pub(crate) addr: std::net::SocketAddr,
    pub(crate) source: PublicAddrSource,
}

/// Reason an externally-observed address could not be used for the direct path.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ObservedAddrReject {
    /// The net report completed but observed no global IPv4 address.
    NoGlobalIpv4,
    /// The observed address differs per probe destination, so it is not punchable.
    MappingVariesByDestination,
}

/// Decide whether a net report yields an address safe to advertise as our direct path.
///
/// `global_v4` is the address a remote probe server observed us from, so it carries
/// the NAT-mapped port rather than the port we happen to have bound inside a
/// container. A mapping that varies by probe destination is address-dependent NAT
/// and cannot be hole-punched, so it is rejected rather than advertised.
pub(crate) fn observed_public_ipv4(
    report: &iroh::unstable_net_report::NetReport,
) -> Result<std::net::SocketAddr, ObservedAddrReject> {
    if report.mapping_varies_by_dest_ipv4 == Some(true) {
        return Err(ObservedAddrReject::MappingVariesByDestination);
    }

    report
        .global_v4
        .map(std::net::SocketAddr::V4)
        .ok_or(ObservedAddrReject::NoGlobalIpv4)
}

/// Pick a locally-enumerated public IPv4 candidate from the endpoint's addresses.
///
/// This is the pre-existing behaviour and is only used when no remote probe can
/// run at all, which is the case when relays are disabled: net reports have no
/// server to ask, so `global_v4` is never populated. Its port is unverified.
fn enumerated_public_ipv4(endpoint_addr: &iroh::EndpointAddr) -> Option<std::net::SocketAddr> {
    endpoint_addr
        .ip_addrs()
        .copied()
        .find(is_public_ipv4_candidate)
}

/// Outcome of inspecting one net report: either we are done, or we keep waiting.
enum ReportVerdict {
    Settled(Option<PublicAddr>),
    KeepWaiting,
}

fn verdict_for(report: Option<&iroh::unstable_net_report::NetReport>) -> ReportVerdict {
    let Some(report) = report else {
        return ReportVerdict::KeepWaiting;
    };
    match observed_public_ipv4(report) {
        Ok(addr) => {
            tracing::info!(%addr, "QUIC endpoint observed public address");
            ReportVerdict::Settled(Some(PublicAddr {
                addr,
                source: PublicAddrSource::Observed,
            }))
        }
        Err(ObservedAddrReject::MappingVariesByDestination) => {
            tracing::warn!(
                "QUIC endpoint public address varies by probe destination — \
                 address-dependent NAT, direct UDP unavailable"
            );
            ReportVerdict::Settled(None)
        }
        Err(ObservedAddrReject::NoGlobalIpv4) => ReportVerdict::KeepWaiting,
    }
}

/// Discover a public IPv4 address to advertise as this node's direct path.
///
/// With relays configured, this waits for a net report and uses the address a
/// remote server observed. With relays disabled there is nobody to probe, so it
/// falls back to interface enumeration and marks the result unverified.
pub(crate) async fn stun_public_addr(
    endpoint: &iroh::Endpoint,
    relay_policy: RelayPolicy,
) -> Option<PublicAddr> {
    if !relay_policy.uses_relay() {
        // No relay map means no QAD probe target, so `global_v4` will never be
        // populated. Preserve the previous enumerated behaviour rather than
        // regressing relay-disabled hosts to no direct address at all.
        let addr = enumerated_public_ipv4(&endpoint.addr())?;
        tracing::info!(
            %addr,
            "Public address from local interfaces; port unverified (no relay to probe from)"
        );
        return Some(PublicAddr {
            addr,
            source: PublicAddrSource::LocallyEnumerated,
        });
    }

    let mut reports = endpoint.net_report();
    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(iroh::NET_REPORT_TIMEOUT);

    loop {
        if let ReportVerdict::Settled(addr) = verdict_for(reports.get().as_ref()) {
            return addr;
        }

        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .unwrap_or_default();
        match tokio::time::timeout(remaining, reports.updated()).await {
            Ok(Ok(_)) => {}
            Ok(Err(_)) | Err(_) => {
                tracing::warn!("QUIC endpoint could not observe a public address");
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddrV4;

    fn report_with(
        global_v4: Option<SocketAddrV4>,
        varies: Option<bool>,
    ) -> iroh::unstable_net_report::NetReport {
        let mut report = iroh::unstable_net_report::NetReport::default();
        report.udp_v4 = global_v4.is_some();
        report.global_v4 = global_v4;
        report.mapping_varies_by_dest_ipv4 = varies;
        report
    }

    #[test]
    fn uses_externally_observed_mapped_port() {
        // The container bound 41842; the observed port is Vast's remap, 23555.
        let observed = SocketAddrV4::new([213, 5, 72, 196].into(), 23_555);
        assert_eq!(
            observed_public_ipv4(&report_with(Some(observed), Some(false))),
            Ok(std::net::SocketAddr::V4(observed))
        );
    }

    #[test]
    fn rejects_report_without_global_ipv4() {
        assert_eq!(
            observed_public_ipv4(&report_with(None, Some(false))),
            Err(ObservedAddrReject::NoGlobalIpv4)
        );
    }

    #[test]
    fn rejects_address_dependent_mapping() {
        let observed = SocketAddrV4::new([213, 5, 72, 196].into(), 23_555);
        assert_eq!(
            observed_public_ipv4(&report_with(Some(observed), Some(true))),
            Err(ObservedAddrReject::MappingVariesByDestination)
        );
    }

    #[test]
    fn accepts_when_variance_is_unknown() {
        // A single-probe report leaves variance unmeasured; that is not evidence of
        // address-dependent NAT, so the observed address is still usable.
        let observed = SocketAddrV4::new([173, 239, 92, 155].into(), 41_842);
        assert_eq!(
            observed_public_ipv4(&report_with(Some(observed), None)),
            Ok(std::net::SocketAddr::V4(observed))
        );
    }

    #[test]
    fn enumerates_public_candidate_when_no_relay_can_be_probed() {
        let endpoint_id = iroh::SecretKey::generate().public();
        let public = std::net::SocketAddr::from(([9, 9, 9, 9], 45_678));
        let addr = iroh::EndpointAddr::new(endpoint_id)
            .with_ip_addr(std::net::SocketAddr::from(([192, 168, 1, 8], 45_678)))
            .with_ip_addr(public);

        assert_eq!(enumerated_public_ipv4(&addr), Some(public));
    }

    #[test]
    fn enumeration_ignores_private_candidates() {
        let endpoint_id = iroh::SecretKey::generate().public();
        let addr = iroh::EndpointAddr::new(endpoint_id)
            .with_ip_addr(std::net::SocketAddr::from(([172, 17, 0, 2], 41_842)));

        assert_eq!(enumerated_public_ipv4(&addr), None);
    }
}
