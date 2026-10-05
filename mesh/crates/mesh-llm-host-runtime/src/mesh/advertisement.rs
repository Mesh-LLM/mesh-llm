use super::*;

/// Structured view of the direct-connect candidates this node advertises in
/// invite tokens, with per-candidate external verification state.
///
/// This mirrors `Node::invite_token`'s merge + bind-IP filter pipeline so
/// diagnostics can never drift from what the token actually carries (#1300).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdvertisementSnapshot {
    /// Public IPv4 candidates in the advertisement, in advertised order.
    pub public_ipv4: Vec<AdvertisedCandidate>,
    /// LAN/private candidates kept for local reachability context.
    pub lan_candidates: Vec<std::net::SocketAddr>,
    /// The externally observed public tuple from startup discovery, if any.
    pub observed_public: Option<std::net::SocketAddr>,
    /// True when raw-STUN discovery ran at startup (not LAN-only mode).
    pub raw_stun_enabled: bool,
}

/// A single advertised public IPv4 tuple.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdvertisedCandidate {
    pub addr: std::net::SocketAddr,
    /// True when this exact tuple was externally verified by the endpoint's
    /// relay probe (QAD). An unverified candidate carries the locally bound
    /// port, which a port-remapping host never preserves (#1300).
    pub externally_verified: bool,
}

/// Split an advertisement into public and LAN candidates, marking the
/// externally observed tuple when it survived the merge.
///
/// Pure helper so the classification is unit-testable without a live endpoint.
pub(crate) fn classify_advertisement(
    addr: &EndpointAddr,
    observed_public: Option<std::net::SocketAddr>,
) -> (Vec<AdvertisedCandidate>, Vec<std::net::SocketAddr>) {
    let mut public_ipv4 = Vec::new();
    let mut lan_candidates = Vec::new();
    for candidate in addr.addrs.iter() {
        if let TransportAddr::Ip(socket) = candidate {
            if is_public_ipv4_candidate(socket) {
                public_ipv4.push(AdvertisedCandidate {
                    addr: *socket,
                    externally_verified: observed_public == Some(*socket),
                });
            } else {
                lan_candidates.push(*socket);
            }
        }
    }
    (public_ipv4, lan_candidates)
}

impl Node {
    /// Snapshot the current advertisement exactly as an invite token would
    /// carry it: same merge of the startup-discovered public address and same
    /// bind-IP filter, applied in the same order as `Node::invite_token`.
    pub(crate) fn advertisement_snapshot(&self) -> AdvertisementSnapshot {
        let mut addr = self.endpoint_addr_for_advertisement();
        if let Some(pub_addr) = self.public_addr.as_ref() {
            merge_public_addr_into_advertisement(&mut addr, pub_addr);
        }
        addr = filter_endpoint_addr_for_bind_ip(
            addr,
            self.quic_bind.ip,
            self.relay_policy.uses_raw_stun(),
        );

        let observed_public = match self.public_addr {
            Some(stun::PublicAddr::Observed(observed)) => Some(observed),
            _ => None,
        };
        let (public_ipv4, lan_candidates) = classify_advertisement(&addr, observed_public);

        AdvertisementSnapshot {
            public_ipv4,
            lan_candidates,
            observed_public,
            raw_stun_enabled: self.relay_policy.uses_raw_stun(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint_addr(sockets: &[std::net::SocketAddr]) -> EndpointAddr {
        let endpoint_id = iroh::SecretKey::generate().public();
        let mut addr = EndpointAddr::new(endpoint_id);
        for socket in sockets {
            addr.addrs.insert(TransportAddr::Ip(*socket));
        }
        addr
    }

    fn socket(octets: [u8; 4], port: u16) -> std::net::SocketAddr {
        std::net::SocketAddr::from((octets, port))
    }

    #[test]
    fn observed_tuple_is_marked_externally_verified() {
        // The #1300 field shape: the container's local interface holds the
        // public IP with the bound port; the merge replaced it with the
        // relay-observed NAT-mapped tuple.
        let enumerated = socket([213, 5, 72, 196], 41_842);
        let observed = socket([213, 5, 72, 196], 23_555);
        let mut addr = endpoint_addr(&[enumerated]);
        merge_public_addr_into_advertisement(&mut addr, &stun::PublicAddr::Observed(observed));

        let (public, lan) = classify_advertisement(&addr, Some(observed));

        assert_eq!(lan, Vec::new());
        assert_eq!(
            public,
            vec![AdvertisedCandidate {
                addr: observed,
                externally_verified: true,
            }]
        );
    }

    #[test]
    fn unverified_public_candidate_is_flagged() {
        // Discovery failed, so the enumerated candidate (bound port) survives
        // with no external verification — exactly the #1300 advertisement.
        let enumerated = socket([213, 5, 72, 196], 41_842);
        let addr = endpoint_addr(&[enumerated]);

        let (public, lan) = classify_advertisement(&addr, None);

        assert_eq!(lan, Vec::new());
        assert_eq!(
            public,
            vec![AdvertisedCandidate {
                addr: enumerated,
                externally_verified: false,
            }]
        );
    }

    #[test]
    fn private_candidates_are_classified_as_lan() {
        let lan = socket([192, 168, 1, 8], 45_678);
        let public = socket([93, 184, 216, 34], 9);
        let addr = endpoint_addr(&[lan, public]);

        let (public_candidates, lan_candidates) = classify_advertisement(&addr, None);

        assert_eq!(lan_candidates, vec![lan]);
        assert_eq!(
            public_candidates,
            vec![AdvertisedCandidate {
                addr: public,
                externally_verified: false,
            }]
        );
    }
}
