#[cfg(test)]
use super::is_public_ipv4_candidate;
use iroh::{EndpointAddr, TransportAddr};
use std::net::{Ipv4Addr, SocketAddr};

pub(super) fn lan_ipv4_candidates(addr: &EndpointAddr) -> Vec<std::net::SocketAddrV4> {
    addr.addrs
        .iter()
        .filter_map(|addr| match addr {
            TransportAddr::Ip(SocketAddr::V4(v4)) if is_private_lan_ipv4(v4.ip()) => Some(*v4),
            _ => None,
        })
        .collect()
}

pub(crate) fn is_private_lan_ipv4(ip: &Ipv4Addr) -> bool {
    ip.is_private()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub(crate) fn private_rfc1918_ranges_are_lan() {
        for ip in [
            Ipv4Addr::new(10, 0, 0, 5),
            Ipv4Addr::new(10, 96, 0, 5),
            Ipv4Addr::new(172, 16, 4, 9),
            Ipv4Addr::new(172, 17, 0, 5),
            Ipv4Addr::new(172, 31, 255, 1),
            Ipv4Addr::new(192, 168, 86, 60),
        ] {
            assert!(is_private_lan_ipv4(&ip), "{ip} should be treated as LAN");
        }
    }

    #[test]
    pub(crate) fn public_cgnat_link_local_and_loopback_are_not_lan() {
        for ip in [
            Ipv4Addr::new(8, 8, 8, 8),
            Ipv4Addr::new(100, 64, 0, 1),
            Ipv4Addr::new(169, 254, 10, 10),
            Ipv4Addr::new(127, 0, 0, 1),
            Ipv4Addr::new(172, 32, 0, 1),
        ] {
            assert!(!is_private_lan_ipv4(&ip), "{ip} must not be treated as LAN");
        }
    }

    #[test]
    pub(crate) fn public_candidate_classifier_excludes_private_and_cgnat() {
        let public = SocketAddr::from(([203, 0, 113, 0], 9));
        assert!(!is_public_ipv4_candidate(&public));
        let real_public = SocketAddr::from(([9, 9, 9, 9], 9));
        assert!(is_public_ipv4_candidate(&real_public));
        let lan = SocketAddr::from(([192, 168, 1, 50], 9));
        assert!(!is_public_ipv4_candidate(&lan));
        let cgnat = SocketAddr::from(([100, 100, 1, 1], 9));
        assert!(!is_public_ipv4_candidate(&cgnat));
    }
}
