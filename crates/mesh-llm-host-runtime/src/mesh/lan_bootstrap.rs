#[cfg(test)]
use super::is_public_ipv4_candidate;
use iroh::{EndpointAddr, TransportAddr};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// Detect the most likely LAN IPv4 address for mDNS-only meshes.
pub fn detect_primary_lan_ipv4() -> Option<IpAddr> {
    if let Some(ip) = default_route_source_ipv4().filter(is_private_lan_interface_ipv4) {
        return Some(IpAddr::V4(ip));
    }
    first_private_lan_interface_ipv4().map(IpAddr::V4)
}

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

pub(crate) fn is_private_lan_interface_ipv4(ip: &Ipv4Addr) -> bool {
    is_private_lan_ipv4(ip) && !is_container_bridge_ipv4(ip)
}

pub(crate) fn is_container_bridge_ipv4(ip: &Ipv4Addr) -> bool {
    matches!(
        ip.octets(),
        [10, 88 | 89, _, _] | [10, 96..=111, _, _] | [10, 244, _, _] | [172, 17, _, _]
    )
}

/// Source IPv4 the kernel would use for the default route, via a connect-trick.
///
/// 192.88.99.1 is a routable, globally-assigned target; connecting a UDP socket
/// to it only drives route/source selection — it sends nothing. Returns `None`
/// when there is no default route or the source is unspecified/loopback.
pub(crate) fn default_route_source_ipv4() -> Option<Ipv4Addr> {
    let socket = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((Ipv4Addr::new(192, 88, 99, 1), 9)).ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(v4) if !v4.is_unspecified() && !v4.is_loopback() => Some(v4),
        _ => None,
    }
}

/// First operational private-LAN IPv4 from the local interface table.
///
/// Skips loopback, link-local, point-to-point, and common container bridge
/// addresses so the result is a host LAN interface peers can directly reach.
pub(crate) fn first_private_lan_interface_ipv4() -> Option<Ipv4Addr> {
    let interfaces = if_addrs::get_if_addrs().ok()?;
    interfaces
        .into_iter()
        .filter(|iface| !iface.is_loopback() && !iface.is_link_local() && !iface.is_p2p())
        .filter_map(|iface| match iface.addr {
            if_addrs::IfAddr::V4(v4) => Some(v4.ip),
            if_addrs::IfAddr::V6(_) => None,
        })
        .find(is_private_lan_interface_ipv4)
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
    pub(crate) fn common_container_bridge_ranges_are_not_selected_as_local_interfaces() {
        for ip in [
            Ipv4Addr::new(172, 17, 0, 1),
            Ipv4Addr::new(10, 88, 0, 1),
            Ipv4Addr::new(10, 96, 0, 1),
            Ipv4Addr::new(10, 244, 0, 1),
        ] {
            assert!(
                !is_private_lan_interface_ipv4(&ip),
                "{ip} must not be selected as a local LAN interface"
            );
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
