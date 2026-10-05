//! IPv4 address classification shared by discovery and invite-token assembly.
//!
//! Moved from host-runtime `mesh/connections.rs`. These predicates decide which
//! candidates are worth advertising as a public direct path (STUN-observed
//! NAT-mapped tuples) versus suppressing (private, loopback, CGNAT, link-local,
//! documentation, and reserved ranges).

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// True when `socket` is a globally routable IPv4 candidate worth advertising
/// as a direct path. IPv6 candidates are never advertised by this classifier.
pub fn is_public_ipv4_candidate(socket: &SocketAddr) -> bool {
    match socket.ip() {
        IpAddr::V4(ip) => is_global_ipv4_candidate(ip),
        IpAddr::V6(_) => false,
    }
}

/// True when `ip` is a globally routable IPv4 address, excluding private,
/// loopback, link-local, multicast, broadcast, unspecified, CGNAT
/// (100.64/10), documentation (192.0.0.0/24, 192.0.2.0/24, 198.18/15,
/// 198.51.100/24, 203.0.113/24) and reserved (>= 240) ranges.
pub fn is_global_ipv4_candidate(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_multicast()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || (a == 100 && (64..=127).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 240)
}
