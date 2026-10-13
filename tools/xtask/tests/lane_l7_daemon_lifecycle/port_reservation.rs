//! Reserve service ports outside the OS automatic client-port range.
use std::net::{Ipv4Addr, TcpListener};

const WIDTH: u16 = 20;

pub(super) fn reserve() -> (u16, Vec<TcpListener>) {
    let (first, last) = ephemeral_range();
    let (low, high) = if first > 10_000 + WIDTH {
        (10_000, first.min(30_000) - WIDTH)
    } else {
        (
            last.checked_add(1).expect("no non-ephemeral fixture ports"),
            u16::MAX - WIDTH,
        )
    };
    assert!(low <= high, "no non-ephemeral fixture port range");
    let mut entropy = [0; 4];
    getrandom::fill(&mut entropy).expect("fixture port entropy");
    let span = u32::from(high - low) + 1;
    let start = u32::from_le_bytes(entropy) % span;
    for offset in 0..span {
        let base = low + u16::try_from((start + offset) % span).unwrap();
        let listeners: Result<Vec<_>, _> = (0..WIDTH)
            .map(|index| TcpListener::bind((Ipv4Addr::LOCALHOST, base + index)))
            .collect();
        if let Ok(listeners) = listeners {
            return (base, listeners);
        }
    }
    panic!("all fixture service port ranges are occupied");
}

#[cfg(target_os = "linux")]
fn ephemeral_range() -> (u16, u16) {
    let source = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range")
        .expect("kernel client-port range");
    let mut fields = source.split_whitespace();
    let first = fields.next().unwrap().parse().unwrap();
    let last = fields.next().unwrap().parse().unwrap();
    assert!(first <= last && fields.next().is_none());
    (first, last)
}

#[cfg(target_os = "macos")]
fn ephemeral_range() -> (u16, u16) {
    fn port(key: &std::ffi::CStr) -> u16 {
        let mut value: libc::c_int = 0;
        let mut size = std::mem::size_of_val(&value);
        // SAFETY: the kernel writes at most the supplied size into one live c_int;
        // the NUL-terminated key is static, and no new sysctl value is supplied.
        let result = unsafe {
            libc::sysctlbyname(
                key.as_ptr(),
                std::ptr::from_mut(&mut value).cast(),
                std::ptr::from_mut(&mut size),
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(
            result,
            0,
            "read kernel client-port range: {}",
            std::io::Error::last_os_error()
        );
        assert_eq!(size, std::mem::size_of_val(&value));
        u16::try_from(value).unwrap()
    }
    let first = port(c"net.inet.ip.portrange.first");
    let last = port(c"net.inet.ip.portrange.last");
    assert!(first <= last);
    (first, last)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn ephemeral_range() -> (u16, u16) {
    // Windows' default dynamic range. Customized Windows port policy remains a
    // native-platform qualification prerequisite for these service-port fixtures.
    (49_152, 65_535)
}
