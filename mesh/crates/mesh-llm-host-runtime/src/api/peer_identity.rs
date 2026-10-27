//! OS-level identity of the process on the other end of a connected socket.
//!
//! The loopback control endpoints accept any local connection, which is the
//! right trust model for a single-user console but not for owner-control: a
//! different OS account on the same host can also reach `127.0.0.1`, and an
//! owner-control credential served to it would let it read config secrets and
//! act as the owner. Where the kernel exposes the connecting process's
//! credentials on the socket (`SO_PEERCRED` on Linux; Unix-domain sockets on
//! macOS) the host additionally requires that the caller runs as the same OS
//! user before handing out the owner-control credential or letting the host
//! sign owner-control traffic on the caller's behalf.
//!
//! macOS and Windows loopback TCP do not expose the peer process identity
//! (verified: macOS `SO_PEERCRED` returns `ENODATA` on TCP sockets), so the
//! check degrades to the bearer-token-only model there; the credential
//! remains required.

#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::fd::AsFd;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::fd::AsRawFd;

/// Whether the connected peer process runs as the current OS user.
///
/// `Some(true)`/`Some(false)` where the platform exposes peer credentials for
/// this socket (Linux loopback TCP; macOS Unix-domain sockets); `Some(false)`
/// also covers extraction failures on those platforms, so the host fails
/// closed there. `None` where the platform does not expose peer credentials
/// for this socket kind (macOS/Windows loopback TCP); callers keep the
/// bearer-credential requirement as the owner-control gate.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn peer_runs_as_current_user(stream: &impl AsFd) -> Option<bool> {
    #[cfg(target_os = "linux")]
    {
        Some(peer_uid(stream) == Some(unsafe { libc::getuid() }))
    }
    #[cfg(target_os = "macos")]
    {
        peer_uid(stream).map(|uid| uid == unsafe { libc::getuid() })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn peer_runs_as_current_user(_stream: &tokio::net::TcpStream) -> Option<bool> {
    None
}

#[cfg(target_os = "linux")]
fn peer_uid(stream: &impl AsFd) -> Option<u32> {
    // Matches the kernel `struct ucred` (SO_PEERCRED) layout.
    #[repr(C)]
    struct PeerCred {
        pid: i32,
        uid: u32,
        gid: u32,
    }

    let fd = stream.as_fd().as_raw_fd();
    let mut cred: PeerCred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<PeerCred>() as libc::socklen_t;
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut cred as *mut _ as *mut libc::c_void,
            &mut len,
        )
    };
    (rc == 0).then_some(cred.uid)
}

// macOS `<sys/socket.h>` defines SO_PEERCRED; the libc crate does not expose
// it for this target.
#[cfg(target_os = "macos")]
const SO_PEERCRED: libc::c_int = 27;

#[cfg(target_os = "macos")]
fn peer_uid(stream: &impl AsFd) -> Option<u32> {
    let fd = stream.as_fd().as_raw_fd();
    let mut peer_pid: libc::pid_t = 0;
    let mut len = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            SO_PEERCRED,
            &mut peer_pid as *mut _ as *mut libc::c_void,
            &mut len,
        )
    };
    if rc != 0 || peer_pid <= 0 {
        return None;
    }
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::proc_pidinfo(
            peer_pid,
            libc::PROC_PIDTBSDINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            std::mem::size_of::<libc::proc_bsdinfo>() as i32,
        )
    };
    (rc > 0).then_some(info.pbi_uid)
}

#[cfg(test)]
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::net::TcpStream;
    #[cfg(target_os = "linux")]
    use std::os::fd::FromRawFd;

    fn loopback_accepted() -> TcpStream {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let client = TcpStream::connect(listener.local_addr().expect("local addr"))
            .expect("connect loopback");
        let (accepted, _) = listener.accept().expect("accept loopback");
        drop(client);
        accepted
    }

    #[test]
    fn loopback_same_process_peer_matches_platform_exposure() {
        let accepted = loopback_accepted();
        #[cfg(target_os = "linux")]
        assert_eq!(
            peer_runs_as_current_user(&accepted),
            Some(true),
            "a same-process loopback peer must be recognized as the current OS user on Linux"
        );
        #[cfg(not(target_os = "linux"))]
        assert_eq!(
            peer_runs_as_current_user(&accepted),
            None,
            "loopback TCP does not expose peer credentials here; degraded mode expected"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unconnected_socket_fails_closed() {
        let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
        assert!(fd >= 0, "socket() must succeed");
        let stream = unsafe { TcpStream::from_raw_fd(fd) };
        assert_eq!(
            peer_runs_as_current_user(&stream),
            Some(false),
            "extraction failure must fail closed on Linux"
        );
    }
}
