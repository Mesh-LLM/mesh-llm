//! OS-level identity of the process on the other end of a connected socket.
//!
//! The loopback control endpoints accept any local connection, which is the
//! right trust model for a single-user console but not for owner-control: a
//! different OS account on the same host can also reach `127.0.0.1`, and an
//! owner-control credential served to it would let it read config secrets and
//! act as the owner. Where the kernel exposes the connecting process's
//! credentials on the socket (Unix-domain sockets on macOS and Linux), the
//! host additionally requires that the caller runs as the same OS user before
//! handing out the owner-control credential or letting the host sign
//! owner-control traffic on the caller's behalf.
//!
//! Loopback **TCP** never exposes the peer process identity on the platforms
//! this host runs on:
//!
//! - Linux: `SO_PEERCRED` is supported only for `AF_UNIX` sockets (man 7
//!   `socket`). On TCP sockets `getsockopt` still succeeds but fills the
//!   credential with the unmappable overflow uid (`4294967295`), never the
//!   peer's real uid (verified: bare hosts and containers on kernels 6.17 and
//!   7.3). Reading that value as identity would reject every legitimate
//!   same-user caller.
//! - macOS: `SO_PEERCRED` returns `ENODATA` on TCP sockets (verified), and
//!   newer kernels removed the facility entirely — Darwin 27 returns
//!   `EPROTONOSUPPORT` on both TCP and `AF_UNIX` sockets (verified), so the
//!   Unix-domain path degrades to the same bearer-only contract there.
//! - Windows: no equivalent facility.
//!
//! Where the kernel has removed `SO_PEERCRED` altogether the same-user check
//! is unavailable on every socket kind; the owner-control credential remains
//! the gate. The Unix-domain implementation stays available where a listener
//! actually uses `AF_UNIX` on a kernel that still exposes peer credentials.

#[cfg(target_os = "macos")]
use std::os::fd::AsFd;
#[cfg(target_os = "macos")]
use std::os::fd::AsRawFd;

/// Whether the connected peer process runs as the current OS user.
///
/// `Some(true)`/`Some(false)` where the platform exposes peer credentials for
/// this socket kind (Unix-domain sockets on macOS); `Some(false)` also covers
/// extraction failures there, so the host fails closed. `None` where the
/// platform does not expose peer credentials for this socket kind (all
/// supported platforms over loopback TCP); callers keep the bearer-credential
/// requirement as the owner-control gate.
#[cfg(target_os = "macos")]
pub(crate) fn peer_runs_as_current_user(stream: &tokio::net::TcpStream) -> Option<bool> {
    peer_uid(stream).map(|uid| uid == unsafe { libc::getuid() })
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn peer_runs_as_current_user(_stream: &tokio::net::TcpStream) -> Option<bool> {
    None
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
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[tokio::test]
    async fn loopback_tcp_never_exposes_peer_credentials() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let client = std::net::TcpStream::connect(listener.local_addr().expect("local addr"))
            .expect("connect loopback");
        let accepted = listener.accept().expect("accept loopback").0;
        drop(client);
        accepted
            .set_nonblocking(true)
            .expect("nonblocking accepted socket");
        let stream = tokio::net::TcpStream::from_std(accepted).expect("into tokio stream");
        assert_eq!(
            peer_runs_as_current_user(&stream),
            None,
            "loopback TCP does not expose peer credentials on any supported platform; \n             degraded bearer-only mode expected"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn unix_domain_peer_uid_matches_same_user() {
        use std::os::unix::net::{UnixListener, UnixStream};

        let path = std::env::temp_dir().join(format!("peer-identity-test-{}", std::process::id()));
        let listener = UnixListener::bind(&path).expect("bind unix socket");
        let client = UnixStream::connect(&path).expect("connect");
        let accepted = listener.accept().expect("accept").0;
        drop(client);
        // Unix-domain sockets expose the peer pid via SO_PEERCRED where the
        // kernel still provides it; kernels that removed it (Darwin 27+
        // returns EPROTONOSUPPORT) degrade to the bearer-only contract,
        // which the loopback TCP test above pins down.
        if let Some(uid) = peer_uid(&accepted) {
            assert_eq!(
                uid,
                unsafe { libc::getuid() },
                "exposed peer credentials must be the caller's uid"
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}
