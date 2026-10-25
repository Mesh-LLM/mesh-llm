use std::{
    io::{Read, Write},
    net::{Ipv4Addr, TcpStream},
    path::Path,
    time::Duration,
};

pub(super) fn query(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let port = std::fs::read_to_string(root.join("primary.port"))?.parse::<u16>()?;
    let mut stream =
        TcpStream::connect_timeout(&(Ipv4Addr::LOCALHOST, port).into(), Duration::from_secs(1))?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    stream
        .write_all(b"GET /api/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut body = String::new();
    stream.read_to_string(&mut body)?;
    if !body.starts_with("HTTP/1.1 200 ") {
        return Err("primary unavailable".into());
    }
    Ok(())
}

#[cfg(not(test))]
pub(super) fn headless(root: &Path, scenario: &str) -> Result<(), Box<dyn std::error::Error>> {
    query(root)?;
    std::fs::write(root.join("overlap.observed"), b"live")?;
    if scenario.starts_with("overlap-") {
        std::fs::write(root.join("overlap.armed"), b"armed")?;
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !root.join("overlap.release").exists() && !super::signals::stopped() {
            if std::time::Instant::now() >= deadline {
                return Err("overlap barrier expired".into());
            }
            std::thread::park_timeout(Duration::from_millis(1));
        }
        if super::signals::stopped() {
            return Ok(());
        }
        query(root)?;
    }
    Ok(())
}
