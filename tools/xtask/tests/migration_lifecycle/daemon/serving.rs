use super::{
    protocol::{Behavior, Plan},
    signals, wait_stop,
};
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Duration;

fn accept(listener: &TcpListener) -> io::Result<Option<(TcpStream, String)>> {
    let (mut stream, _) = match listener.accept() {
        Ok(pair) => pair,
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
        Err(error) => return Err(error),
    };
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut request = Vec::new();
    while !request.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        stream.read_exact(&mut byte)?;
        request.push(byte[0]);
        if request.len() > 8192 {
            return Err(io::ErrorKind::InvalidData.into());
        }
    }
    Ok(Some((
        stream,
        String::from_utf8(request).map_err(|_| io::ErrorKind::InvalidData)?,
    )))
}

pub fn status(listener: &TcpListener, context: (&Plan, u16), count: &mut usize) -> io::Result<()> {
    let Some((mut stream, request)) = accept(listener)? else {
        return Ok(());
    };
    assert!(request.starts_with("GET /api/status HTTP/1.1\r\n"));
    assert!(!request.contains("x-request-id"));
    let (plan, api) = context;
    if let Some(wire) = &plan.status_wire {
        return stream.write_all(wire);
    }
    let code = if *count < plan.status_failures {
        503
    } else {
        200
    };
    *count += 1;
    let body = plan
        .status
        .replace("PID", &std::process::id().to_string())
        .replace("API", &api.to_string());
    write!(
        stream,
        "HTTP/1.1 {code} Fixture\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

pub fn models(listener: &TcpListener, context: (&Plan, &Path)) -> io::Result<Option<String>> {
    let Some((mut stream, request)) = accept(listener)? else {
        return Ok(None);
    };
    let (plan, root) = context;
    assert!(request.starts_with("GET /v1/models HTTP/1.1\r\n"));
    let ids: Vec<_> = request
        .lines()
        .filter_map(|line| line.strip_prefix("x-request-id: "))
        .collect();
    assert_eq!(ids.len(), 1);
    let id = ids[0];
    assert_eq!(uuid::Uuid::parse_str(id).unwrap().get_version_num(), 4);
    let count_path = root.join("models.count");
    assert!(
        !count_path.exists(),
        "models must be requested exactly once"
    );
    std::fs::write(count_path, b"1")?;
    std::fs::write(root.join("models.armed"), b"armed")?;
    if matches!(plan.behavior, Behavior::ExitModels) {
        #[cfg(unix)]
        {
            let child = std::process::Command::new(std::env::current_exe()?)
                .arg("--hold-connection")
                .stdin(std::process::Stdio::from(std::os::fd::OwnedFd::from(
                    stream,
                )))
                .spawn()?;
            std::fs::write(root.join("leaf.pid"), child.id().to_string())?;
            drop(child);
        }
        std::process::exit(0);
    }
    if matches!(plan.behavior, Behavior::SlowHeaders) {
        wait_stop()?;
        return Ok(None);
    }
    let terminal = plan
        .record
        .replace("ID", id)
        .replace("CODE", &plan.models_code.to_string())
        .replace(
            "EVENT",
            if plan.models_code < 300 {
                "request_completed"
            } else {
                "request_failed"
            },
        )
        .replace(
            "OUTCOME",
            if plan.models_code < 300 {
                "completed"
            } else {
                "failed"
            },
        );
    if plan.before_body && !matches!(plan.behavior, Behavior::CleanupRecord) {
        emit(&terminal)?;
    }
    if matches!(plan.behavior, Behavior::SlowBody) {
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n")?;
        wait_stop()?;
        return Ok(Some(terminal));
    }
    if matches!(plan.behavior, Behavior::EndlessBody) {
        stream.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")?;
        while !signals::stopped() {
            if stream.write_all(b"1\r\nx\r\n").is_err() {
                break;
            }
        }
        return Ok(Some(terminal));
    }
    let result = match &plan.wire {
        Some(wire) => stream.write_all(wire),
        None => {
            write!(
                stream,
                "HTTP/1.1 {} Fixture\r\nContent-Length: {}\r\n\r\n",
                plan.models_code,
                plan.models_body.len()
            )?;
            stream.write_all(&plan.models_body)
        }
    };
    if let Err(error) = result
        && ![io::ErrorKind::BrokenPipe, io::ErrorKind::ConnectionReset].contains(&error.kind())
    {
        return Err(error);
    }
    drop(stream);
    if !plan.before_body && !matches!(plan.behavior, Behavior::CleanupRecord) {
        emit(&terminal)?;
    }
    if plan.record.ends_with("EOF") {
        signals::close_stdout()?;
    }
    Ok(Some(terminal))
}

fn emit(record: &str) -> io::Result<()> {
    io::stdout().write_all(b"\n")?;
    io::stdout().write_all(record.trim_end_matches("EOF").as_bytes())?;
    io::stdout().flush()
}
