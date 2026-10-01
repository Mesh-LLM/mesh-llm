mod protocol;
mod serving;
#[path = "../signals.rs"]
mod signals;

use protocol::{Behavior, Plan};
use std::io::{self, Write};
use std::net::{Ipv4Addr, TcpListener, UdpSocket};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let root =
        std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("native root missing")?;
    let root = Path::new(&root);
    signals::install()?;
    if arguments == ["--hold-connection"] {
        return wait_stop().map_err(Into::into);
    }
    if arguments == ["--sentinel"] {
        return wait_stop().map_err(Into::into);
    }
    let plan: Plan = serde_json::from_slice(&std::fs::read(root.join("daemon.json"))?)?;
    let audit = protocol::Audit {
        pid: std::process::id(),
        cwd: std::env::current_dir()?,
        arguments: arguments.clone(),
        environment: std::env::vars()
            .map(|(key, value)| {
                let retained = if ["HOME", "MESH_LLM_RUNTIME_ROOT", "MESH_LLM_EPHEMERAL_KEY"]
                    .contains(&key.as_str())
                {
                    value
                } else {
                    "<present>".into()
                };
                (key, retained)
            })
            .collect(),
    };
    std::fs::write(root.join("audit.json"), serde_json::to_vec(&audit)?)?;
    let (api, console, quic) = match arguments.as_slice() {
        [
            log,
            json,
            port_flag,
            api,
            console_flag,
            console,
            bind,
            quic,
            bind_ip,
            loopback,
            headless,
            serve,
            discovery,
            mdns,
        ] if log == "--log-format"
            && json == "json"
            && port_flag == "--port"
            && console_flag == "--console"
            && bind == "--bind-port"
            && bind_ip == "--bind-ip"
            && loopback == "127.0.0.1"
            && headless == "--headless"
            && serve == "serve"
            && discovery == "--mesh-discovery-mode"
            && mdns == "mdns" =>
        {
            (
                api.parse::<u16>()?,
                console.parse::<u16>()?,
                quic.parse::<u16>()?,
            )
        }
        _ => return Err("unexpected daemon argv".into()),
    };
    match plan.behavior {
        Behavior::EarlyZero => return Ok(()),
        Behavior::EarlyNonzero => std::process::exit(23),
        _ => (),
    }
    let _quic = UdpSocket::bind((Ipv4Addr::LOCALHOST, quic))?;
    if matches!(plan.behavior, Behavior::HoldBind | Behavior::ExternalModels) {
        std::fs::write(root.join("bind.armed"), b"armed")?;
        let until = Instant::now() + Duration::from_secs(5);
        while !root.join("bind.release").exists() {
            if Instant::now() >= until {
                return Err("bind release deadline".into());
            }
            std::thread::park_timeout(Duration::from_millis(1));
        }
    }
    let status = TcpListener::bind((Ipv4Addr::LOCALHOST, console))?;
    let models = TcpListener::bind((
        Ipv4Addr::LOCALHOST,
        if matches!(plan.behavior, Behavior::ExternalModels) {
            0
        } else {
            api
        },
    ))?;
    status.set_nonblocking(true)?;
    models.set_nonblocking(true)?;
    std::fs::write(root.join("startup.armed"), b"armed")?;
    let mut status_count = 0;
    let mut terminal = String::new();
    let until = Instant::now() + Duration::from_secs(20);
    while !signals::stopped() {
        if Instant::now() >= until {
            return Err("fixture deadline".into());
        }
        serving::status(&status, (&plan, api), &mut status_count)?;
        if let Some(record) = serving::models(&models, (&plan, root))? {
            terminal = record;
        }
        if matches!(plan.behavior, Behavior::Flood) {
            io::stdout().write_all(&[b'x'; 4096])?;
            io::stderr().write_all(&[b'y'; 4096])?;
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    std::fs::write(root.join("handler"), b"graceful")?;
    match plan.behavior {
        Behavior::Stubborn => std::thread::sleep(Duration::from_secs(20)),
        Behavior::Nonzero => std::process::exit(23),
        Behavior::CleanupRecord => io::stdout().write_all(terminal.as_bytes())?,
        Behavior::HoldCleanup => {
            std::fs::write(root.join("cleanup.armed"), b"armed")?;
            while !root.join("cleanup.release").exists() && Instant::now() < until {
                std::thread::park_timeout(Duration::from_millis(1));
            }
        }
        Behavior::DeleteFailure => {
            let home = std::env::var_os("HOME").ok_or("HOME missing")?;
            let state = Path::new(&home).parent().ok_or("state missing")?;
            std::fs::rename(state, state.with_extension("displaced"))?;
            std::fs::write(state, b"obstruction")?;
        }
        _ => (),
    }
    Ok(())
}

pub fn wait_stop() -> io::Result<()> {
    let until = Instant::now() + Duration::from_secs(20);
    while !signals::stopped() {
        if Instant::now() >= until {
            return Err(io::ErrorKind::TimedOut.into());
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    Ok(())
}
