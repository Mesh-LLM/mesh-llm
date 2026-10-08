mod cleanup_git;
mod fixture_interruption;
mod protocol;
mod signals;

use protocol::{Audit, Behavior, Destination, Plan};
use std::io::{self, Write};
use std::net::{Ipv4Addr, TcpListener};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().is_some_and(|argument| argument == "-C") {
        return cleanup_git::run(&arguments);
    }
    if arguments == ["--help"] {
        println!(
            "migration_client_fixture consumes fixture.json from MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR; otherwise accepts the private client argv"
        );
        return Ok(());
    }
    let root = std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")
        .ok_or("fixture requires private native root")?;
    let root = Path::new(&root);
    signals::install()?;
    match arguments.as_slice() {
        [mode] if mode == "--leaf" => return leaf(root),
        [mode] if mode == "--sentinel" => {
            std::fs::write(root.join("sentinel.armed"), b"armed")?;
            wait_stop(false)?;
            return Ok(());
        }
        _ => (),
    }
    let plan: Plan = serde_json::from_slice(&std::fs::read(root.join("fixture.json"))?)?;
    audit(root, &arguments)?;
    let port = match arguments.as_slice() {
        [log, json, port_flag, port, console, client, discovery, mdns]
            if log == "--log-format"
                && json == "json"
                && port_flag == "--port"
                && console == "--no-console"
                && client == "client"
                && discovery == "--mesh-discovery-mode"
                && mdns == "mdns" =>
        {
            port.parse::<u16>()?
        }
        _ => return Err("unexpected fixture arguments".into()),
    };
    let occupation = match plan.behavior {
        Behavior::OccupiedPort => Some(TcpListener::bind((Ipv4Addr::LOCALHOST, port))?),
        Behavior::Clean
        | Behavior::Stubborn
        | Behavior::Nonzero
        | Behavior::EarlyZero
        | Behavior::EarlyNonzero
        | Behavior::LateReady
        | Behavior::LateNewline
        | Behavior::UnterminatedEof
        | Behavior::DescendantAfterExit
        | Behavior::Flood
        | Behavior::StateDeletionFailure
        | Behavior::HeldCleanup
        | Behavior::LiveDescendant
        | Behavior::StubbornDescendant => None,
    };
    let _listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
    drop(occupation);
    match plan.behavior {
        Behavior::EarlyZero => return Ok(()),
        Behavior::EarlyNonzero => {
            io::stderr()
                .write_all(b"fixture startup diagnostic\npassword=synthetic-never-print\n")?;
            std::process::exit(23);
        }
        Behavior::DescendantAfterExit => return branch(root),
        Behavior::LiveDescendant | Behavior::StubbornDescendant => {
            return fixture_interruption::branch(root, plan.behavior);
        }
        Behavior::Flood => loop {
            io::stdout().write_all(&[b'x'; 4096])?;
            io::stderr().write_all(&[b'y'; 4096])?;
        },
        Behavior::Clean
        | Behavior::Stubborn
        | Behavior::Nonzero
        | Behavior::OccupiedPort
        | Behavior::LateReady
        | Behavior::LateNewline
        | Behavior::UnterminatedEof
        | Behavior::StateDeletionFailure
        | Behavior::HeldCleanup => (),
    }
    for record in plan.records {
        match record.stream {
            Destination::Stdout => {
                io::stdout().write_all(&record.bytes)?;
                io::stdout().flush()?;
            }
            Destination::Stderr => {
                io::stderr().write_all(&record.bytes)?;
                io::stderr().flush()?;
            }
        }
    }
    if matches!(plan.behavior, Behavior::UnterminatedEof) {
        signals::close_stdout()?;
    }
    std::fs::write(root.join("startup.armed"), b"armed")?;
    wait_stop(matches!(plan.behavior, Behavior::Stubborn))?;
    std::fs::write(root.join("handler"), b"graceful")?;
    match plan.behavior {
        Behavior::Nonzero => std::process::exit(23),
        Behavior::LateReady => io::stderr().write_all(b"{\"message\":\"Client ready\"}\n")?,
        Behavior::LateNewline => io::stdout().write_all(b"\n")?,
        Behavior::StateDeletionFailure => break_state_deletion()?,
        Behavior::HeldCleanup => fixture_interruption::hold_cleanup(root)?,
        Behavior::Clean
        | Behavior::Stubborn
        | Behavior::EarlyZero
        | Behavior::EarlyNonzero
        | Behavior::OccupiedPort
        | Behavior::UnterminatedEof
        | Behavior::DescendantAfterExit
        | Behavior::LiveDescendant
        | Behavior::StubbornDescendant
        | Behavior::Flood => (),
    }
    Ok(())
}

fn audit(root: &Path, arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let audit = Audit {
        pid: std::process::id(),
        cwd: std::env::current_dir()?,
        arguments: arguments.to_vec(),
        environment: std::env::vars()
            .map(|(key, value)| {
                let retained = match key.as_str() {
                    "HOME"
                    | "USERPROFILE"
                    | "APPDATA"
                    | "LOCALAPPDATA"
                    | "MESH_LLM_CONFIG"
                    | "MESH_LLM_RUNTIME_ROOT"
                    | "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"
                    | "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR"
                    | "XDG_CACHE_HOME"
                    | "XDG_CONFIG_HOME"
                    | "XDG_RUNTIME_DIR"
                    | "TMPDIR"
                    | "TEMP"
                    | "TMP" => value,
                    _ => "<present>".to_owned(),
                };
                (key, retained)
            })
            .collect(),
    };
    std::fs::write(root.join("audit.json"), serde_json::to_vec(&audit)?)?;
    Ok(())
}

fn wait_stop(ignore: bool) -> io::Result<()> {
    let until = Instant::now() + Duration::from_secs(20);
    while ignore || !signals::stopped() {
        if Instant::now() >= until {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "fixture stop deadline",
            ));
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    Ok(())
}

fn branch(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let child = std::process::Command::new(std::env::current_exe()?)
        .arg("--leaf")
        .spawn()?;
    let until = Instant::now() + Duration::from_secs(5);
    while !root.join("leaf.armed").is_file() {
        if Instant::now() >= until {
            return Err("leaf startup deadline".into());
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    drop(child);
    std::process::exit(0);
}

fn leaf(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(root.join("leaf.pid"), std::process::id().to_string())?;
    std::fs::write(root.join("leaf.armed"), b"armed")?;
    wait_stop(root.join("stubborn-leaf").exists())?;
    io::stderr().write_all(b"{\"message\":\"Client ready\"}\n")?;
    std::fs::write(root.join("leaf.handler"), b"graceful")?;
    Ok(())
}

fn break_state_deletion() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var_os("HOME").ok_or("missing private HOME")?;
    let state = Path::new(&home).parent().ok_or("missing private state")?;
    let displaced = state.with_extension("fixture-displaced");
    std::fs::rename(state, &displaced)?;
    std::fs::write(state, b"fixture replacement blocks remove_dir_all")?;
    Ok(())
}
