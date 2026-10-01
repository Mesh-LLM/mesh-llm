mod overlap;
#[path = "../../migration_lifecycle/signals.rs"]
#[expect(
    dead_code,
    reason = "shared fixture signals include stdout closure unused by smoke"
)]
mod signals;
mod wire;

use std::{
    io,
    net::{Ipv4Addr, TcpListener},
    path::PathBuf,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    signals::install()?;
    let args: Vec<_> = std::env::args().skip(1).collect();
    let value = |flag| {
        args.windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
            .ok_or("missing fixture flag")
    };
    if value("--log-format")? != "json" || !args.iter().any(|arg| arg == "--no-draft") {
        return Err("invalid smoke launch".into());
    }
    let api = value("--port")?.parse::<u16>()?;
    let console = value("--console")?.parse::<u16>()?;
    let root = PathBuf::from(
        std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR").ok_or("missing root")?,
    );
    let scenario = std::fs::read_to_string(root.join("scenario"))?;
    let headless = args.iter().any(|arg| arg == "--headless");
    if headless {
        overlap::headless(&root, &scenario)?;
    } else {
        std::fs::write(root.join("primary.port"), console.to_string())?;
    }
    if scenario == "early" {
        return Ok(());
    }
    let model = value("--model")?;
    let audit = serde_json::json!({"arguments": &args, "stack": std::env::var("MESH_TOKIO_STACK_SIZE").ok(),
        "config": std::fs::read_to_string(std::env::var_os("MESH_LLM_CONFIG").ok_or("config")?).ok()});
    std::fs::write(
        root.join(if headless {
            "headless-audit.json"
        } else {
            "primary-audit.json"
        }),
        serde_json::to_vec(&audit)?,
    )?;
    let listeners = [
        TcpListener::bind((Ipv4Addr::LOCALHOST, api))?,
        TcpListener::bind((Ipv4Addr::LOCALHOST, console))?,
    ];
    for listener in &listeners {
        listener.set_nonblocking(true)?;
    }
    let started = Instant::now();
    let mut chats = 0;
    while !signals::stopped() {
        if !headless && scenario == "overlap-early" && root.join("primary.exit").exists() {
            return Ok(());
        }
        if started.elapsed() > Duration::from_secs(20) {
            return Err("fixture expired".into());
        }
        for listener in &listeners {
            match listener.accept() {
                Ok((stream, _)) => wire::serve(
                    stream,
                    &mut chats,
                    wire::Context {
                        api,
                        scenario: &scenario,
                        model,
                        headless,
                    },
                )?,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => (),
                Err(error) => return Err(error.into()),
            }
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    match scenario.as_str() {
        "headless-spawn" => (),
        "stubborn" => std::thread::sleep(Duration::from_secs(20)),
        "nonzero" => std::process::exit(23),
        "delete" => {
            let home = PathBuf::from(std::env::var_os("HOME").ok_or("home")?);
            let state = home.parent().ok_or("state")?;
            std::fs::rename(state, state.with_extension("displaced"))?;
            std::fs::write(state, b"obstruction")?;
        }
        _ => (),
    }
    Ok(())
}
