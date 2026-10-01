use std::io::{Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments == ["hold"] {
        loop {
            std::thread::park();
        }
    }
    if arguments != ["metadata", "--locked", "--no-deps", "--format-version=1"] {
        return Err("unexpected metadata argv".into());
    }
    let cwd = std::env::current_dir()?;
    let record = serde_json::json!({
        "argv": arguments,
        "cwd": cwd,
        "environment": std::env::vars().collect::<std::collections::BTreeMap<_, _>>(),
    });
    std::fs::write("invocation.json", serde_json::to_vec(&record)?)?;
    let mut stdin = Vec::new();
    std::io::stdin().read_to_end(&mut stdin)?;
    if !stdin.is_empty() {
        return Err("metadata stdin was not null".into());
    }
    std::io::stderr().write_all(b"token fixture-private-diagnostic\nordinary diagnostic\n")?;
    match std::fs::read_to_string("mode")?.trim() {
        "success" => std::io::stdout().write_all(&std::fs::read("payload.json")?)?,
        "crash" => {
            std::io::stdout().write_all(&std::fs::read("payload.json")?)?;
            std::process::exit(17);
        }
        "invalid" => std::io::stdout().write_all(b"{\"password-sensitive-payload\":")?,
        "overflow" => {
            let mut stdout = std::io::stdout().lock();
            for _ in 0..4097 {
                stdout.write_all(&[b'x'; 4096])?;
            }
        }
        mode @ ("tree" | "signal-tree") => {
            #[cfg(unix)]
            let mut lease = if mode == "signal-tree" {
                Some(std::os::unix::net::UnixStream::connect("ready.sock")?)
            } else {
                None
            };
            let mut child = std::process::Command::new(std::env::current_exe()?)
                .arg("hold")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()?;
            #[cfg(unix)]
            if let Some(ready) = &mut lease {
                let result = (|| -> std::io::Result<()> {
                    std::fs::write("descendant.pid", child.id().to_string())?;
                    std::fs::write("leader.pid", std::process::id().to_string())?;
                    writeln!(ready, "{}\n{}", std::process::id(), child.id())?;
                    let mut release = [0];
                    ready.read_exact(&mut release)
                })();
                child.kill()?;
                child.wait()?;
                return match result {
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(()),
                    result => result.map_err(Into::into),
                };
            }
            #[cfg(not(unix))]
            let _ = mode;
            std::fs::write("descendant.pid", child.id().to_string())?;
            child.wait()?;
        }
        _ => return Err("unknown metadata fixture mode".into()),
    }
    Ok(())
}
