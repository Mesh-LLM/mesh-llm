use std::io::Write;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        #[cfg(unix)]
        [mode] if mode == "held-pipe" => {
            use std::os::unix::process::CommandExt;
            let mut holder = std::process::Command::new(std::env::current_exe()?);
            holder
                .arg("pipe-holder")
                .process_group(0)
                .stdin(std::process::Stdio::null());
            let _holder = holder.spawn()?;
            let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !std::path::Path::new("holder-ready").is_file() {
                if std::time::Instant::now() >= until {
                    return Err("pipe holder readiness deadline".into());
                }
                std::thread::park_timeout(std::time::Duration::from_millis(1));
            }
            std::io::stdout().write_all(b"HELD_PIPE\n")?;
        }
        #[cfg(unix)]
        [mode] if mode == "pipe-holder" => {
            std::fs::write("holder-ready", b"ready")?;
            let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !std::path::Path::new("release-holder").is_file() {
                if std::time::Instant::now() >= until {
                    return Err("pipe holder release deadline".into());
                }
                std::thread::park_timeout(std::time::Duration::from_millis(1));
            }
            std::fs::write("holder-done", b"done")?;
        }
        [mode] if mode == "raw" => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(b"token\0\xff\r\npassword secret authorization invite\n")?;
            stdout.write_all(&vec![b'x'; 9000])?;
            stdout.write_all(b"\r\nend\0")?;
            std::io::stderr().write_all(b"token diagnostic\nordinary diagnostic\r\n")?;
        }
        [mode] if mode == "held" => {
            std::io::stdout().write_all(b"token\0\xff\r\n")?;
            loop {
                std::thread::park();
            }
        }
        [mode] if mode == "large" => {
            let mut stdout = std::io::stdout().lock();
            for _ in 0..4097 {
                stdout.write_all(&[b'x'; 4096])?;
            }
        }
        [verb, external, binary, index, base, separator, path]
            if verb == "diff"
                && external == "--no-ext-diff"
                && binary == "--binary"
                && index == "--full-index"
                && base == "fixture-base"
                && separator == "--"
                && path == "src/models" =>
        {
            let bytes = std::fs::read("input.diff")?;
            std::io::stdout().write_all(&bytes)?;
            std::io::stderr().write_all(b"token diagnostic\n")?;
        }
        _ => return Err("unexpected fixture argument vector".into()),
    }
    Ok(())
}
