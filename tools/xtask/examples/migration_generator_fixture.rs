use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut trace = OpenOptions::new()
        .create(true)
        .append(true)
        .open("trace.jsonl")?;
    writeln!(trace, "{}", serde_json::to_string(&arguments)?)?;
    match arguments.as_slice() {
        [command, porcelain, untracked]
            if command == "status"
                && porcelain == "--porcelain"
                && untracked == "--untracked-files=no" =>
        {
            io::stdout().write_all(&fs::read("status.txt")?)?;
        }
        [command, head] if command == "rev-parse" && head == "HEAD" => {
            if Path::new("head.txt").exists() {
                io::stdout().write_all(&fs::read("head.txt")?)?;
            } else {
                io::stdout().write_all(b"fallback-identity\r\n")?;
            }
        }
        [command, no_ext, binary, full, base, separator, models]
            if command == "diff"
                && no_ext == "--no-ext-diff"
                && binary == "--binary"
                && full == "--full-index"
                && base == "fixture-base"
                && separator == "--"
                && models == "src/models" =>
        {
            io::stdout().write_all(&fs::read("input.diff")?)?;
        }
        [first, ..] if first == "--source-root" => rewrite(&arguments)?,
        _ => return Err("unexpected generator fixture argv".into()),
    }
    Ok(())
}

fn rewrite(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let pass = if arguments.iter().any(|argument| argument == "--apply") {
        "first"
    } else {
        "second"
    };
    if Path::new(&format!("fail-{pass}")).exists() {
        std::process::exit(23);
    }
    let index = arguments
        .iter()
        .position(|argument| argument == "--report")
        .ok_or("report flag missing")?;
    let report = arguments.get(index + 1).ok_or("report value missing")?;
    fs::copy(format!("{pass}.json"), report)?;
    Ok(())
}
