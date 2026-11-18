use std::env;
use std::ffi::OsString;
use std::io::{self, ErrorKind, Write};
use std::path::PathBuf;

use mesh_llm_cli::{check_cli_inventory, write_cli_inventory};

fn main() {
    // Clap reflection needs more stack than the Windows main thread provides.
    let result = std::thread::Builder::new()
        .name("cli-inventory".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| run(env::args_os().skip(1).collect()))
        .and_then(|thread| {
            thread
                .join()
                .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
        });
    if let Err(error) = result {
        let message = format!("mesh-llm-cli-inventory: {error}\n");
        let _ = io::stderr().write_all(message.as_bytes());
        std::process::exit(1);
    }
}

fn run(args: Vec<OsString>) -> io::Result<()> {
    match args.as_slice() {
        [path] => write_cli_inventory(PathBuf::from(path)),
        [flag, path] if flag == "--check" => check_cli_inventory(PathBuf::from(path)),
        _ => Err(io::Error::new(
            ErrorKind::InvalidInput,
            "usage: mesh-llm-cli-inventory [--check] <output-path>",
        )),
    }
}
