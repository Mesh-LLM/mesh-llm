use std::env;
use std::ffi::OsString;
use std::io::{self, ErrorKind, Write};
use std::path::PathBuf;

use mesh_llm_cli::{check_cli_inventory, write_cli_inventory};

fn main() {
    if let Err(error) = run(env::args_os().skip(1).collect()) {
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
