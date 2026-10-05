use std::io::Write;

use trajectory_reader::{
    DynResult, parquet_input,
    wire::{Request, Response, SCHEMA_VERSION},
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> DynResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--help"] {
        println!("trajectory-reader --request FILE --response FILE");
        return Ok(());
    }
    let [flag, file, response_flag, response_file] = args.as_slice() else {
        return Err("usage: trajectory-reader --request FILE".into());
    };
    if flag != "--request" || response_flag != "--response" {
        return Err("usage: trajectory-reader --request FILE".into());
    }
    if !std::path::Path::new(response_file).is_absolute() {
        return Err("reader response requires an absolute output path".into());
    }
    let request: Request = serde_json::from_slice(&std::fs::read(file)?)?;
    if request.schema_version != SCHEMA_VERSION || !request.dataset_file.is_absolute() {
        return Err("reader request requires schema 1 and an absolute dataset path".into());
    }
    let rows = parquet_input::select(&request.dataset_file, &request.selection)?;
    let response = Response {
        schema_version: SCHEMA_VERSION,
        rows: rows.into_iter().map(Into::into).collect(),
    };
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(response_file)?;
    serde_json::to_writer(&mut output, &response)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}
