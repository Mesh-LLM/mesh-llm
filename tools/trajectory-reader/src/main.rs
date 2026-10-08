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
        println!(
            "trajectory-reader --request FILE --response FILE\n{}",
            trajectory_reader::cohorts::cli::USAGE
        );
        #[cfg(feature = "corpus-input")]
        println!("{}", trajectory_reader::corpus::cli::USAGE);
        return Ok(());
    }
    #[cfg(feature = "corpus-input")]
    if args.first().is_some_and(|arg| arg == "corpus") {
        return trajectory_reader::corpus::cli::run(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "cohorts") {
        return trajectory_reader::cohorts::cli::run(&args[1..]);
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
