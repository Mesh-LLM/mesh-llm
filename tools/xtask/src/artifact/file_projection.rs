use crate::repository::check_report::CheckReport;
use sha2::{Digest, Sha256};
use std::io::Read;

pub(super) fn run(args: &[String]) -> CheckReport {
    if args == ["--help"] {
        return CheckReport::success(
            "usage: artifact file-projection {sha256 PATH|-|canonical-path PATH}\n".to_owned(),
        );
    }
    project(args).map_or_else(
        |error| CheckReport::failure(String::new(), format!("{error}\n")),
        |text| CheckReport::success(format!("{text}\n")),
    )
}

fn project(args: &[String]) -> crate::command::DynResult<String> {
    match args {
        [operation, path] if operation == "canonical-path" => {
            Ok(std::fs::canonicalize(path)?.to_string_lossy().into_owned())
        }
        [operation, path] if operation == "sha256" => {
            let mut reader: Box<dyn Read> = if path == "-" {
                Box::new(std::io::stdin().lock())
            } else {
                Box::new(std::fs::File::open(path)?)
            };
            let mut digest = Sha256::new();
            let mut buffer = [0; 8192];
            loop {
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    return Ok(hex::encode(digest.finalize()));
                }
                digest.update(&buffer[..count]);
            }
        }
        _ => Err("usage: artifact file-projection {sha256 PATH|-|canonical-path PATH}".into()),
    }
}
