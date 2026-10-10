use std::path::PathBuf;
use std::{io, io::ErrorKind};

pub(super) const USAGE: &str =
    "usage: cargo xtool hf-converted-artifact preflight --artifact-dir <directory>";

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PreflightArgs {
    pub(super) artifact_dir: PathBuf,
}

fn invalid_args(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, format!("{message}\n{USAGE}"))
}

pub(super) fn parse(args: &[String]) -> io::Result<Option<PreflightArgs>> {
    match args {
        [flag] if matches!(flag.as_str(), "--help" | "-h") => Ok(None),
        [flag, path] if flag == "--artifact-dir" && !path.starts_with("--") => {
            Ok(Some(PreflightArgs {
                artifact_dir: PathBuf::from(path.as_str()),
            }))
        }
        [] => Err(invalid_args(
            "the following arguments are required: --artifact-dir",
        )),
        [flag] if flag == "--artifact-dir" => Err(invalid_args(
            "argument --artifact-dir: expected one argument",
        )),
        _ => Err(invalid_args(&format!(
            "unrecognized arguments: {}",
            args.join(" ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn migration_hf_converted_artifact_args_parse_explicit_local_path() {
        let parsed = parse(&strings(&["--artifact-dir", "local artifact"]));

        assert!(matches!(
            parsed,
            Ok(Some(PreflightArgs { artifact_dir }))
                if artifact_dir.as_path() == Path::new("local artifact")
        ));
    }

    #[test]
    fn migration_hf_converted_artifact_args_expose_help_without_a_path() {
        assert!(matches!(parse(&strings(&["--help"])), Ok(None)));
        assert!(matches!(parse(&strings(&["-h"])), Ok(None)));
    }

    #[test]
    fn migration_hf_converted_artifact_args_require_directory_and_reject_remote_flags() {
        assert!(parse(&[]).is_err());
        assert!(parse(&strings(&["--artifact-dir"])).is_err());
        assert!(
            parse(&strings(&[
                "--artifact-dir",
                "/tmp/artifact",
                "--upload-only"
            ]))
            .is_err()
        );
        assert!(
            parse(&strings(&[
                "--artifact-dir",
                "/tmp/artifact",
                "--target-repo",
                "x/y"
            ]))
            .is_err()
        );
    }
}
