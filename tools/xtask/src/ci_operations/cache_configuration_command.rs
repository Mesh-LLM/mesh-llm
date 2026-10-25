use super::cache_configuration::{Configuration, configuration};
use crate::{command::DynResult, repository::check_report::CheckReport};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if !args.is_empty() {
        return CheckReport::usage("ci-ops configure-canary-cache", "no arguments are accepted")
            .emit();
    }
    let result = execute();
    match result {
        Ok(()) => Ok(()),
        Err(message) => CheckReport::failure(
            String::new(),
            format!("canary cache preflight: {message}\n"),
        )
        .emit(),
    }
}

fn execute() -> Result<(), String> {
    let names = [
        "HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_CACHE",
        "HF_HUB_CACHE",
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "GITHUB_ENV",
    ];
    let mut env = BTreeMap::new();
    for name in names {
        match std::env::var(name) {
            Ok(value) => {
                env.insert(name.to_owned(), value);
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(_) => return Err(format!("{name} must be UTF-8")),
        }
    }
    let config = configuration(&env)?;
    let path = env
        .get("GITHUB_ENV")
        .filter(|path| !path.is_empty())
        .ok_or("GITHUB_ENV is required")?;
    export(&config, Path::new(path), &mut io::stdout().lock()).map_err(|error| error.to_string())
}

fn export(config: &Configuration, path: &Path, output: &mut impl Write) -> io::Result<()> {
    if let Some((_, token)) = config.values.iter().find(|(name, _)| *name == "HF_TOKEN") {
        writeln!(output, "::add-mask::{}", token.replace('%', "%25"))?;
        output.flush()?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    for (name, value) in &config.values {
        writeln!(file, "{name}={value}")?;
    }
    writeln!(
        output,
        "Using existing Hugging Face cache: {} (offline certification)",
        config.hub.display()
    )?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Observer<'a> {
        path: &'a Path,
        bytes: Vec<u8>,
        flushed: bool,
    }
    impl Write for Observer<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if !self.flushed {
                assert!(!self.path.exists());
                self.flushed = true;
            }
            Ok(())
        }
    }

    #[test]
    fn token_mask_is_flushed_before_environment_file_creation() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("environment");
        let config = Configuration {
            hub: root.path().join("hub"),
            values: vec![("HF_TOKEN", "fixture%token".into())],
        };
        let mut observer = Observer {
            path: &path,
            bytes: Vec::new(),
            flushed: false,
        };
        export(&config, &path, &mut observer).unwrap();
        assert!(observer.flushed);
        assert!(observer.bytes.starts_with(b"::add-mask::fixture%25token\n"));
        assert_eq!(
            std::fs::read_to_string(path).unwrap(),
            "HF_TOKEN=fixture%token\n"
        );
    }
}
