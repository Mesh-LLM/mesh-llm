use std::{collections::BTreeMap, fs, path::Path, time::Duration};

use serde_json::{Value as Json, json};

use super::{
    fixture_catalog as catalog,
    fixture_fetch::{self, Adapter},
    fixture_materialization,
};
use crate::{automation::command_interrupt::Interrupt, command::DynResult, process::Value};

pub(super) fn run(verb: &str, args: &[String]) -> DynResult<()> {
    let mut values = BTreeMap::new();
    let mut remaining = args;
    while !remaining.is_empty() {
        let [flag, value, rest @ ..] = remaining else {
            return Err("fixture fetch option requires a value".into());
        };
        if ![
            "--catalog",
            "--profile",
            "--hf-bin",
            "--cache-dir",
            "--timeout",
            "--output",
        ]
        .contains(&flag.as_str())
        {
            return Err(format!("unknown fixture fetch option {flag}").into());
        }
        if values.insert(flag.as_str(), value.as_str()).is_some() {
            return Err(format!("duplicate fixture fetch option {flag}").into());
        }
        remaining = rest;
    }
    let required = |name| -> DynResult<&str> {
        values
            .get(name)
            .copied()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("missing {name}").into())
    };
    let profile = required("--profile")?;
    let input: Json = serde_json::from_slice(&fs::read(required("--catalog")?)?)?;
    catalog::validate(&input)?;
    let selected = input["profiles"]
        .get(profile)
        .ok_or("unknown scheduler fixture profile")?;
    let corpus = catalog::object(selected, "corpus")?;
    if catalog::text(corpus, "kind")? != "hf" {
        return Err("fixture fetch requires an HF corpus".into());
    }
    let dataset = &input["datasets"][catalog::text(corpus, "dataset")?];
    let seconds: u64 = values.get("--timeout").copied().unwrap_or("600").parse()?;
    if !(1..=86400).contains(&seconds) {
        return Err("fixture timeout must be in 1..=86400 seconds".into());
    }
    let executable = Path::new(required("--hf-bin")?);
    if !executable.is_absolute() {
        return Err("--hf-bin must name an absolute ecosystem executable".into());
    }
    // Validate publication inputs before allowing the optional download.
    let output = if verb == "prepare-fixture" {
        Some(required("--output")?)
    } else if values.contains_key("--output") {
        return Err("fetch-fixture does not accept --output".into());
    } else {
        None
    };
    let adapter = Adapter {
        executable: executable.canonicalize()?,
        cwd: std::env::current_dir()?.canonicalize()?,
        environment: environment(),
        cache_dir: values.get("--cache-dir").map(|value| (*value).into()),
        budget: Duration::from_secs(seconds),
    };
    let interrupt = Interrupt::install()?;
    let result = fixture_fetch::fetch(dataset, &adapter, &interrupt.cancellation());
    interrupt.finish()?;
    let parquet = result?;
    if let Some(output) = output {
        let digest =
            fixture_materialization::materialize(&input, profile, &parquet, Path::new(output))?;
        println!("{}", json!({"output":output,"sha256":digest}));
    } else {
        println!("{}", parquet.display());
    }
    Ok(())
}

fn environment() -> BTreeMap<std::ffi::OsString, Value> {
    let public = [
        "PATH",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "SYSTEMROOT",
        "TMPDIR",
        "TMP",
        "TEMP",
        "HF_HOME",
        "HF_HUB_CACHE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ];
    let secret = [
        "HF_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ];
    std::env::vars_os()
        .filter_map(|(key, value)| {
            if value.is_empty() {
                None
            } else if public.iter().any(|name| key == *name) {
                Some((key, Value::Public(value)))
            } else if secret.iter().any(|name| key == *name) {
                Some((key, Value::Secret(value)))
            } else {
                None
            }
        })
        .collect()
}
