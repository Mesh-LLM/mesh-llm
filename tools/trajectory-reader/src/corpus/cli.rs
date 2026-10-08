use super::{acquisition, config::Configuration, digest, document, sampling};
use crate::DynResult;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

pub const USAGE: &str = "trajectory-reader corpus TIER [--config FILE] [--out-root DIR] [--hf-dir DIR] [--artifact-manifest FILE] [--seed N] [--max-prompt-chars N] [--sample-multiplier N]";
pub fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let options = Options::parse(args)?;
    let destination = options.output.join(&options.tier);
    require_fresh(&destination)?;
    let config_bytes = bounded(&options.config, 8 * 1024 * 1024)?;
    let config: Configuration = serde_json::from_slice(&config_bytes)?;
    let tier = config.validate(&options.tier)?;
    let seed = options.seed.unwrap_or(config.seed);
    let max = options.max.unwrap_or(tier.max_prompt_chars.unwrap_or(6000));
    super::prompt_budget("", max, tier.target_prompt_chars)?;
    let mut rows = Vec::new();
    let mut sources = Vec::new();
    let mut retained_output_bytes = 0usize;
    for source in &config.sources {
        let quota = source.quota.get(&options.tier).copied().unwrap_or_default();
        if quota == 0 {
            continue;
        }
        let acquired = acquisition::acquire(source, &options.cache, options.manifest.as_deref())?;
        let limit = quota
            .checked_mul(options.multiplier)
            .ok_or("sample size overflow")?
            .max(quota.checked_add(25).ok_or("sample size overflow")?);
        let candidates = sampling::sample(&acquired.artifacts, source, seed, limit)?;
        let mut accepted = 0;
        let mut generated = 0;
        for candidate in candidates {
            let projected = document::normalize(
                source,
                &options.tier,
                accepted,
                &candidate,
                max,
                tier.target_prompt_chars,
            )?;
            if projected.is_empty() {
                continue;
            }
            for row in &projected {
                retained_output_bytes = retained_output_bytes
                    .checked_add(serde_json::to_vec(row)?.len() + 1)
                    .ok_or("corpus size overflow")?;
                if retained_output_bytes > 128 * 1024 * 1024 {
                    return Err("corpus output exceeds 128MiB".into());
                }
            }
            generated += projected.len();
            rows.extend(projected);
            accepted += 1;
            if accepted == quota {
                break;
            }
        }
        if accepted != quota {
            return Err(format!("{} produced {accepted}/{quota} sessions", source.name).into());
        }
        sources.push(json!({"name":source.name,"dataset":source.dataset,"config":source.config,"split":source.split,"revision":source.revision,"resolved_revision":source.revision,"family":source.family,"adapter":source.adapter,"routing_hint":source.routing_hint,"quota":quota,"generated_rows":generated,"artifacts":acquired.artifacts,"acquisition":acquired.provenance}));
    }
    let mut corpus = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut corpus, row)?;
        corpus.push(b'\n');
        if corpus.len() > 128 * 1024 * 1024 {
            return Err("corpus output exceeds 128MiB".into());
        }
    }
    let manifest = json!({"schema_version":1,"tier":options.tier,"seed":seed,"generator":"trajectory-reader corpus","sampling_algorithm":sampling::ALGORITHM,"configuration_sha256":digest(&config_bytes),"corpus_sha256":digest(&corpus),"row_count":rows.len(),"max_prompt_chars":max,"target_prompt_chars":tier.target_prompt_chars,"sources":sources});
    publish(&destination, &corpus, &manifest)?;
    println!(
        "generated {} rows\ncorpus: {}\nmanifest: {}",
        rows.len(),
        options
            .output
            .join(&options.tier)
            .join("corpus.jsonl")
            .display(),
        options
            .output
            .join(&options.tier)
            .join("manifest.json")
            .display()
    );
    Ok(())
}
struct Options {
    tier: String,
    config: PathBuf,
    output: PathBuf,
    cache: PathBuf,
    manifest: Option<PathBuf>,
    seed: Option<u64>,
    max: Option<usize>,
    multiplier: usize,
}
impl Options {
    fn parse(args: &[String]) -> DynResult<Self> {
        let (tier, rest) = args.split_first().ok_or(USAGE)?;
        if !super::config::component(tier) {
            return Err("invalid corpus tier".into());
        }
        let mut values = BTreeMap::new();
        let (pairs, remainder) = rest.as_chunks::<2>();
        for pair in pairs {
            if ![
                "--config",
                "--out-root",
                "--hf-dir",
                "--artifact-manifest",
                "--seed",
                "--max-prompt-chars",
                "--sample-multiplier",
            ]
            .contains(&pair[0].as_str())
                || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("unknown or duplicate corpus option".into());
            }
        }
        if !remainder.is_empty() {
            return Err("corpus option requires a value".into());
        }
        let number = |key: &str| -> DynResult<Option<u64>> {
            values
                .get(key)
                .map(|v| v.parse().map_err(Into::into))
                .transpose()
        };
        let multiplier = usize::try_from(number("--sample-multiplier")?.unwrap_or(20))?;
        if multiplier == 0 || multiplier > 1000 {
            return Err("sample multiplier requires 1..1000".into());
        }
        Ok(Self {
            tier: tier.clone(),
            config: PathBuf::from(
                values
                    .get("--config")
                    .copied()
                    .unwrap_or("skippy/crates/skippy-bench/corpora/bench_corpus_sources.json"),
            ),
            output: PathBuf::from(
                values
                    .get("--out-root")
                    .copied()
                    .unwrap_or("target/bench-corpora"),
            ),
            cache: PathBuf::from(
                values
                    .get("--hf-dir")
                    .copied()
                    .unwrap_or("target/hf-datasets"),
            ),
            manifest: values.get("--artifact-manifest").map(PathBuf::from),
            seed: number("--seed")?,
            max: number("--max-prompt-chars")?
                .map(usize::try_from)
                .transpose()?,
            multiplier,
        })
    }
}
fn bounded(path: &Path, limit: usize) -> DynResult<Vec<u8>> {
    let mut bytes = Vec::new();
    acquisition::regular(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("metadata exceeds byte budget".into());
    }
    Ok(bytes)
}
fn require_fresh(output: &Path) -> DynResult<()> {
    match fs::symlink_metadata(output) {
        Ok(_) => {
            Err("corpus tier output already exists; regenerate with a fresh --out-root".into())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}
fn publish(output: &Path, corpus: &[u8], manifest: &Value) -> DynResult<()> {
    require_fresh(output)?;
    let parent = output.parent().ok_or("corpus output parent missing")?;
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".corpus-stage-")
        .tempdir_in(parent)?;
    let mut data = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(stage.path().join("corpus.jsonl"))?;
    data.write_all(corpus)?;
    data.sync_all()?;
    let mut receipt = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(stage.path().join("manifest.json"))?;
    serde_json::to_writer_pretty(&mut receipt, manifest)?;
    receipt.write_all(b"\n")?;
    receipt.sync_all()?;
    drop(data);
    drop(receipt);
    require_fresh(output)?;
    fs::rename(stage.path(), output)?;
    Ok(())
}
