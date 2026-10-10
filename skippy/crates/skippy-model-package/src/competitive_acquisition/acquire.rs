use super::contract::{check, path, pin, tree};
use anyhow::{Result, bail};
use futures::StreamExt as _;
use hf_hub::{
    HFClient,
    repository::{HFRepository, RepoType},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Write as _, path::Path, time::Instant};
pub(crate) fn client(token: String, cache: &Path) -> Result<HFClient> {
    skippy_model_hf::configure_hf_tls_provider();
    let http = reqwest::Client::builder()
        .no_proxy()
        .https_only(true)
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|_| anyhow::anyhow!("HF client refused"))?;
    let mut builder = HFClient::builder().endpoint("https://huggingface.co");
    if token.is_empty() {
        if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").as_deref() != Ok("true") {
            bail!("anonymous HF requires isolated implicit-token disable admission");
        }
    } else {
        builder = builder.token(token);
    }
    builder
        .cache_dir(cache)
        .cache_enabled(false)
        .client(http)
        .retry_max_attempts(0)
        .build()
        .map_err(|_| anyhow::anyhow!("native HF client refused"))
}
fn repo_parts(repo: &str) -> Result<(&str, &str)> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("repository grammar"))?;
    if !path(repo) || name.contains('/') {
        bail!("repository grammar");
    }
    Ok((owner, name))
}
pub(crate) async fn one<T: RepoType>(
    repo: &HFRepository<T>,
    revision: &str,
    name: &str,
    output: &Path,
    expected: Option<&str>,
    maximum: u64,
    deadline: Instant,
) -> Result<(String, u64)> {
    check(deadline)?;
    if !pin(revision, 40) || !path(name) {
        bail!("immutable download refused");
    }
    let metadata = repo
        .get_file_metadata()
        .filepath(name)
        .revision(revision)
        .send()
        .await
        .map_err(|_| anyhow::anyhow!("HF metadata unavailable"))?;
    if metadata.commit_hash != revision || metadata.file_size > maximum {
        bail!("resolved source revision/size refused");
    }
    check(deadline)?;
    let (length, mut stream) = repo
        .download_file_stream()
        .filename(name)
        .revision(revision)
        .send()
        .await
        .map_err(|_| anyhow::anyhow!("native HF stream refused"))?;
    if length.is_some_and(|n| n != metadata.file_size) {
        bail!("download size identity refused");
    }
    std::fs::create_dir_all(
        output
            .parent()
            .ok_or_else(|| anyhow::anyhow!("download parent"))?,
    )?;
    let mut staged = tempfile::NamedTempFile::new_in(output.parent().unwrap())?;
    use sha2::Digest as _;
    let mut hash = sha2::Sha256::new();
    let mut size = 0_u64;
    while let Some(chunk) = stream.next().await {
        check(deadline)?;
        let chunk = chunk.map_err(|_| anyhow::anyhow!("native HF stream failed"))?;
        size = size
            .checked_add(chunk.len() as u64)
            .filter(|n| *n <= maximum && *n <= metadata.file_size)
            .ok_or_else(|| anyhow::anyhow!("download byte bound"))?;
        hash.update(&chunk);
        staged.write_all(&chunk)?;
    }
    let observed: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if size != metadata.file_size || expected.is_some_and(|p| p != observed) {
        bail!("download final content pin/length mismatch");
    }
    check(deadline)?;
    staged.as_file().sync_all()?;
    staged
        .persist_noclobber(output)
        .map_err(|_| anyhow::anyhow!("fresh download publication refused"))?;
    Ok((observed, size))
}
pub(super) struct Snapshot<'a> {
    pub repo: &'a str,
    pub revision: &'a str,
    pub output: &'a Path,
    pub complete: bool,
    pub maximum: u64,
}
pub(super) async fn snapshot(
    client: &HFClient,
    listing: &super::listing::Listing,
    input: Snapshot<'_>,
    deadline: Instant,
) -> Result<Value> {
    let Snapshot {
        repo,
        revision,
        output,
        complete,
        maximum,
    } = input;
    if !pin(revision, 40) {
        bail!("snapshot immutable pin refused");
    }
    let (owner, name) = repo_parts(repo)?;
    let repository = client.model(owner, name);
    let roster = listing.roster(repo, revision, deadline).await?;
    let total = roster
        .iter()
        .filter(|(name, _)| selected(name, complete))
        .try_fold(0_u64, |total, (_, size)| {
            total
                .checked_add(*size)
                .filter(|n| *n <= maximum)
                .ok_or_else(|| anyhow::anyhow!("snapshot declared byte bound"))
        })?;
    if roster.is_empty() {
        bail!("empty snapshot refused");
    }
    std::fs::create_dir(output)?;
    let mut observed = BTreeMap::new();
    for (name, size) in &roster {
        if !selected(name, complete) {
            continue;
        }
        let (hash, actual) = one(
            &repository,
            revision,
            name,
            &output.join(name),
            None,
            *size,
            deadline,
        )
        .await?;
        if actual != *size {
            bail!("snapshot file size drift");
        }
        observed.insert(name.clone(), hash);
    }
    let actual = tree(&observed)?;
    Ok(
        json!({"repo":repo,"revision":revision,"tree_sha256":actual,"files":observed,"bytes":total,"complete_roster":complete,"README_basename_excluded":complete,"listed_paths":roster.keys().collect::<Vec<_>>()}),
    )
}
pub(super) async fn artifact(
    client: &HFClient,
    row: &Value,
    dataset: bool,
    root: &Path,
    maximum: u64,
    deadline: Instant,
) -> Result<Value> {
    let repo = row["repo"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("artifact repo"))?;
    let revision = row["revision"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("artifact revision"))?;
    let file = row["filename"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("artifact filename"))?;
    let expected = row["sha256"]
        .as_str()
        .filter(|s| pin(s, 64))
        .ok_or_else(|| anyhow::anyhow!("artifact pin"))?;
    let (owner, name) = repo_parts(repo)?;
    let (hash, size) = if dataset {
        one(
            &client.dataset(owner, name),
            revision,
            file,
            &root.join(file),
            Some(expected),
            maximum,
            deadline,
        )
        .await?
    } else {
        one(
            &client.model(owner, name),
            revision,
            file,
            &root.join(file),
            Some(expected),
            maximum,
            deadline,
        )
        .await?
    };
    Ok(
        json!({"repo":repo,"revision":revision,"filename":file,"sha256":hash,"bytes":size,"dataset":dataset}),
    )
}
pub(crate) fn recheck(
    root: &Path,
    rows: &BTreeMap<String, String>,
    deadline: Instant,
) -> Result<()> {
    for (name, pin) in rows {
        check(deadline)?;
        let mut f = super::read::open(&root.join(name), 1024 * 1024 * 1024 * 1024, false)?;
        use sha2::Digest as _;
        use std::io::Read as _;
        let mut h = sha2::Sha256::new();
        let mut b = [0; 65536];
        loop {
            check(deadline)?;
            let n = f.read(&mut b)?;
            if n == 0 {
                break;
            }
            h.update(&b[..n]);
        }
        let actual: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if actual != *pin {
            bail!("acquired source changed");
        }
    }
    Ok(())
}

fn selected(name: &str, complete: bool) -> bool {
    if complete {
        Path::new(name)
            .file_name()
            .is_some_and(|v| v != "README.md")
    } else {
        [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ]
        .contains(&name)
    }
}
