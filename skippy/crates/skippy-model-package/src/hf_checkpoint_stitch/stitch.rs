use super::contract::{Request, Source, TOKENIZERS, checkpoint_file};
use crate::competitive_acquisition::{
    acquire,
    contract::{check, digest},
    listing::Listing,
};
use crate::snapshot_promotion::local_publisher::regular_input as read;
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Instant};
pub(super) struct Clients {
    pub client: hf_hub::HFClient,
    pub listing: Listing,
}
fn profile(input: &Request) -> Result<Vec<u8>> {
    let mut fd = read::open(&input.tokenizer_profile, 1024 * 1024, false)?;
    let bytes = read::read(&mut fd, 1024 * 1024)?;
    if digest(&bytes) != input.tokenizer_profile_sha256 {
        bail!("tokenizer profile content pin refused");
    }
    Ok(bytes)
}
async fn source(
    clients: &Clients,
    input: &Source,
    root: &Path,
    maximum: u64,
    deadline: Instant,
    checkpoint: bool,
) -> Result<Value> {
    let inventory = clients
        .listing
        .roster(&input.repo, &input.revision, deadline)
        .await?;
    if checkpoint
        && inventory
            .keys()
            .any(|n| n.contains('/') && (n.ends_with(".json") || n.ends_with(".safetensors")))
    {
        bail!("nested checkpoint layout unsupported by native immediate-roster conversion");
    }
    let selected: BTreeMap<_, _> = inventory
        .into_iter()
        .filter(|(n, _)| {
            if checkpoint {
                checkpoint_file(n)
            } else {
                TOKENIZERS.contains(&n.as_str())
            }
        })
        .collect();
    if selected.keys().ne(input.files.keys()) {
        bail!("immutable selected source roster mismatch");
    }
    let total = selected.values().try_fold(0_u64, |sum, n| {
        sum.checked_add(*n)
            .filter(|v| *v <= maximum)
            .ok_or_else(|| anyhow::anyhow!("source byte budget refused"))
    })?;
    let (owner, name) = input
        .repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("source repo"))?;
    let repo = clients.client.model(owner, name);
    std::fs::create_dir(root)?;
    for (name, size) in &selected {
        acquire::one(
            &repo,
            &input.revision,
            name,
            &root.join(name),
            Some(&input.files[name]),
            *size,
            deadline,
        )
        .await?;
    }
    acquire::recheck(root, &input.files, deadline)?;
    Ok(
        json!({"repo":input.repo,"revision":input.revision,"files":input.files,"bytes":total,"selected_roster_complete":true}),
    )
}
fn copy(source: &Path, destination: &Path, pin: &str, max: u64, deadline: Instant) -> Result<()> {
    use sha2::Digest as _;
    use std::io::{Read as _, Write as _};
    let mut fd = read::open(source, max, false)?;
    let mut staged = tempfile::NamedTempFile::new_in(destination.parent().unwrap())?;
    let mut h = sha2::Sha256::new();
    let mut size = 0_u64;
    let mut buffer = [0; 65536];
    loop {
        check(deadline)?;
        let n = fd.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        size = size
            .checked_add(n as u64)
            .filter(|n| *n <= max)
            .ok_or_else(|| anyhow::anyhow!("stitch byte budget"))?;
        h.update(&buffer[..n]);
        staged.write_all(&buffer[..n])?;
    }
    let actual: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if actual != pin {
        bail!("stitch source content drift");
    }
    check(deadline)?;
    staged.as_file().sync_all()?;
    staged
        .persist_noclobber(destination)
        .map_err(|_| anyhow::anyhow!("fresh stitch leaf refused"))?;
    Ok(())
}
fn admit_profile(bytes: &[u8], pins: &BTreeMap<String, String>) -> Result<()> {
    let value = crate::competitive_acquisition::json::unique(bytes)?;
    let keys = [
        "schema_version",
        "config_sha256",
        "tokenizer_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
        "pre",
    ];
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("profile object"))?;
    if object.len() != keys.len()
        || keys.iter().any(|k| !object.contains_key(*k))
        || value["schema_version"] != 1
        || !["llama-bpe", "llama3", "qwen2", "dbrx"].contains(&value["pre"].as_str().unwrap_or(""))
    {
        bail!("explicit native tokenizer profile refused");
    }
    for (key, name) in [
        ("config_sha256", "config.json"),
        ("tokenizer_sha256", "tokenizer.json"),
        ("tokenizer_config_sha256", "tokenizer_config.json"),
        ("chat_template_sha256", "chat_template.jinja"),
    ] {
        let expected = pins.get(name).map_or(Value::Null, |s| json!(s));
        if value[key] != expected {
            bail!("stitched tokenizer profile byte binding refused");
        }
    }
    Ok(())
}
pub(super) async fn execute(
    input: &Request,
    deadline: Instant,
    evidence: &mut Value,
    supplied: Option<Clients>,
) -> Result<()> {
    input.validate()?;
    check(deadline)?;
    let profile_bytes = profile(input)?;
    let mut expected = input.checkpoint.files.clone();
    for (n, h) in &input.tokenizer_source.files {
        expected.entry(n.clone()).or_insert(h.clone());
    }
    admit_profile(&profile_bytes, &expected)?;
    let token = match &input.credential_file {
        Some(p) => {
            let mut fd = read::open(p, 4096, true)?;
            let b = read::read(&mut fd, 4096)?;
            let s = std::str::from_utf8(&b)?.trim().to_string();
            if s.is_empty() || s.chars().any(char::is_control) {
                bail!("credential grammar");
            }
            s
        }
        None => String::new(),
    };
    std::fs::create_dir(&input.output_directory)?;
    evidence["output_owned"] = json!(true);
    let clients = match supplied {
        Some(c) => c,
        None => Clients {
            listing: Listing::new(token.clone())?,
            client: acquire::client(token, &input.output_directory.join("private-cache"))?,
        },
    };
    let checkpoint = input.output_directory.join("checkpoint-source");
    let tokenizer = input.output_directory.join("tokenizer-source");
    evidence["checkpoint_source"] = source(
        &clients,
        &input.checkpoint,
        &checkpoint,
        input.maximum_bytes,
        deadline,
        true,
    )
    .await?;
    let remaining_bytes = input
        .maximum_bytes
        .checked_sub(
            evidence["checkpoint_source"]["bytes"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("checkpoint byte observation absent"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("whole acquisition byte budget exhausted"))?;
    evidence["tokenizer_source"] = source(
        &clients,
        &input.tokenizer_source,
        &tokenizer,
        remaining_bytes,
        deadline,
        false,
    )
    .await?;
    if evidence["checkpoint_source"]["bytes"]
        .as_u64()
        .unwrap_or(u64::MAX)
        .checked_add(
            evidence["tokenizer_source"]["bytes"]
                .as_u64()
                .unwrap_or(u64::MAX),
        )
        .is_none_or(|n| n > input.maximum_bytes)
    {
        bail!("combined source byte budget refused");
    }
    let staged = input.output_directory.join("mtp-src");
    std::fs::create_dir(&staged)?;
    let mut pins = input.checkpoint.files.clone();
    let mut lineage = BTreeMap::new();
    for (name, pin) in &input.checkpoint.files {
        copy(
            &checkpoint.join(name),
            &staged.join(name),
            pin,
            input.maximum_bytes,
            deadline,
        )?;
        lineage.insert(name.clone(), "checkpoint");
    }
    for (name, pin) in &input.tokenizer_source.files {
        if !pins.contains_key(name) {
            copy(
                &tokenizer.join(name),
                &staged.join(name),
                pin,
                input.maximum_bytes,
                deadline,
            )?;
            pins.insert(name.clone(), pin.clone());
            lineage.insert(name.clone(), "tokenizer-source");
        }
    }
    admit_profile(&profile_bytes, &pins)?;
    crate::competitive_acquisition::publish(
        &input.output_directory.join("tokenizer-profile.json"),
        &profile_bytes,
    )?;
    acquire::recheck(&checkpoint, &input.checkpoint.files, deadline)?;
    acquire::recheck(&tokenizer, &input.tokenizer_source.files, deadline)?;
    acquire::recheck(&staged, &pins, deadline)?;
    if profile(input)? != profile_bytes {
        bail!("tokenizer profile changed");
    }
    evidence["checkpoint_directory"] = json!(staged);
    evidence["checkpoint_files"] = json!(
        pins.iter()
            .map(|(n, h)| json!({"path":staged.join(n),"sha256":h}))
            .collect::<Vec<_>>()
    );
    evidence["tokenizer_profile"] = json!({"path":input.output_directory.join("tokenizer-profile.json"),"sha256":input.tokenizer_profile_sha256});
    evidence["lineage"] = json!(lineage);
    evidence["loaded_model_qualification"] = json!(false);
    check(deadline)
}
