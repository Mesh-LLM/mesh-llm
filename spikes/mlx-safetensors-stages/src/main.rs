use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use reqwest::StatusCode;
use reqwest::Url;
use reqwest::blocking::{Client, Response};
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, RANGE};
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};

const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Parser)]
#[command(about = "Plan exact SafeTensors byte ranges for one transformer layer stage")]
struct Args {
    /// Hugging Face model repository, for example Qwen/Qwen3-235B-A22B.
    #[arg(long)]
    repo: String,

    /// Repository revision. Use an immutable commit SHA for reproducible plans.
    #[arg(long, default_value = "main")]
    revision: String,

    /// First transformer layer owned by this stage.
    #[arg(long)]
    layer_start: u32,

    /// Exclusive end of the transformer layer range.
    #[arg(long)]
    layer_end: u32,

    /// Include additional exact tensor-name prefixes, such as model.embed_tokens.
    #[arg(long = "include-prefix")]
    include_prefixes: Vec<String>,

    /// Merge ranges separated by at most this many unneeded bytes.
    #[arg(long, default_value_t = 0)]
    coalesce_gap_bytes: u64,

    #[arg(long, default_value = "https://huggingface.co")]
    endpoint: String,

    #[arg(long)]
    json: bool,

    /// Materialize the selected tensors as output/model.safetensors.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct SafetensorIndex {
    #[serde(default)]
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct IndexMetadata {
    total_size: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct RemoteHeader {
    header_len: u64,
    tensors: BTreeMap<String, TensorHeader>,
}

#[derive(Debug)]
struct CheckpointLayout {
    index: SafetensorIndex,
    index_bytes: u64,
    headers: BTreeMap<String, RemoteHeader>,
}

#[derive(Debug)]
struct PreparedStage {
    plan: StagePlan,
    tensors: Vec<SelectedTensor>,
}

#[derive(Debug)]
struct SelectedTensor {
    name: String,
    source_file: String,
    source_range: ByteRange,
    header: TensorHeader,
}

#[derive(Debug, Serialize)]
struct StagePlan {
    repo: String,
    revision: String,
    layer_start: u32,
    layer_end: u32,
    include_prefixes: Vec<String>,
    total_model_tensor_bytes: Option<u64>,
    index_bytes: u64,
    selected_tensor_count: usize,
    selected_tensor_bytes: u64,
    largest_selected_tensor_bytes: u64,
    source_shard_count: usize,
    source_shard_bytes: u64,
    range_request_count: usize,
    range_payload_bytes: u64,
    header_probe_bytes: u64,
    planned_download_bytes: u64,
    source_shard_bytes_avoided: u64,
    full_model_tensor_bytes_avoided: Option<u64>,
    shards: Vec<ShardPlan>,
}

#[derive(Debug, Serialize)]
struct ShardPlan {
    file: String,
    file_bytes: u64,
    header_probe_bytes: u64,
    selected_tensor_count: usize,
    selected_tensor_bytes: u64,
    largest_selected_tensor_bytes: u64,
    range_request_count: usize,
    range_payload_bytes: u64,
    ranges: Vec<ByteRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ByteRange {
    start: u64,
    end_exclusive: u64,
}

impl ByteRange {
    fn len(&self) -> u64 {
        self.end_exclusive - self.start
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.layer_start < args.layer_end,
        "--layer-start must be less than --layer-end"
    );
    let client = build_client()?;
    let prepared = prepare_stage(&client, &args)?;
    let plan = &prepared.plan;
    if let Some(output) = &args.output {
        materialize_stage(&client, &args, &prepared, output)?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        print_human_plan(plan);
    }
    Ok(())
}

fn build_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .redirect(Policy::limited(10))
        .user_agent("mesh-llm-mlx-safetensors-stage-plan/0")
        .build()
        .context("build HTTP client")
}

fn prepare_stage(client: &Client, args: &Args) -> Result<PreparedStage> {
    let mut layout = load_checkpoint_layout(client, args)?;
    let selected = select_tensors(&layout.index.weight_map, args);
    ensure!(
        !selected.is_empty(),
        "no tensors matched layers {}..{} or the requested prefixes",
        args.layer_start,
        args.layer_end
    );

    let mut by_shard: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for name in &selected {
        let shard = layout
            .index
            .weight_map
            .get(*name)
            .with_context(|| format!("selected tensor {name} is absent from the weight map"))?;
        by_shard.entry(shard).or_default().insert(*name);
    }

    let mut shards = Vec::with_capacity(by_shard.len());
    let mut tensors = Vec::with_capacity(selected.len());
    for (file, names) in by_shard {
        if !layout.headers.contains_key(file) {
            let url = resolve_url(&args.endpoint, &args.repo, &args.revision, file)?;
            let header =
                fetch_safetensor_header(client, &url).with_context(|| format!("inspect {file}"))?;
            layout.headers.insert(file.to_string(), header);
        }
        let header = layout
            .headers
            .get(file)
            .with_context(|| format!("missing inspected header for {file}"))?;
        shards.push(plan_shard(file, header, &names, args.coalesce_gap_bytes)?);
        tensors.extend(selected_tensors(file, header, &names)?);
    }
    let plan = summarize_plan(args, &layout.index, layout.index_bytes, shards);
    Ok(PreparedStage { plan, tensors })
}

fn load_checkpoint_layout(client: &Client, args: &Args) -> Result<CheckpointLayout> {
    let index_url = resolve_url(
        &args.endpoint,
        &args.repo,
        &args.revision,
        "model.safetensors.index.json",
    )?;
    if let Some(index_bytes) = fetch_optional_small_file(client, index_url, MAX_INDEX_BYTES)? {
        let index: SafetensorIndex =
            serde_json::from_slice(&index_bytes).context("parse model.safetensors.index.json")?;
        return Ok(CheckpointLayout {
            index,
            index_bytes: index_bytes.len() as u64,
            headers: BTreeMap::new(),
        });
    }

    let file = "model.safetensors";
    let url = resolve_url(&args.endpoint, &args.repo, &args.revision, file)?;
    let header = fetch_safetensor_header(client, &url)
        .context("inspect unsharded model.safetensors checkpoint")?;
    let total_size = header
        .tensors
        .values()
        .map(|tensor| tensor.data_offsets[1])
        .max();
    let weight_map = header
        .tensors
        .keys()
        .map(|name| (name.clone(), file.to_string()))
        .collect();
    Ok(CheckpointLayout {
        index: SafetensorIndex {
            metadata: IndexMetadata { total_size },
            weight_map,
        },
        index_bytes: 0,
        headers: BTreeMap::from([(file.to_string(), header)]),
    })
}

fn summarize_plan(
    args: &Args,
    index: &SafetensorIndex,
    index_bytes: u64,
    shards: Vec<ShardPlan>,
) -> StagePlan {
    let selected_tensor_count = shards.iter().map(|shard| shard.selected_tensor_count).sum();
    let selected_tensor_bytes = shards.iter().map(|shard| shard.selected_tensor_bytes).sum();
    let largest_selected_tensor_bytes = shards
        .iter()
        .map(|shard| shard.largest_selected_tensor_bytes)
        .max()
        .unwrap_or(0);
    let source_shard_bytes = shards.iter().map(|shard| shard.file_bytes).sum();
    let range_request_count = shards.iter().map(|shard| shard.range_request_count).sum();
    let range_payload_bytes = shards.iter().map(|shard| shard.range_payload_bytes).sum();
    let header_probe_bytes = shards.iter().map(|shard| shard.header_probe_bytes).sum();
    let planned_download_bytes = index_bytes + header_probe_bytes + range_payload_bytes;
    StagePlan {
        repo: args.repo.clone(),
        revision: args.revision.clone(),
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        include_prefixes: args.include_prefixes.clone(),
        total_model_tensor_bytes: index.metadata.total_size,
        index_bytes,
        selected_tensor_count,
        selected_tensor_bytes,
        largest_selected_tensor_bytes,
        source_shard_count: shards.len(),
        source_shard_bytes,
        range_request_count,
        range_payload_bytes,
        header_probe_bytes,
        planned_download_bytes,
        source_shard_bytes_avoided: source_shard_bytes
            .saturating_sub(header_probe_bytes + range_payload_bytes),
        full_model_tensor_bytes_avoided: index
            .metadata
            .total_size
            .map(|total| total.saturating_sub(selected_tensor_bytes)),
        shards,
    }
}

fn select_tensors<'a>(weight_map: &'a BTreeMap<String, String>, args: &Args) -> BTreeSet<&'a str> {
    weight_map
        .keys()
        .filter(|name| {
            layer_index(name)
                .is_some_and(|layer| layer >= args.layer_start && layer < args.layer_end)
                || args
                    .include_prefixes
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
        })
        .map(String::as_str)
        .collect()
}

fn layer_index(name: &str) -> Option<u32> {
    let parts = name.split('.').collect::<Vec<_>>();
    parts.windows(2).find_map(|pair| {
        matches!(pair[0], "layers" | "layer" | "h")
            .then(|| pair[1].parse().ok())
            .flatten()
    })
}

fn fetch_safetensor_header(client: &Client, url: &Url) -> Result<RemoteHeader> {
    let len_bytes = fetch_range(client, url.clone(), 0..8)?;
    let header_len = u64::from_le_bytes(
        len_bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("invalid 8-byte SafeTensors header length"))?,
    );
    ensure!(
        header_len <= MAX_HEADER_BYTES,
        "SafeTensors header is unexpectedly large: {header_len} bytes"
    );
    let header_end = 8_u64
        .checked_add(header_len)
        .context("SafeTensors header range overflow")?;
    let header_bytes = fetch_range(client, url.clone(), 8..header_end)?;
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header_bytes).context("parse SafeTensors header")?;
    let tensors = raw
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, value)| {
            serde_json::from_value(value)
                .map(|tensor| (name, tensor))
                .context("parse SafeTensors tensor header")
        })
        .collect::<Result<_>>()?;
    Ok(RemoteHeader {
        header_len,
        tensors,
    })
}

fn plan_shard(
    file: &str,
    header: &RemoteHeader,
    selected: &BTreeSet<&str>,
    coalesce_gap_bytes: u64,
) -> Result<ShardPlan> {
    let data_start = 8_u64
        .checked_add(header.header_len)
        .context("SafeTensors data offset overflow")?;
    let file_bytes = header
        .tensors
        .values()
        .map(|tensor| tensor.data_offsets[1])
        .max()
        .unwrap_or(0)
        .checked_add(data_start)
        .context("SafeTensors file length overflow")?;
    let mut ranges = Vec::with_capacity(selected.len());
    let mut selected_tensor_bytes = 0_u64;
    let mut largest_selected_tensor_bytes = 0_u64;
    for name in selected {
        let tensor = header
            .tensors
            .get(*name)
            .with_context(|| format!("weight-map tensor {name} is absent from {file}"))?;
        validate_tensor(name, tensor)?;
        let start = data_start
            .checked_add(tensor.data_offsets[0])
            .with_context(|| format!("absolute offset overflow for {name}"))?;
        let end_exclusive = data_start
            .checked_add(tensor.data_offsets[1])
            .with_context(|| format!("absolute offset overflow for {name}"))?;
        let tensor_bytes = end_exclusive - start;
        selected_tensor_bytes += tensor_bytes;
        largest_selected_tensor_bytes = largest_selected_tensor_bytes.max(tensor_bytes);
        ranges.push(ByteRange {
            start,
            end_exclusive,
        });
    }
    let ranges = coalesce_ranges(ranges, coalesce_gap_bytes);
    let range_payload_bytes = ranges.iter().map(ByteRange::len).sum();
    Ok(ShardPlan {
        file: file.to_string(),
        file_bytes,
        header_probe_bytes: data_start,
        selected_tensor_count: selected.len(),
        selected_tensor_bytes,
        largest_selected_tensor_bytes,
        range_request_count: ranges.len(),
        range_payload_bytes,
        ranges,
    })
}

fn selected_tensors(
    file: &str,
    header: &RemoteHeader,
    selected: &BTreeSet<&str>,
) -> Result<Vec<SelectedTensor>> {
    let data_start = 8_u64
        .checked_add(header.header_len)
        .context("SafeTensors data offset overflow")?;
    selected
        .iter()
        .map(|name| {
            let tensor = header
                .tensors
                .get(*name)
                .with_context(|| format!("weight-map tensor {name} is absent from {file}"))?;
            validate_tensor(name, tensor)?;
            let start = data_start
                .checked_add(tensor.data_offsets[0])
                .with_context(|| format!("absolute offset overflow for {name}"))?;
            let end_exclusive = data_start
                .checked_add(tensor.data_offsets[1])
                .with_context(|| format!("absolute offset overflow for {name}"))?;
            Ok(SelectedTensor {
                name: (*name).to_string(),
                source_file: file.to_string(),
                source_range: ByteRange {
                    start,
                    end_exclusive,
                },
                header: tensor.clone(),
            })
        })
        .collect()
}

fn validate_tensor(name: &str, tensor: &TensorHeader) -> Result<()> {
    ensure!(
        tensor.data_offsets[0] <= tensor.data_offsets[1],
        "invalid data offsets for tensor {name}"
    );
    ensure!(!tensor.dtype.is_empty(), "tensor {name} has no dtype");
    let _rank = tensor.shape.len();
    Ok(())
}

fn coalesce_ranges(mut ranges: Vec<ByteRange>, max_gap: u64) -> Vec<ByteRange> {
    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<ByteRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && range.start.saturating_sub(previous.end_exclusive) <= max_gap
        {
            previous.end_exclusive = previous.end_exclusive.max(range.end_exclusive);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn fetch_small_file(client: &Client, url: Url, max_bytes: u64) -> Result<Vec<u8>> {
    let response = authorized(client.get(url))
        .send()
        .context("send HTTP request")?;
    let response = response
        .error_for_status()
        .context("download metadata file")?;
    if let Some(length) = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        ensure!(
            length <= max_bytes,
            "metadata file is too large: {length} bytes"
        );
    }
    let bytes = response.bytes().context("read metadata response")?;
    ensure!(
        bytes.len() as u64 <= max_bytes,
        "metadata response exceeded {max_bytes} bytes"
    );
    Ok(bytes.to_vec())
}

fn fetch_optional_small_file(client: &Client, url: Url, max_bytes: u64) -> Result<Option<Vec<u8>>> {
    let response = authorized(client.get(url))
        .send()
        .context("send HTTP request")?;
    if response.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    read_small_response(response, max_bytes).map(Some)
}

fn read_small_response(response: Response, max_bytes: u64) -> Result<Vec<u8>> {
    let response = response
        .error_for_status()
        .context("download metadata file")?;
    if let Some(length) = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        ensure!(
            length <= max_bytes,
            "metadata file is too large: {length} bytes"
        );
    }
    let bytes = response.bytes().context("read metadata response")?;
    ensure!(
        bytes.len() as u64 <= max_bytes,
        "metadata response exceeded {max_bytes} bytes"
    );
    Ok(bytes.to_vec())
}

fn fetch_range(client: &Client, url: Url, range: Range<u64>) -> Result<Vec<u8>> {
    ensure!(range.start < range.end, "HTTP byte range must not be empty");
    let header = format!("bytes={}-{}", range.start, range.end - 1);
    let response = authorized(client.get(url).header(RANGE, header.clone()))
        .send()
        .with_context(|| format!("request HTTP range {header}"))?;
    ensure_partial_content(&response, &header)?;
    let bytes = response.bytes().context("read HTTP range response")?;
    ensure!(
        bytes.len() as u64 == range.end - range.start,
        "HTTP range {header} returned {} bytes, expected {}",
        bytes.len(),
        range.end - range.start
    );
    Ok(bytes.to_vec())
}

fn ensure_partial_content(response: &Response, range: &str) -> Result<()> {
    ensure!(
        response.status() == StatusCode::PARTIAL_CONTENT,
        "server did not honor {range}; status was {} (refusing a possible full-shard download)",
        response.status()
    );
    Ok(())
}

fn materialize_stage(
    client: &Client,
    args: &Args,
    prepared: &PreparedStage,
    output: &Path,
) -> Result<()> {
    fs::create_dir_all(output)
        .with_context(|| format!("create output directory {}", output.display()))?;
    let mut tensors = prepared.tensors.iter().collect::<Vec<_>>();
    tensors.sort_by(|left, right| {
        (&left.source_file, left.source_range.start)
            .cmp(&(&right.source_file, right.source_range.start))
    });

    let mut output_offset = 0_u64;
    let mut output_header = BTreeMap::new();
    for tensor in &tensors {
        let len = tensor.source_range.len();
        let end = output_offset
            .checked_add(len)
            .context("partial SafeTensors output offset overflow")?;
        let mut header = tensor.header.clone();
        header.data_offsets = [output_offset, end];
        output_header.insert(tensor.name.clone(), header);
        output_offset = end;
    }
    let mut header_bytes = serde_json::to_vec(&output_header)?;
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let header_len = u64::try_from(header_bytes.len()).context("output header is too large")?;

    let destination = output.join("model.safetensors");
    let partial = output.join("model.safetensors.partial");
    let mut writer = BufWriter::new(
        File::create(&partial).with_context(|| format!("create {}", partial.display()))?,
    );
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_bytes)?;

    let spans = materialization_spans(&tensors);
    let mut payload_bytes = 0_u64;
    for span in &spans {
        let url = resolve_url(
            &args.endpoint,
            &args.repo,
            &args.revision,
            &span.source_file,
        )?;
        payload_bytes += fetch_range_into(client, url, span.range.clone(), &mut writer)?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(&partial, &destination).with_context(|| {
        format!(
            "move completed partial SafeTensors file to {}",
            destination.display()
        )
    })?;

    let config_url = resolve_url(&args.endpoint, &args.repo, &args.revision, "config.json")?;
    let config = fetch_small_file(client, config_url, MAX_INDEX_BYTES)?;
    fs::write(output.join("config.json"), config)?;
    fs::write(
        output.join("stage-plan.json"),
        serde_json::to_vec_pretty(&prepared.plan)?,
    )?;
    ensure!(
        payload_bytes == prepared.plan.selected_tensor_bytes,
        "materialized {payload_bytes} payload bytes, planned {}",
        prepared.plan.selected_tensor_bytes
    );
    eprintln!(
        "materialized {} tensors ({} payload) in {} HTTP spans to {}",
        tensors.len(),
        human_bytes(payload_bytes),
        spans.len(),
        destination.display()
    );
    Ok(())
}

#[derive(Debug)]
struct MaterializationSpan {
    source_file: String,
    range: Range<u64>,
}

fn materialization_spans(tensors: &[&SelectedTensor]) -> Vec<MaterializationSpan> {
    let mut spans: Vec<MaterializationSpan> = Vec::new();
    for tensor in tensors {
        if let Some(previous) = spans.last_mut()
            && previous.source_file == tensor.source_file
            && previous.range.end == tensor.source_range.start
        {
            previous.range.end = tensor.source_range.end_exclusive;
        } else {
            spans.push(MaterializationSpan {
                source_file: tensor.source_file.clone(),
                range: tensor.source_range.start..tensor.source_range.end_exclusive,
            });
        }
    }
    spans
}

fn fetch_range_into(
    client: &Client,
    url: Url,
    range: Range<u64>,
    writer: &mut impl Write,
) -> Result<u64> {
    ensure!(range.start < range.end, "HTTP byte range must not be empty");
    let expected = range.end - range.start;
    let header = format!("bytes={}-{}", range.start, range.end - 1);
    let mut response = authorized(client.get(url).header(RANGE, header.clone()))
        .send()
        .with_context(|| format!("request HTTP range {header}"))?;
    ensure_partial_content(&response, &header)?;
    let written = std::io::copy(&mut response, writer).context("stream HTTP tensor range")?;
    ensure!(
        written == expected,
        "HTTP range {header} returned {written} bytes, expected {expected}"
    );
    Ok(written)
}

fn authorized(builder: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
    if let Some(token) = hf_token() {
        builder.header(AUTHORIZATION, format!("Bearer {token}"))
    } else {
        builder
    }
}

fn hf_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|name| env::var(name).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn resolve_url(endpoint: &str, repo: &str, revision: &str, file: &str) -> Result<Url> {
    let mut url = Url::parse(endpoint).context("parse Hugging Face endpoint")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("Hugging Face endpoint cannot be a base URL"))?;
        segments.pop_if_empty();
        segments.extend(repo.split('/'));
        segments.push("resolve");
        segments.push(revision);
        segments.extend(file.split('/'));
    }
    Ok(url)
}

fn print_human_plan(plan: &StagePlan) {
    println!(
        "{}@{} layers {}..{}",
        plan.repo, plan.revision, plan.layer_start, plan.layer_end
    );
    if let Some(total) = plan.total_model_tensor_bytes {
        println!("full checkpoint tensors: {}", human_bytes(total));
    }
    println!(
        "selected: {} tensors, {}",
        plan.selected_tensor_count,
        human_bytes(plan.selected_tensor_bytes)
    );
    println!(
        "largest selected tensor: {}",
        human_bytes(plan.largest_selected_tensor_bytes)
    );
    println!(
        "whole source shards: {} files, {}",
        plan.source_shard_count,
        human_bytes(plan.source_shard_bytes)
    );
    println!(
        "exact ranged payload: {} requests, {}",
        plan.range_request_count,
        human_bytes(plan.range_payload_bytes)
    );
    println!(
        "planned download including index/headers: {}",
        human_bytes(plan.planned_download_bytes)
    );
    println!(
        "avoided versus whole shards: {}",
        human_bytes(plan.source_shard_bytes_avoided)
    );
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_common_transformer_layer_paths() {
        assert_eq!(layer_index("model.layers.42.mlp.up_proj.weight"), Some(42));
        assert_eq!(layer_index("transformer.h.7.attn.weight"), Some(7));
        assert_eq!(layer_index("model.embed_tokens.weight"), None);
    }

    #[test]
    fn coalesces_adjacent_and_small_gap_ranges() {
        let ranges = vec![
            ByteRange {
                start: 20,
                end_exclusive: 30,
            },
            ByteRange {
                start: 0,
                end_exclusive: 10,
            },
            ByteRange {
                start: 10,
                end_exclusive: 20,
            },
            ByteRange {
                start: 32,
                end_exclusive: 40,
            },
        ];
        assert_eq!(
            coalesce_ranges(ranges, 2),
            vec![ByteRange {
                start: 0,
                end_exclusive: 40
            }]
        );
    }

    #[test]
    fn plans_only_selected_tensor_payloads() {
        let header = RemoteHeader {
            header_len: 100,
            tensors: BTreeMap::from([
                (
                    "model.layers.1.weight".into(),
                    tensor("BF16", &[2, 2], [0, 8]),
                ),
                (
                    "model.layers.2.weight".into(),
                    tensor("BF16", &[2, 2], [8, 16]),
                ),
                (
                    "model.layers.3.weight".into(),
                    tensor("BF16", &[2, 2], [16, 24]),
                ),
            ]),
        };
        let selected = BTreeSet::from(["model.layers.2.weight"]);
        let plan = plan_shard("model.safetensors", &header, &selected, 0).unwrap();
        assert_eq!(plan.file_bytes, 132);
        assert_eq!(plan.selected_tensor_bytes, 8);
        assert_eq!(plan.largest_selected_tensor_bytes, 8);
        assert_eq!(
            plan.ranges,
            vec![ByteRange {
                start: 116,
                end_exclusive: 124
            }]
        );
    }

    #[test]
    fn materialization_spans_merge_only_contiguous_ranges_in_one_source() {
        let tensors = [
            selected_tensor("a", "one.safetensors", 10, 20),
            selected_tensor("b", "one.safetensors", 20, 30),
            selected_tensor("c", "one.safetensors", 40, 50),
            selected_tensor("d", "two.safetensors", 50, 60),
        ];
        let references = tensors.iter().collect::<Vec<_>>();
        let spans = materialization_spans(&references);

        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].source_file, "one.safetensors");
        assert_eq!(spans[0].range, 10..30);
        assert_eq!(spans[1].range, 40..50);
        assert_eq!(spans[2].source_file, "two.safetensors");
        assert_eq!(spans[2].range, 50..60);
    }

    fn selected_tensor(name: &str, file: &str, start: u64, end: u64) -> SelectedTensor {
        SelectedTensor {
            name: name.to_string(),
            source_file: file.to_string(),
            source_range: ByteRange {
                start,
                end_exclusive: end,
            },
            header: tensor("BF16", &[1], [0, end - start]),
        }
    }

    fn tensor(dtype: &str, shape: &[u64], data_offsets: [u64; 2]) -> TensorHeader {
        TensorHeader {
            dtype: dtype.into(),
            shape: shape.to_vec(),
            data_offsets,
        }
    }
}
