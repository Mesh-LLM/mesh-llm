use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, anyhow, ensure};
use model_artifact::safetensors::{
    IndexMetadata, SafetensorsIndex, TensorHeader, parse_header, parse_index,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    http::RemoteSource,
    types::{
        ByteRange, PreparedStage, SafetensorsCheckpointDescriptor, SafetensorsShardPlan,
        SafetensorsSourceShard, SafetensorsStagePlan, SafetensorsStageRequest, SelectedTensor,
    },
};

pub(crate) const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StageModelFamily {
    Llama,
    NemotronH,
}

#[derive(Debug, Deserialize)]
struct RawStageModelConfig {
    model_type: String,
    num_hidden_layers: u32,
    hidden_size: u32,
    #[serde(default)]
    max_position_embeddings: u32,
    #[serde(default)]
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: u32,
    #[serde(default)]
    head_dim: u32,
}

#[derive(Clone, Copy, Debug)]
struct StageModelLayout {
    family: StageModelFamily,
    num_hidden_layers: u32,
}

pub(crate) fn describe(
    remote: &RemoteSource,
    repo: &str,
    revision: &str,
) -> Result<SafetensorsCheckpointDescriptor> {
    let probe = SafetensorsStageRequest {
        repo: repo.to_string(),
        revision: revision.to_string(),
        layer_start: 0,
        layer_end: 1,
        include_prefixes: Vec::new(),
    }
    .normalized()?;
    let config_url = remote.url(repo, revision, "config.json")?;
    let config_file = remote
        .small_file(config_url, MAX_INDEX_BYTES)
        .context("download SafeTensors model config")?;
    let config = parse_raw_stage_model_config(&config_file.bytes)?;
    ensure!(
        config.model_type == "llama",
        "automatic distributed MLX planning currently supports model_type=llama, got {:?}",
        config.model_type
    );
    validate_planning_config(&config)?;

    let full = prepare(
        remote,
        &SafetensorsStageRequest {
            layer_end: config.num_hidden_layers,
            ..probe
        },
    )?;
    let dense_tensor_bytes = full
        .plan
        .total_model_tensor_bytes
        .context("SafeTensors checkpoint does not declare total tensor bytes")?;
    let stage_layout = StageModelLayout {
        family: StageModelFamily::Llama,
        num_hidden_layers: config.num_hidden_layers,
    };
    let estimated_affine4_weight_bytes =
        checked_sum(full.tensors.iter().map(estimated_affine4_tensor_bytes))?;
    let estimated_affine4_layer_bytes = estimated_affine4_layer_bytes(stage_layout, &full.tensors)?;
    let kv_heads = if config.num_key_value_heads > 0 {
        config.num_key_value_heads
    } else {
        config.num_attention_heads
    };
    let head_dim = if config.head_dim > 0 {
        config.head_dim
    } else {
        config.hidden_size / config.num_attention_heads
    };
    let kv_bytes_per_token_bf16 = u64::from(config.num_hidden_layers)
        .checked_mul(u64::from(kv_heads))
        .and_then(|bytes| bytes.checked_mul(u64::from(head_dim)))
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| bytes.checked_mul(2))
        .context("SafeTensors KV byte estimate overflow")?;

    Ok(SafetensorsCheckpointDescriptor {
        checkpoint_sha256: full.checkpoint_sha256,
        repo: repo.to_string(),
        revision: revision.to_string(),
        model_type: config.model_type,
        layer_count: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        native_context_length: if config.max_position_embeddings == 0 {
            2_048
        } else {
            config.max_position_embeddings
        },
        dense_tensor_bytes,
        estimated_affine4_weight_bytes,
        estimated_affine4_layer_bytes,
        kv_bytes_per_token_bf16,
    })
}

fn estimated_affine4_layer_bytes(
    layout: StageModelLayout,
    tensors: &[SelectedTensor],
) -> Result<Vec<u64>> {
    let mut layer_bytes = vec![0_u64; layout.num_hidden_layers as usize];
    let final_layer = layer_bytes
        .len()
        .checked_sub(1)
        .context("SafeTensors model has no layers")?;
    for tensor in tensors {
        let bytes = estimated_affine4_tensor_bytes(tensor);
        if let Some(layer) = layout.layer_index(&tensor.name) {
            add_estimated_bytes(&mut layer_bytes[layer as usize], bytes)?;
        } else if tensor.name.starts_with(layout.embedding_prefix()) {
            add_estimated_bytes(&mut layer_bytes[0], bytes)?;
            if layout.family == StageModelFamily::Llama && final_layer != 0 {
                // The current Llama final-stage selection reloads embeddings
                // for tied readout compatibility, even when lm_head is also
                // present. Charge the planner for what the stage really loads.
                add_estimated_bytes(&mut layer_bytes[final_layer], bytes)?;
            }
        } else {
            // Final norm/readout and any unclassified boundary tensors are
            // selected only by the final stage.
            add_estimated_bytes(&mut layer_bytes[final_layer], bytes)?;
        }
    }
    Ok(layer_bytes)
}

fn estimated_affine4_tensor_bytes(tensor: &SelectedTensor) -> u64 {
    let shape = &tensor.header.shape;
    let is_dense_float_weight = tensor.name.ends_with(".weight")
        && shape.len() == 2
        && shape[1].is_multiple_of(64)
        && matches!(tensor.header.dtype.as_str(), "F16" | "BF16" | "F32");
    if !is_dense_float_weight {
        return tensor.source_range.len();
    }
    let elements = shape.iter().copied().fold(1_u64, u64::saturating_mul);
    // Packed 4-bit values plus a deliberately conservative allowance for
    // group-64 scales, biases, alignment, and SafeTensors metadata.
    elements.saturating_mul(3).div_ceil(4)
}

fn add_estimated_bytes(total: &mut u64, bytes: u64) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .context("SafeTensors affine-4 layer byte estimate overflow")?;
    Ok(())
}

fn validate_planning_config(config: &RawStageModelConfig) -> Result<()> {
    ensure!(
        config.hidden_size > 0,
        "SafeTensors hidden_size must be positive"
    );
    ensure!(
        config.num_hidden_layers > 0,
        "SafeTensors num_hidden_layers must be positive"
    );
    ensure!(
        config.num_attention_heads > 0,
        "SafeTensors num_attention_heads must be positive"
    );
    ensure!(
        config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
            || config.head_dim > 0,
        "SafeTensors hidden_size is not divisible by num_attention_heads and head_dim is absent"
    );
    Ok(())
}

impl StageModelLayout {
    fn layer_index(self, name: &str) -> Option<u32> {
        self.layer_prefixes()
            .iter()
            .find_map(|prefix| name.strip_prefix(prefix)?.split_once('.')?.0.parse().ok())
    }

    fn layer_prefixes(self) -> &'static [&'static str] {
        match self.family {
            StageModelFamily::Llama => &["model.layers."],
            StageModelFamily::NemotronH => &["backbone.layers.", "model.backbone.layers."],
        }
    }

    fn embedding_prefix(self) -> &'static str {
        match self.family {
            StageModelFamily::Llama => "model.embed_tokens.",
            StageModelFamily::NemotronH => "backbone.embeddings.",
        }
    }

    fn final_norm_prefix(self) -> &'static str {
        match self.family {
            StageModelFamily::Llama => "model.norm.",
            StageModelFamily::NemotronH => "backbone.norm_f.",
        }
    }

    fn readout_prefix(self) -> &'static str {
        "lm_head."
    }
}

#[derive(Clone, Debug)]
struct RemoteHeader {
    header_len: u64,
    header_sha256: String,
    file_bytes: u64,
    etag: String,
    tensors: BTreeMap<String, TensorHeader>,
}

struct CheckpointLayout {
    index: SafetensorsIndex,
    layout_sha256: String,
    index_bytes: u64,
    index_sha256: Option<String>,
    index_etag: Option<String>,
    headers: BTreeMap<String, RemoteHeader>,
}

pub(crate) fn prepare(
    remote: &RemoteSource,
    request: &SafetensorsStageRequest,
) -> Result<PreparedStage> {
    let config_url = remote.url(&request.repo, &request.revision, "config.json")?;
    let config = remote
        .small_file(config_url, MAX_INDEX_BYTES)
        .context("download SafeTensors model config")?;
    let model_config = parse_stage_model_layout(&config.bytes)?;
    validate_layer_range(request, &model_config)?;
    let config_sha256 = sha256_hex(&config.bytes);
    let mut selection_request = request.clone();
    add_required_prefixes(&mut selection_request, &model_config);

    let mut layout = load_checkpoint_layout(remote, request)?;
    let checkpoint_sha256 = checkpoint_sha256(request, &config_sha256, &layout.layout_sha256)?;
    validate_layer_coverage(&layout.index, request, model_config)?;
    let selected = select_tensors(&layout.index.weight_map, &selection_request, model_config);
    ensure!(
        !selected.is_empty(),
        "no tensors matched layers {}..{} or requested prefixes",
        request.layer_start,
        request.layer_end
    );
    validate_required_tensors(&selected, request, &model_config)?;

    let mut by_shard: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for name in &selected {
        let shard = layout
            .index
            .weight_map
            .get(*name)
            .with_context(|| format!("selected tensor {name} is absent from weight map"))?;
        by_shard.entry(shard).or_default().insert(*name);
    }

    let mut shards = Vec::with_capacity(by_shard.len());
    let mut tensors = Vec::with_capacity(selected.len());
    for (file, names) in by_shard {
        if !layout.headers.contains_key(file) {
            let header = fetch_safetensor_header(remote, request, file)
                .with_context(|| format!("inspect {file}"))?;
            layout.headers.insert(file.to_string(), header);
        }
        let header = layout
            .headers
            .get(file)
            .with_context(|| format!("missing inspected header for {file}"))?;
        shards.push(plan_shard(file, header, &names)?);
        tensors.extend(selected_tensors(file, header, &names)?);
    }
    let source_shards = shards
        .iter()
        .map(|shard| {
            let header = layout
                .headers
                .get(&shard.file)
                .expect("planned shard has inspected header");
            SafetensorsSourceShard {
                file: shard.file.clone(),
                file_bytes: header.file_bytes,
                etag: Some(header.etag.clone()),
            }
        })
        .collect();
    let plan = summarize_plan(
        &selection_request,
        &checkpoint_sha256,
        &layout.index,
        config.bytes.len() as u64,
        layout.index_bytes,
        shards,
    )?;
    Ok(PreparedStage {
        checkpoint_sha256,
        plan,
        tensors,
        config: config.bytes,
        config_sha256,
        config_etag: config.etag,
        index_sha256: layout.index_sha256,
        index_etag: layout.index_etag,
        source_shards,
    })
}

fn parse_stage_model_layout(bytes: &[u8]) -> Result<StageModelLayout> {
    let config = parse_raw_stage_model_config(bytes)?;
    ensure!(
        config.num_hidden_layers > 0,
        "SafeTensors model num_hidden_layers must be non-zero"
    );
    let family = match config.model_type.as_str() {
        "llama" => StageModelFamily::Llama,
        "nemotron_h" => StageModelFamily::NemotronH,
        model_type => anyhow::bail!(
            "MLX partial SafeTensors currently supports model_type=llama or nemotron_h, got {model_type:?}"
        ),
    };
    Ok(StageModelLayout {
        family,
        num_hidden_layers: config.num_hidden_layers,
    })
}

fn parse_raw_stage_model_config(bytes: &[u8]) -> Result<RawStageModelConfig> {
    let config: RawStageModelConfig = match serde_json::from_slice(bytes) {
        Ok(config) => config,
        Err(strict_error) => {
            let text =
                std::str::from_utf8(bytes).context("SafeTensors model config is not UTF-8")?;
            json5::from_str(text).with_context(|| {
                format!("parse SafeTensors model config as strict JSON ({strict_error}) or JSON5")
            })?
        }
    };
    Ok(config)
}

fn validate_layer_range(
    request: &SafetensorsStageRequest,
    config: &StageModelLayout,
) -> Result<()> {
    ensure!(
        request.layer_end <= config.num_hidden_layers,
        "stage layer end {} exceeds model layer count {}",
        request.layer_end,
        config.num_hidden_layers
    );
    Ok(())
}

fn add_required_prefixes(request: &mut SafetensorsStageRequest, config: &StageModelLayout) {
    if request.layer_start == 0
        || (config.family == StageModelFamily::Llama
            && request.layer_end == config.num_hidden_layers)
    {
        request
            .include_prefixes
            .push(config.embedding_prefix().to_string());
    }
    if request.layer_end == config.num_hidden_layers {
        request
            .include_prefixes
            .push(config.final_norm_prefix().to_string());
        request
            .include_prefixes
            .push(config.readout_prefix().to_string());
    }
    request.include_prefixes.sort();
    request.include_prefixes.dedup();
}

fn validate_layer_coverage(
    index: &SafetensorsIndex,
    request: &SafetensorsStageRequest,
    config: StageModelLayout,
) -> Result<()> {
    for layer in request.layer_start..request.layer_end {
        ensure!(
            index
                .weight_map
                .keys()
                .any(|name| config.layer_index(name) == Some(layer)),
            "SafeTensors checkpoint has no tensors for requested layer {layer}"
        );
    }
    Ok(())
}

fn validate_required_tensors(
    selected: &BTreeSet<&str>,
    request: &SafetensorsStageRequest,
    config: &StageModelLayout,
) -> Result<()> {
    let has_prefix = |prefix: &str| selected.iter().any(|name| name.starts_with(prefix));
    if request.layer_start == 0 {
        ensure!(
            has_prefix(config.embedding_prefix()),
            "first MLX stage requires {} tensors",
            config.embedding_prefix()
        );
    }
    if request.layer_end == config.num_hidden_layers {
        ensure!(
            has_prefix(config.final_norm_prefix()),
            "final MLX stage requires {} tensors",
            config.final_norm_prefix()
        );
        let has_readout = has_prefix(config.readout_prefix())
            || (config.family == StageModelFamily::Llama && has_prefix(config.embedding_prefix()));
        ensure!(
            has_readout,
            "final MLX stage requires {} tensors{}",
            config.readout_prefix(),
            if config.family == StageModelFamily::Llama {
                " or tied embeddings"
            } else {
                ""
            }
        );
    }
    Ok(())
}

fn load_checkpoint_layout(
    remote: &RemoteSource,
    request: &SafetensorsStageRequest,
) -> Result<CheckpointLayout> {
    let index_url = remote.url(
        &request.repo,
        &request.revision,
        "model.safetensors.index.json",
    )?;
    if let Some(index_file) = remote.optional_small_file(index_url, MAX_INDEX_BYTES)? {
        let index = parse_index(&index_file.bytes)?;
        let index_sha256 = sha256_hex(&index_file.bytes);
        return Ok(CheckpointLayout {
            index,
            layout_sha256: index_sha256.clone(),
            index_bytes: index_file.bytes.len() as u64,
            index_sha256: Some(index_sha256),
            index_etag: index_file.etag,
            headers: BTreeMap::new(),
        });
    }

    let file = "model.safetensors";
    let header = fetch_safetensor_header(remote, request, file)
        .context("inspect unsharded SafeTensors checkpoint")?;
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
        index: SafetensorsIndex {
            metadata: IndexMetadata { total_size },
            weight_map,
        },
        layout_sha256: header.header_sha256.clone(),
        index_bytes: 0,
        index_sha256: None,
        index_etag: None,
        headers: BTreeMap::from([(file.to_string(), header)]),
    })
}

fn fetch_safetensor_header(
    remote: &RemoteSource,
    request: &SafetensorsStageRequest,
    file: &str,
) -> Result<RemoteHeader> {
    let url = remote.url(&request.repo, &request.revision, file)?;
    let len_response = remote.exact_range(url.clone(), 0..8)?;
    let file_bytes = len_response.total_file_bytes;
    let len_etag = len_response
        .etag()
        .context("SafeTensors header-length response omitted ETag")?
        .to_string();
    let len_bytes = len_response.into_bytes()?;
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
    ensure!(
        header_end <= file_bytes,
        "SafeTensors header exceeds source file length"
    );
    let header_response = remote.exact_range_if_range(url, 8..header_end, &len_etag)?;
    ensure!(
        header_response.total_file_bytes == file_bytes,
        "source file size changed while reading SafeTensors header"
    );
    let header_etag = header_response
        .etag()
        .context("SafeTensors header response omitted ETag")?;
    ensure_matching_etag(&len_etag, header_etag, file)?;
    let etag = header_etag.to_string();
    let header_bytes = header_response.into_bytes()?;
    let header_sha256 = sha256_hex(&header_bytes);
    let data_bytes = file_bytes - header_end;
    let tensors = parse_header(&header_bytes, data_bytes)?;
    Ok(RemoteHeader {
        header_len,
        header_sha256,
        file_bytes,
        etag,
        tensors,
    })
}

fn ensure_matching_etag(left: &str, right: &str, file: &str) -> Result<()> {
    ensure!(
        left == right,
        "source identity changed while reading SafeTensors shard {file}"
    );
    Ok(())
}

fn select_tensors<'a>(
    weight_map: &'a BTreeMap<String, String>,
    request: &SafetensorsStageRequest,
    config: StageModelLayout,
) -> BTreeSet<&'a str> {
    weight_map
        .keys()
        .filter(|name| {
            config
                .layer_index(name)
                .is_some_and(|layer| layer >= request.layer_start && layer < request.layer_end)
                || request
                    .include_prefixes
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
        })
        .map(String::as_str)
        .collect()
}

fn plan_shard(
    file: &str,
    header: &RemoteHeader,
    selected: &BTreeSet<&str>,
) -> Result<SafetensorsShardPlan> {
    let data_start = 8_u64
        .checked_add(header.header_len)
        .context("SafeTensors data offset overflow")?;
    let mut ranges = Vec::with_capacity(selected.len());
    let mut selected_tensor_bytes = 0_u64;
    let mut largest_selected_tensor_bytes = 0_u64;
    for name in selected {
        let tensor = header
            .tensors
            .get(*name)
            .with_context(|| format!("weight-map tensor {name} is absent from {file}"))?;
        let start = data_start
            .checked_add(tensor.data_offsets[0])
            .with_context(|| format!("absolute offset overflow for {name}"))?;
        let end_exclusive = data_start
            .checked_add(tensor.data_offsets[1])
            .with_context(|| format!("absolute offset overflow for {name}"))?;
        let tensor_bytes = end_exclusive - start;
        selected_tensor_bytes = selected_tensor_bytes
            .checked_add(tensor_bytes)
            .context("selected tensor byte count overflow")?;
        largest_selected_tensor_bytes = largest_selected_tensor_bytes.max(tensor_bytes);
        ranges.push(ByteRange {
            start,
            end_exclusive,
        });
    }
    let ranges = coalesce_contiguous_ranges(ranges);
    Ok(SafetensorsShardPlan {
        file: file.to_string(),
        file_bytes: header.file_bytes,
        header_probe_bytes: data_start,
        selected_tensor_count: selected.len(),
        selected_tensor_bytes,
        largest_selected_tensor_bytes,
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
            Ok(SelectedTensor {
                name: (*name).to_string(),
                source_file: file.to_string(),
                source_range: ByteRange {
                    start: data_start
                        .checked_add(tensor.data_offsets[0])
                        .with_context(|| format!("absolute offset overflow for {name}"))?,
                    end_exclusive: data_start
                        .checked_add(tensor.data_offsets[1])
                        .with_context(|| format!("absolute offset overflow for {name}"))?,
                },
                header: tensor.clone(),
            })
        })
        .collect()
}

fn coalesce_contiguous_ranges(mut ranges: Vec<ByteRange>) -> Vec<ByteRange> {
    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<ByteRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && range.start == previous.end_exclusive
        {
            previous.end_exclusive = range.end_exclusive;
        } else {
            merged.push(range);
        }
    }
    merged
}

fn summarize_plan(
    request: &SafetensorsStageRequest,
    checkpoint_sha256: &str,
    index: &SafetensorsIndex,
    config_bytes: u64,
    index_bytes: u64,
    shards: Vec<SafetensorsShardPlan>,
) -> Result<SafetensorsStagePlan> {
    let selected_tensor_count = shards.iter().map(|shard| shard.selected_tensor_count).sum();
    let selected_tensor_bytes =
        checked_sum(shards.iter().map(|shard| shard.selected_tensor_bytes))?;
    let largest_selected_tensor_bytes = shards
        .iter()
        .map(|shard| shard.largest_selected_tensor_bytes)
        .max()
        .unwrap_or(0);
    let source_shard_bytes = checked_sum(shards.iter().map(|shard| shard.file_bytes))?;
    let range_request_count = shards.iter().map(|shard| shard.ranges.len()).sum();
    let range_payload_bytes = checked_sum(
        shards
            .iter()
            .flat_map(|shard| shard.ranges.iter())
            .map(ByteRange::len),
    )?;
    let header_probe_bytes = checked_sum(shards.iter().map(|shard| shard.header_probe_bytes))?;
    let planned_download_bytes = config_bytes
        .checked_add(index_bytes)
        .and_then(|bytes| bytes.checked_add(header_probe_bytes))
        .and_then(|bytes| bytes.checked_add(range_payload_bytes))
        .context("planned download byte count overflow")?;
    Ok(SafetensorsStagePlan {
        checkpoint_sha256: checkpoint_sha256.to_string(),
        repo: request.repo.clone(),
        revision: request.revision.clone(),
        layer_start: request.layer_start,
        layer_end: request.layer_end,
        include_prefixes: request.include_prefixes.clone(),
        total_model_tensor_bytes: index.metadata.total_size,
        config_bytes,
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
    })
}

fn checkpoint_sha256(
    request: &SafetensorsStageRequest,
    config_sha256: &str,
    layout_sha256: &str,
) -> Result<String> {
    let identity = serde_json::to_vec(&(
        "mesh-mlx-safetensors-checkpoint-v1",
        &request.repo,
        &request.revision,
        config_sha256,
        layout_sha256,
    ))?;
    Ok(sha256_hex(&identity))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).context("byte count overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planning_tensor(name: &str, shape: Vec<u64>, source_bytes: u64) -> SelectedTensor {
        SelectedTensor {
            name: name.to_string(),
            source_file: "model.safetensors".to_string(),
            source_range: ByteRange {
                start: 0,
                end_exclusive: source_bytes,
            },
            header: TensorHeader {
                dtype: "BF16".to_string(),
                shape,
                data_offsets: [0, source_bytes],
            },
        }
    }

    #[test]
    fn recognizes_family_specific_layer_paths() {
        let llama = parse_stage_model_layout(
            br#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":48}"#,
        )
        .unwrap();
        assert_eq!(
            llama.layer_index("model.layers.42.mlp.up_proj.weight"),
            Some(42)
        );
        assert_eq!(llama.layer_index("backbone.layers.7.mixer.weight"), None);
        assert_eq!(llama.layer_index("model.embed_tokens.weight"), None);

        let nemotron = parse_stage_model_layout(
            br#"{"model_type":"nemotron_h","hidden_size":64,"num_hidden_layers":52}"#,
        )
        .unwrap();
        assert_eq!(
            nemotron.layer_index("backbone.layers.7.mixer.up_proj.weight"),
            Some(7)
        );
        assert_eq!(
            nemotron.layer_index("model.backbone.layers.8.mixer.weight"),
            Some(8)
        );
        assert_eq!(
            nemotron.layer_index("model.layers.42.mlp.up_proj.weight"),
            None
        );
    }

    #[test]
    fn parses_hugging_face_nonfinite_config_values_without_using_them() {
        let layout = parse_stage_model_layout(
            br#"{
                "model_type":"nemotron_h",
                "hidden_size":2688,
                "num_hidden_layers":52,
                "time_step_limit":[0.0, Infinity]
            }"#,
        )
        .unwrap();

        assert_eq!(layout.family, StageModelFamily::NemotronH);
        assert_eq!(layout.num_hidden_layers, 52);
    }

    #[test]
    fn coalesces_only_contiguous_ranges() {
        assert_eq!(
            coalesce_contiguous_ranges(vec![
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
            ]),
            vec![
                ByteRange {
                    start: 0,
                    end_exclusive: 30,
                },
                ByteRange {
                    start: 32,
                    end_exclusive: 40,
                },
            ]
        );
    }

    #[test]
    fn affine4_layer_estimates_charge_boundary_tensors_to_loading_stages() {
        let layout = StageModelLayout {
            family: StageModelFamily::Llama,
            num_hidden_layers: 2,
        };
        let tensors = vec![
            planning_tensor("model.embed_tokens.weight", vec![128, 64], 16_384),
            planning_tensor("model.layers.0.mlp.weight", vec![64, 64], 8_192),
            planning_tensor("model.layers.1.mlp.weight", vec![64, 64], 8_192),
            planning_tensor("model.norm.weight", vec![64], 128),
            planning_tensor("lm_head.weight", vec![128, 64], 16_384),
        ];

        let estimates = estimated_affine4_layer_bytes(layout, &tensors).unwrap();

        assert_eq!(estimates, vec![9_216, 15_488]);
        assert!(estimates[1] > estimates[0]);
    }

    #[test]
    fn assigns_embedding_and_readout_tensors_to_final_llama_stage() {
        let mut request = SafetensorsStageRequest {
            repo: "org/model".to_string(),
            revision: "a".repeat(40),
            layer_start: 1,
            layer_end: 2,
            include_prefixes: Vec::new(),
        };
        let config = StageModelLayout {
            family: StageModelFamily::Llama,
            num_hidden_layers: 2,
        };

        add_required_prefixes(&mut request, &config);

        assert_eq!(
            request.include_prefixes,
            vec![
                "lm_head.".to_string(),
                "model.embed_tokens.".to_string(),
                "model.norm.".to_string(),
            ]
        );
    }

    #[test]
    fn assigns_nemotron_embeddings_and_readout_to_their_own_boundaries() {
        let config = StageModelLayout {
            family: StageModelFamily::NemotronH,
            num_hidden_layers: 2,
        };
        let mut first = SafetensorsStageRequest {
            repo: "org/model".to_string(),
            revision: "a".repeat(40),
            layer_start: 0,
            layer_end: 1,
            include_prefixes: Vec::new(),
        };
        add_required_prefixes(&mut first, &config);
        assert_eq!(
            first.include_prefixes,
            vec!["backbone.embeddings.".to_string()]
        );

        let mut final_stage = SafetensorsStageRequest {
            layer_start: 1,
            layer_end: 2,
            ..first
        };
        final_stage.include_prefixes.clear();
        add_required_prefixes(&mut final_stage, &config);
        assert_eq!(
            final_stage.include_prefixes,
            vec!["backbone.norm_f.".to_string(), "lm_head.".to_string()]
        );
    }
}
