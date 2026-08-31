use std::{
    io::{BufReader, BufWriter, Read, Write},
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use skippy_cache::{
    ExactStateIdentityParams, HandoffManifest, HandoffSegmentRef, HandoffSegmentStore,
    KvFetchClient, exact_state_identity, serve_store,
};
use skippy_runtime::{
    GGML_TYPE_F16, MtpSource, RuntimeConfig, RuntimeKvPageDesc, StageModel, StageSession,
};

use crate::{
    cli::{RemoteHandoffArgs, RemoteHandoffRole, StatePayloadKind},
    report::{RemoteHandoffReceiverTimings, RemoteHandoffReport},
};

use super::native_mtp::emit_report;
use super::stage_execution::{
    PackageStageSpec, elapsed_ms, ensure_matches, runtime_flash_attn, runtime_load_mode,
    runtime_model_identity, stage_id_for_index, stage_model_resolution, status,
};
use super::state_handoff::state_handoff_tokens;

const PROTOCOL_VERSION: u32 = 1;
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_SEGMENT_BYTES: u64 = 256 * 1024 * 1024;
const STREAM_BUFFER_BYTES: usize = 1024 * 1024;

mod frame_kind {
    pub const HELLO: u8 = 1;
    pub const HELLO_ACK: u8 = 2;
    pub const SEGMENT: u8 = 3;
    pub const COMMIT: u8 = 4;
    pub const VERIFY: u8 = 5;
    pub const RESULT: u8 = 6;
    pub const PAGE: u8 = 7;
    pub const RECURRENT: u8 = 8;
}

#[derive(Serialize, Deserialize)]
struct HelloHeader {
    protocol_version: u32,
    model_id: String,
    layer_end: u32,
    ctx_size: u32,
    state_payload_kind: String,
    flash_attn: String,
    decode_token_count: usize,
    lane_count: u32,
    state_identity: String,
}

#[derive(Serialize, Deserialize)]
struct HelloAckHeader {
    ok: bool,
    reason: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct SegmentHeader {
    index: usize,
    offset: u64,
    payload_bytes: u64,
    blake3: String,
}

#[derive(Serialize, Deserialize)]
struct PageHeader {
    index: usize,
    token_start: u64,
    token_count: u64,
    payload_bytes: u64,
    blake3: String,
    kv_desc: RuntimeKvPageDesc,
}

#[derive(Serialize, Deserialize)]
struct RecurrentHeader {
    payload_bytes: u64,
    blake3: String,
}

/// Per-segment metadata carried in `HandoffSegmentRef::meta_json` for
/// page-stream manifests.
#[derive(Serialize, Deserialize)]
struct PageSegmentMeta {
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kv_desc: Option<RuntimeKvPageDesc>,
    #[serde(default)]
    token_start: u64,
    #[serde(default)]
    token_count: u64,
}

#[derive(Serialize, Deserialize)]
struct CommitHeader {
    segment_count: usize,
    total_bytes: u64,
    payload_blake3: String,
    prompt_token_count: u64,
    continuation_token: i32,
    kv_bytes: u64,
    recurrent_bytes: u64,
    kv_desc: Option<RuntimeKvPageDesc>,
    prefix_tokens: Vec<i32>,
    run_baseline: bool,
    #[serde(default)]
    streaming: bool,
    #[serde(default)]
    page_count: usize,
}

#[derive(Serialize, Deserialize)]
struct VerifyHeader {
    source_tokens: Vec<i32>,
}

#[derive(Serialize, Deserialize)]
struct ResultHeader {
    ok: bool,
    reason: Option<String>,
    restored_tokens: Vec<i32>,
    baseline_tokens: Vec<i32>,
    tokens_match: bool,
    baseline_matches: Option<bool>,
    timings: RemoteHandoffReceiverTimings,
}

struct StatePayload {
    kv_desc: Option<RuntimeKvPageDesc>,
    kv: Vec<u8>,
    recurrent: Vec<u8>,
}

impl StatePayload {
    fn full_state(bytes: Vec<u8>) -> Self {
        Self {
            kv_desc: None,
            kv: bytes,
            recurrent: Vec::new(),
        }
    }

    fn wire_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.kv.len() + self.recurrent.len());
        bytes.extend_from_slice(&self.kv);
        bytes.extend_from_slice(&self.recurrent);
        bytes
    }
}

fn effective_lane_count(args: &RemoteHandoffArgs) -> u32 {
    // Full-state blobs cover the whole context, whose KV layout depends on
    // the lane count, so sender and receiver must open identically shaped
    // contexts.
    args.runtime_lane_count.unwrap_or(2).max(2)
}

pub fn remote_handoff(args: RemoteHandoffArgs) -> Result<()> {
    match args.role {
        RemoteHandoffRole::Send if args.streaming => run_sender_streaming(args),
        RemoteHandoffRole::Send => run_sender(args),
        RemoteHandoffRole::Recv => run_receiver(args),
        RemoteHandoffRole::Restore => run_restore(args),
        RemoteHandoffRole::Serve => run_serve(args),
        RemoteHandoffRole::Fetch => run_fetch(args),
    }
}

/// Serve the local L3 store to peers over the `skippy-kv/1` fetch protocol.
fn run_serve(args: RemoteHandoffArgs) -> Result<()> {
    let store_dir = args
        .store_dir
        .clone()
        .context("--role serve requires --store-dir")?;
    let store = HandoffSegmentStore::open(&store_dir, args.store_budget_bytes)?;
    let listener = std::net::TcpListener::bind(args.listen)
        .with_context(|| format!("failed to bind skippy-kv listener on {}", args.listen))?;
    eprintln!(
        "skippy-kv store server ready on {} serving {} ({} manifests)",
        args.listen,
        store_dir.display(),
        store.list_manifests()?.len()
    );
    serve_store(&store, &listener, args.accept_count)
}

/// Pull a manifest and its segments from a peer's store into the local one,
/// then restore and decode from it to prove the fetched state is usable —
/// cross-node prefix reuse without a push handoff.
fn run_fetch(mut args: RemoteHandoffArgs) -> Result<()> {
    let peer = args
        .peer
        .context("--role fetch requires --peer <store-server-address>")?;
    let store_dir = args
        .store_dir
        .clone()
        .context("--role fetch requires --store-dir")?;
    let store = HandoffSegmentStore::open(&store_dir, args.store_budget_bytes)?;
    let fetch_started = Instant::now();
    let mut client = KvFetchClient::connect(&peer.to_string())?;
    let (manifest, stats) = client
        .fetch_into_store(args.manifest.as_deref(), &store)
        .context("failed to fetch manifest from peer")?;
    drop(client);
    let fetch_ms = elapsed_ms(fetch_started);
    eprintln!(
        "skippy-kv fetched manifest {} from {peer}: {} segments pulled, {} already local, {:.1} MiB in {fetch_ms:.0} ms ({:.2} Gbps)",
        manifest.payload_digest,
        stats.segments_fetched,
        stats.segments_skipped,
        stats.bytes_fetched as f64 / (1024.0 * 1024.0),
        transfer_gbps(stats.bytes_fetched as usize, fetch_ms),
    );
    args.manifest = Some(manifest.payload_digest.clone());
    args.streaming = manifest.payload_kind == "kv-page-stream";
    run_restore(args)
}

fn effective_payload_kind(args: &RemoteHandoffArgs) -> &'static str {
    if args.streaming {
        "kv-page-stream"
    } else {
        payload_kind_name(args.state_payload_kind)
    }
}

/// Content digest of the served artifact, memoized per path: two harness
/// processes serving different local GGUFs behind the same display model id
/// must never share a state identity. Directories (layer-package refs) are
/// not hashed here; their identity rides the package manifest via the
/// model-identity fields.
fn artifact_sha256_cached(path: &std::path::Path) -> Option<String> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<std::path::PathBuf, Option<String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(path)
    {
        return cached.clone();
    }
    let digest = (|| -> Option<String> {
        if !path.is_file() {
            return None;
        }
        let mut file = std::fs::File::open(path).ok()?;
        let mut hasher = sha2::Sha256::new();
        std::io::copy(&mut file, &mut hasher).ok()?;
        use sha2::Digest as _;
        Some(format!("{:x}", hasher.finalize()))
    })();
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(path.to_path_buf(), digest.clone());
    digest
}

/// The numerical identity of the state this configuration produces or
/// accepts. The harness pins F16 KV and does not resolve a concrete backend
/// device, so the platform tag inside `exact_state_identity` (arch,
/// endianness, pointer width) is the cross-machine guard — and the served
/// artifact's content digest guards against two different local files
/// behind the same display model id.
fn state_identity_for(
    args: &RemoteHandoffArgs,
    identity: &model_artifact::ModelIdentity,
) -> String {
    let source_model_sha256 = artifact_sha256_cached(&args.runtime.model);
    exact_state_identity(&ExactStateIdentityParams {
        model_id: &identity.model_id,
        model_revision: identity.source_revision.as_deref(),
        model_file: identity.source_file.as_deref(),
        manifest_sha256: None,
        source_model_sha256: source_model_sha256.as_deref(),
        package_ref: None,
        cache_type_k: "f16",
        cache_type_v: "f16",
        flash_attn_type: protocol_flash_attn_type(args.runtime.flash_attn),
        n_gpu_layers: args.runtime.n_gpu_layers,
        backend_device: None,
        layer_start: 0,
        layer_end: args.runtime.layer_end,
        ctx_size: args.runtime.ctx_size,
        lane_count: effective_lane_count(args),
        payload_kind: effective_payload_kind(args),
    })
}

fn protocol_flash_attn_type(
    value: crate::cli::FlashAttentionArg,
) -> skippy_protocol::FlashAttentionType {
    match value {
        crate::cli::FlashAttentionArg::Auto => skippy_protocol::FlashAttentionType::Auto,
        crate::cli::FlashAttentionArg::Disabled => skippy_protocol::FlashAttentionType::Disabled,
        crate::cli::FlashAttentionArg::Enabled => skippy_protocol::FlashAttentionType::Enabled,
    }
}

fn open_store(args: &RemoteHandoffArgs) -> Result<Option<HandoffSegmentStore>> {
    args.store_dir
        .as_ref()
        .map(|dir| HandoffSegmentStore::open(dir, args.store_budget_bytes))
        .transpose()
}

fn manifest_from_commit(
    commit: &CommitHeader,
    segments: Vec<HandoffSegmentRef>,
    state_identity: String,
    payload_kind: &str,
    expected_tokens: Vec<i32>,
) -> Result<HandoffManifest> {
    let mut manifest = HandoffManifest::new(state_identity, payload_kind.to_string());
    manifest.total_bytes = commit.total_bytes;
    manifest.payload_digest = commit.payload_blake3.clone();
    manifest.segments = segments;
    manifest.kv_bytes = commit.kv_bytes;
    manifest.recurrent_bytes = commit.recurrent_bytes;
    manifest.kv_desc_json = commit
        .kv_desc
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .context("failed to serialize kv desc for manifest")?;
    manifest.token_count = commit.prompt_token_count;
    manifest.continuation_token = commit.continuation_token;
    manifest.expected_tokens = expected_tokens;
    Ok(manifest)
}

fn open_full_model(args: &RemoteHandoffArgs, lane_count: u32) -> Result<StageModel> {
    let model_identity = runtime_model_identity(&args.runtime)?;
    let spec = PackageStageSpec {
        topology_id: "correctness-remote-handoff",
        stage_id: stage_id_for_index(0),
        stage_index: 0,
        layer_start: 0,
        layer_end: args.runtime.layer_end,
        include_embeddings: true,
        include_output: true,
    };
    let resolution = stage_model_resolution(
        &args.runtime.model,
        args.runtime.stage_model.as_ref(),
        args.runtime.stage_load_mode,
        &model_identity,
        spec,
    )?;
    let runtime_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.runtime.layer_end,
        ctx_size: args.runtime.ctx_size,
        lane_count,
        n_batch: args.runtime.n_batch,
        n_ubatch: args.runtime.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.runtime.n_gpu_layers,
        mmap: None,
        mlock: false,
        repack: false,
        op_offload: None,
        no_host_buffer: false,
        check_tensors: false,
        direct_io: false,
        main_gpu: None,
        split_mode: skippy_runtime::SplitMode::Auto,
        selected_backend_device: None,
        load_mode: runtime_load_mode(args.runtime.stage_load_mode),
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
        include_embeddings: true,
        include_output: true,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: false,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: runtime_flash_attn(args.runtime.flash_attn),
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    StageModel::open(&resolution.path, &runtime_config)
        .context("failed to open remote handoff model")
}

fn greedy_decode(session: &mut StageSession, first_token: i32, count: usize) -> Result<Vec<i32>> {
    let mut tokens = Vec::with_capacity(count);
    let mut next = first_token;
    for _ in 0..count {
        let predicted = session
            .decode_step(next)
            .context("greedy decode step failed")?;
        tokens.push(predicted);
        next = predicted;
    }
    Ok(tokens)
}

fn export_state_payload(
    session: &mut StageSession,
    args: &RemoteHandoffArgs,
    token_count: u64,
) -> Result<StatePayload> {
    match args.state_payload_kind {
        StatePayloadKind::FullState => Ok(StatePayload::full_state(
            session.export_full_state(0, args.runtime.layer_end as i32)?,
        )),
        StatePayloadKind::KvRecurrent => {
            let page = session.export_kv_page(0, args.runtime.layer_end as i32, 0, token_count)?;
            let recurrent = session.export_recurrent_state()?;
            Ok(StatePayload {
                kv_desc: Some(page.desc),
                kv: page.payload,
                recurrent,
            })
        }
        other => bail!("remote handoff does not support state payload kind {other:?}"),
    }
}

fn import_state_payload(
    session: &mut StageSession,
    args: &RemoteHandoffArgs,
    commit: &CommitHeader,
    bytes: &[u8],
) -> Result<()> {
    let kv_bytes = usize::try_from(commit.kv_bytes).context("kv byte count exceeds usize")?;
    let recurrent_bytes =
        usize::try_from(commit.recurrent_bytes).context("recurrent byte count exceeds usize")?;
    if kv_bytes + recurrent_bytes != bytes.len() {
        bail!(
            "commit component sizes {} + {} do not cover payload of {} bytes",
            kv_bytes,
            recurrent_bytes,
            bytes.len()
        );
    }
    match &commit.kv_desc {
        None => session.import_full_state_for_token_count(
            0,
            args.runtime.layer_end as i32,
            &bytes[..kv_bytes],
            commit.prompt_token_count,
        ),
        Some(kv_desc) => {
            session.import_kv_page(kv_desc, &bytes[..kv_bytes])?;
            session.import_recurrent_state_for_token_count(
                &bytes[kv_bytes..],
                commit.prompt_token_count,
            )
        }
    }
}

fn run_sender(args: RemoteHandoffArgs) -> Result<()> {
    let peer = args
        .peer
        .context("--role send requires --peer <receiver-address>")?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let report_out = args.output.report_out.clone();

    let model_load_started = Instant::now();
    let model = open_full_model(&args, effective_lane_count(&args))?;
    let model_load_ms = elapsed_ms(model_load_started);

    let tokenize_started = Instant::now();
    let tokens = state_handoff_tokens(&model, &args.runtime.prompt, args.prefix_token_count)
        .context("failed to tokenize remote handoff prompt")?;
    let split = args.prefix_token_count.unwrap_or(tokens.len() - 1);
    let prefix = tokens[..split].to_vec();
    let continuation = tokens[split];
    let tokenize_ms = elapsed_ms(tokenize_started);

    let stream = TcpStream::connect(peer)
        .with_context(|| format!("failed to connect to receiver at {peer}"))?;
    stream.set_nodelay(true).ok();
    let mut reader = BufReader::with_capacity(STREAM_BUFFER_BYTES, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER_BYTES, stream);

    write_frame(
        &mut writer,
        frame_kind::HELLO,
        &HelloHeader {
            protocol_version: PROTOCOL_VERSION,
            model_id: model_identity.model_id.clone(),
            layer_end: args.runtime.layer_end,
            ctx_size: args.runtime.ctx_size,
            state_payload_kind: effective_payload_kind(&args).to_string(),
            flash_attn: format!("{:?}", args.runtime.flash_attn),
            decode_token_count: args.decode_tokens,
            lane_count: effective_lane_count(&args),
            state_identity: state_identity_for(&args, &model_identity),
        },
        &[],
    )?;
    writer.flush().context("failed to flush hello frame")?;
    let (ack, _) = read_frame_expect::<HelloAckHeader>(&mut reader, frame_kind::HELLO_ACK)?;
    if !ack.ok {
        bail!(
            "receiver rejected handoff: {}",
            ack.reason.unwrap_or_else(|| "no reason given".to_string())
        );
    }

    let mut session = model
        .create_session()
        .context("failed to create sender session")?;
    let prefill_started = Instant::now();
    session
        .prefill_chunked(&prefix)
        .context("sender prefill failed")?;
    let source_prefill_ms = elapsed_ms(prefill_started);

    let export_started = Instant::now();
    let payload = export_state_payload(&mut session, &args, prefix.len() as u64)?;
    let state_export_ms = elapsed_ms(export_started);

    let wire_bytes = payload.wire_bytes();
    let payload_digest = digest_hex(&wire_bytes);
    let segment_bytes = args.segment_bytes.max(1);
    let transfer_started = Instant::now();
    let mut segment_count = 0usize;
    for (index, chunk) in wire_bytes.chunks(segment_bytes).enumerate() {
        write_frame(
            &mut writer,
            frame_kind::SEGMENT,
            &SegmentHeader {
                index,
                offset: (index * segment_bytes) as u64,
                payload_bytes: chunk.len() as u64,
                blake3: digest_hex(chunk),
            },
            chunk,
        )?;
        segment_count += 1;
    }
    write_frame(
        &mut writer,
        frame_kind::COMMIT,
        &CommitHeader {
            segment_count,
            total_bytes: wire_bytes.len() as u64,
            payload_blake3: payload_digest.clone(),
            prompt_token_count: prefix.len() as u64,
            continuation_token: continuation,
            kv_bytes: payload.kv.len() as u64,
            recurrent_bytes: payload.recurrent.len() as u64,
            kv_desc: payload.kv_desc.clone(),
            prefix_tokens: if args.baseline {
                prefix.clone()
            } else {
                Vec::new()
            },
            run_baseline: args.baseline,
            streaming: false,
            page_count: 0,
        },
        &[],
    )?;
    writer.flush().context("failed to flush handoff stream")?;
    let transfer_ms = elapsed_ms(transfer_started);

    // Spill to the local L3 store off the transfer critical path: the
    // receiver is already importing while these writes land on disk.
    let store = open_store(&args)?;
    let store_started = Instant::now();
    let mut segment_refs = Vec::new();
    if let Some(store) = &store {
        for (index, chunk) in wire_bytes.chunks(segment_bytes).enumerate() {
            let (digest, _) = store
                .put_segment(chunk)
                .context("sender segment spill failed")?;
            segment_refs.push(HandoffSegmentRef {
                index: index as u32,
                offset: (index * segment_bytes) as u64,
                bytes: chunk.len() as u64,
                digest,
                meta_json: None,
            });
        }
    }
    let mut store_ms = elapsed_ms(store_started);

    // The receiver imports and decodes while the sender produces the
    // reference continuation, mirroring how both phases overlap in serving.
    let source_decode_started = Instant::now();
    let source_tokens = greedy_decode(&mut session, continuation, args.decode_tokens)?;
    let source_decode_ms = elapsed_ms(source_decode_started);

    if let Some(store) = &store {
        let commit_started = Instant::now();
        let manifest = manifest_from_commit(
            &CommitHeader {
                segment_count,
                total_bytes: wire_bytes.len() as u64,
                payload_blake3: payload_digest.clone(),
                prompt_token_count: prefix.len() as u64,
                continuation_token: continuation,
                kv_bytes: payload.kv.len() as u64,
                recurrent_bytes: payload.recurrent.len() as u64,
                kv_desc: payload.kv_desc.clone(),
                prefix_tokens: Vec::new(),
                run_baseline: false,
                streaming: false,
                page_count: 0,
            },
            segment_refs,
            state_identity_for(&args, &model_identity),
            payload_kind_name(args.state_payload_kind),
            source_tokens.clone(),
        )?;
        store
            .commit(&manifest)
            .context("sender manifest commit failed")?;
        store_ms += elapsed_ms(commit_started);
    }

    write_frame(
        &mut writer,
        frame_kind::VERIFY,
        &VerifyHeader {
            source_tokens: source_tokens.clone(),
        },
        &[],
    )?;
    writer.flush().context("failed to flush verify frame")?;

    let (result, _) = read_frame_expect::<ResultHeader>(&mut reader, frame_kind::RESULT)?;
    if !result.ok {
        bail!(
            "receiver failed to complete handoff: {}",
            result
                .reason
                .unwrap_or_else(|| "no reason given".to_string())
        );
    }

    let timings = result.timings;
    let ttft_disaggregated_ms = source_prefill_ms
        + state_export_ms
        + transfer_ms
        + timings.kv_attach_ms
        + timings.first_decode_ms;
    let ttft_local_ms = (timings.baseline_prefill_ms > 0.0)
        .then_some(timings.baseline_prefill_ms + timings.baseline_first_decode_ms);
    let matches = result.tokens_match && result.baseline_matches.unwrap_or(true);
    let report = RemoteHandoffReport {
        mode: "remote-handoff",
        status: status(matches),
        role: "send",
        model_identity,
        matches,
        tokens_match: result.tokens_match,
        baseline_matches: result.baseline_matches,
        state_payload_kind: payload_kind_name(args.state_payload_kind),
        prompt_token_count: prefix.len(),
        decode_token_count: args.decode_tokens,
        continuation_token: continuation,
        source_tokens,
        restored_tokens: result.restored_tokens,
        state_bytes: wire_bytes.len(),
        state_bytes_per_prompt_token: wire_bytes.len() as f64 / prefix.len().max(1) as f64,
        kv_bytes: payload.kv.len(),
        recurrent_bytes: payload.recurrent.len(),
        segment_count,
        segment_bytes,
        payload_digest,
        model_load_ms,
        tokenize_ms,
        source_prefill_ms,
        state_export_ms,
        transfer_ms,
        transfer_gbps: transfer_gbps(wire_bytes.len(), transfer_ms),
        source_decode_ms,
        store_ms: store.is_some().then_some(store_ms),
        overlap_wall_ms: None,
        receiver: timings,
        ttft_disaggregated_ms,
        ttft_local_ms,
        ttft_speedup: ttft_local_ms.map(|local| local / ttft_disaggregated_ms.max(f64::EPSILON)),
    };
    emit_report(&report, report_out.as_deref())?;
    ensure_matches(matches, args.allow_mismatch)?;
    Ok(())
}

/// Streaming sender: after each prefill chunk, the KV page for that token
/// range is exported and streamed while the next chunk computes, so transfer
/// hides behind prefill. The recurrent snapshot (when the family has one)
/// and the commit record are the uncovered tail; the receiver stages pages
/// as they arrive but cannot generate until the commit validates.
fn run_sender_streaming(args: RemoteHandoffArgs) -> Result<()> {
    let peer = args
        .peer
        .context("--role send requires --peer <receiver-address>")?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let report_out = args.output.report_out.clone();

    let model_load_started = Instant::now();
    let model = open_full_model(&args, effective_lane_count(&args))?;
    let model_load_ms = elapsed_ms(model_load_started);

    let tokenize_started = Instant::now();
    let tokens = state_handoff_tokens(&model, &args.runtime.prompt, args.prefix_token_count)
        .context("failed to tokenize remote handoff prompt")?;
    let split = args.prefix_token_count.unwrap_or(tokens.len() - 1);
    let prefix = tokens[..split].to_vec();
    let continuation = tokens[split];
    let tokenize_ms = elapsed_ms(tokenize_started);

    let stream = TcpStream::connect(peer)
        .with_context(|| format!("failed to connect to receiver at {peer}"))?;
    stream.set_nodelay(true).ok();
    let mut reader = BufReader::with_capacity(STREAM_BUFFER_BYTES, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER_BYTES, stream);

    write_frame(
        &mut writer,
        frame_kind::HELLO,
        &HelloHeader {
            protocol_version: PROTOCOL_VERSION,
            model_id: model_identity.model_id.clone(),
            layer_end: args.runtime.layer_end,
            ctx_size: args.runtime.ctx_size,
            state_payload_kind: effective_payload_kind(&args).to_string(),
            flash_attn: format!("{:?}", args.runtime.flash_attn),
            decode_token_count: args.decode_tokens,
            lane_count: effective_lane_count(&args),
            state_identity: state_identity_for(&args, &model_identity),
        },
        &[],
    )?;
    writer.flush().context("failed to flush hello frame")?;
    let (ack, _) = read_frame_expect::<HelloAckHeader>(&mut reader, frame_kind::HELLO_ACK)?;
    if !ack.ok {
        bail!(
            "receiver rejected handoff: {}",
            ack.reason.unwrap_or_else(|| "no reason given".to_string())
        );
    }

    let mut session = model
        .create_session()
        .context("failed to create sender session")?;
    let chunk_tokens = args.stream_chunk_tokens.max(1);
    let store = open_store(&args)?;
    let overlap_started = Instant::now();
    let mut source_prefill_ms = 0.0f64;
    let mut state_export_ms = 0.0f64;
    let mut transfer_ms = 0.0f64;
    let mut store_ms = 0.0f64;
    let mut payload_hasher = blake3::Hasher::new();
    let mut segment_refs: Vec<HandoffSegmentRef> = Vec::new();
    let mut total_bytes = 0u64;
    let mut page_count = 0usize;
    for (index, chunk) in prefix.chunks(chunk_tokens).enumerate() {
        let token_start = (index * chunk_tokens) as u64;
        let prefill_started = Instant::now();
        session
            .prefill_chunked(chunk)
            .context("sender streaming prefill chunk failed")?;
        source_prefill_ms += elapsed_ms(prefill_started);

        let export_started = Instant::now();
        let page = session
            .export_kv_page(
                0,
                args.runtime.layer_end as i32,
                token_start,
                chunk.len() as u64,
            )
            .with_context(|| format!("failed to export KV page for chunk {index}"))?;
        state_export_ms += elapsed_ms(export_started);

        let digest = digest_hex(&page.payload);
        payload_hasher.update(&page.payload);
        let stream_started = Instant::now();
        write_frame(
            &mut writer,
            frame_kind::PAGE,
            &PageHeader {
                index,
                token_start,
                token_count: chunk.len() as u64,
                payload_bytes: page.payload.len() as u64,
                blake3: digest.clone(),
                kv_desc: page.desc.clone(),
            },
            &page.payload,
        )?;
        // Flush per page so the receiver imports while later chunks prefill.
        writer.flush().context("failed to flush page frame")?;
        transfer_ms += elapsed_ms(stream_started);

        if let Some(store) = &store {
            let store_started = Instant::now();
            store
                .put_segment(&page.payload)
                .context("sender page spill failed")?;
            store_ms += elapsed_ms(store_started);
            segment_refs.push(HandoffSegmentRef {
                index: index as u32,
                offset: total_bytes,
                bytes: page.payload.len() as u64,
                digest,
                meta_json: Some(serde_json::to_string(&PageSegmentMeta {
                    kind: "kv-page".to_string(),
                    kv_desc: Some(page.desc),
                    token_start,
                    token_count: chunk.len() as u64,
                })?),
            });
        }
        total_bytes += page.payload.len() as u64;
        page_count += 1;
    }

    // Recurrent/SSM state is only final once the whole prompt has prefilled —
    // the uncovered tail of the overlap.
    let export_started = Instant::now();
    let recurrent: Vec<u8> = session.export_recurrent_state().unwrap_or_default();
    state_export_ms += elapsed_ms(export_started);
    if !recurrent.is_empty() {
        let digest = digest_hex(&recurrent);
        payload_hasher.update(&recurrent);
        let stream_started = Instant::now();
        write_frame(
            &mut writer,
            frame_kind::RECURRENT,
            &RecurrentHeader {
                payload_bytes: recurrent.len() as u64,
                blake3: digest.clone(),
            },
            &recurrent,
        )?;
        transfer_ms += elapsed_ms(stream_started);
        if let Some(store) = &store {
            store
                .put_segment(&recurrent)
                .context("sender recurrent spill failed")?;
            segment_refs.push(HandoffSegmentRef {
                index: page_count as u32,
                offset: total_bytes,
                bytes: recurrent.len() as u64,
                digest,
                meta_json: Some(serde_json::to_string(&PageSegmentMeta {
                    kind: "recurrent".to_string(),
                    kv_desc: None,
                    token_start: 0,
                    token_count: prefix.len() as u64,
                })?),
            });
        }
        total_bytes += recurrent.len() as u64;
    }

    let payload_digest = format!("{}", payload_hasher.finalize().to_hex());
    let commit = CommitHeader {
        segment_count: page_count + usize::from(!recurrent.is_empty()),
        total_bytes,
        payload_blake3: payload_digest.clone(),
        prompt_token_count: prefix.len() as u64,
        continuation_token: continuation,
        kv_bytes: total_bytes - recurrent.len() as u64,
        recurrent_bytes: recurrent.len() as u64,
        kv_desc: None,
        prefix_tokens: if args.baseline {
            prefix.clone()
        } else {
            Vec::new()
        },
        run_baseline: args.baseline,
        streaming: true,
        page_count,
    };
    write_frame(&mut writer, frame_kind::COMMIT, &commit, &[])?;
    writer.flush().context("failed to flush commit frame")?;
    let overlap_wall_ms = elapsed_ms(overlap_started);

    let source_decode_started = Instant::now();
    let source_tokens = greedy_decode(&mut session, continuation, args.decode_tokens)?;
    let source_decode_ms = elapsed_ms(source_decode_started);

    if let Some(store) = &store {
        let commit_started = Instant::now();
        let manifest = manifest_from_commit(
            &commit,
            segment_refs,
            state_identity_for(&args, &model_identity),
            effective_payload_kind(&args),
            source_tokens.clone(),
        )?;
        store
            .commit(&manifest)
            .context("sender manifest commit failed")?;
        store_ms += elapsed_ms(commit_started);
    }

    write_frame(
        &mut writer,
        frame_kind::VERIFY,
        &VerifyHeader {
            source_tokens: source_tokens.clone(),
        },
        &[],
    )?;
    writer.flush().context("failed to flush verify frame")?;

    let (result, _) = read_frame_expect::<ResultHeader>(&mut reader, frame_kind::RESULT)?;
    if !result.ok {
        bail!(
            "receiver failed to complete handoff: {}",
            result
                .reason
                .unwrap_or_else(|| "no reason given".to_string())
        );
    }

    let timings = result.timings;
    // Streaming TTFT: page transfer and import hide inside the prefill wall;
    // only the commit tail and the first decode remain serial.
    let ttft_disaggregated_ms =
        overlap_wall_ms + timings.attach_residual_ms + timings.first_decode_ms;
    let ttft_local_ms = (timings.baseline_prefill_ms > 0.0)
        .then_some(timings.baseline_prefill_ms + timings.baseline_first_decode_ms);
    let matches = result.tokens_match && result.baseline_matches.unwrap_or(true);
    let report = RemoteHandoffReport {
        mode: "remote-handoff",
        status: status(matches),
        role: "send",
        model_identity,
        matches,
        tokens_match: result.tokens_match,
        baseline_matches: result.baseline_matches,
        state_payload_kind: effective_payload_kind(&args),
        prompt_token_count: prefix.len(),
        decode_token_count: args.decode_tokens,
        continuation_token: continuation,
        source_tokens,
        restored_tokens: result.restored_tokens,
        state_bytes: total_bytes as usize,
        state_bytes_per_prompt_token: total_bytes as f64 / prefix.len().max(1) as f64,
        kv_bytes: (total_bytes - recurrent.len() as u64) as usize,
        recurrent_bytes: recurrent.len(),
        segment_count: page_count,
        segment_bytes: chunk_tokens,
        payload_digest,
        model_load_ms,
        tokenize_ms,
        source_prefill_ms,
        state_export_ms,
        transfer_ms,
        transfer_gbps: transfer_gbps(total_bytes as usize, transfer_ms),
        source_decode_ms,
        store_ms: store.is_some().then_some(store_ms),
        overlap_wall_ms: Some(overlap_wall_ms),
        receiver: timings,
        ttft_disaggregated_ms,
        ttft_local_ms,
        ttft_speedup: ttft_local_ms.map(|local| local / ttft_disaggregated_ms.max(f64::EPSILON)),
    };
    emit_report(&report, report_out.as_deref())?;
    ensure_matches(matches, args.allow_mismatch)?;
    Ok(())
}

fn run_receiver(args: RemoteHandoffArgs) -> Result<()> {
    let model_identity = runtime_model_identity(&args.runtime)?;

    let model_load_started = Instant::now();
    let model = open_full_model(&args, effective_lane_count(&args))?;
    let model_load_ms = elapsed_ms(model_load_started);

    let store = open_store(&args)?;
    let local_state_identity = state_identity_for(&args, &model_identity);
    let listener = TcpListener::bind(args.listen)
        .with_context(|| format!("failed to bind receiver listener on {}", args.listen))?;
    eprintln!(
        "remote-handoff receiver ready on {} (model loaded in {model_load_ms:.0} ms)",
        args.listen
    );
    let mut served = 0usize;
    let mut failures = 0usize;
    loop {
        let (stream, sender_addr) = listener.accept().context("failed to accept sender")?;
        served += 1;
        eprintln!("remote-handoff sender connected from {sender_addr} (handoff {served})");
        let report_out = per_connection_report_path(&args, served);
        match handle_receiver_connection(
            &model,
            &args,
            &model_identity,
            model_load_ms,
            stream,
            report_out.as_deref(),
            store.as_ref(),
            &local_state_identity,
        ) {
            Ok(true) => eprintln!("remote-handoff handoff {served}: MATCH"),
            Ok(false) => {
                failures += 1;
                eprintln!("remote-handoff handoff {served}: MISMATCH");
            }
            Err(error) => {
                failures += 1;
                eprintln!("remote-handoff handoff {served} failed: {error:#}");
            }
        }
        if args.accept_count != 0 && served >= args.accept_count {
            break;
        }
    }
    if failures > 0 {
        ensure_matches(false, args.allow_mismatch)
            .with_context(|| format!("{failures} of {served} handoffs failed"))?;
    }
    Ok(())
}

fn per_connection_report_path(
    args: &RemoteHandoffArgs,
    connection_index: usize,
) -> Option<std::path::PathBuf> {
    let base = args.output.report_out.as_ref()?;
    if args.accept_count == 1 {
        return Some(base.clone());
    }
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let extension = base
        .extension()
        .map(|extension| format!(".{}", extension.to_string_lossy()))
        .unwrap_or_default();
    Some(base.with_file_name(format!("{stem}-{connection_index}{extension}")))
}

#[allow(clippy::too_many_arguments)]
fn handle_receiver_connection(
    model: &StageModel,
    args: &RemoteHandoffArgs,
    model_identity: &model_artifact::ModelIdentity,
    model_load_ms: f64,
    stream: TcpStream,
    report_out: Option<&std::path::Path>,
    store: Option<&HandoffSegmentStore>,
    local_state_identity: &str,
) -> Result<bool> {
    stream.set_nodelay(true).ok();
    // A per-read socket property: it stays in force for every subsequent
    // read on this connection, so a sender that stalls mid-stream errors
    // out after this long rather than wedging the accept loop.
    stream
        .set_read_timeout(Some(Duration::from_secs(args.handshake_timeout_secs)))
        .ok();
    let mut reader = BufReader::with_capacity(STREAM_BUFFER_BYTES, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER_BYTES, stream);

    let (hello, _) = read_frame_expect::<HelloHeader>(&mut reader, frame_kind::HELLO)?;
    if let Err(error) = validate_hello(&hello, args, &model_identity.model_id)
        .and_then(|()| validate_state_identity(&hello, local_state_identity))
    {
        write_frame(
            &mut writer,
            frame_kind::HELLO_ACK,
            &HelloAckHeader {
                ok: false,
                reason: Some(error.to_string()),
            },
            &[],
        )?;
        writer.flush().ok();
        return Err(error);
    }
    write_frame(
        &mut writer,
        frame_kind::HELLO_ACK,
        &HelloAckHeader {
            ok: true,
            reason: None,
        },
        &[],
    )?;
    writer.flush().context("failed to flush hello ack")?;

    // Phase one: accumulate segments — into the L3 store when configured
    // (write-behind), in memory otherwise. Nothing touches a session until
    // the commit record validates completeness.
    let receive_started = Instant::now();
    let mut payload: Vec<u8> = Vec::new();
    let mut received_bytes = 0u64;
    let mut segment_refs: Vec<HandoffSegmentRef> = Vec::new();
    let mut store_ms = 0.0f64;
    let mut segments_seen = 0usize;
    let mut staged: Option<StagedStream> = None;
    let commit = loop {
        let (kind, header, body) = read_frame(&mut reader)?;
        match kind {
            frame_kind::PAGE => {
                let page: PageHeader =
                    serde_json::from_value(header).context("malformed page header")?;
                if digest_hex(&body) != page.blake3 {
                    bail!("page {} failed digest verification", page.index);
                }
                let stage = match staged.as_mut() {
                    Some(stage) => stage,
                    None => staged.insert(StagedStream::new(model)?),
                };
                if page.token_start != stage.imported_tokens {
                    bail!(
                        "page {} starts at token {} but {} tokens are staged",
                        page.index,
                        page.token_start,
                        stage.imported_tokens
                    );
                }
                stage.hasher.update(&body);
                // Import into the staging session while the sender is still
                // prefilling later chunks. Staged state cannot generate:
                // decode is gated on the commit record validating below.
                let import_started = Instant::now();
                stage
                    .session
                    .import_kv_page(&page.kv_desc, &body)
                    .with_context(|| format!("failed to stage KV page {}", page.index))?;
                stage.kv_attach_ms += elapsed_ms(import_started);
                stage.imported_tokens += page.token_count;
                if let Some(store) = store {
                    let put_started = Instant::now();
                    let (digest, _) = store
                        .put_segment(&body)
                        .context("receiver page write-behind failed")?;
                    store_ms += elapsed_ms(put_started);
                    segment_refs.push(HandoffSegmentRef {
                        index: page.index as u32,
                        offset: received_bytes,
                        bytes: body.len() as u64,
                        digest,
                        meta_json: Some(serde_json::to_string(&PageSegmentMeta {
                            kind: "kv-page".to_string(),
                            kv_desc: Some(page.kv_desc),
                            token_start: page.token_start,
                            token_count: page.token_count,
                        })?),
                    });
                }
                received_bytes += body.len() as u64;
                segments_seen += 1;
            }
            frame_kind::RECURRENT => {
                let recurrent: RecurrentHeader =
                    serde_json::from_value(header).context("malformed recurrent header")?;
                if digest_hex(&body) != recurrent.blake3 {
                    bail!("recurrent snapshot failed digest verification");
                }
                let stage = match staged.as_mut() {
                    Some(stage) => stage,
                    None => staged.insert(StagedStream::new(model)?),
                };
                stage.hasher.update(&body);
                if let Some(store) = store {
                    let put_started = Instant::now();
                    let (digest, _) = store
                        .put_segment(&body)
                        .context("receiver recurrent write-behind failed")?;
                    store_ms += elapsed_ms(put_started);
                    segment_refs.push(HandoffSegmentRef {
                        index: segments_seen as u32,
                        offset: received_bytes,
                        bytes: body.len() as u64,
                        digest,
                        meta_json: Some(serde_json::to_string(&PageSegmentMeta {
                            kind: "recurrent".to_string(),
                            kv_desc: None,
                            token_start: 0,
                            token_count: stage.imported_tokens,
                        })?),
                    });
                }
                received_bytes += body.len() as u64;
                segments_seen += 1;
                stage.recurrent = body;
            }
            frame_kind::SEGMENT => {
                let segment: SegmentHeader =
                    serde_json::from_value(header).context("malformed segment header")?;
                if segment.index != segments_seen {
                    bail!(
                        "segment {} arrived out of order (expected {})",
                        segment.index,
                        segments_seen
                    );
                }
                if segment.offset != received_bytes {
                    bail!(
                        "segment {} offset {} does not match received byte count {received_bytes}",
                        segment.index,
                        segment.offset,
                    );
                }
                match store {
                    Some(store) => {
                        let put_started = Instant::now();
                        let (digest, _) = store
                            .put_segment(&body)
                            .context("receiver segment write-behind failed")?;
                        store_ms += elapsed_ms(put_started);
                        if digest != segment.blake3 {
                            bail!("segment {} failed digest verification", segment.index);
                        }
                        segment_refs.push(HandoffSegmentRef {
                            index: segment.index as u32,
                            offset: segment.offset,
                            bytes: body.len() as u64,
                            digest,
                            meta_json: None,
                        });
                    }
                    None => {
                        if digest_hex(&body) != segment.blake3 {
                            bail!("segment {} failed digest verification", segment.index);
                        }
                        payload.extend_from_slice(&body);
                    }
                }
                received_bytes += body.len() as u64;
                segments_seen += 1;
            }
            frame_kind::COMMIT => {
                break serde_json::from_value::<CommitHeader>(header)
                    .context("malformed commit header")?;
            }
            other => bail!("unexpected frame kind {other} while receiving segments"),
        }
    };
    let transfer_receive_ms = elapsed_ms(receive_started);

    let mut committed_manifest = None;
    let outcome = if commit.streaming {
        (|| -> Result<ReceiverAttachOutcome> {
            let mut stage = staged
                .take()
                .context("streaming commit arrived before any staged pages")?;
            if stage.imported_tokens != commit.prompt_token_count {
                bail!(
                    "staged pages cover {} tokens but commit records {}",
                    stage.imported_tokens,
                    commit.prompt_token_count
                );
            }
            if segments_seen != commit.segment_count {
                bail!(
                    "commit expected {} stream segments but {segments_seen} arrived",
                    commit.segment_count
                );
            }
            if stage.recurrent.len() as u64 != commit.recurrent_bytes {
                bail!(
                    "commit records {} recurrent bytes but {} arrived",
                    commit.recurrent_bytes,
                    stage.recurrent.len()
                );
            }
            let running_digest = format!("{}", stage.hasher.finalize().to_hex());
            if running_digest != commit.payload_blake3 {
                bail!("streamed payload failed commit digest verification");
            }
            let residual_started = Instant::now();
            if stage.recurrent.is_empty() {
                stage
                    .session
                    .set_position(commit.prompt_token_count)
                    .context("failed to finalize staged position")?;
            } else {
                let recurrent = std::mem::take(&mut stage.recurrent);
                stage
                    .session
                    .import_recurrent_state_for_token_count(&recurrent, commit.prompt_token_count)
                    .context("failed to import staged recurrent state")?;
            }
            if let Some(store) = store {
                let commit_started = Instant::now();
                let manifest = manifest_from_commit(
                    &commit,
                    std::mem::take(&mut segment_refs),
                    local_state_identity.to_string(),
                    effective_payload_kind(args),
                    Vec::new(),
                )?;
                store
                    .commit(&manifest)
                    .context("receiver streaming manifest commit failed")?;
                store_ms += elapsed_ms(commit_started);
                committed_manifest = Some(manifest);
            }
            let mut timings = RemoteHandoffReceiverTimings {
                kv_attach_ms: stage.kv_attach_ms,
                ..RemoteHandoffReceiverTimings::default()
            };
            timings.attach_residual_ms = elapsed_ms(residual_started);
            decode_and_baseline(model, args, &commit, stage.session, timings)
        })()
    } else {
        (|| -> Result<ReceiverAttachOutcome> {
            let payload_bytes = match store {
                Some(store) => {
                    let commit_started = Instant::now();
                    let manifest = manifest_from_commit(
                        &commit,
                        std::mem::take(&mut segment_refs),
                        local_state_identity.to_string(),
                        payload_kind_name(args.state_payload_kind),
                        Vec::new(),
                    )?;
                    store
                        .commit(&manifest)
                        .context("receiver manifest commit failed")?;
                    // Import from the store, not the socket buffer: the disk
                    // backend is the path under test, and `assemble` re-verifies
                    // every segment plus the whole-payload digest.
                    let assembled = store
                        .assemble(&manifest)
                        .context("receiver manifest assembly failed")?;
                    store_ms += elapsed_ms(commit_started);
                    committed_manifest = Some(manifest);
                    assembled
                }
                None => {
                    validate_commit(&commit, segments_seen, &payload)?;
                    std::mem::take(&mut payload)
                }
            };
            run_receiver_attach_and_decode(model, args, &commit, &payload_bytes)
        })()
    };
    let (attach, reason) = match outcome {
        Ok(attach) => (Some(attach), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };

    let (verify, _) = read_frame_expect::<VerifyHeader>(&mut reader, frame_kind::VERIFY)?;
    if let (Some(store), Some(mut manifest)) = (store, committed_manifest) {
        // Second-phase manifest update: record the reference continuation so
        // an offline restore can self-verify. Not part of the handoff's
        // success criteria.
        manifest.expected_tokens = verify.source_tokens.clone();
        if let Err(error) = store.commit(&manifest) {
            eprintln!("remote-handoff: failed to record expected tokens in manifest: {error:#}");
        }
    }
    let (restored_tokens, baseline_tokens, mut timings) = match attach {
        Some(attach) => (
            attach.restored_tokens,
            attach.baseline_tokens,
            attach.timings,
        ),
        None => (
            Vec::new(),
            Vec::new(),
            RemoteHandoffReceiverTimings::default(),
        ),
    };
    timings.model_load_ms = model_load_ms;
    timings.transfer_receive_ms = transfer_receive_ms;
    timings.store_ms = store_ms;
    let tokens_match = !restored_tokens.is_empty() && restored_tokens == verify.source_tokens;
    let baseline_matches = commit
        .run_baseline
        .then(|| !baseline_tokens.is_empty() && baseline_tokens == verify.source_tokens);
    let ok = reason.is_none();
    write_frame(
        &mut writer,
        frame_kind::RESULT,
        &ResultHeader {
            ok,
            reason: reason.clone(),
            restored_tokens: restored_tokens.clone(),
            baseline_tokens,
            tokens_match,
            baseline_matches,
            timings: timings.clone(),
        },
        &[],
    )?;
    writer.flush().context("failed to flush result frame")?;

    let matches = ok && tokens_match && baseline_matches.unwrap_or(true);
    let report = RemoteHandoffReport {
        mode: "remote-handoff",
        status: status(matches),
        role: "recv",
        model_identity: model_identity.clone(),
        matches,
        tokens_match,
        baseline_matches,
        state_payload_kind: payload_kind_name(args.state_payload_kind),
        prompt_token_count: commit.prompt_token_count as usize,
        decode_token_count: hello.decode_token_count,
        continuation_token: commit.continuation_token,
        source_tokens: verify.source_tokens,
        restored_tokens,
        state_bytes: received_bytes as usize,
        state_bytes_per_prompt_token: received_bytes as f64
            / (commit.prompt_token_count as f64).max(1.0),
        kv_bytes: commit.kv_bytes as usize,
        recurrent_bytes: commit.recurrent_bytes as usize,
        segment_count: segments_seen,
        segment_bytes: args.segment_bytes,
        payload_digest: commit.payload_blake3.clone(),
        model_load_ms,
        tokenize_ms: 0.0,
        source_prefill_ms: 0.0,
        state_export_ms: 0.0,
        transfer_ms: transfer_receive_ms,
        transfer_gbps: transfer_gbps(received_bytes as usize, transfer_receive_ms),
        source_decode_ms: 0.0,
        store_ms: None,
        overlap_wall_ms: None,
        receiver: timings,
        ttft_disaggregated_ms: 0.0,
        ttft_local_ms: None,
        ttft_speedup: None,
    };
    emit_report(&report, report_out)?;
    if let Some(reason) = reason {
        bail!("remote handoff receive failed: {reason}");
    }
    Ok(matches)
}

/// Reattach continuation state purely from the local L3 store — no network,
/// no exporter. Restart survival: any manifest the store holds can be
/// imported into a fresh process and decoded, and when the manifest records
/// the exporter's continuation the run self-verifies determinism.
fn run_restore(args: RemoteHandoffArgs) -> Result<()> {
    let store_dir = args
        .store_dir
        .clone()
        .context("--role restore requires --store-dir")?;
    let store = HandoffSegmentStore::open(&store_dir, args.store_budget_bytes)?;
    let key = match args.manifest.clone() {
        Some(key) => key,
        None => store
            .list_manifests()?
            .into_iter()
            .next()
            .with_context(|| format!("store at {} holds no manifests", store_dir.display()))?,
    };
    let manifest = store.load_manifest(&key)?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let local_state_identity = state_identity_for(&args, &model_identity);
    if manifest.state_identity != local_state_identity {
        bail!(
            "manifest {key} was produced under state identity {} but this configuration is {local_state_identity}",
            manifest.state_identity
        );
    }

    let model_load_started = Instant::now();
    let model = open_full_model(&args, effective_lane_count(&args))?;
    let model_load_ms = elapsed_ms(model_load_started);

    let commit = CommitHeader {
        segment_count: manifest.segments.len(),
        total_bytes: manifest.total_bytes,
        payload_blake3: manifest.payload_digest.clone(),
        prompt_token_count: manifest.token_count,
        continuation_token: manifest.continuation_token,
        kv_bytes: manifest.kv_bytes,
        recurrent_bytes: manifest.recurrent_bytes,
        kv_desc: manifest
            .kv_desc_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .context("malformed kv desc in manifest")?,
        prefix_tokens: Vec::new(),
        run_baseline: false,
        streaming: manifest.payload_kind == "kv-page-stream",
        page_count: 0,
    };
    let (attach, state_bytes, store_ms) = if commit.streaming {
        // Page-stream manifests restore page by page, exactly as the
        // network path staged them.
        let restore_started = Instant::now();
        let mut stage = StagedStream::new(&model)?;
        let mut recurrent: Vec<u8> = Vec::new();
        let mut restored_bytes = 0usize;
        for segment in &manifest.segments {
            let meta: PageSegmentMeta = serde_json::from_str(
                segment
                    .meta_json
                    .as_deref()
                    .context("page-stream segment is missing metadata")?,
            )
            .context("malformed page-stream segment metadata")?;
            let bytes = store.read_segment(&segment.digest)?;
            restored_bytes += bytes.len();
            match meta.kind.as_str() {
                "kv-page" => {
                    let kv_desc = meta
                        .kv_desc
                        .context("kv-page segment is missing its descriptor")?;
                    if meta.token_start != stage.imported_tokens {
                        bail!(
                            "page-stream segment starts at token {} but {} tokens are staged",
                            meta.token_start,
                            stage.imported_tokens
                        );
                    }
                    stage
                        .session
                        .import_kv_page(&kv_desc, &bytes)
                        .context("failed to restore staged KV page")?;
                    stage.imported_tokens += meta.token_count;
                }
                "recurrent" => recurrent = bytes,
                other => bail!("unknown page-stream segment kind {other}"),
            }
        }
        if stage.imported_tokens != manifest.token_count {
            bail!(
                "page-stream manifest covers {} tokens but records {}",
                stage.imported_tokens,
                manifest.token_count
            );
        }
        if recurrent.is_empty() {
            stage
                .session
                .set_position(manifest.token_count)
                .context("failed to finalize restored position")?;
        } else {
            stage
                .session
                .import_recurrent_state_for_token_count(&recurrent, manifest.token_count)
                .context("failed to restore recurrent state")?;
        }
        let store_ms = elapsed_ms(restore_started);
        let timings = RemoteHandoffReceiverTimings::default();
        (
            decode_and_baseline(&model, &args, &commit, stage.session, timings)?,
            restored_bytes,
            store_ms,
        )
    } else {
        let assemble_started = Instant::now();
        let payload = store
            .assemble(&manifest)
            .context("manifest assembly failed")?;
        let store_ms = elapsed_ms(assemble_started);
        let state_bytes = payload.len();
        (
            run_receiver_attach_and_decode(&model, &args, &commit, &payload)?,
            state_bytes,
            store_ms,
        )
    };

    let expected = &manifest.expected_tokens;
    let compared = expected.len().min(attach.restored_tokens.len());
    let tokens_match = compared > 0 && attach.restored_tokens[..compared] == expected[..compared];
    let matches = expected.is_empty() || tokens_match;
    let mut timings = attach.timings;
    timings.model_load_ms = model_load_ms;
    timings.store_ms = store_ms;
    let report = RemoteHandoffReport {
        mode: "remote-handoff",
        status: status(matches),
        role: "restore",
        model_identity,
        matches,
        tokens_match,
        baseline_matches: None,
        state_payload_kind: effective_payload_kind(&args),
        prompt_token_count: manifest.token_count as usize,
        decode_token_count: args.decode_tokens,
        continuation_token: manifest.continuation_token,
        source_tokens: expected.clone(),
        restored_tokens: attach.restored_tokens,
        state_bytes,
        state_bytes_per_prompt_token: state_bytes as f64 / (manifest.token_count as f64).max(1.0),
        kv_bytes: manifest.kv_bytes as usize,
        recurrent_bytes: manifest.recurrent_bytes as usize,
        segment_count: manifest.segments.len(),
        segment_bytes: args.segment_bytes,
        payload_digest: manifest.payload_digest.clone(),
        model_load_ms,
        tokenize_ms: 0.0,
        source_prefill_ms: 0.0,
        state_export_ms: 0.0,
        transfer_ms: 0.0,
        transfer_gbps: 0.0,
        source_decode_ms: 0.0,
        store_ms: Some(store_ms),
        overlap_wall_ms: None,
        receiver: timings,
        ttft_disaggregated_ms: 0.0,
        ttft_local_ms: None,
        ttft_speedup: None,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(matches, args.allow_mismatch)?;
    Ok(())
}

struct ReceiverAttachOutcome {
    restored_tokens: Vec<i32>,
    baseline_tokens: Vec<i32>,
    timings: RemoteHandoffReceiverTimings,
}

/// A session staged from streamed KV pages. Holds no generation authority:
/// callers only decode after the commit record validates, and dropping the
/// struct discards uncommitted state.
struct StagedStream {
    session: StageSession,
    imported_tokens: u64,
    hasher: blake3::Hasher,
    recurrent: Vec<u8>,
    kv_attach_ms: f64,
}

impl StagedStream {
    fn new(model: &StageModel) -> Result<Self> {
        Ok(Self {
            session: model
                .create_session()
                .context("failed to create staging session")?,
            imported_tokens: 0,
            hasher: blake3::Hasher::new(),
            recurrent: Vec::new(),
            kv_attach_ms: 0.0,
        })
    }
}

fn run_receiver_attach_and_decode(
    model: &StageModel,
    args: &RemoteHandoffArgs,
    commit: &CommitHeader,
    payload: &[u8],
) -> Result<ReceiverAttachOutcome> {
    let attach_started = Instant::now();
    let mut session = model
        .create_session()
        .context("failed to create receiver session")?;
    import_state_payload(&mut session, args, commit, payload)
        .context("failed to import handoff state")?;
    let kv_attach_ms = elapsed_ms(attach_started);
    let timings = RemoteHandoffReceiverTimings {
        kv_attach_ms,
        ..RemoteHandoffReceiverTimings::default()
    };
    decode_and_baseline(model, args, commit, session, timings)
}

fn decode_and_baseline(
    model: &StageModel,
    args: &RemoteHandoffArgs,
    commit: &CommitHeader,
    mut session: StageSession,
    mut timings: RemoteHandoffReceiverTimings,
) -> Result<ReceiverAttachOutcome> {
    let first_decode_started = Instant::now();
    let first = session
        .decode_step(commit.continuation_token)
        .context("receiver first decode failed")?;
    let first_decode_ms = elapsed_ms(first_decode_started);
    let mut restored_tokens = vec![first];
    let decode_started = Instant::now();
    if args.decode_tokens > 1 {
        restored_tokens.extend(greedy_decode(&mut session, first, args.decode_tokens - 1)?);
    }
    timings.first_decode_ms = first_decode_ms;
    timings.decode_ms = first_decode_ms + elapsed_ms(decode_started);
    drop(session);

    let mut baseline_tokens = Vec::new();
    if commit.run_baseline {
        if commit.prefix_tokens.is_empty() {
            bail!("baseline requested but commit carried no prefix tokens");
        }
        let mut baseline = model
            .create_session()
            .context("failed to create baseline session")?;
        let baseline_prefill_started = Instant::now();
        baseline
            .prefill_chunked(&commit.prefix_tokens)
            .context("baseline prefill failed")?;
        timings.baseline_prefill_ms = elapsed_ms(baseline_prefill_started);
        let baseline_first_started = Instant::now();
        let first = baseline
            .decode_step(commit.continuation_token)
            .context("baseline first decode failed")?;
        timings.baseline_first_decode_ms = elapsed_ms(baseline_first_started);
        baseline_tokens.push(first);
        if args.decode_tokens > 1 {
            baseline_tokens.extend(greedy_decode(&mut baseline, first, args.decode_tokens - 1)?);
        }
    }

    Ok(ReceiverAttachOutcome {
        restored_tokens,
        baseline_tokens,
        timings,
    })
}

fn validate_hello(hello: &HelloHeader, args: &RemoteHandoffArgs, model_id: &str) -> Result<()> {
    if hello.protocol_version != PROTOCOL_VERSION {
        bail!(
            "protocol version mismatch: sender {} vs receiver {PROTOCOL_VERSION}",
            hello.protocol_version
        );
    }
    if hello.model_id != model_id {
        bail!(
            "model mismatch: sender serves {} but receiver serves {model_id}",
            hello.model_id
        );
    }
    if hello.layer_end != args.runtime.layer_end {
        bail!(
            "layer_end mismatch: sender {} vs receiver {}",
            hello.layer_end,
            args.runtime.layer_end
        );
    }
    if hello.ctx_size != args.runtime.ctx_size {
        bail!(
            "ctx_size mismatch: sender {} vs receiver {}",
            hello.ctx_size,
            args.runtime.ctx_size
        );
    }
    if hello.state_payload_kind != effective_payload_kind(args) {
        bail!(
            "state payload kind mismatch: sender {} vs receiver {}",
            hello.state_payload_kind,
            effective_payload_kind(args)
        );
    }
    if hello.decode_token_count != args.decode_tokens {
        bail!(
            "decode token count mismatch: sender {} vs receiver {}",
            hello.decode_token_count,
            args.decode_tokens
        );
    }
    if hello.lane_count != effective_lane_count(args) {
        bail!(
            "lane count mismatch: sender {} vs receiver {}",
            hello.lane_count,
            effective_lane_count(args)
        );
    }
    Ok(())
}

fn validate_state_identity(hello: &HelloHeader, local_state_identity: &str) -> Result<()> {
    if hello.state_identity != local_state_identity {
        bail!(
            "state identity mismatch: sender {} vs receiver {local_state_identity} — the \
             numerical configurations differ even though the per-field checks passed",
            hello.state_identity
        );
    }
    Ok(())
}

fn validate_commit(commit: &CommitHeader, segments_seen: usize, payload: &[u8]) -> Result<()> {
    if commit.segment_count != segments_seen {
        bail!(
            "commit expected {} segments but {} arrived",
            commit.segment_count,
            segments_seen
        );
    }
    if commit.total_bytes != payload.len() as u64 {
        bail!(
            "commit expected {} bytes but {} arrived",
            commit.total_bytes,
            payload.len()
        );
    }
    if digest_hex(payload) != commit.payload_blake3 {
        bail!("assembled payload failed commit digest verification");
    }
    Ok(())
}

fn payload_kind_name(kind: StatePayloadKind) -> &'static str {
    match kind {
        StatePayloadKind::ResidentKv => "resident-kv",
        StatePayloadKind::FullState => "full-state",
        StatePayloadKind::RecurrentOnly => "recurrent-only",
        StatePayloadKind::KvRecurrent => "kv-recurrent",
    }
}

fn transfer_gbps(bytes: usize, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0) / (elapsed_ms / 1000.0) / 1e9
}

fn write_frame(
    writer: &mut impl Write,
    kind: u8,
    header: &impl Serialize,
    payload: &[u8],
) -> Result<()> {
    let header_bytes = serde_json::to_vec(header).context("failed to encode frame header")?;
    if header_bytes.len() as u64 > MAX_HEADER_BYTES {
        bail!("frame header of {} bytes exceeds limit", header_bytes.len());
    }
    writer.write_all(&[kind])?;
    writer.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&header_bytes)?;
    writer.write_all(&(payload.len() as u64).to_le_bytes())?;
    writer.write_all(payload)?;
    Ok(())
}

fn read_frame(reader: &mut impl Read) -> Result<(u8, serde_json::Value, Vec<u8>)> {
    let mut kind = [0u8; 1];
    reader
        .read_exact(&mut kind)
        .context("handoff stream closed while reading frame kind")?;
    let mut header_len = [0u8; 4];
    reader.read_exact(&mut header_len)?;
    let header_len = u32::from_le_bytes(header_len) as u64;
    if header_len > MAX_HEADER_BYTES {
        bail!("frame header of {header_len} bytes exceeds limit");
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    reader.read_exact(&mut header_bytes)?;
    let header = serde_json::from_slice(&header_bytes).context("malformed frame header")?;
    let mut payload_len = [0u8; 8];
    reader.read_exact(&mut payload_len)?;
    let payload_len = u64::from_le_bytes(payload_len);
    if payload_len > MAX_SEGMENT_BYTES {
        bail!("frame payload of {payload_len} bytes exceeds limit");
    }
    let mut payload = vec![0u8; payload_len as usize];
    reader.read_exact(&mut payload)?;
    Ok((kind[0], header, payload))
}

fn read_frame_expect<T: DeserializeOwned>(
    reader: &mut impl Read,
    expected_kind: u8,
) -> Result<(T, Vec<u8>)> {
    let (kind, header, payload) = read_frame(reader)?;
    if kind != expected_kind {
        bail!("expected frame kind {expected_kind}, got {kind}");
    }
    Ok((
        serde_json::from_value(header).context("malformed frame header for expected kind")?,
        payload,
    ))
}

fn digest_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[cfg(test)]
mod state_identity_tests {
    use super::*;
    use crate::cli::{
        FlashAttentionArg, OutputArgs, RemoteHandoffArgs, RemoteHandoffRole, RuntimeArgs,
        StageLoadMode, StatePayloadKind,
    };

    fn args_for(model: std::path::PathBuf) -> RemoteHandoffArgs {
        RemoteHandoffArgs {
            runtime: RuntimeArgs {
                model,
                model_id: None,
                stage_model: None,
                stage_load_mode: StageLoadMode::RuntimeSlice,
                layer_end: 28,
                ctx_size: 2048,
                n_gpu_layers: 99,
                n_batch: None,
                n_ubatch: None,
                prompt: "Hello".to_string(),
                flash_attn: FlashAttentionArg::Auto,
            },
            output: OutputArgs { report_out: None },
            role: RemoteHandoffRole::Send,
            listen: "0.0.0.0:19081".parse().expect("addr"),
            peer: None,
            state_payload_kind: StatePayloadKind::FullState,
            prefix_token_count: None,
            decode_tokens: 16,
            segment_bytes: 8 * 1024 * 1024,
            baseline: false,
            runtime_lane_count: None,
            handshake_timeout_secs: 600,
            accept_count: 1,
            store_dir: None,
            store_budget_bytes: 0,
            manifest: None,
            streaming: false,
            stream_chunk_tokens: 512,
            allow_mismatch: false,
        }
    }

    /// Two different local files behind the same display model id must not
    /// share a handoff identity — the content digest, not the name, decides.
    #[test]
    fn different_file_contents_behind_one_model_id_change_identity() {
        let dir = std::env::temp_dir()
            .join("skippy-remote-handoff-identity-tests")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&dir).expect("temp dir");
        let first = dir.join("model-a.gguf");
        let second = dir.join("model-b.gguf");
        std::fs::write(&first, b"weights generation one").expect("write first");
        std::fs::write(&second, b"weights generation two").expect("write second");
        let identity = model_artifact::ModelIdentity::from_model_id("org/model:Q4_K_M");

        let first_identity = state_identity_for(&args_for(first.clone()), &identity);
        let second_identity = state_identity_for(&args_for(second), &identity);
        assert_ne!(
            first_identity, second_identity,
            "same model id over different file contents must not share identity"
        );

        // Stable for the same content (memoized path re-queried).
        assert_eq!(
            first_identity,
            state_identity_for(&args_for(first), &identity)
        );
    }
}
