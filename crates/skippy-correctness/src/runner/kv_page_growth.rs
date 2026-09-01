//! Does a growing prefix append to the exported KV page, or re-lay it out?
//!
//! Issue #1576 §13.4 gates the disk tier at 1.2x write amplification: physical
//! bytes written must stay close to the newly committed unique payload bytes.
//! The L3 store cuts content-addressed segments at fixed byte offsets, so that
//! gate holds only if turn N+1's payload keeps turn N's bytes at the same
//! offsets. If the runtime lays the page out layer-major, every layer block
//! after the first shifts as tokens are added, every downstream segment
//! re-digests, and each turn rewrites the whole prefix.
//!
//! This runner answers that empirically before the on-disk format is fixed: it
//! grows one session a turn at a time, exports the full page after each turn,
//! and reports how much of the previous export survives — byte-identical
//! prefix, and reusable segments at the store's segment size. It also probes
//! the alternative design, exporting only the new token window
//! (`token_start > 0`), which would make growth append-only at the manifest
//! level regardless of internal layout.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::{
    GGML_TYPE_F16, MtpSource, RuntimeConfig, RuntimeKvPage, StageModel, StageSession,
};

use super::{
    stage_execution::{runtime_flash_attn, runtime_load_mode},
    state_handoff::state_handoff_tokens,
};
use crate::cli::KvPageGrowthArgs;

#[derive(Debug, Serialize)]
struct TurnReport {
    turn: usize,
    token_count: u64,
    payload_bytes: u64,
    /// Bytes of this payload identical to the previous turn's from offset 0.
    common_prefix_bytes: u64,
    segments: usize,
    /// Segments whose digest was already stored by an earlier turn.
    reused_segments: usize,
    /// What the store would physically write for this turn.
    physical_bytes: u64,
    /// What an ideal append-only layout would write: the payload growth.
    ideal_bytes: u64,
    /// physical / ideal. The §13.4 gate is 1.2.
    amplification: f64,
    /// Whether exporting just this turn's token window succeeded.
    windowed_export: WindowedProbe,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
enum WindowedProbe {
    Ok { payload_bytes: u64 },
    Unsupported { error: String },
    Skipped,
}

#[derive(Debug, Serialize)]
struct GrowthReport {
    model: String,
    segment_bytes: u64,
    base_tokens: usize,
    turn_tokens: usize,
    turns: Vec<TurnReport>,
    /// Worst per-turn amplification: the number the gate has to clear.
    max_amplification: f64,
    /// True when every turn kept the previous payload byte-identical at offset 0.
    append_only: bool,
}

pub fn kv_page_growth(args: KvPageGrowthArgs) -> Result<()> {
    if args.turns == 0 {
        bail!("--turns must be at least 1");
    }
    if args.turn_tokens == 0 {
        bail!("--turn-tokens must be at least 1");
    }
    if args.segment_bytes == 0 {
        bail!("--segment-bytes must be greater than zero");
    }
    let total_tokens = args
        .base_tokens
        .checked_add(args.turns.saturating_mul(args.turn_tokens))
        .context("token budget overflow")?;
    if u64::try_from(total_tokens)? > u64::from(args.runtime.ctx_size) {
        bail!(
            "prefix of {total_tokens} tokens exceeds --ctx-size {}; raise the context or lower --turns",
            args.runtime.ctx_size
        );
    }

    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.runtime.layer_end,
        ctx_size: args.runtime.ctx_size,
        lane_count: 1,
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
        include_output: false,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: true,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: runtime_flash_attn(args.runtime.flash_attn),
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let model = StageModel::open(&args.runtime.model, &config)
        .context("failed to open stage model for KV page growth probe")?;
    let tokens = state_handoff_tokens(&model, &args.runtime.prompt, Some(total_tokens))
        .context("failed to build growth prefix")?;
    if tokens.len() < total_tokens {
        bail!(
            "prefix expansion produced {} tokens, needed {total_tokens}",
            tokens.len()
        );
    }
    let layer_end = i32::try_from(args.runtime.layer_end).context("layer_end exceeds i32")?;
    let mut session = model
        .create_session()
        .context("failed to create growth probe session")?;

    let mut seen_digests: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut previous_payload: Vec<u8> = Vec::new();
    let mut turns = Vec::with_capacity(args.turns);
    let mut cursor = 0usize;

    for turn in 0..=args.turns {
        let next = if turn == 0 {
            args.base_tokens
        } else {
            cursor + args.turn_tokens
        };
        session
            .prefill_chunked(&tokens[cursor..next])
            .with_context(|| format!("prefill failed on turn {turn}"))?;
        let previous_cursor = cursor;
        cursor = next;

        let page = export_page(&mut session, layer_end, 0, cursor as u64)
            .with_context(|| format!("full-prefix KV export failed on turn {turn}"))?;
        let payload = page.payload;

        let common_prefix_bytes = common_prefix_len(&previous_payload, &payload) as u64;
        let mut segments = 0usize;
        let mut reused_segments = 0usize;
        let mut physical_bytes = 0u64;
        for chunk in payload.chunks(usize::try_from(args.segment_bytes)?) {
            segments += 1;
            let digest = blake3::hash(chunk).to_hex().to_string();
            if seen_digests.insert(digest) {
                physical_bytes += chunk.len() as u64;
            } else {
                reused_segments += 1;
            }
        }
        let ideal_bytes = (payload.len() as u64).saturating_sub(previous_payload.len() as u64);
        // Turn 0 commits the whole base prefix, which is genuinely new data.
        let ideal_bytes = if turn == 0 {
            payload.len() as u64
        } else {
            ideal_bytes
        };
        let amplification = if ideal_bytes == 0 {
            f64::INFINITY
        } else {
            physical_bytes as f64 / ideal_bytes as f64
        };

        // The alternative design: store only the new window rather than
        // re-cutting the whole prefix. Turn 0 has no preceding window.
        let windowed_export = if turn == 0 {
            WindowedProbe::Skipped
        } else {
            match export_page(
                &mut session,
                layer_end,
                previous_cursor as u64,
                (cursor - previous_cursor) as u64,
            ) {
                Ok(page) => WindowedProbe::Ok {
                    payload_bytes: page.payload.len() as u64,
                },
                Err(error) => WindowedProbe::Unsupported {
                    error: format!("{error:#}"),
                },
            }
        };

        println!(
            "turn={turn} tokens={cursor} payload={} common_prefix={common_prefix_bytes} \
             segments={segments} reused={reused_segments} physical={physical_bytes} \
             ideal={ideal_bytes} amplification={amplification:.2}",
            payload.len()
        );

        turns.push(TurnReport {
            turn,
            token_count: cursor as u64,
            payload_bytes: payload.len() as u64,
            common_prefix_bytes,
            segments,
            reused_segments,
            physical_bytes,
            ideal_bytes,
            amplification,
            windowed_export,
        });
        previous_payload = payload;
    }

    let append_only = turns.windows(2).all(|pair| {
        let [previous, current] = pair else {
            return true;
        };
        current.common_prefix_bytes >= previous.payload_bytes
    });
    let max_amplification = turns
        .iter()
        .skip(1)
        .map(|turn| turn.amplification)
        .fold(0.0_f64, f64::max);

    let report = GrowthReport {
        model: args.runtime.model.display().to_string(),
        segment_bytes: args.segment_bytes,
        base_tokens: args.base_tokens,
        turn_tokens: args.turn_tokens,
        turns,
        max_amplification,
        append_only,
    };
    println!(
        "kv_page_growth append_only={append_only} max_amplification={max_amplification:.2} \
         gate=1.20 verdict={}",
        if append_only && max_amplification <= 1.2 {
            "fixed-offset chunking holds"
        } else {
            "fixed-offset chunking fails the gate"
        }
    );
    if let Some(path) = args.json.as_deref() {
        let rendered = serde_json::to_string_pretty(&report).context("render growth report")?;
        std::fs::write(path, rendered)
            .with_context(|| format!("write growth report {}", path.display()))?;
    }
    Ok(())
}

fn export_page(
    session: &mut StageSession,
    layer_end: i32,
    token_start: u64,
    token_count: u64,
) -> Result<RuntimeKvPage> {
    session.export_kv_page(0, layer_end, token_start, token_count)
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}
