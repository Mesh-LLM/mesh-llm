use std::path::Path;

use model_artifact::ModelIdentity;
use skippy_cache::{ExactStateIdentityParams, exact_state_identity};

use crate::cli::{FlashAttentionArg, RemoteHandoffArgs};

use super::{effective_lane_count, protocol_load_mode};

pub(super) fn effective_payload_kind(args: &RemoteHandoffArgs) -> &'static str {
    if args.streaming {
        "kv-page-stream"
    } else {
        super::payload_kind_name(args.state_payload_kind)
    }
}

/// Content digest of the served artifact, memoized per path: two harness
/// processes serving different local GGUFs behind the same display model id
/// must never share a state identity. Directories (layer-package refs) are
/// not hashed here; their identity rides the package manifest via the
/// model-identity fields.
fn artifact_sha256_cached(path: &Path) -> Option<String> {
    use sha2::Digest as _;
    use std::collections::HashMap;
    use std::io::Read as _;
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
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).ok()?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Some(hex::encode(hasher.finalize()))
    })();
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(path.to_path_buf(), digest.clone());
    digest
}

/// Numerical identity of the state this harness produces or accepts.
pub(super) fn state_identity_for(args: &RemoteHandoffArgs, identity: &ModelIdentity) -> String {
    let source_model_sha256 = artifact_sha256_cached(&args.runtime.model);
    exact_state_identity(&ExactStateIdentityParams {
        model_id: &identity.model_id,
        model_revision: identity.source_revision.as_deref(),
        model_file: identity.source_file.as_deref(),
        manifest_sha256: None,
        source_model_sha256: source_model_sha256.as_deref(),
        package_ref: None,
        load_mode: protocol_load_mode(args.runtime.stage_load_mode),
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

fn protocol_flash_attn_type(value: FlashAttentionArg) -> skippy_protocol::FlashAttentionType {
    match value {
        FlashAttentionArg::Auto => skippy_protocol::FlashAttentionType::Auto,
        FlashAttentionArg::Disabled => skippy_protocol::FlashAttentionType::Disabled,
        FlashAttentionArg::Enabled => skippy_protocol::FlashAttentionType::Enabled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{OutputArgs, RemoteHandoffRole, RuntimeArgs, StageLoadMode, StatePayloadKind};

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
            listen: "127.0.0.1:19081".parse().expect("addr"),
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
        let identity = ModelIdentity::from_model_id("org/model:Q4_K_M");

        let first_identity = state_identity_for(&args_for(first.clone()), &identity);
        let second_identity = state_identity_for(&args_for(second), &identity);
        assert_ne!(first_identity, second_identity);
        assert_eq!(
            first_identity,
            state_identity_for(&args_for(first), &identity)
        );
    }

    #[test]
    fn load_mode_changes_handoff_identity() {
        let dir = std::env::temp_dir()
            .join("skippy-remote-handoff-load-mode-tests")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&dir).expect("temp dir");
        let model = dir.join("model.gguf");
        std::fs::write(&model, b"weights").expect("write model");
        let identity = ModelIdentity::from_model_id("org/model:Q4_K_M");
        let runtime = args_for(model.clone());
        let mut artifact = args_for(model);
        artifact.runtime.stage_load_mode = StageLoadMode::ArtifactSlice;

        assert_ne!(
            state_identity_for(&runtime, &identity),
            state_identity_for(&artifact, &identity)
        );
    }
}
