use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
struct Fixture {
    root: PathBuf,
    config: Value,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "nemotron-profile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let config = serde_json::json!({"model_type":"nemotron_h_moe","num_nextn_predict_layers":1,"num_hidden_layers":3,"hybrid_override_pattern":"M*E","hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"moe_intermediate_size":16,"n_routed_experts":2,"num_experts_per_tok":1,"moe_shared_expert_intermediate_size":8,"n_shared_experts":1,"n_group":1,"norm_topk_prob":true,"routed_scaling_factor":1.0,"layer_norm_epsilon":0.00001,"num_heads":2,"mamba_head_dim":4,"n_groups":2,"conv_kernel":4,"ssm_state_size":8,"vocab_size":2});
        fs::write(root.join("tokenizer.json"),serde_json::to_vec(&serde_json::json!({"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]},"decoder":{"type":"ByteLevel"}})).unwrap()).unwrap();
        let fixture = Self { root, config };
        fixture.repin();
        fixture
    }
    fn profile(&self) -> PathBuf {
        self.root.join("profile.json")
    }
    fn repin(&self) {
        fs::write(
            self.root.join("config.json"),
            serde_json::to_vec(&self.config).unwrap(),
        )
        .unwrap();
        let pin = |name: &str| digest(&fs::read(self.root.join(name)).unwrap());
        fs::write(self.profile(),serde_json::to_vec(&serde_json::json!({"schema_version":1,"config_sha256":pin("config.json"),"tokenizer_sha256":pin("tokenizer.json"),"tokenizer_config_sha256":null,"chat_template_sha256":null,"pre":"llama-bpe"})).unwrap()).unwrap();
    }
    fn prepare(&self) -> Result<(Vec<GgufKv>, TensorNameMap)> {
        super::prepare(&self.root, &self.profile(), 7)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove owned metadata fixture");
    }
}
#[test]
fn nemotron_bound_profile_emits_hybrid_nextn_ssm_and_padded_bpe_metadata() {
    let fixture = Fixture::new();
    let (metadata, map) = fixture.prepare().unwrap();
    assert!(matches!(
        map,
        TensorNameMap::NemotronHMoeMtp { layer_start: 3 }
    ));
    assert!(metadata.contains(&GgufKv::u32("nemotron_h_moe.block_count", 4)));
    assert!(metadata.contains(&GgufKv::array_u32(
        "nemotron_h_moe.attention.head_count_kv",
        vec![0, 1, 0, 1]
    )));
    assert!(metadata.contains(&GgufKv::array_u32(
        "nemotron_h_moe.feed_forward_length",
        vec![0, 0, 16, 16]
    )));
    assert!(metadata.contains(&GgufKv::u32("nemotron_h_moe.ssm.inner_size", 8)));
    assert!(metadata.contains(&GgufKv::string("tokenizer.ggml.pre", "llama-bpe")));
    let tokens = metadata
        .iter()
        .find_map(|kv| match kv {
            GgufKv::ArrayString { key, value } if key == "tokenizer.ggml.tokens" => Some(value),
            _ => None,
        })
        .unwrap();
    assert_eq!(tokens.len(), 8);
    assert_eq!(&tokens[..2], &["a", "b"]);
    assert!(metadata.contains(&GgufKv::string(
        "skippy.convert.tokenizer_profile_sha256",
        &digest(&fs::read(fixture.profile()).unwrap())
    )));
}
#[test]
fn nemotron_profile_refuses_byte_drift_unbound_optional_and_unknown_pre() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("tokenizer.json"), b"{}").unwrap();
    assert!(
        fixture
            .prepare()
            .unwrap_err()
            .to_string()
            .contains("pin mismatch")
    );
    fixture.repin();
    fs::write(fixture.root.join("chat_template.jinja"), b"unbound").unwrap();
    assert!(
        fixture
            .prepare()
            .unwrap_err()
            .to_string()
            .contains("unbound optional")
    );
    fs::remove_file(fixture.root.join("chat_template.jinja")).unwrap();
    let mut profile: Value = serde_json::from_slice(&fs::read(fixture.profile()).unwrap()).unwrap();
    profile["pre"] = serde_json::json!("nemotron-guessed");
    fs::write(fixture.profile(), serde_json::to_vec(&profile).unwrap()).unwrap();
    assert!(fixture.prepare().is_err());
}
#[test]
fn nemotron_metadata_refuses_multinextn_pattern_alias_and_ssm_overflow() {
    let mut fixture = Fixture::new();
    let original = fixture.config.clone();
    for (key, value) in [
        ("num_nextn_predict_layers", serde_json::json!(2)),
        ("hybrid_override_pattern", serde_json::json!("M*")),
        ("attention_head_dim", serde_json::json!(7)),
        ("num_heads", serde_json::json!(u32::MAX)),
        ("n_heads", serde_json::json!("2")),
    ] {
        fixture.config = original.clone();
        fixture.config[key] = value;
        fixture.repin();
        assert!(fixture.prepare().is_err(), "accepted {key}");
    }
}
#[cfg(unix)]
#[test]
fn nemotron_profile_refuses_writerless_fifo_before_open() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let fixture = Fixture::new();
    fs::remove_file(fixture.profile()).unwrap();
    let path = CString::new(fixture.profile().as_os_str().as_bytes()).unwrap();
    // SAFETY: owned path is NUL-terminated and has no writer/child process.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    assert!(
        fixture
            .prepare()
            .unwrap_err()
            .to_string()
            .contains("regular file")
    );
}

fn complete_roster() -> Vec<String> {
    let required = [
        "mtp.layers.0.enorm.weight",
        "mtp.layers.0.hnorm.weight",
        "mtp.layers.0.eh_proj.weight",
        "mtp.layers.0.norm.weight",
        "mtp.layers.0.mixer.q_proj.weight",
        "mtp.layers.0.mixer.k_proj.weight",
        "mtp.layers.0.mixer.v_proj.weight",
        "mtp.layers.0.mixer.o_proj.weight",
        "mtp.layers.1.norm.weight",
        "mtp.layers.1.final_layernorm.weight",
        "mtp.layers.1.mixer.gate.weight",
        "mtp.layers.1.mixer.gate.e_score_correction.bias",
        "mtp.layers.1.mixer.shared_experts.up_proj.weight",
        "mtp.layers.1.mixer.shared_experts.down_proj.weight",
    ];
    let mut names = required
        .iter()
        .map(|name| name.to_string())
        .collect::<Vec<_>>();
    for id in 0..2 {
        for projection in ["up_proj", "down_proj"] {
            names.push(format!(
                "mtp.layers.1.mixer.experts.{id}.{projection}.weight"
            ));
        }
    }
    names
}
#[test]
fn nemotron_roster_requires_both_configured_expert_projections() {
    let mut names = complete_roster();
    let metadata = vec![GgufKv::u32("nemotron_h_moe.expert_count", 2)];
    validate_roster(&metadata, names.iter().map(String::as_str)).unwrap();
    names.pop();
    assert!(validate_roster(&metadata, names.iter().map(String::as_str)).is_err());
    names.push("mtp.layers.1.mixer.experts.2.down_proj.weight".into());
    assert!(validate_roster(&metadata, names.iter().map(String::as_str)).is_err());
}

#[test]
fn nemotron_roster_requires_router_bias_latent_pair_and_refuses_scaled_sidecars() {
    let mut names = complete_roster();
    let metadata = vec![
        GgufKv::u32("nemotron_h_moe.expert_count", 2),
        GgufKv::u32("nemotron_h_moe.moe_latent_size", 4),
    ];
    assert!(validate_roster(&metadata, names.iter().map(String::as_str)).is_err());
    names.push("mtp.layers.1.mixer.fc1_latent_proj.weight".into());
    assert!(validate_roster(&metadata, names.iter().map(String::as_str)).is_err());
    names.push("mtp.layers.1.mixer.fc2_latent_proj.weight".into());
    validate_roster(&metadata, names.iter().map(String::as_str)).unwrap();
    let bias = "mtp.layers.1.mixer.gate.e_score_correction.bias";
    let incomplete = names
        .iter()
        .filter(|name| name.as_str() != bias)
        .map(String::as_str);
    assert!(
        validate_roster(&metadata, incomplete)
            .unwrap_err()
            .to_string()
            .contains("required NextN projection absent")
    );
    assert!(validate_roster(&metadata[..1], names.iter().map(String::as_str)).is_err());
    for suffix in [
        "weight_scale",
        "weight_scale_2",
        "weight_scale_inv",
        "input_scale",
        "input_global_scale",
        "weight_global_scale",
        "weight_packed",
    ] {
        names.push(format!("lm_head.{suffix}"));
        assert!(
            validate_roster(&metadata, names.iter().map(String::as_str))
                .unwrap_err()
                .to_string()
                .contains("unsupported scaled/packed")
        );
        names.pop();
    }
    names.push("mtp.layers.0.mixer.q_proj.weight_scale".into());
    assert!(validate_roster(&metadata, names.iter().map(String::as_str)).is_err());
    let mut fixture = Fixture::new();
    fixture.config.as_object_mut().unwrap().remove("n_groups");
    fixture.config["num_groups"] = serde_json::json!(2);
    fixture.repin();
    assert!(fixture.prepare().is_ok());
}
