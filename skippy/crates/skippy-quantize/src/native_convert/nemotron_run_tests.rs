//! Actual native-run frontdoor and full synthetic latent-width SafeTensors writer.
//! No native model loading or actual Super checkpoint/tokenizer qualification.
use super::*;
use clap::Parser as _;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("owned native conversion fixture cleanup");
    }
}
fn root(label: &str) -> Root {
    let root = Root(std::env::temp_dir().join(format!(
        "native-nemotron-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    fs::create_dir(&root.0).unwrap();
    root
}
fn shape(name: &str) -> Vec<usize> {
    if name.ends_with("e_score_correction.bias") {
        vec![2]
    } else if name.ends_with(".eh_proj.weight") {
        vec![8, 16]
    } else if name.ends_with("fc1_latent_proj.weight") {
        vec![4, 8]
    } else if name.ends_with("fc2_latent_proj.weight") {
        vec![8, 4]
    } else if name.contains(".experts.") {
        if name.ends_with("up_proj.weight") {
            vec![16, 4]
        } else {
            vec![4, 16]
        }
    } else if name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight") {
        vec![4, 8]
    } else if name.ends_with("gate.weight") {
        vec![2, 8]
    } else if name.ends_with("norm.weight")
        || name.ends_with("enorm.weight")
        || name.ends_with("hnorm.weight")
    {
        vec![8]
    } else if name == "lm_head.weight_scale" {
        vec![1]
    } else {
        vec![8, 8]
    }
}
fn checkpoint(root: &Path, mode: &str) {
    fs::write(root.join("config.json"), CONFIG).unwrap();
    fs::write(root.join("tokenizer.json"), TOKENIZER).unwrap();
    fs::write(root.join("profile.json"), PROFILE).unwrap();
    let mut names = [
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
        "mtp.layers.1.mixer.fc1_latent_proj.weight",
        "mtp.layers.1.mixer.fc2_latent_proj.weight",
        "mtp.layers.1.mixer.shared_experts.up_proj.weight",
        "mtp.layers.1.mixer.shared_experts.down_proj.weight",
    ]
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
    match mode {
        "missing-bias" => {
            names.retain(|name| name != "mtp.layers.1.mixer.gate.e_score_correction.bias")
        }
        "missing-latent" => {
            names.retain(|name| name != "mtp.layers.1.mixer.fc2_latent_proj.weight")
        }
        "scaled" => names.push("lm_head.weight_scale".into()),
        "success" => (),
        _ => panic!("unknown owned fixture mode"),
    }
    let mut entries = serde_json::Map::new();
    let mut payload = Vec::new();
    for name in names {
        let dimensions = shape(&name);
        let start = payload.len();
        for _ in 0..dimensions.iter().product::<usize>() {
            payload.extend(1_f32.to_le_bytes());
        }
        entries.insert(name,serde_json::json!({"dtype":"F32","shape":dimensions,"data_offsets":[start,payload.len()]}));
    }
    let header = serde_json::to_vec(&entries).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(root.join("head.safetensors"), bytes).unwrap();
}
fn runner_manifest(root: &Path) -> (ConvertRunnerArgs, Manifest) {
    let profile = root.join("profile.json");
    let parsed = crate::Args::try_parse_from([
        "skippy-quantize",
        "run-convert-window",
        "--manifest",
        "unused.json",
        "--backend",
        "native-rust",
        "--mtp",
        "--nemotron-mtp-tokenizer-profile",
        profile.to_str().unwrap(),
    ])
    .unwrap();
    let crate::Command::RunConvertWindow(args) = parsed.command else {
        panic!("wrong CLI variant");
    };
    let runner = crate::prepare_convert_runner(args.runner).unwrap();
    let manifest = Manifest {
        schema_version: 1,
        kind: crate::types::JobKind::ConvertHf,
        source: root.into(),
        source_prefix: None,
        target: root.into(),
        target_prefix: "BF16".into(),
        output_basename: "mtp".into(),
        expected_splits: 1,
        window_size: 1,
        quant: None,
        output_type: Some(crate::types::ConvertOutputType::Bf16),
        tensor_type_file: None,
        tensor_type_recipe: None,
    };
    (runner, manifest)
}
#[test]
fn native_nemotron_profile_run_reaches_bf16_writer_and_refuses_source_drift() {
    let root = root("success");
    checkpoint(&root.0, "success");
    let (runner, manifest) = runner_manifest(&root.0);
    let output = root.0.join("mtp.gguf");
    let window = SplitWindow {
        first_split: 1,
        last_split: 1,
    };
    let status = run_native_convert(&runner, &manifest, window, &output).unwrap();
    crate::backend::ensure_success(status, &[]).unwrap();
    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[..4], b"GGUF");
    for expected in [
        "nemotron_h_moe",
        "blk.3.nextn.eh_proj.weight",
        "blk.3.ffn_up_exps.weight",
        "tokenizer.ggml.pre",
        "llama-bpe",
        "skippy.convert.tokenizer_profile_sha256",
    ] {
        assert!(
            bytes
                .windows(expected.len())
                .any(|value| value == expected.as_bytes()),
            "missing {expected}"
        );
    }
    let catalog = skippy_model::gguf_catalog::read_gguf_catalog(&output).unwrap();
    assert_eq!(
        catalog.metadata["nemotron_h_moe.moe_latent_size"],
        serde_json::json!(4)
    );
    for (name, dimensions) in [
        ("blk.3.exp_probs_b.bias", vec![2]),
        ("blk.3.post_attention_norm.weight", vec![8]),
        ("blk.3.ffn_latent_down.weight", vec![8, 4]),
        ("blk.3.ffn_latent_up.weight", vec![4, 8]),
        ("blk.3.ffn_up_exps.weight", vec![4, 16, 2]),
        ("blk.3.ffn_down_exps.weight", vec![16, 4, 2]),
    ] {
        let tensor = catalog
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap();
        assert_eq!(tensor.dimensions, dimensions, "{name}");
    }
    let projected = build_native_convert_command(&runner, &manifest, &output, window);
    let profile = root.0.join("profile.json");
    assert!(
        projected
            .windows(2)
            .any(|pair| pair[0] == "--nemotron-mtp-tokenizer-profile"
                && pair[1] == profile.to_str().unwrap())
    );
    fs::write(root.0.join("tokenizer.json"), b"{}").unwrap();
    let refused = root.0.join("refused.gguf");
    assert!(
        run_native_convert(&runner, &manifest, window, &refused)
            .unwrap_err()
            .to_string()
            .contains("pin mismatch")
    );
    assert!(!refused.exists());
    assert_eq!(fs::read(output).unwrap(), bytes);
}
#[test]
fn native_nemotron_profile_frontdoor_refuses_missing_bias_latent_half_or_scaled_source_before_output()
 {
    let root = root("refusal");
    for (mode, diagnostic) in [
        ("missing-bias", "required NextN projection absent"),
        (
            "missing-latent",
            "latent metadata requires both projections",
        ),
        ("scaled", "unsupported scaled/packed"),
    ] {
        checkpoint(&root.0, mode);
        let (runner, manifest) = runner_manifest(&root.0);
        let output = root.0.join(format!("{mode}.gguf"));
        let error = run_native_convert(
            &runner,
            &manifest,
            SplitWindow {
                first_split: 1,
                last_split: 1,
            },
            &output,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains(diagnostic), "{error:#}");
        assert!(!output.exists());
    }
}

const CONFIG: &str = r#"{"model_type":"nemotron_h_moe","num_nextn_predict_layers":1,"num_hidden_layers":3,"hybrid_override_pattern":"M*E","hidden_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"moe_intermediate_size":16,"n_routed_experts":2,"num_experts_per_tok":1,"moe_shared_expert_intermediate_size":8,"n_shared_experts":1,"n_group":1,"norm_topk_prob":true,"routed_scaling_factor":1.0,"layer_norm_epsilon":1.0e-05,"num_heads":2,"mamba_head_dim":4,"n_groups":2,"conv_kernel":4,"ssm_state_size":8,"vocab_size":2,"moe_latent_size":4}"#;
const TOKENIZER: &str = r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]},"decoder":{"type":"ByteLevel"}}"#;
const PROFILE: &str = r#"{"schema_version":1,"config_sha256":"4a44b9c6307791b9d5f6c032d4903a2aee909100e72ac4c1c5f5a7a30ac2665f","tokenizer_sha256":"b720848e4a67bf622850aa03b21ea4a7411e43184497b886d8aefa2cd6670242","tokenizer_config_sha256":null,"chat_template_sha256":null,"pre":"llama-bpe"}"#;
