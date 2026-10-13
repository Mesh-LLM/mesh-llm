//! Inert certification process boundary; no model execution.
use std::{path::PathBuf, time::Duration};
pub(super) fn run(arguments: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let value = |name| {
        arguments
            .windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| pair[1].as_str())
            .ok_or("fixture argument")
    };
    let root = PathBuf::from(value("--cert-root")?);
    std::fs::create_dir_all(&root)?;
    std::fs::write(root.join("arguments.json"), serde_json::to_vec(arguments)?)?;
    let names = [
        "LLAMA_STAGE_BACKEND",
        "SKIPPY_LLAMA_BACKEND",
        "LLAMA_STAGE_LINK_MODE",
        "SKIPPY_LLAMA_LINK_MODE",
        "LLAMA_STAGE_CUDA_ARCHITECTURES",
        "SKIPPY_CUDA_ARCHITECTURES",
        "LLAMA_STAGE_AMDGPU_TARGETS",
        "SKIPPY_AMDGPU_TARGETS",
        "LLAMA_STAGE_GGML_NATIVE",
        "SKIPPY_GGML_NATIVE",
        "GGML_CUDA_NO_VMM",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HF_TOKEN",
        "LLAMA_STAGE_BUILD_DIR",
        "SKIPPY_LLAMA_BUILD_DIR",
        "CUDA_PATH",
        "HIP_PATH",
        "ROCM_PATH",
        "LLVMInstallDir",
        "VULKAN_SDK",
        "GIT_MASTER",
        "GIT_OPTIONAL_LOCKS",
        "PATH",
    ];
    let mut environment = serde_json::Map::new();
    for name in names {
        if let Ok(value) = std::env::var(name) {
            environment.insert(name.into(), serde_json::json!(value));
        }
    }
    std::fs::write(
        root.join("environment.json"),
        serde_json::to_vec(&environment)?,
    )?;
    match value("--prompt")? {
        "fail" => std::process::exit(23),
        "missing-manifest" => return Ok(()),
        "hold" => {
            super::signals::install()?;
            #[cfg(unix)]
            std::fs::write(
                root.join("parent.pid"),
                unsafe { libc::getppid() }.to_string(),
            )?;
            std::fs::write(root.join("ready"), b"held")?;
            while !super::signals::stopped() {
                std::thread::sleep(Duration::from_millis(5));
            }
            return Ok(());
        }
        _ => (),
    }
    let family = if value("--prompt")? == "wrong-identity" {
        "wrong"
    } else {
        value("--family")?
    };
    let capability = root.join("capability-draft.json");
    let declaration = serde_json::json!({"schema_version":1,"status":"draft","generated_by":"scripts/family-certify.sh","family":family,"model_id":value("--model-id")?,"target_model":value("--target-model")?});
    std::fs::write(&capability, serde_json::to_vec(&declaration)?)?;
    let output = serde_json::json!({"run_id":value("--run-id")?,"family":family,"model_id":value("--model-id")?,"target_model":value("--target-model")?,"target_model_identity":{"model_id":value("--model-id")?},"draft_model":null,"output_dir":root,"git_commit":"fixture-observation","correctness":{"layer_end":value("--layer-end")?,"split_layer":value("--split-layer")?,"splits":value("--splits")?,"activation_width":value("--activation-width")?,"ctx_size":value("--ctx-size")?,"n_gpu_layers":value("--n-gpu-layers")?},"speculative":{"skipped_by_policy":true},"capability_draft":capability,"commands":[]});
    std::fs::write(root.join("manifest.json"), serde_json::to_vec(&output)?)?;
    if value("--prompt")? == "replace-toolkit" {
        let path = PathBuf::from(std::env::var("CUDA_PATH")?);
        let old = path.with_extension("previous-toolkit");
        std::fs::rename(&path, &old)?;
        std::fs::create_dir(&path)?;
    }
    if value("--prompt")? == "sanitized-path" {
        // Match the outer shell summary path shape, not native per-lane JSON.
        println!("summary: /inert/token-named-output/summary.md");
    }
    if value("--prompt")? == "source-guard-drift" {
        let source = std::env::current_dir()?;
        let path = source.join("docs/skippy/llama-parity-candidates.json");
        let mut document: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        document["defaults"]["prompt"] = "observed source changed after harness".into();
        std::fs::write(path, serde_json::to_vec(&document)?)?;
        std::fs::write(source.join("source-guard-mutated"), b"observed")?;
    }
    println!("inert harness completed");
    Ok(())
}
