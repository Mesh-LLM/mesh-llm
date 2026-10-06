use super::Projection;
use anyhow::{Result, bail};
use skippy_package_format::PackageManifest;
fn cell(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace('`', "&#96;")
        .replace('\n', "<br>")
        .replace('\r', "")
}
pub(super) fn render(
    manifest: &PackageManifest,
    projection: &Projection,
    target: &str,
    pipeline: &str,
    license: Option<&str>,
) -> Result<String> {
    super::repo(target)?;
    if pipeline.is_empty()
        || pipeline.len() > 128
        || !pipeline
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        bail!("pipeline tag refused");
    }
    let license_line = match license {
        Some(value)
            if !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control) =>
        {
            format!("license: {}\n", serde_json::to_string(value)?)
        }
        Some(_) => bail!("source license refused"),
        None => String::new(),
    };
    let mut card = format!(
        "---\nlibrary_name: mesh-llm\n{license_line}base_model:\n- {}\npipeline_tag: {}\ntags:\n- gguf\n- mesh-llm\n- layer-package\n- skippy\n- distributed-inference\n- local-inference\n- openai-compatible\n{}---\n\n# {}\n\nDistributed GGUF inference package for Mesh LLM.\n\n",
        serde_json::to_string(&projection.source_repo)?,
        serde_json::to_string(pipeline)?,
        if projection.experimental {
            "- experimental\n"
        } else {
            ""
        },
        cell(&projection.model_id)
    );
    if projection.experimental {
        card.push_str("> [!WARNING]\n> Experimental package: artifact integrity, runtime, split-correctness and multimodal qualification are separate. This card does not promote an entry into meshllm/catalog@main.\n\n");
    }
    card.push_str("## Highlights\n\nPrivate local inference, pooled memory/compute and OpenAI-compatible serving.\n\n## Model Overview\n\n| Property | Value |\n|---|---|\n");
    for (name, value) in [
        ("Source model", projection.source_repo.clone()),
        ("Model id", projection.model_id.clone()),
        ("Layer count", projection.layer_count.to_string()),
        ("Declared package bytes", projection.total_bytes.to_string()),
        ("Source revision", projection.source_revision.clone()),
        ("Package repo", target.into()),
    ] {
        card.push_str(&format!("| {} | {} |\n", name, cell(&value)));
    }
    card.push_str(&format!("\n## Recommended Use\n\nLocal/private and multi-machine serving. Review upstream architecture, license, templates and sampling guidance at https://huggingface.co/{}. The source license, when present, is declared from the admitted source metadata; no base-model fallback or license was inferred.\n\n## Quickstart\n\n```bash\nmesh-llm serve --model \"{}\" --split\n```\n\n## Package Variant\n\n| Property | Value |\n|---|---|\n| Format | {} |\n| Package identity | {} |\n| Manifest SHA-256 | {} |\n| Source identity | {} |\n| Native ABI | {} |\n",projection.source_repo,target,cell(&manifest.format),cell(&projection.package_id),projection.manifest_sha256,cell(&projection.source_identity),cell(&manifest.native_abi_version)));
    if let Some(spec) = manifest
        .generation
        .as_ref()
        .and_then(|g| g.speculative_decoding.as_ref())
    {
        card.push_str(&format!(
            "| Speculative decoding declaration | {} |\n",
            cell(&serde_json::to_string(spec)?)
        ));
    }
    card.push_str(
        "\n## What Is Included\n\n| Artifact id | Path | Bytes | SHA-256 |\n|---|---|---|---|\n",
    );
    for artifact in &projection.artifacts {
        card.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            cell(&artifact.id),
            cell(&artifact.path),
            artifact.byte_size,
            artifact.sha256
        ));
    }
    card.push_str("\n## Validation\n\nThe native job validates the package root, canonical package identity and catalog/source bindings. Listed hashes are declared artifact identities. Upload confirmation, immutable remote verification, independently verified source tensors, ABI/runtime behavior and catalog promotion require their own receipts.\n\n## Links\n\n- https://www.meshllm.cloud\n- https://github.com/Mesh-LLM/mesh-llm\n- https://huggingface.co/datasets/meshllm/catalog\n");
    if card.len() > 2 * 1024 * 1024 {
        bail!("model card size refused");
    }
    Ok(card)
}

pub(super) struct CardContext<'a> {
    pub target: &'a str,
    pub pipeline: &'a str,
    pub source_file: &'a str,
    pub mesh_ref: &'a str,
    pub license: &'a super::License,
}
pub(super) fn render_complete(
    manifest: &PackageManifest,
    projection: &Projection,
    context: CardContext<'_>,
) -> Result<String> {
    let CardContext {
        target,
        pipeline,
        source_file,
        mesh_ref,
        license,
    } = context;
    if [source_file, mesh_ref]
        .iter()
        .any(|s| s.is_empty() || s.len() > 4096 || s.chars().any(char::is_control))
    {
        bail!("model card context refused");
    }
    let mut card = render(
        manifest,
        projection,
        target,
        pipeline,
        license.value.as_deref(),
    )?;
    card=card.replace("The source license, when present, is declared from the admitted source metadata; no base-model fallback or license was inferred.","License metadata is resolved from the immutable source card, then its first explicitly declared base model; missing or unavailable metadata does not invent a license.");
    let display = manifest
        .source_model
        .distribution_id
        .as_deref()
        .unwrap_or(&manifest.model_id);
    let family = [
        "Qwen3", "Qwen2.5", "DeepSeek", "Kimi", "Gemma", "GLM", "Llama",
    ]
    .into_iter()
    .find(|f| {
        display
            .to_ascii_lowercase()
            .contains(&f.to_ascii_lowercase())
    })
    .unwrap_or_else(|| display.split('-').next().unwrap_or("Unknown"));
    card.push_str("\n## Source and Build Context\n\n| Property | Value |\n|---|---|\n");
    for (label, value) in [
        ("Display name", display),
        ("Family", family),
        ("Source file", source_file),
        ("Source SHA-256", manifest.source_model.sha256.as_str()),
        ("Mesh LLM ref", mesh_ref),
        ("Generator version", manifest.generator_version.as_str()),
    ] {
        card.push_str(&format!("| {} | {} |\n", label, cell(value)));
    }
    card.push_str(&format!(
        "| Parameter scale | {} |\n| Quantization | {} |\n",
        cell(&parameter_scale(display)),
        cell(&quantization(display, source_file))
    ));
    if let Some(repo) = &license.repo {
        card.push_str(&format!(
            "| License source | https://huggingface.co/{} |\n",
            cell(repo)
        ));
    }
    if let Some(pin) = &license.revision {
        card.push_str(&format!("| License source revision | {} |\n", cell(pin)));
    }
    if let Some(warning) = license.warning {
        card.push_str(&format!("\n> WARNING: {}\n", cell(warning)));
    }
    card.push_str(&format!(
        "\n## Native Model Metadata\n\n```json\n{}\n```\n",
        serde_json::to_string_pretty(&manifest.model_metadata)?
    ));
    card.push_str(
        "\n## Tensor Catalog\n\n| Tensor | Shape | Native type | Layer |\n|---|---|---|---|\n",
    );
    for tensor in &manifest.tensor_catalog.entries {
        card.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            cell(&tensor.name),
            cell(&serde_json::to_string(&tensor.dimensions)?),
            tensor.ggml_type,
            tensor
                .layer_ordinal
                .map_or_else(|| "shared".into(), |n| n.to_string())
        ));
        if card.len() > 2 * 1024 * 1024 {
            bail!("model card size refused");
        }
    }
    card.push_str("\n## Operator Checks\n\n```bash\ncurl -s http://localhost:3131/api/status\ncurl -s http://localhost:3131/v1/models\n```\n\nUse the returned model name with /v1/chat/completions and max_tokens 128. Artifact identities and successful publication are separate from runtime, ABI, split correctness and model qualification.\n");
    if card.len() > 2 * 1024 * 1024 {
        bail!("model card size refused");
    }
    Ok(card)
}

fn parameter_scale(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    for start in 0..chars.len() {
        if !chars[start].is_ascii_digit() {
            continue;
        }
        let mut end = start;
        while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == '.') {
            end += 1;
        }
        if end < chars.len() && matches!(chars[end].to_ascii_uppercase(), 'B' | 'M') {
            end += 1;
            if chars.get(end) == Some(&'-')
                && chars
                    .get(end + 1)
                    .is_some_and(|c| c.eq_ignore_ascii_case(&'A'))
            {
                let mut active = end + 2;
                while active < chars.len()
                    && (chars[active].is_ascii_digit() || chars[active] == '.')
                {
                    active += 1;
                }
                if chars
                    .get(active)
                    .is_some_and(|c| c.eq_ignore_ascii_case(&'B'))
                {
                    end = active + 1;
                }
            }
            return chars[start..end].iter().collect();
        }
    }
    "not recorded".into()
}
fn quantization(name: &str, file: &str) -> String {
    let combined = format!("{name}/{file}");
    let chars: Vec<char> = combined.chars().collect();
    for prefix in ["UD-Q", "Q", "IQ", "BF16", "F16"] {
        let p: Vec<char> = prefix.chars().collect();
        for start in 0..chars.len() {
            if start + p.len() > chars.len()
                || !chars[start..start + p.len()]
                    .iter()
                    .zip(&p)
                    .all(|(a, b)| a.eq_ignore_ascii_case(b))
            {
                continue;
            }
            let mut end = start + p.len();
            if prefix == "BF16" || prefix == "F16" {
                return chars[start..end].iter().collect();
            }
            let digits = end;
            while end < chars.len() && chars[end].is_ascii_digit() {
                end += 1;
            }
            if end == digits {
                continue;
            }
            let mut groups = 0;
            while groups < 2 && chars.get(end) == Some(&'_') {
                let begin = end + 1;
                let mut next = begin;
                while next < chars.len() && chars[next].is_ascii_alphabetic() {
                    next += 1;
                }
                if next == begin {
                    break;
                }
                end = next;
                groups += 1;
            }
            if groups > 0 {
                return chars[start..end].iter().collect();
            }
        }
    }
    "not recorded".into()
}
