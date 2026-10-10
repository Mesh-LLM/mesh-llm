//! Capability-owned postimage hunk contracts; mail prose and removed lines cannot qualify.
use std::{fs, path::Path};
fn patch(suffix: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../skippy/llama_cpp/patches");
    let directory = if suffix.starts_with("model_support/") {
        root.join("model_support")
    } else {
        root
    };
    let suffix = suffix.strip_prefix("model_support/").unwrap_or(suffix);
    let candidates: Vec<_> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(suffix)
        })
        .collect();
    assert_eq!(candidates.len(), 1, "unique capability patch: {suffix}");
    fs::read_to_string(&candidates[0]).unwrap()
}
fn postimage(mail: &str, owner: &str) -> String {
    let header = format!("diff --git a/{owner} b/{owner}");
    let mut selected = false;
    let mut hunk = false;
    let mut found = 0;
    let mut lines = Vec::new();
    for line in mail.lines() {
        if line.starts_with("diff --git ") {
            selected = line == header;
            found += usize::from(selected);
            hunk = false;
        } else if selected && line.starts_with("@@ ") {
            hunk = true;
        } else if selected && hunk {
            if let Some(line) = line.strip_prefix('+').or_else(|| line.strip_prefix(' ')) {
                lines.push(line);
            } else if !line.starts_with('-') && !line.starts_with("\\ No newline") {
                hunk = false;
            }
        }
    }
    assert_eq!(found, 1, "unique patch owner: {owner}");
    assert!(!lines.is_empty());
    lines.join("\n")
}
fn owned(suffix: &str, owner: &str) -> String {
    postimage(&patch(suffix), owner)
}
// Converter policy belongs to the complete immutable external research project.
// Native admission validates its commit, manifest, physical roster and file hashes.
fn converter_source(owner: &str) -> String {
    assert!(matches!(
        owner,
        "conversion/base.py" | "conversion/inkling.py"
    ));
    let fixture = super::Fixture::new();
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(super::root())
        .env(
            "MESH_PYTHON_RESEARCH_SOURCE",
            std::env::var_os("MESH_PYTHON_RESEARCH_SOURCE")
                .expect("prepare the immutable external research source"),
        )
        .args([
            "automation",
            "smoke-observation",
            "research-source",
            "--kind",
            "root",
        ]);
    let receipt = fixture.run(command);
    assert!(
        receipt.status.success(),
        "external converter source admission failed: {}",
        String::from_utf8_lossy(&receipt.stderr)
    );
    let root = std::path::PathBuf::from(String::from_utf8(receipt.stdout).unwrap().trim());
    assert!(root.is_absolute() && root.is_dir());
    fs::read_to_string(root.join("model-converters").join(owner)).unwrap()
}
fn compact(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}
fn required(body: &str, needle: &str) {
    assert!(
        compact(body).contains(&compact(needle)),
        "missing owner contract: {needle}"
    );
}
fn ordered(body: &str, before: &str, after: &str) {
    let body = compact(body);
    assert!(body.find(&compact(before)).unwrap() < body.find(&compact(after)).unwrap());
}
#[test]
fn regression_postimage_hunks_exclude_mail_prose_and_removed_code() {
    let body = postimage(
        "Subject: required_marker\ndiff --git a/src/x.cpp b/src/x.cpp\n--- a/src/x.cpp\n+++ b/src/x.cpp\n@@ -1,2 +1,2 @@\n-required_marker\n+replacement\n context\n-- \n2.0\n",
        "src/x.cpp",
    );
    assert_eq!(body, "replacement\ncontext");
}
#[test]
fn regression_generated_tensor_guard_precedes_preparation_and_rejects_bad_index_membership() {
    let body = converter_source("conversion/base.py");
    for needle in [
        "self.has_model_weight_shards = bool(self.model_tensors)",
        "allow_no_weight_shards: bool = False",
        "if not self.has_model_weight_shards and not self.allow_no_weight_shards:",
        "duplicate tensor",
        "refusing to load a wrong-shard copy",
    ] {
        required(&body, needle);
    }
    ordered(
        &body,
        "if not self.has_model_weight_shards and not self.allow_no_weight_shards:",
        "self.prepare_tensors()",
    );
}
#[test]
fn regression_minimax_msa_requires_all_indexer_tensors_when_metadata_declares_heads_and_width() {
    let body = owned(
        "-models-expose-stage-independent-graph-semantics.patch",
        "src/models/minimax-m3.cpp",
    );
    required(
        &body,
        "const int indexer_flags = hparams.indexer_n_head > 0 && hparams.indexer_head_size > 0 ? 0 : TENSOR_NOT_REQUIRED",
    );
    for name in [
        "index_q_proj",
        "index_k_proj",
        "index_q_norm",
        "index_k_norm",
    ] {
        let assignment = body
            .lines()
            .find(|line| line.contains(&format!("layer.{name} = create_tensor(")))
            .unwrap();
        required(assignment, "indexer_flags);");
    }
}
#[test]
fn regression_inkling_weight_and_metadata_vocab_fallbacks_stay_consistent() {
    let body = converter_source("conversion/inkling.py");
    required(
        &body,
        "n_unpadded = self.hparams.get(\"unpadded_vocab_size\") or n_vocab",
    );
    required(
        &body,
        "n_unpadded = hp.get(\"unpadded_vocab_size\") or hp[\"vocab_size\"]",
    );
    required(
        &body,
        "add_uint32(f\"{arch}.unpadded_vocab_size\", n_unpadded)",
    );
}
#[test]
fn regression_diffusion_failed_decode_cannot_promote_completed_generation_count() {
    let body = owned(
        "model_support/-llama-port-diffusion-gemma-support.patch",
        "examples/diffusion/diffusion.cpp",
    );
    let signature = "void diffusion_generate_entropy_bound(";
    assert_eq!(body.matches(signature).count(), 1);
    let body = body.split_once(signature).unwrap().1;
    required(body, "decode_failed = true;");
    required(body, "finish();");
    required(
        body,
        "if (!decode_failed) { n_generated = params.max_length; }",
    );
    ordered(body, "decode_failed = true;", "if (!decode_failed)");
    ordered(
        body,
        "if (!decode_failed)",
        "n_generated = params.max_length;",
    );
}
#[test]
fn regression_diffusion_counts_and_size_overflow_are_rejected_before_self_condition_allocation() {
    let body = owned(
        "model_support/-llama-port-diffusion-gemma-support.patch",
        "examples/diffusion-gemma-server/diffusion-gemma-server.cpp",
    );
    for needle in [
        "const int64_t N64 = (int64_t) P + C;",
        "P < 0 || C <= 0",
        "std::numeric_limits<size_t>::max() / (size_t) n_vocab",
    ] {
        required(&body, needle);
    }
    ordered(&body, "P < 0 || C <= 0", "sc_cache.assign(sc_size");
    ordered(
        &body,
        "std::numeric_limits<size_t>::max()",
        "sc_cache.assign(sc_size",
    );
}
#[test]
fn regression_system_one_label_coverage_rejects_overlap_and_holes_before_claiming_lane() {
    let body = owned(
        "model_support/-skippy-add-diffusion-gemma-system-one-reads.patch",
        "src/skippy/system_one.cpp",
    );
    for needle in [
        "seen_label_positions(label_token_count, 0)",
        "seen_label_positions[flattened_index] != 0",
        "system-one label ranges must not overlap",
        "must cover the flattened label-token array exactly",
    ] {
        required(&body, needle);
    }
    ordered(
        &body,
        "seen_label_positions[flattened_index] != 0",
        "skippy_claim_execution_lane(model)",
    );
    ordered(
        &body,
        "must cover the flattened label-token array exactly",
        "skippy_claim_execution_lane(model)",
    );
}
#[test]
fn regression_pooled_reset_detaches_sampler_and_native_gate_repeats_multi_token_prefill() {
    let mail = patch("-fix-skippy-detach-backend-sampler-before-pooled-pre.patch");
    let session = postimage(&mail, "src/skippy/session.cpp");
    required(
        &session,
        "session->sampling_backend_enabled || !skippy_reset_reusable_sampling(session)",
    );
    ordered(
        &session,
        "session->sampling_backend_enabled",
        "skippy_clear_chat_sampling(session)",
    );
    let gate = postimage(&mail, "src/skippy/tests/runtime_events.cpp");
    for needle in [
        "CHECK(session->sampling_backend_enabled);",
        "CHECK(!session->sampling_backend_enabled);",
        "CHECK(session->sampling_chain == nullptr);",
        "CHECK(session->n_past == 0);",
    ] {
        required(&gate, needle);
    }
    let reset = gate.find("skippy_session_reset(session").unwrap();
    assert!(gate[..reset].contains("skippy_prefill_chunk("));
    assert!(gate[reset..].contains("skippy_prefill_chunk("));
    assert!(gate[..reset].contains("CHECK(session->n_past == 2);"));
    assert!(gate[reset..].contains("CHECK(session->n_past == 2);"));
}
