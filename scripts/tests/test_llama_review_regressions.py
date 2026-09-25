"""Lock down the native patch-queue invariants found during PR review."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATCHES = ROOT / "third_party/llama.cpp/patches"


def patch_text(relative_path: str) -> str:
    return (PATCHES / relative_path).read_text(encoding="utf-8")


class LlamaReviewRegressionTests(unittest.TestCase):
    def test_generated_tensors_cannot_hide_an_empty_source_index(self):
        patch = patch_text("0004-models-expose-stage-independent-graph-semantics.patch")
        guard = "if not self.has_model_weight_shards and not self.allow_no_weight_shards:"
        self.assertIn("self.has_model_weight_shards = bool(self.model_tensors)", patch)
        self.assertIn(guard, patch)
        self.assertLess(patch.index(guard), patch.index("self.prepare_tensors()"))
        self.assertIn("duplicate tensor", patch)
        self.assertIn("refusing to load a wrong-shard copy", patch)

    def test_minimax_requires_indexer_tensors_when_msa_metadata_is_present(self):
        patch = patch_text("0004-models-expose-stage-independent-graph-semantics.patch")
        self.assertIn("const int indexer_flags = hparams.indexer_n_head > 0", patch)
        self.assertIn("hparams.indexer_head_size > 0", patch)
        self.assertIn("? 0 : TENSOR_NOT_REQUIRED", patch)
        for tensor in ("index_q_proj", "index_k_proj", "index_q_norm", "index_k_norm"):
            self.assertRegex(patch, rf"layer\.{tensor} = create_tensor\([^\n]+, indexer_flags\);")

    def test_inkling_unpadded_vocab_has_the_same_fallback_for_metadata(self):
        patch = patch_text("model_support/0001-models-add-inkling-family-support.patch")
        self.assertIn('n_unpadded = hp.get("unpadded_vocab_size") or hp["vocab_size"]', patch)
        self.assertIn('add_uint32(f"{arch}.unpadded_vocab_size", n_unpadded)', patch)

    def test_diffusion_decode_failure_does_not_report_a_complete_generation(self):
        patch = patch_text("model_support/0003-llama-port-diffusion-gemma-support.patch")
        self.assertIn("decode_failed = true;", patch)
        self.assertIn("finish();", patch)
        self.assertIn("if (!decode_failed)", patch)
        self.assertLess(patch.index("if (!decode_failed)"), patch.rindex("n_generated = params.max_length;"))

    def test_diffusion_server_validates_counts_before_sizing_self_conditioning(self):
        patch = patch_text("model_support/0003-llama-port-diffusion-gemma-support.patch")
        self.assertIn("const int64_t N64 = (int64_t) P + C;", patch)
        self.assertIn("P < 0 || C <= 0", patch)
        self.assertIn("std::numeric_limits<size_t>::max()", patch)
        self.assertLess(patch.index("P < 0 || C <= 0"), patch.index("sc_cache.assign(sc_size"))

    def test_system_one_label_ranges_cover_each_output_once(self):
        patch = patch_text("model_support/0004-skippy-add-diffusion-gemma-system-one-reads.patch")
        self.assertIn("seen_label_positions", patch)
        self.assertIn("system-one label ranges must not overlap", patch)
        self.assertIn("must cover the flattened label-token array exactly", patch)

    def test_pooled_session_detaches_backend_sampler_before_next_prefill(self):
        patch = patch_text("0029-fix-skippy-detach-backend-sampler-before-pooled-pre.patch")
        self.assertIn(
            "session->sampling_backend_enabled || !skippy_reset_reusable_sampling(session)",
            patch,
        )
        self.assertIn("CHECK(session->sampling_backend_enabled);", patch)
        self.assertIn("CHECK(!session->sampling_backend_enabled);", patch)
        self.assertIn("CHECK(session->sampling_chain == nullptr);", patch)
        self.assertEqual(patch.count("skippy_prefill_chunk("), 2)
        self.assertIn("CHECK(session->n_past == 0);", patch)
        self.assertEqual(patch.count("CHECK(session->n_past == 2);"), 2)


if __name__ == "__main__":
    unittest.main()
