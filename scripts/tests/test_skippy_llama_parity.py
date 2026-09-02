from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "skippy-llama-parity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("skippy_llama_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SkippyLlamaParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = load_module()

    def test_resolves_first_gguf_shard_for_split_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cache = os.environ.get("HF_HUB_CACHE")
            cache_root = Path(tmp) / "hub"
            snapshot = (
                cache_root
                / "models--DevQuasar--CohereLabs.command-a-plus-05-2026-bf16-GGUF"
                / "snapshots"
                / "abc123"
            )
            snapshot.mkdir(parents=True)
            first_shard = (
                snapshot
                / "CohereLabs.command-a-plus-05-2026-bf16-Q4_K_M-00001-of-00009.gguf"
            )
            last_shard = (
                snapshot
                / "CohereLabs.command-a-plus-05-2026-bf16-Q4_K_M-00009-of-00009.gguf"
            )
            mmproj = snapshot / "mmproj-CohereLabs.command-a-plus-05-2026-bf16.gguf"
            first_shard.write_bytes(b"larger-first-shard")
            last_shard.write_bytes(b"x")
            mmproj.write_bytes(b"")
            os.environ["HF_HUB_CACHE"] = str(cache_root)
            try:
                resolved = self.parity.resolve_candidate_file(
                    {
                        "repo": "DevQuasar/CohereLabs.command-a-plus-05-2026-bf16-GGUF",
                        "include": "*Q4_K_M*.gguf",
                    }
                )
            finally:
                if previous_cache is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = previous_cache

        self.assertEqual(resolved, first_shard)

    def test_runtime_slice_admission_rejects_architecture_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(
                    "if (model->arch != LLM_ARCH_LLAMA) { return SKIPPY_STATUS_UNSUPPORTED; }"
                ),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_requires_realized_contract_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source().replace(
                    "stage graph did not expose a stable input activation boundary",
                    "missing input boundary",
                ),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_diagnostic_literals_without_controls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(controls=False), encoding="utf-8"
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 6)

    def test_runtime_slice_admission_rejects_detached_failure_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(detached_failure=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_accepts_architecture_independent_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(self.runtime_slice_admission_source(), encoding="utf-8")

            failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 0)

    def test_runtime_slice_admission_allows_architecture_specific_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(
                    implementation=(
                        "if (model->arch == LLM_ARCH_GLM_DSA) { configure_graph(); }"
                    )
                ),
                encoding="utf-8",
            )

            failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 0)

    @staticmethod
    def runtime_slice_admission_source(
        extra: str = "",
        implementation: str = "",
        controls: bool = True,
        detached_failure: bool = False,
    ) -> str:
        checks = (
            "layer_end exceeds model layer count",
            "only the first runtime slice may include token embeddings",
            "the first runtime slice must include token embeddings",
            "only the final runtime slice may include output tensors",
            "stage graph did not expose a stable output activation boundary",
            "stage graph did not expose a stable input activation boundary",
        )
        invalid_argument_checks = (
            (
                "config->layer_end > n_layer",
                checks[0],
            ),
            (
                "config->include_embeddings && config->layer_start != 0 && !config->include_output",
                checks[1],
            ),
            (
                "config->layer_start == 0 && !config->include_embeddings",
                checks[2],
            ),
            (
                "config->include_output && config->layer_end != n_layer",
                checks[3],
            ),
        )
        def invalid_argument_failure(message: str) -> str:
            return (
                "llama_model_free(model); "
                f'const char * message = "{message}"; '
                "skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message); "
                "return SKIPPY_STATUS_INVALID_ARGUMENT;"
            )

        control_lines = tuple(
            f"if ({guard}) {{ {invalid_argument_failure(message)} }}"
            for guard, message in invalid_argument_checks
        )
        if detached_failure:
            guard, message = invalid_argument_checks[0]
            control_lines = (
                f"if ({guard}) {{ }}",
                f"{{ {invalid_argument_failure(message)} }}",
                *control_lines[1:],
            )
        if controls:
            boundary_lines = (
                "if (!stage_model->ctx->get_activation_boundary(type, elements, bytes)) { "
                f'return fail_boundary_load("{checks[4]}"); }}',
                "if (!stage_model->ctx->get_input_activation_boundary(type, elements, bytes)) { "
                f'return fail_boundary_load("{checks[5]}"); }}',
            )
        else:
            control_lines = ()
            boundary_lines = tuple(
                f'const char * diagnostic = "{check}";' for check in checks
            )
        return "\n".join(
            (
                "static enum skippy_status skippy_finish_model_open(",
                extra,
                *control_lines,
                "skippy_model * stage_model = new skippy_model{};",
                implementation,
                *boundary_lines,
                "enum skippy_status skippy_model_open_impl(",
            )
        )


if __name__ == "__main__":
    unittest.main()
