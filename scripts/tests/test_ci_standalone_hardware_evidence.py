from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "skippy/scripts/ci-standalone-model-qualification.py"
SPEC = importlib.util.spec_from_file_location("ci_standalone_model_qualification", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

# Captured verbatim from the composed macOS Metal product's own `doctor --output
# json` (product source f18c273c82). Keep this fixture in the shape the runtime
# actually emits, not a hand-written ideal.
METAL_PROFILE = {
    "arch": "aarch64",
    "available_flavors": ["cpu", "metal"],
    "gpus": [
        {"backend_device": None, "display_name": "Metal Support: Metal 4",
         "stable_id": None, "unified_memory": True, "vram_bytes": None},
        {"backend_device": None, "display_name": "Type: GPU",
         "stable_id": None, "unified_memory": True, "vram_bytes": None},
    ],
    "os": "macos",
    "target_triple": None,
}
CUDA_PROFILE = {
    "arch": "x86_64",
    "gpus": [{"display_name": "NVIDIA H100 PCIe", "cuda_sm": "sm_90"}],
    "cuda": {"toolkit_majors": [12], "driver_version": "550.54.15", "gpu_arches": ["sm_90"]},
}
AVAILABLE = {"schema_version": 1, "state": "available", "row_id": "linux-cuda", "runner": "gpu-nvidia"}


class HardwareEvidenceTests(unittest.TestCase):
    def row(self, row_id: str) -> dict:
        return {"id": row_id, "backend": row_id.split("-", 1)[1], "target": "test", "architecture": "test"}

    def test_selected_backend_device_reads_the_runtime_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "serve.log"
            log.write_text(
                "\n".join([
                    '{"data":{"message":"preparing"},"type":"status"}',
                    '{"data":{"current":1},"type":"progress"}',
                    '{"data":{"detail":"MTL0"},"type":"backend_device_selected"}',
                    '{"data":{"api_base":"http://127.0.0.1:1/v1"},"type":"ready"}',
                ]),
                encoding="utf-8",
            )
            self.assertEqual(driver.selected_backend_device(log), "MTL0")

    def test_selected_backend_device_absent_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "serve.log"
            log.write_text('{"data":{"message":"preparing"},"type":"status"}\n', encoding="utf-8")
            self.assertIsNone(driver.selected_backend_device(log))
            self.assertIsNone(driver.selected_backend_device(Path(temporary) / "missing.log"))

    def test_wait_for_backend_device_reads_a_late_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "serve.log"
            log.write_text('{"data":{"message":"preparing"},"type":"status"}\n', encoding="utf-8")
            self.assertIsNone(driver.selected_backend_device(log))
            with log.open("a", encoding="utf-8") as handle:
                handle.write('{"data":{"detail":"CPU"},"type":"backend_device_selected"}\n')
            self.assertEqual(driver.wait_for_backend_device(log, timeout=5.0), "CPU")

    def test_wait_for_backend_device_times_out_without_the_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "serve.log"
            self.assertIsNone(driver.wait_for_backend_device(missing, timeout=0.6))

    def test_cpu_row_records_the_runtime_cpu_device(self) -> None:
        hardware = driver.build_hardware(
            self.row("linux-cpu"),
            {"state": "available", "row_id": "linux-cpu", "runner": "linux-hosted"},
            {"arch": "x86_64", "gpus": []}, "CPU",
        )
        self.assertEqual(hardware, {"actual_backend": "cpu", "runner": "linux-hosted",
                                    "device": "CPU", "selected_device": "CPU"})

    def test_metal_row_derives_device_runtime_and_driver(self) -> None:
        with mock.patch.object(driver, "_apple_chip_name", return_value="Apple M1 Ultra"), \
             mock.patch.object(driver, "_macos_version", return_value="27.0.1"):
            hardware = driver.build_hardware(
                self.row("macos-metal"),
                {"state": "available", "row_id": "macos-metal", "runner": "macos-hosted"},
                METAL_PROFILE, "MTL0",
            )
        self.assertEqual(hardware["actual_backend"], "metal")
        self.assertEqual(hardware["runner"], "macos-hosted")
        self.assertEqual(hardware["device"], "Apple M1 Ultra")
        self.assertEqual(hardware["selected_device"], "MTL0")
        self.assertEqual(hardware["runtime_version"], "Metal 4")
        self.assertEqual(hardware["device_architecture"], "aarch64")
        self.assertTrue(hardware["driver"].startswith("macOS "))

    def test_cuda_row_records_driver_runtime_and_architecture(self) -> None:
        hardware = driver.build_hardware(
            self.row("linux-cuda"), AVAILABLE, CUDA_PROFILE, "CUDA0",
        )
        self.assertEqual(hardware["device"], "NVIDIA H100 PCIe")
        self.assertEqual(hardware["driver"], "550.54.15")
        self.assertEqual(hardware["runtime_version"], "CUDA 12")
        self.assertEqual(hardware["device_architecture"], "sm_90")

    def test_cpu_fallback_cannot_satisfy_a_gpu_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected device"):
            driver.build_hardware(self.row("macos-metal"),
                                  {"state": "available", "row_id": "macos-metal",
                                   "runner": "macos-hosted"}, METAL_PROFILE, "CPU")

    def test_unavailable_row_cannot_produce_hardware_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy-available"):
            driver.build_hardware(self.row("linux-rocm"),
                                  {"state": "hardware-unavailable", "row_id": "linux-rocm"},
                                  {}, "ROCm0")

    def test_missing_device_identity_fails_closed(self) -> None:
        with mock.patch.object(driver, "_apple_chip_name", return_value=None):
            with self.assertRaisesRegex(ValueError, "hardware device identity"):
                driver.build_hardware(self.row("macos-metal"),
                                      {"state": "available", "row_id": "macos-metal",
                                       "runner": "macos-hosted"},
                                      {"arch": "aarch64", "gpus": []}, "MTL0")

    def test_metal_runtime_without_a_version_token_uses_the_host_label(self) -> None:
        # A headless runner can enumerate the Metal stack without a version
        # number; the row must still carry a non-empty runtime identity.
        with mock.patch.object(driver, "_apple_chip_name", return_value="Apple M2"):
            hardware = driver.build_hardware(
                self.row("macos-metal"),
                {"state": "available", "row_id": "macos-metal", "runner": "macos-hosted"},
                {"arch": "aarch64",
                 "gpus": [{"display_name": "Metal Support: Metal"},
                          {"display_name": "Type: GPU"}]},
                "MTL0",
            )
        self.assertEqual(hardware["runtime_version"], "Metal Support: Metal")

    def test_metal_runtime_without_any_gpu_uses_the_os_metal_stack(self) -> None:
        with mock.patch.object(driver, "_apple_chip_name", return_value="Apple M2"), \
             mock.patch.object(driver, "_macos_version", return_value="14.6"):
            hardware = driver.build_hardware(
                self.row("macos-metal"),
                {"state": "available", "row_id": "macos-metal", "runner": "macos-hosted"},
                {"arch": "aarch64", "gpus": [{"display_name": "Type: GPU"}]},
                "MTL0",
            )
        self.assertEqual(hardware["runtime_version"], "Metal (macOS 14.6)")
        self.assertEqual(hardware["driver"], "macOS 14.6")

    def test_unidentified_metal_host_fails_closed(self) -> None:
        # When the host exposes neither a GPU label nor a readable macOS
        # version, the row cannot prove its runtime identity and must fail.
        with mock.patch.object(driver, "_apple_chip_name", return_value="Apple M2"), \
             mock.patch.object(driver, "_macos_version", return_value=None):
            with self.assertRaisesRegex(ValueError, "driver identity"):
                driver.build_hardware(self.row("macos-metal"),
                                      {"state": "available", "row_id": "macos-metal",
                                       "runner": "macos-hosted"},
                                      {"arch": "aarch64", "gpus": [{"display_name": "Type: GPU"}]},
                                      "MTL0")

    def test_gpu_row_without_cuda_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "driver identity"):
            driver.build_hardware(self.row("linux-cuda"), AVAILABLE,
                                  {"gpus": [{"display_name": "NVIDIA H100 PCIe"}]}, "CUDA0")


if __name__ == "__main__":
    unittest.main()
