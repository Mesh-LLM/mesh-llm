from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate-skippy-api-doc.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_skippy_api_doc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateSkippyApiDocTests(unittest.TestCase):
    def test_parses_exports_from_llama_and_common_libraries(self) -> None:
        generator = load_module()
        with tempfile.TemporaryDirectory(prefix="mesh-skippy-api-doc-") as temp_dir:
            header_path = Path(temp_dir) / "sample.h"
            header_path.write_text(
                """/**
 * @file sample.h
 * @brief Test exports from both native libraries.
 */

/** @brief Exported by llama. */
LLAMA_API int skippy_llama_export(void);

/** @brief Exported by llama-common. */
SKIPPY_COMMON_API int skippy_common_export(void);
""",
                encoding="utf-8",
            )

            header = generator.parse_header(header_path)

        self.assertEqual(
            [function.name for function in header.functions],
            ["skippy_llama_export", "skippy_common_export"],
        )


if __name__ == "__main__":
    unittest.main()
