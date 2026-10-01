"""Reject malformed SDK responses before awarding embedding certification."""

from __future__ import annotations

import base64
import contextlib
import io
from pathlib import Path
import runpy
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SMOKE = runpy.run_path(str(SCRIPTS / "ci-openai-embeddings-smoke.py"))


class EmbeddingSdkSmokeTests(unittest.TestCase):
    """Exercise the actual CLI validator with deterministic SDK-shaped responses."""

    def numeric_response(self) -> SimpleNamespace:
        """Supply a valid normalized vector for every canonical smoke input."""
        return SimpleNamespace(
            object="list", model="fixture",
            data=[SimpleNamespace(object="embedding", index=index, embedding=[1.0, 0.0])
                  for index in range(len(SMOKE["EMBEDDING_INPUTS"]))],
            usage=SimpleNamespace(prompt_tokens=2),
        )

    def encoded_response(self) -> SimpleNamespace:
        """Return the single-input base64 representation of the numeric fixture."""
        return SimpleNamespace(
            object="list", model="fixture",
            data=[SimpleNamespace(object="embedding", index=0,
                                  embedding=base64.b64encode(struct.pack("<2f", 1, 0)).decode())],
        )

    def run_smoke(self, encoded: SimpleNamespace) -> str:
        """Run both SDK requests without a network dependency or installed SDK."""
        client = Mock()
        client.embeddings.create.side_effect = [self.numeric_response(), encoded]
        sdk = SimpleNamespace(OpenAI=Mock(return_value=client))
        output = io.StringIO()
        with patch.dict(sys.modules, {"openai": sdk}), patch.object(
            sys, "argv", ["smoke", "--base-url", "http://127.0.0.1:9337/v1", "--model", "fixture"]
        ), contextlib.redirect_stdout(output):
            SMOKE["main"]()
        self.assertEqual(client.embeddings.create.call_count, 2)
        self.assertEqual(client.embeddings.create.call_args.kwargs, {
            "model": "fixture", "input": SMOKE["EMBEDDING_INPUTS"][0], "encoding_format": "base64",
        })
        return output.getvalue()

    def test_valid_single_item_response_passes(self) -> None:
        """A correctly labelled, sized and encoded response remains certifiable."""
        self.assertIn("smoke passed", self.run_smoke(self.encoded_response()))

    def test_numeric_batch_rejects_invalid_indexes_and_boolean_components(self) -> None:
        """The float response must enforce wire types as strictly as base64 metadata."""
        for field, value, message in (
            ("index", False, "invalid item metadata"),
            ("index", 0.0, "invalid item metadata"),
            ("embedding", [True, 0.0], "finite vector"),
        ):
            with self.subTest(field=field, value=value):
                numeric = self.numeric_response()
                setattr(numeric.data[0], field, value)
                with patch.object(self, "numeric_response", return_value=numeric):
                    with self.assertRaisesRegex(RuntimeError, message):
                        self.run_smoke(self.encoded_response())

    def test_empty_and_surplus_batches_fail(self) -> None:
        """Do not certify a nonempty response that silently adds extra vectors."""
        for size in (0, 2):
            with self.subTest(size=size):
                response = self.encoded_response()
                response.data *= size
                with self.assertRaisesRegex(RuntimeError, "wrong batch size"):
                    self.run_smoke(response)

    def test_response_metadata_must_match_request(self) -> None:
        """A correct vector cannot mask the wrong envelope or selected model."""
        for field, value in (("object", "embedding"), ("model", "another-model")):
            with self.subTest(field=field):
                response = self.encoded_response()
                setattr(response, field, value)
                with self.assertRaisesRegex(RuntimeError, "wrong object or model"):
                    self.run_smoke(response)

    def test_item_metadata_must_match_single_input(self) -> None:
        """Reject missing, negative and surplus indexes and invalid item objects."""
        for field, value in (("object", "list"), ("object", None),
                             ("index", -1), ("index", 1), ("index", None),
                             ("index", False), ("index", 0.0), ("index", "0")):
            with self.subTest(field=field, value=value):
                response = self.encoded_response()
                setattr(response.data[0], field, value)
                with self.assertRaisesRegex(RuntimeError, "invalid item metadata"):
                    self.run_smoke(response)

    def test_invalid_payloads_still_fail(self) -> None:
        """Metadata validation must retain type, length and finite-value checks."""
        for payload, message in (
            ([1.0, 0.0], "string"),
            (base64.b64encode(struct.pack("<f", 1)).decode(), "byte length"),
            (base64.b64encode(struct.pack("<2f", float("nan"), 0)).decode(), "non-finite"),
        ):
            with self.subTest(message=message):
                response = self.encoded_response()
                response.data[0].embedding = payload
                with self.assertRaisesRegex(RuntimeError, message):
                    self.run_smoke(response)

    def test_base64_values_must_match_float_response(self) -> None:
        """A finite normalized vector of the right size still needs value parity."""
        response = self.encoded_response()
        response.data[0].embedding = base64.b64encode(struct.pack("<2f", 0, 1)).decode()
        with self.assertRaisesRegex(RuntimeError, "differs from float"):
            self.run_smoke(response)

    def test_base64_parity_allows_float32_rounding(self) -> None:
        """Float32 rounding must not reject otherwise equivalent representations."""
        response = self.encoded_response()
        response.data[0].embedding = base64.b64encode(
            struct.pack("<2f", 1 + 1e-7, 1e-7)
        ).decode()
        self.run_smoke(response)


if __name__ == "__main__":
    unittest.main()
