"""Regression tests for deterministic OCR media and multimodal parity checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import zlib


ROOT = Path(__file__).resolve().parents[2]


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = import_script("ocr_oracle_fixture", ROOT / "scripts" / "generate-ocr-oracle-fixture.py")
oracle = import_script("skippy_ocr_asr_oracle", ROOT / "scripts" / "skippy-ocr-asr-oracle.py")


class OcrFixtureTests(unittest.TestCase):
    def test_png_is_deterministic_and_text_bearing(self):
        png = fixture.png_bytes()
        self.assertEqual(png, fixture.png_bytes())
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(fixture.TEXT, "MESH 42")
        size = struct.unpack(">II", png[16:24])
        self.assertEqual(size, (fixture.WIDTH, fixture.HEIGHT))

        image_data_start = png.index(b"IDAT") + 4
        compressed_length = struct.unpack(">I", png[image_data_start - 8:image_data_start - 4])[0]
        raw = zlib.decompress(png[image_data_start:image_data_start + compressed_length])
        self.assertEqual(len(raw), fixture.HEIGHT * (1 + fixture.WIDTH * 3))
        self.assertIn(b"\x00\x00\x00", raw)
        self.assertIn(b"\xff\xff\xff", raw)


class MultimodalOracleTests(unittest.TestCase):
    def test_ocr_requires_both_parity_and_known_text(self):
        self.assertIn("mesh 42", oracle.compare_text("MESH 42", "Mesh 42.", "MESH 42"))
        with self.assertRaisesRegex(RuntimeError, "differs from monolithic"):
            oracle.compare_text("MESH 42", "MESH 43", "MESH 42")
        with self.assertRaisesRegex(RuntimeError, "does not exactly match independently known"):
            oracle.compare_text("unrelated", "Unrelated.", "MESH 42")
        with self.assertRaisesRegex(RuntimeError, "does not exactly match independently known"):
            oracle.compare_text("MESH 42 extra", "Mesh 42 extra.", "MESH 42")

    def test_asr_unlabeled_fixture_does_not_claim_accuracy(self):
        detail = oracle.compare_text("The mesh is ready.", "the mesh is ready", None)
        self.assertIn("no accuracy claim", detail)
        with self.assertRaisesRegex(RuntimeError, "returned empty text"):
            oracle.compare_text("!", "?", None)

    def test_asr_normalizes_only_known_decorative_prefixes(self):
        detail = oracle.compare_text(
            'The text is: "The mesh is ready"',
            'The audio is: "The mesh is ready"',
            None,
            transcript=True,
        )
        self.assertIn("identical normalized text", detail)
        with self.assertRaisesRegex(RuntimeError, "differs from monolithic"):
            oracle.compare_text("The text is: mesh ready", "The audio is: mesh not ready",
                                None, transcript=True)
        with self.assertRaisesRegex(RuntimeError, "differs from monolithic"):
            oracle.compare_text("Transcribed text: mesh ready", "The audio is: mesh ready",
                                None, transcript=True)

    def test_asr_matching_refusals_do_not_pass_as_transcripts(self):
        with self.assertRaisesRegex(RuntimeError, "not a transcript"):
            oracle.compare_text("I can't fulfill this request.",
                                "I can't fulfill this request.", None, transcript=True)
        with self.assertRaisesRegex(RuntimeError, "not a transcript"):
            oracle.compare_text("I can help you with transcribing audio to text.",
                                "I can help you with transcribing audio to text.",
                                None, transcript=True)

    def test_asr_repeated_reference_content_is_not_normalized_away(self):
        with self.assertRaisesRegex(RuntimeError, "differs from monolithic"):
            oracle.compare_text("The mesh is ready", "The mesh is ready. The mesh is ready.",
                                None, transcript=True)

    def test_ocr_sends_same_request_to_both_servers(self):
        reply = {"choices": [{"message": {"content": "MESH 42"}}]}
        with patch.object(oracle, "request_json", side_effect=[reply, reply]) as request:
            oracle.compare_ocr("http://candidate/v1", "http://reference/v1", "ocr", b"png", "MESH 42")
        self.assertEqual(request.call_count, 2)
        candidate_call, reference_call = request.call_args_list
        self.assertEqual(candidate_call.args[1:], reference_call.args[1:])
        self.assertEqual(candidate_call.args[1], "/chat/completions")

    def test_asr_aligns_reference_chat_prompt_with_candidate_audio_route(self):
        with (
            patch.object(oracle, "request_multipart", return_value={"text": "The mesh is ready."}) as candidate,
            patch.object(oracle, "request_json", return_value={
                "choices": [{"message": {"content": "The mesh is ready."}}]
            }) as reference,
        ):
            oracle.compare_asr("http://candidate/v1", "http://reference/v1", "asr", b"wav", None)
        self.assertEqual(candidate.call_args.args,
                         ("http://candidate/v1", "/audio/transcriptions", "asr", b"wav"))
        self.assertEqual(reference.call_args.args[:2],
                         ("http://reference/v1", "/chat/completions"))
        payload = reference.call_args.args[2]
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0]["text"], oracle.ASR_PROMPT + "\n")
        self.assertEqual(parts[1]["input_audio"]["data"], "d2F2")
        self.assertEqual(payload["max_tokens"], oracle.ASR_MAX_TOKENS)
        # The multipart API has no max_tokens field; set the candidate server's
        # default explicitly instead of comparing its CLI default of 16 to 128.
        runner = (ROOT / "scripts/skippy-workload-certify.sh").read_text()
        self.assertIn(f"--default-max-tokens {oracle.ASR_MAX_TOKENS}", runner)

    def test_asr_multipart_contains_deterministic_fields_and_audio(self):
        with patch.object(oracle, "response_json", return_value={"text": "ok"}) as response:
            oracle.request_multipart("http://localhost/v1/", "/audio/transcriptions", "asr", b"RIFF\x00audio")
        request = response.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost/v1/audio/transcriptions")
        body = request.data
        self.assertIn(b'name="model"\r\n\r\nasr\r\n', body)
        self.assertIn(b'name="response_format"\r\n\r\njson\r\n', body)
        self.assertIn(b'name="temperature"\r\n\r\n0\r\n', body)
        self.assertIn(b"RIFF\x00audio", body)
        with self.assertRaisesRegex(RuntimeError, "collides with multipart boundary"):
            oracle.request_multipart("http://localhost/v1", "/audio/transcriptions", "asr",
                                     oracle.BOUNDARY.encode("ascii"))


if __name__ == "__main__":
    unittest.main()
