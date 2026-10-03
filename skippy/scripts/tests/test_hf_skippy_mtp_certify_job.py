import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "hf-skippy-mtp-certify-job.py"
SPEC = importlib.util.spec_from_file_location("hf_skippy_mtp_certify_job", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProjectorDownloadSafetyTests(unittest.TestCase):
    @mock.patch.object(MODULE.socket, "getaddrinfo")
    def test_accepts_public_hugging_face_https_url(self, getaddrinfo):
        getaddrinfo.return_value = [
            (MODULE.socket.AF_INET, MODULE.socket.SOCK_STREAM, 6, "", ("13.33.88.1", 443))
        ]

        parsed = MODULE.validate_projector_url(
            "https://huggingface.co/org/model/resolve/main/mmproj.gguf?download=true"
        )

        self.assertEqual(parsed.hostname, "huggingface.co")

    def test_rejects_untrusted_projector_host(self):
        with self.assertRaisesRegex(RuntimeError, "untrusted projector URL host"):
            MODULE.validate_projector_url("https://example.com/mmproj.gguf")

    @mock.patch.object(MODULE.socket, "getaddrinfo")
    def test_rejects_private_resolution_for_trusted_host(self, getaddrinfo):
        getaddrinfo.return_value = [
            (MODULE.socket.AF_INET, MODULE.socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))
        ]

        with self.assertRaisesRegex(RuntimeError, "non-public address"):
            MODULE.validate_projector_url("https://huggingface.co/mmproj.gguf")

    def test_copy_rejects_body_over_limit(self):
        response = mock.Mock()
        response.headers = {}
        response.read = mock.Mock(
            side_effect=[b"GGUF", b"x", b""],
        )
        output = io.BytesIO()

        with mock.patch.object(MODULE, "PROJECTOR_DOWNLOAD_MAX_BYTES", 4):
            with self.assertRaisesRegex(RuntimeError, "maximum supported size"):
                MODULE.copy_projector_response(response, output)


if __name__ == "__main__":
    unittest.main()
