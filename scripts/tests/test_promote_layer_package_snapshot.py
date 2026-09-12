from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "promote_layer_package_snapshot.py"
SPEC = importlib.util.spec_from_file_location("promote_layer_package_snapshot", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@dataclass
class CopyOperation:
    src_path_in_repo: str
    path_in_repo: str
    src_revision: str


class FakeApi:
    def __init__(self) -> None:
        self.parent = "a" * 40
        self.refs = {
            "main": {
                "shared/metadata.gguf": b"old-artifact",
                "model-package.json": b"old-manifest",
            }
        }

    def model_info(self, repo_id: str, revision: str):
        del repo_id
        assert revision == "main"
        return type("Info", (), {"sha": self.parent})()

    def create_branch(self, repo_id: str, branch: str, revision: str, repo_type: str) -> None:
        del repo_id, repo_type
        assert revision == self.parent
        self.refs[branch] = dict(self.refs["main"])

    def create_commit(self, **kwargs) -> None:
        assert kwargs["revision"] == "main"
        assert kwargs["parent_commit"] == self.parent
        promoted = dict(self.refs["main"])
        for operation in kwargs["operations"]:
            promoted[operation.path_in_repo] = self.refs[operation.src_revision][
                operation.src_path_in_repo
            ]
        self.refs["main"] = promoted

    def delete_branch(self, repo_id: str, branch: str, repo_type: str) -> None:
        del repo_id, repo_type
        del self.refs[branch]


class SnapshotPromotionTests(unittest.TestCase):
    def test_partial_or_failed_replacement_leaves_main_readable(self) -> None:
        api = FakeApi()
        revision, _ = MODULE.prepare_snapshot(api, "meshllm/model", "b" * 40, "run-1")
        api.refs[revision]["shared/metadata.gguf"] = b"new-artifact"

        self.assertEqual(api.refs["main"]["shared/metadata.gguf"], b"old-artifact")
        self.assertEqual(api.refs["main"]["model-package.json"], b"old-manifest")

    def test_complete_snapshot_promotes_in_one_parent_guarded_commit(self) -> None:
        api = FakeApi()
        revision, parent = MODULE.prepare_snapshot(api, "meshllm/model", "b" * 40, "run-2")
        manifest = {
            "artifact_catalog": {"entries": [{"path": "shared/metadata.gguf"}]}
        }
        api.refs[revision]["shared/metadata.gguf"] = b"new-artifact"
        api.refs[revision]["model-package.json"] = json.dumps(manifest).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-package.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            MODULE.promote_snapshot(
                api, CopyOperation, "meshllm/model", path, revision, parent
            )

        self.assertEqual(api.refs["main"]["shared/metadata.gguf"], b"new-artifact")
        self.assertEqual(api.refs["main"]["model-package.json"], json.dumps(manifest).encode())
        self.assertNotIn(revision, api.refs)


if __name__ == "__main__":
    unittest.main()
