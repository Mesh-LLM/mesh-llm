import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
import zipfile

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[3]
TOOL = ROOT / "mesh" / "scripts" / "bundled-plugins.py"
PACKAGE_RELEASE = ROOT / "mesh" / "scripts" / "package-release.sh"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
PINS = ROOT / "ci" / "bundled-plugins.json"

UNIX = ("aarch64-apple-darwin", "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu")


def archive_bytes(target: str) -> bytes:
    """A small, deterministic stand-in for one plugin release archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        data = f"plugin for {target}\n".encode()
        info = tarfile.TarInfo(f"example/{target}")
        info.size = len(data)
        info.mtime = 0
        tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fixture:
    """A pin file, a fake plugin release and a stub gh that serves it."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.release = root / "release"
        self.release.mkdir()
        self.archives = {t: archive_bytes(t) for t in UNIX}
        for target, data in self.archives.items():
            (self.release / f"example-1.2.3-{target}.tar.gz").write_bytes(data)
        self.write_sums()
        self.pins = root / "pins.json"
        self.write_pins()
        self.gh = root / "gh"
        self.gh.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import os, shutil, sys
                args = sys.argv[1:]
                with open({str(root / 'gh-calls.log')!r}, 'a') as log:
                    log.write(' '.join(args) + '\\n')
                if args[:2] == ['release', 'download']:
                    out = args[args.index('-D') + 1]
                    patterns = [args[i + 1] for i, a in enumerate(args) if a == '-p']
                    for name in patterns:
                        src = os.path.join({str(self.release)!r}, name)
                        if os.path.exists(src):
                            shutil.copyfile(src, os.path.join(out, name))
                    sys.exit(0)
                if args[:2] == ['attestation', 'verify']:
                    refused = os.environ.get('REFUSE_ATTESTATION', '')
                    sys.exit(1 if refused and args[2].endswith(refused) else 0)
                sys.exit(2)
                """
            ),
            encoding="utf-8",
        )
        self.gh.chmod(0o755)

    def write_sums(self, override: dict[str, str] | None = None) -> None:
        lines = []
        for target, data in self.archives.items():
            name = f"example-1.2.3-{target}.tar.gz"
            lines.append(f"{(override or {}).get(name, sha(data))}  {name}")
        (self.release / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_pins(self, archives: dict[str, str] | None = None) -> None:
        self.pins.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "plugins": [
                        {
                            "name": "example",
                            "repository": "Example/example",
                            "tag": "v1.2.3",
                            "version": "1.2.3",
                            "signer_workflow": "Example/example/.github/workflows/release.yml",
                            "archives": archives or {t: sha(d) for t, d in self.archives.items()},
                            "absent": {"x86_64-pc-windows-msvc": "no Windows build"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def run(self, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOL), "--pins", str(self.pins), *args],
            env={**os.environ, "BUNDLED_PLUGINS_GH": str(self.gh), **(env or {})},
            check=False,
            text=True,
            capture_output=True,
        )


class BundledPluginsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.fx = Fixture(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def fetched(self) -> pathlib.Path:
        out = self.root / "fetched"
        result = self.fx.run("fetch", "--out", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        return out

    def test_the_repository_pin_file_is_valid_and_lists_windows_as_absent(self) -> None:
        pins = json.loads(PINS.read_text(encoding="utf-8"))
        result = subprocess.run(
            [sys.executable, str(TOOL), "--pins", str(PINS), "verify", str(self.root)],
            check=False, text=True, capture_output=True,
        )
        self.assertNotIn("expected schema_version", result.stderr)
        self.assertIn("no release archives", result.stderr, "the pins load; only the empty directory is refused")
        for plugin in pins["plugins"]:
            self.assertEqual(sorted(plugin["archives"]), sorted(UNIX))
            self.assertIn("x86_64-pc-windows-msvc", plugin["absent"])
            self.assertEqual(plugin["tag"], "v" + plugin["version"])

    def test_fetch_verifies_sums_pin_and_attestation(self) -> None:
        out = self.fetched()
        for target, data in self.fx.archives.items():
            self.assertEqual((out / "example" / f"example-1.2.3-{target}.tar.gz").read_bytes(), data)
        calls = (self.root / "gh-calls.log").read_text(encoding="utf-8")
        self.assertIn("release download v1.2.3 -R Example/example", calls)
        self.assertEqual(calls.count("attestation verify"), 3)
        self.assertIn("--signer-workflow Example/example/.github/workflows/release.yml", calls)
        self.assertEqual(calls.count("--source-ref refs/tags/v1.2.3 --deny-self-hosted-runners"), 3)

    def test_fetch_refuses_what_it_cannot_stand_behind(self) -> None:
        name = "example-1.2.3-x86_64-unknown-linux-gnu.tar.gz"
        cases = {
            "a SHA256SUMS line that disagrees with the file": lambda: self.fx.write_sums({name: "0" * 64}),
            "a release archive that is not the pinned one": lambda: self.fx.write_pins({**{t: sha(d) for t, d in self.fx.archives.items()}, "x86_64-unknown-linux-gnu": "1" * 64}),
            "a pinned archive the release does not have": lambda: (self.fx.release / name).unlink(),
        }
        for label, break_it in cases.items():
            with self.subTest(label):
                self.tearDown()
                self.setUp()
                break_it()
                result = self.fx.run("fetch", "--out", str(self.root / "fetched"))
                self.assertEqual(result.returncode, 1, label)
                self.assertIn("bundled-plugins:", result.stderr, label)
        with self.subTest("an attestation that does not verify"):
            self.tearDown()
            self.setUp()
            result = self.fx.run("fetch", "--out", str(self.root / "fetched"), env={"REFUSE_ATTESTATION": name})
            self.assertEqual(result.returncode, 1)
            self.assertIn("attestation does not verify", result.stderr)

    def test_place_bundles_the_pinned_archive_with_a_manifest(self) -> None:
        out = self.fetched()
        bundle = self.root / "mesh-bundle"
        result = self.fx.run("place", "--from", str(out), "--target", "aarch64-unknown-linux-gnu", "--bundle", str(bundle))
        self.assertEqual(result.returncode, 0, result.stderr)
        placed = bundle / "plugins" / "example-1.2.3-aarch64-unknown-linux-gnu.tar.gz"
        self.assertEqual(placed.read_bytes(), self.fx.archives["aarch64-unknown-linux-gnu"])
        manifest = json.loads((bundle / "plugins" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["target"], "aarch64-unknown-linux-gnu")
        self.assertEqual([p["archive"] for p in manifest["plugins"]], [placed.name])
        self.assertEqual(manifest["plugins"][0]["sha256"], sha(self.fx.archives["aarch64-unknown-linux-gnu"]))
        self.assertEqual(manifest["absent"], [])

    def test_place_for_an_absent_target_bundles_nothing_and_says_why(self) -> None:
        out = self.fetched()
        bundle = self.root / "mesh-bundle"
        result = self.fx.run("place", "--from", str(out), "--target", "x86_64-pc-windows-msvc", "--bundle", str(bundle))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(p.name for p in (bundle / "plugins").iterdir()), ["manifest.json"])
        manifest = json.loads((bundle / "plugins" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["plugins"], [])
        self.assertEqual(manifest["absent"], [{"name": "example", "version": "1.2.3", "reason": "no Windows build"}])

    def test_place_refuses_a_missing_or_changed_archive(self) -> None:
        out = self.fetched()
        archive = out / "example" / "example-1.2.3-x86_64-unknown-linux-gnu.tar.gz"
        archive.write_bytes(b"not the pinned archive")
        result = self.fx.run("place", "--from", str(out), "--target", "x86_64-unknown-linux-gnu", "--bundle", str(self.root / "b1"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("pinned", result.stderr)
        archive.unlink()
        result = self.fx.run("place", "--from", str(out), "--target", "x86_64-unknown-linux-gnu", "--bundle", str(self.root / "b2"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("run fetch first", result.stderr)

    def release_archive(self, artifacts: pathlib.Path, name: str, members: dict[str, bytes]) -> None:
        artifacts.mkdir(exist_ok=True)
        path = artifacts / name
        if name.endswith(".zip"):
            with zipfile.ZipFile(path, "w") as zf:
                for member, data in members.items():
                    zf.writestr(member, data)
            return
        with tarfile.open(path, "w:gz") as tf:
            for member, data in members.items():
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    def placed_members(self, target: str) -> dict[str, bytes]:
        out = self.root / "fetched"
        if not out.exists():
            self.fetched()
        bundle = self.root / f"bundle-{target}" / "mesh-bundle"
        self.assertEqual(self.fx.run("place", "--from", str(out), "--target", target, "--bundle", str(bundle)).returncode, 0)
        members = {"mesh-bundle/mesh-llm": b"host"}
        for path in (bundle / "plugins").iterdir():
            members[f"mesh-bundle/plugins/{path.name}"] = path.read_bytes()
        return members

    def full_release(self, artifacts: pathlib.Path) -> None:
        for target in UNIX:
            members = self.placed_members(target)
            self.release_archive(artifacts, f"mesh-llm-v9.9.9-{target}.tar.gz", members)
            self.release_archive(artifacts, f"mesh-llm-{target}.tar.gz", members)
        self.release_archive(artifacts, "mesh-llm-x86_64-unknown-linux-gnu-cuda-12.tar.gz", self.placed_members("x86_64-unknown-linux-gnu"))
        self.release_archive(artifacts, "mesh-llm-v9.9.9-x86_64-pc-windows-msvc.zip", {"mesh-bundle/mesh-llm.exe": b"host"})
        self.release_archive(artifacts, "mesh-llm-node-sdk-addon-9.9.9-win32-x64.tar.gz", {"addon/x.node": b"addon"})

    def test_verify_accepts_a_release_whose_archives_carry_the_pinned_plugins(self) -> None:
        artifacts = self.root / "artifacts"
        self.full_release(artifacts)
        result = self.fx.run("verify", str(artifacts))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("node-sdk-addon", result.stdout)
        self.assertIn("ok mesh-llm-v9.9.9-x86_64-pc-windows-msvc.zip", result.stdout)

    def test_verify_fails_a_release_missing_a_plugin(self) -> None:
        cases = {
            "an archive without the plugin": ("mesh-llm-x86_64-unknown-linux-gnu-cuda-12.tar.gz", {"mesh-bundle/mesh-llm": b"host"}, "does not carry"),
            "an archive with a changed plugin": (
                "mesh-llm-aarch64-apple-darwin.tar.gz",
                {"mesh-bundle/mesh-llm": b"host", "mesh-bundle/plugins/example-1.2.3-aarch64-apple-darwin.tar.gz": b"changed"},
                "pinned",
            ),
            "a Windows archive carrying plugins": ("mesh-llm-x86_64-pc-windows-msvc.zip", {"mesh-bundle/plugins/manifest.json": b"{}"}, "lists as absent"),
        }
        for label, (name, members, words) in cases.items():
            with self.subTest(label):
                artifacts = self.root / ("artifacts-" + str(abs(hash(label))))
                self.full_release(artifacts)
                (artifacts / name).unlink(missing_ok=True)
                self.release_archive(artifacts, name, members)
                result = self.fx.run("verify", str(artifacts))
                self.assertEqual(result.returncode, 1, label)
                self.assertIn(words, result.stderr)

    def test_verify_fails_a_product_archive_it_cannot_read_a_target_from(self) -> None:
        artifacts = self.root / "artifacts"
        self.full_release(artifacts)
        self.release_archive(artifacts, "mesh-llm-v9.9.9-riscv64gc-unknown-linux-gnu.tar.gz", {"mesh-bundle/mesh-llm": b"host"})
        result = self.fx.run("verify", str(artifacts))
        self.assertEqual(result.returncode, 1)
        self.assertIn("target is not recognized", result.stderr)

    def test_verify_fails_when_a_pinned_target_has_no_release_archive(self) -> None:
        artifacts = self.root / "artifacts"
        self.full_release(artifacts)
        for path in artifacts.glob("mesh-llm-*aarch64-apple-darwin*"):
            path.unlink()
        result = self.fx.run("verify", str(artifacts))
        self.assertEqual(result.returncode, 1)
        self.assertIn("no release archive for aarch64-apple-darwin", result.stderr)

    def test_package_release_bundles_plugins_only_when_given_them(self) -> None:
        out = self.fetched()
        for env, expect in ((None, False), (str(out), True)):
            with self.subTest(bundled=expect):
                bundle = self.root / ("bundle-" + str(expect)) / "mesh-bundle"
                bundle.mkdir(parents=True)
                result = subprocess.run(
                    ["bash", "-c", f'source "{PACKAGE_RELEASE}"; resolve_release_target; bundle_plugins "{bundle}"'],
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "MESH_RELEASE_OS": "Linux",
                        "MESH_RELEASE_ARCH": "x86_64",
                        "MESH_RELEASE_FLAVOR": "cuda",
                        "MESH_CUDA_VERSION": "12.9.2",
                        "MESH_LLM_BUNDLED_PLUGINS_PINS": str(self.fx.pins),
                        **({"MESH_LLM_BUNDLED_PLUGINS_DIR": env} if env else {}),
                    },
                    check=False, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((bundle / "plugins").exists(), expect)
                if expect:
                    self.assertTrue((bundle / "plugins" / "example-1.2.3-x86_64-unknown-linux-gnu.tar.gz").is_file(), "a GPU flavour carries its target's plugin")

    def test_package_release_bundles_before_writing_either_archive(self) -> None:
        script = PACKAGE_RELEASE.read_text(encoding="utf-8")
        main = script[script.index("\nmain() {"):]
        call = main.index('    bundle_plugins "$bundle_dir"')
        self.assertLess(call, main.index('create_archive "$bundle_dir" "$output_dir/$versioned_asset"'))
        self.assertLess(call, main.index('create_archive "$bundle_dir" "$output_dir/$STABLE_ASSET"'))

    def test_release_workflow_bundles_into_every_unix_product_and_verifies_before_publishing(self) -> None:
        workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
        jobs = workflow["jobs"]
        fetch = jobs["bundled_plugins"]
        self.assertEqual(fetch["permissions"], {"contents": "read", "attestations": "read"})
        self.assertIn("scripts/bundled-plugins.py fetch --out bundled-plugins", "\n".join(s.get("run", "") for s in fetch["steps"]))
        unix = ["compose_cpu_products", "compose_linux_arm64_cpu", "compose_linux_aarch64_cuda", "compose_linux_cuda", "compose_linux_rocm", "compose_linux_vulkan"]
        for name in unix:
            with self.subTest(name):
                job = jobs[name]
                self.assertIn("bundled_plugins", job["needs"])
                package = [s for s in job["steps"] if "scripts/package-release.sh" in s.get("run", "")]
                self.assertEqual(len(package), 1)
                self.assertEqual(package[0]["env"]["MESH_LLM_BUNDLED_PLUGINS_DIR"], "${{ github.workspace }}/bundled-plugins")
        for name in ("compose_windows_cpu", "compose_windows_gpu"):
            self.assertNotIn("bundled_plugins", jobs[name].get("needs", []), "Windows is listed as absent")
        publish_runs = [s.get("run", "") for s in jobs["publish"]["steps"]]
        verify_at = next(i for i, r in enumerate(publish_runs) if "scripts/bundled-plugins.py verify release-artifacts" in r)
        publish_at = next(i for i, s in enumerate(jobs["publish"]["steps"]) if s.get("name") == "Publish GitHub release")
        self.assertLess(verify_at, publish_at)


if __name__ == "__main__":
    unittest.main()
