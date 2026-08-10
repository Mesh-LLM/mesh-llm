#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "publish-crates.sh"


def _publish_chain_crates() -> list[str]:
    """The crate order the publish script itself declares.

    Mirrors the array parsing in `tools/xtask/src/publish_consistency.rs`:
    comment lines are skipped and surrounding quotes are stripped, so a quoted
    entry still matches the package name the script publishes.
    """
    contents = SCRIPT.read_text(encoding="utf-8")
    array = re.search(r"publish_crates=\(\n(.*?)\n\)", contents, re.S)
    if array is None:
        raise AssertionError("scripts/publish-crates.sh: missing publish_crates array")
    entries = []
    for line in array.group(1).splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        entries.append(trimmed.strip('"'))
    return entries


PUBLISH_CHAIN_CRATES = _publish_chain_crates()

# Path dependencies the fixture models by default. These mirror a few real
# workspace edges so the derived skip logic has something to resolve.
DEFAULT_WORKSPACE_DEPS = {
    "model-artifact": ["model-ref"],
    "model-hf": ["model-artifact", "model-ref"],
    "mesh-llm-api-server": ["mesh-llm-api-client", "mesh-llm-node"],
}


class PublishCratesScriptTests(unittest.TestCase):
    def test_publish_verification_uses_dynamic_llama_link_mode(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({})
            fixture.write_fake_cargo()
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(["--dry-run", "--allow-dirty"])

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            link_modes = fixture.read_log("cargo-link-mode.log").splitlines()
            self.assertTrue(link_modes)
            self.assertEqual(set(link_modes), {"dynamic"})

    def test_retries_cargo_publish_429_then_continues_chain(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({})
            fixture.write_fake_cargo(
                fail_crates={"model-artifact": 1},
                failure_output=CRATES_IO_429,
            )
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(
                env={
                    "CARGO_REGISTRY_TOKEN": "test-token",
                    "CRATES_IO_PUBLISH_MAX_ATTEMPTS": "3",
                    "CRATES_IO_PUBLISH_SETTLE_SECONDS": "0",
                }
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            cargo_log = fixture.read_log("cargo.log")
            self.assertEqual(cargo_log.count("-p model-artifact"), 2)
            self.assertIn("-p model-hf", cargo_log)
            self.assertIn("-p mesh-llm-client", cargo_log)
            self.assertIn("-p mesh-llm-api-server", cargo_log)
            self.assertRegex(fixture.read_log("sleep.log"), r"^[1-9][0-9]*$")
            self.assertIn(
                "crates.io rate limit hit for model-artifact@0.68.0",
                result.stderr,
            )

    def test_exhausts_429_retries_and_fails_loudly_without_continuing(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({})
            fixture.write_fake_cargo(
                fail_crates={"model-artifact": 5},
                failure_output=CRATES_IO_429,
            )
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(
                env={
                    "CARGO_REGISTRY_TOKEN": "test-token",
                    "CRATES_IO_PUBLISH_MAX_ATTEMPTS": "2",
                    "CRATES_IO_PUBLISH_SETTLE_SECONDS": "0",
                }
            )

            self.assertNotEqual(result.returncode, 0)
            cargo_log = fixture.read_log("cargo.log")
            self.assertEqual(cargo_log.count("-p model-artifact"), 2)
            self.assertNotIn("-p model-hf", cargo_log)
            self.assertIn(
                "retry limit exceeded for model-artifact@0.68.0 after 2 attempts",
                result.stderr,
            )

    def test_dry_run_skips_dependents_of_a_brand_new_workspace_crate(self) -> None:
        """A crate added to the workspace is not yet on crates.io.

        v0.75.1 failed here: `skippy-tokenizer` was new, `skippy-protocol`
        depends on it, and the hand-maintained dependency map had no
        `skippy-protocol` entry, so its dry-run was not skipped and
        `cargo publish` could not resolve the dependency from the registry.
        Deriving the map from cargo metadata must skip the dependent instead.
        """
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({})
            fixture.write_fake_cargo(
                workspace_deps={"skippy-protocol": ["skippy-tokenizer"]}
            )
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(["--dry-run", "--allow-dirty"])

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn(
                "dry-run cannot verify skippy-protocol until "
                "skippy-tokenizer@0.68.0 exists in crates.io",
                result.stdout,
            )
            self.assertNotIn(
                "-p skippy-protocol --dry-run",
                fixture.read_log("cargo.log"),
            )

    def test_dry_run_skips_crates_with_unpublished_registry_deps_without_sleeping(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({"model-ref": 404})
            fixture.write_fake_cargo()
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(["--dry-run", "--allow-dirty"])

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn(
                "dry-run cannot verify model-artifact until model-ref@0.68.0 exists in crates.io",
                result.stdout,
            )
            self.assertIn(
                "publish --locked -p model-ref --dry-run --allow-dirty",
                fixture.read_log("cargo.log"),
            )
            self.assertEqual(fixture.read_log("sleep.log"), "")

    def test_real_publish_requires_registry_token_before_any_cargo_call(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({})
            fixture.write_fake_cargo()
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run()

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CARGO_REGISTRY_TOKEN is required", result.stderr)
            self.assertFalse((fixture.tmp_path / "cargo.log").exists())

    def test_real_publish_uses_cargo_to_detect_already_uploaded_versions(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({"model-ref": 500})
            fixture.write_fake_cargo(
                fail_crates={"model-ref": 1},
                failure_output="error: crate version is already uploaded\n",
            )
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(
                env={
                    "CARGO_REGISTRY_TOKEN": "test-token",
                    "CRATES_IO_PUBLISH_SETTLE_SECONDS": "0",
                }
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn(
                "model-ref@0.68.0 already published according to cargo; continuing",
                result.stdout,
            )
            self.assertIn("-p model-ref", fixture.read_log("cargo.log"))
            self.assertEqual(fixture.read_log("curl.log"), "")

    def test_real_publish_does_not_probe_registry_before_cargo_publish(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({"model-ref": 500})
            fixture.write_fake_cargo()
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(
                env={
                    "CARGO_REGISTRY_TOKEN": "test-token",
                    "CRATES_IO_PUBLISH_SETTLE_SECONDS": "0",
                }
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("-p model-ref", fixture.read_log("cargo.log"))
            self.assertEqual(fixture.read_log("curl.log"), "")

    def test_cargo_failure_output_redacts_registry_token(self) -> None:
        with PublishCratesFixture() as fixture:
            fixture.write_curl_statuses({})
            fixture.write_fake_cargo(
                fail_crates={"model-ref": 1},
                failure_output="fatal: registry token secret-token leaked in diagnostic\n",
            )
            fixture.write_fake_sleep()
            fixture.write_fake_date()

            result = fixture.run(env={"CARGO_REGISTRY_TOKEN": "secret-token"})

            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("secret-token", result.stderr)
            self.assertIn("<redacted>", result.stderr)


class PublishCratesFixture:
    def __init__(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.bin_dir = self.tmp_path / "bin"
        self.bin_dir.mkdir()
        (self.tmp_path / "Cargo.toml").write_text(
            '[workspace.package]\nversion = "0.68.0"\n',
            encoding="utf-8",
        )

    def __enter__(self) -> "PublishCratesFixture":
        return self

    def __exit__(self, *args: object) -> None:
        self.tmpdir.cleanup()

    def run(
        self,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        merged_env.update(env or {})
        merged_env["PATH"] = f"{self.bin_dir}{os.pathsep}{merged_env['PATH']}"
        return subprocess.run(
            ["bash", str(SCRIPT), *(args or [])],
            cwd=self.tmp_path,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def write_fake_cargo(
        self,
        *,
        fail_crates: dict[str, int] | None = None,
        failure_output: str = "",
        workspace_deps: dict[str, list[str]] | None = None,
    ) -> None:
        failure_path = self.tmp_path / "cargo-failure.txt"
        failure_path.write_text(failure_output, encoding="utf-8")
        metadata_path = self.tmp_path / "cargo-metadata.json"
        metadata_path.write_text(
            self._workspace_metadata_json(workspace_deps), encoding="utf-8"
        )
        fail_cases = "\n".join(
            f"{crate}:{count}" for crate, count in (fail_crates or {}).items()
        )
        self._write_executable(
            "cargo",
            f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "metadata" ]]; then
    cat "{metadata_path}"
    exit 0
fi
crate=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "-p" ]]; then
        crate="$arg"
        break
    fi
    prev="$arg"
done
echo "$*" >> "{self.tmp_path}/cargo.log"
echo "${{LLAMA_STAGE_LINK_MODE:-}}" >> "{self.tmp_path}/cargo-link-mode.log"
case "$crate" in
{self._cargo_case_arms(fail_cases, failure_path)}
esac
exit 0
""",
        )

    def _cargo_case_arms(self, fail_cases: str, failure_path: Path) -> str:
        arms: list[str] = []
        for line in fail_cases.splitlines():
            crate, count = line.split(":", 1)
            state_file = self.tmp_path / f"cargo-{crate}.count"
            arms.append(
                textwrap.dedent(
                    f"""
                    {crate})
                        current=0
                        if [[ -f "{state_file}" ]]; then
                            current="$(cat "{state_file}")"
                        fi
                        current="$((current + 1))"
                        echo "$current" > "{state_file}"
                        if [[ "$current" -le "{count}" ]]; then
                            cat "{failure_path}" >&2
                            exit 101
                        fi
                        ;;
                    """
                ).strip()
            )
        return "\n".join(arms)

    def write_curl_statuses(self, statuses: dict[str, int]) -> None:
        cases = "\n".join(
            f"*crates/{crate}/0.68.0*) status={status} ;;"
            for crate, status in statuses.items()
        )
        self._write_executable(
            "curl",
            f"""#!/usr/bin/env bash
set -euo pipefail
url="${{@: -1}}"
status=404
case "$url" in
{cases}
esac
echo "$url" >> "{self.tmp_path}/curl.log"
printf '%s' "$status"
""",
        )

    def write_fake_sleep(self) -> None:
        self._write_executable(
            "sleep",
            f"""#!/usr/bin/env bash
set -euo pipefail
echo "$1" >> "{self.tmp_path}/sleep.log"
""",
        )

    def write_fake_date(self) -> None:
        self._write_executable(
            "date",
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"Fri, 22 May 2026 09:58:23 GMT"* ]]; then
    printf '%s\n' 1779443903
elif [[ "$*" == *"+%s"* ]]; then
    printf '%s\n' 1779443600
else
    /bin/date "$@"
fi
""",
        )

    def read_log(self, name: str) -> str:
        path = self.tmp_path / name
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _workspace_metadata_json(
        self, workspace_deps: dict[str, list[str]] | None
    ) -> str:
        """Fake `cargo metadata --no-deps` output for the publish chain.

        The script derives each crate's publishable workspace dependencies from
        this, so the fixture models path dependencies the same way Cargo does:
        `path` points at the dependency's manifest directory.
        """
        deps = dict(DEFAULT_WORKSPACE_DEPS if workspace_deps is None else workspace_deps)
        crate_names = set(PUBLISH_CHAIN_CRATES)
        for crate, crate_deps in deps.items():
            crate_names.add(crate)
            crate_names.update(crate_deps)

        packages = []
        for crate in sorted(crate_names):
            packages.append(
                {
                    "name": crate,
                    "version": "0.68.0",
                    "manifest_path": f"{self.tmp_path}/crates/{crate}/Cargo.toml",
                    "dependencies": [
                        {
                            "name": dep,
                            "req": "^0.68.0",
                            "kind": None,
                            "path": f"{self.tmp_path}/crates/{dep}",
                        }
                        for dep in deps.get(crate, [])
                    ],
                }
            )
        return json.dumps({"packages": packages})

    def _write_executable(self, name: str, content: str) -> None:
        path = self.bin_dir / name
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


CRATES_IO_429 = """error: failed to publish model-artifact v0.68.0
status 429 Too Many Requests:
"You have published too many new crates in a short period of time.
Please try again after Fri, 22 May 2026 09:58:23 GMT
and see https://crates.io/docs/rate-limits for more details."
"""


if __name__ == "__main__":
    unittest.main()
