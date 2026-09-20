"""Contract tests for the System One (OpenJEV) canary smoke.

Two layers, because the smoke has two halves:

* `SystemOneCaseDriverTests` runs `scripts/skippy-system-one-cases.py` against a
  stub HTTP server. The stub implements the request contract exactly as
  `crates/skippy-server/src/frontend/system_one.rs` documents it, and each
  mutation test flips one documented behaviour to prove the driver actually
  detects the regression instead of agreeing with whatever it is handed. This
  cannot prove the Rust matches the stub; it proves the assertions are live.
* `SystemOneSmokeGateTests` runs `scripts/skippy-system-one-smoke.sh` in its
  hermetic modes and asserts the gate semantics: a non-qualified backend and a
  missing cache entry are loud, reported outcomes rather than silent skips, and
  a red contract part always fails.
* `SystemOneCanaryWiringTests` asserts the smoke is actually wired into the
  unchanged-pin canary, the changed-pin repair gates, and the independent
  verification pass, and that nothing here claims family certification.
"""

from __future__ import annotations

import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts" / "skippy-system-one-cases.py"
SMOKE = ROOT / "scripts" / "skippy-system-one-smoke.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "llama-upstream-canary.yml"
WRAPPER = ROOT / "scripts" / "llama-canary-agent-repair.sh"
REGISTRY = ROOT / "ci" / "model-artifacts" / "registry.json"
SUITE_MANIFEST = ROOT / "ci" / "model-artifacts" / "manifests" / "skippy-system-one-smoke.json"
FAMILY_MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.json"
FAMILY_MAP = ROOT / "ci" / "llama-canary" / "generated-family-map.json"

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DENSE_MODEL = "Qwen/Qwen3-0.6B-GGUF:Q8_0"
ALIAS = "openjev-latest"
UNSUPPORTED_CODE = "unsupported_model_feature"
INVALID_CODE = "invalid_value"


def default_behaviour() -> dict[str, object]:
    """Every knob the stub honours. Flipping one produces a mutation."""
    return {
        "choice_min": 2,
        "choice_max": len(CHOICE_LABELS),
        "score_min": 2,
        "score_max": 10,
        "unsupported_status": 400,
        "unsupported_code": UNSUPPORTED_CODE,
        "invalid_status": 400,
        "invalid_code": INVALID_CODE,
        "method_not_allowed_status": 405,
        "arch_status": 502,
        "arch_type": "server_error",
        "arch_code": "service_unavailable",
        "arch_message": "system-one reads require a DiffusionGemma model",
        "normalize_probabilities": True,
        "deterministic_repeat": True,
        "constant_read": False,
        "choice_is_argmax": True,
        "score_is_expectation": True,
        "drop_usage": False,
        "echo_requested_model": True,
        "output_tokens": 0,
    }


class _SystemOneStub(BaseHTTPRequestHandler):
    behaviour: dict[str, object] = default_behaviour()
    calls: int = 0
    mode = "contract"

    def log_message(self, *args: object) -> None:  # keep the test output clean
        pass

    # -- helpers ---------------------------------------------------------
    def _respond(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, error_type: str, code: str) -> None:
        self._respond(
            status,
            {"error": {"message": message, "type": error_type, "param": None, "code": code}},
        )

    def _unsupported(self, message: str) -> None:
        behaviour = self.behaviour
        self._error(
            int(behaviour["unsupported_status"]),
            message,
            "invalid_request_error",
            str(behaviour["unsupported_code"]),
        )

    def _invalid(self, message: str) -> None:
        behaviour = self.behaviour
        self._error(
            int(behaviour["invalid_status"]),
            message,
            "invalid_request_error",
            str(behaviour["invalid_code"]),
        )

    # -- routing ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.split("?")[0] == "/v1/systemone":
            self._error(
                int(self.behaviour["method_not_allowed_status"]),
                "method not allowed: GET",
                "invalid_request_error",
                "method_not_allowed",
            )
            return
        self._respond(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if self.path.split("?")[0] != "/v1/systemone":
            self._respond(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._invalid("invalid JSON request body")
            return
        if not isinstance(request, dict):
            self._invalid("request body must be an object")
            return
        self._handle(request)

    # -- the documented contract ----------------------------------------
    def _handle(self, request: dict[str, object]) -> None:
        behaviour = self.behaviour
        model = request.get("model")
        if model not in (DENSE_MODEL, ALIAS):
            self._invalid(f"model {model!r} is not loaded; use {DENSE_MODEL!r} or openjev-latest")
            return
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            self._invalid("System One needs at least one question")
            return
        images = request.get("images")
        if (
            (isinstance(images, list) and images)
            or request.get("steps") not in (None, 1)
            or request.get("samples") not in (None, 1)
            or request.get("think") not in (None, 0)
            or request.get("sequential") is True
        ):
            self._unsupported(
                "this PoC supports one text-only System One read; images, multiple steps/"
                "samples, thinking, and sequential reads are not yet supported"
            )
            return
        prepared: list[tuple[str, str, list[str]]] = []
        for key, question in questions.items():
            if not isinstance(question, dict):
                self._invalid(f"question {key!r} must be an object")
                return
            kind = question.get("type")
            if kind == "noul":
                prepared.append((key, "noul", ["yes", "no"]))
            elif kind == "choice":
                criteria = question.get("criteria")
                count = len(criteria) if isinstance(criteria, dict) else 0
                if not int(behaviour["choice_min"]) <= count <= int(behaviour["choice_max"]):
                    self._invalid(
                        f"question {key!r}: choice criteria must contain "
                        f"{behaviour['choice_min']} to {behaviour['choice_max']} options"
                    )
                    return
                prepared.append((key, "choice", list(criteria)))
            elif kind == "score":
                criteria = question.get("criteria")
                count = len(criteria) if isinstance(criteria, list) else 0
                if not int(behaviour["score_min"]) <= count <= int(behaviour["score_max"]):
                    self._invalid(
                        f"question {key!r}: score criteria must contain "
                        f"{behaviour['score_min']} to {behaviour['score_max']} levels"
                    )
                    return
                prepared.append((key, "score", [str(index) for index in range(count)]))
            else:
                self._invalid(f"question {key!r}: unknown question type {kind!r}")
                return

        if self.mode == "contract":
            self._error(
                int(behaviour["arch_status"]),
                str(behaviour["arch_message"]),
                str(behaviour["arch_type"]),
                str(behaviour["arch_code"]),
            )
            return

        type(self).calls += 1
        answers: dict[str, object] = {}
        for key, kind, labels in prepared:
            probabilities = self._probabilities(request, key, kind, labels)
            if kind == "noul":
                answers[key] = {"type": "noul", "noul": probabilities[0]}
            elif kind == "choice":
                winner = max(range(len(labels)), key=lambda index: probabilities[index])
                if not behaviour["choice_is_argmax"]:
                    winner = (winner + 1) % len(labels)
                answers[key] = {
                    "type": "choice",
                    "choice": labels[winner],
                    "probabilities": dict(zip(labels, probabilities)),
                    "confidence": 0.5,
                }
            else:
                expectation = sum(
                    index * probability for index, probability in enumerate(probabilities)
                )
                answers[key] = {
                    "type": "score",
                    "score": expectation if behaviour["score_is_expectation"] else 0.0,
                    "legend": {str(index): f"level {index}" for index in range(len(labels))},
                    "probabilities": dict(zip(labels, probabilities)),
                    "confidence": 0.5,
                }
        response: dict[str, object] = {
            "model": model if behaviour["echo_requested_model"] else (
                "openjev-0.1" if model == ALIAS else model
            ),
            "answers": answers,
        }
        if not behaviour["drop_usage"]:
            response["usage"] = {"input_tokens": 42, "output_tokens": behaviour["output_tokens"]}
        self._respond(200, response)

    def _probabilities(
        self, request: dict[str, object], key: str, kind: str, labels: list[str]
    ) -> list[float]:
        behaviour = self.behaviour
        if behaviour["constant_read"]:
            seed_source = "constant"
        else:
            seed_source = json.dumps(
                {"state": request.get("state"), "key": key, "kind": kind}, sort_keys=True
            )
        if not behaviour["deterministic_repeat"]:
            seed_source = f"{seed_source}#{type(self).calls}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
        # A stable, request-dependent, strictly positive distribution.
        weights = [1.0 + digest[index % len(digest)] / 32.0 for index in range(len(labels))]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        if not behaviour["normalize_probabilities"]:
            probabilities = [probability * 0.5 for probability in probabilities]
        return probabilities


class _StubServer:
    def __init__(self, mode: str, **overrides: object) -> None:
        behaviour = default_behaviour()
        behaviour.update(overrides)
        self._handler = type(
            "_BoundHandler",
            (_SystemOneStub,),
            {"behaviour": behaviour, "calls": 0, "mode": mode},
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def run_driver(
    base_url: str, mode: str, model: str = DENSE_MODEL, alias: str = ALIAS
) -> tuple[int, dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="system-one-driver-") as directory:
        report_path = Path(directory) / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                str(DRIVER),
                "--base-url",
                base_url,
                "--model",
                model,
                "--alias",
                alias,
                "--mode",
                mode,
                "--timeout",
                "30",
                "--json-out",
                str(report_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return result.returncode, report


class SystemOneCaseDriverTests(unittest.TestCase):
    def _run(self, mode: str, **overrides: object) -> tuple[int, dict[str, object]]:
        stub = _StubServer(mode, **overrides)
        self.addCleanup(stub.stop)
        return run_driver(stub.base_url, mode)

    def test_contract_mode_passes_against_the_documented_contract(self) -> None:
        status, report = self._run("contract")
        self.assertEqual(0, status, report["failures"])
        self.assertEqual("pass", report["status"])

    def test_full_read_mode_passes_against_a_healthy_reader(self) -> None:
        status, report = self._run("full-read")
        self.assertEqual(0, status, report["failures"])
        self.assertEqual("pass", report["status"])

    def test_contract_mode_detects_a_missing_rejection(self) -> None:
        # A server that answers where it must refuse is the failure this lane
        # exists for: `sequential` reads silently changing meaning.
        status, report = self._run(
            "contract", unsupported_status=200, unsupported_code="ok"
        )
        self.assertEqual(1, status)
        self.assertEqual("fail", report["status"])
        self.assertTrue(report["failures"])

    def test_contract_mode_detects_a_different_error_code(self) -> None:
        status, _ = self._run("contract", unsupported_code="service_unavailable")
        self.assertEqual(1, status)

    def test_contract_mode_detects_shifted_choice_bounds(self) -> None:
        status, _ = self._run("contract", choice_min=1)
        self.assertEqual(1, status)

    def test_contract_mode_detects_shifted_score_bounds(self) -> None:
        status, _ = self._run("contract", score_max=11)
        self.assertEqual(1, status)

    def test_contract_mode_detects_a_wrong_architecture_diagnostic(self) -> None:
        status, _ = self._run("contract", arch_message="boom")
        self.assertEqual(1, status)

    def test_contract_mode_detects_a_non_typed_architecture_refusal(self) -> None:
        status, _ = self._run("contract", arch_status=200)
        self.assertEqual(1, status)

    def test_full_read_mode_detects_unnormalized_probabilities(self) -> None:
        status, _ = self._run("full-read", normalize_probabilities=False)
        self.assertEqual(1, status)

    def test_full_read_mode_detects_leaked_diffusion_state(self) -> None:
        status, report = self._run("full-read", deterministic_repeat=False)
        self.assertEqual(1, status)
        self.assertIn("identical reads", " ".join(report["failures"]))

    def test_full_read_mode_detects_a_constant_reader(self) -> None:
        status, report = self._run("full-read", constant_read=True)
        self.assertEqual(1, status)
        self.assertIn("different states", " ".join(report["failures"]))

    def test_full_read_mode_detects_an_answer_outside_the_distribution(self) -> None:
        status, _ = self._run("full-read", choice_is_argmax=False)
        self.assertEqual(1, status)

    def test_full_read_mode_detects_a_score_that_is_not_the_expectation(self) -> None:
        status, _ = self._run("full-read", score_is_expectation=False)
        self.assertEqual(1, status)

    def test_full_read_mode_detects_generated_output_tokens(self) -> None:
        status, _ = self._run("full-read", output_tokens=7)
        self.assertEqual(1, status)

    def test_full_read_mode_detects_a_substituted_model_name(self) -> None:
        status, report = self._run("full-read", echo_requested_model=False)
        self.assertEqual(1, status)
        self.assertIn("expected model", " ".join(report["failures"]))

    def test_full_read_mode_detects_a_missing_usage_object(self) -> None:
        status, _ = self._run("full-read", drop_usage=True)
        self.assertEqual(1, status)


class SystemOneSmokeGateTests(unittest.TestCase):
    """The shell gate is hermetic in these modes: an unqualified backend means
    the full-model read never touches a native build."""

    def _run_smoke(self, **env: str) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="system-one-gate-") as directory:
            report_path = Path(directory) / "system-one.json"
            environment = {
                **os.environ,
                # Deterministic and fast: no `build-llama.sh --print-build-dir`,
                # and no inherited cache or annotation state.
                "LLAMA_STAGE_BUILD_DIR": str(Path(directory) / "absent-llama-build"),
                "HF_CACHE": "",
                "HF_HUB_CACHE": "",
                "GITHUB_ACTIONS": "",
                "SYSTEMONE_SMOKE_REPORT": str(report_path),
                "WORK_DIR": directory,
            }
            environment.update(env)
            result = subprocess.run(
                ["bash", str(SMOKE)],
                capture_output=True,
                text=True,
                cwd=ROOT,
                env=environment,
                check=False,
            )
            report = (
                json.loads(report_path.read_text(encoding="utf-8"))
                if report_path.exists()
                else {}
            )
        return result, report

    def test_unqualified_full_read_is_reported_not_silently_skipped(self) -> None:
        result, report = self._run_smoke(
            SYSTEMONE_SMOKE_SKIP_CONTRACT="1",
            SYSTEMONE_SMOKE_BUILD_BACKEND="metal",
            GITHUB_ACTIONS="1",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("unqualified", report["status"])
        self.assertEqual("unqualified", report["full_model_read"]["status"])
        self.assertIn("NOT CERTIFIED", result.stdout)
        # The lane must never pass quietly.
        self.assertIn("::warning title=System One smoke::", result.stdout)

    def test_require_qualified_turns_an_unqualified_read_into_a_failure(self) -> None:
        result, report = self._run_smoke(
            SYSTEMONE_SMOKE_SKIP_CONTRACT="1",
            SYSTEMONE_SMOKE_BUILD_BACKEND="metal",
            SYSTEMONE_SMOKE_REQUIRE_QUALIFIED="1",
        )
        self.assertEqual(1, result.returncode)
        self.assertEqual("fail", report["status"])

    def test_a_declared_backend_without_the_pinned_artifact_still_reports(self) -> None:
        # Declared qualified, artifact absent from the offline cache: an honest
        # NOT CERTIFIED, not a failure and not a skip.
        result, report = self._run_smoke(
            SYSTEMONE_SMOKE_SKIP_CONTRACT="1",
            SYSTEMONE_SMOKE_BUILD_BACKEND="cuda",
            HF_CACHE="/nonexistent-hf-cache",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("unqualified", report["status"])

    def test_missing_native_build_fails_the_contract_part(self) -> None:
        # The contract part cannot run: that is an unusable environment, which
        # must be red rather than a skip.
        result, report = self._run_smoke()
        self.assertEqual(1, result.returncode)
        self.assertEqual("fail", report["status"])
        self.assertEqual("error", report["contract"]["status"])

    def test_prewarm_plan_names_the_pinned_revision_and_digest(self) -> None:
        result = subprocess.run(
            ["bash", str(SMOKE), "--prewarm"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("f4183a2c7a354128d02545752303c4354d165bf0", result.stdout)
        self.assertIn(
            "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5", result.stdout
        )
        self.assertIn("hf download", result.stdout)


class SystemOneCanaryWiringTests(unittest.TestCase):
    def test_unchanged_pin_workflow_runs_and_gates_the_smoke(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        smoke_step = workflow[
            workflow.index("      - name: System One (OpenJEV) smoke\n") :
        ]
        smoke_step = smoke_step[: smoke_step.index("\n      - name: ", 10)]
        self.assertIn("id: system_one", smoke_step)
        self.assertIn("scripts/skippy-system-one-smoke.sh", smoke_step)
        self.assertIn("continue-on-error: true", smoke_step)
        self.assertIn("steps.sha.outputs.certify == 'true'", smoke_step)
        self.assertIn(
            "SYSTEMONE_SMOKE_REQUIRE_QUALIFIED: ${{ vars.LLAMA_CANARY_SYSTEMONE_REQUIRE_QUALIFIED }}",
            smoke_step,
        )
        self.assertIn(
            "SYSTEMONE_SMOKE_CADENCE: ${{ steps.sha.outputs.cadence }}", smoke_step
        )
        # The forced and nightly success reports only claim certification when
        # the System One lane is green.
        for cadence in ("manual-full", "nightly"):
            report = workflow[workflow.index(f"cadence == '{cadence}'") :]
            report = report[: report.index("\n      - name: ", 10)]
            self.assertIn("steps.system_one.outcome == 'success'", report, cadence)
        # And a red lane blocks the unchanged-pin run.
        failure_gate = workflow[workflow.index("      - name: Fail unsuccessful unchanged-pin certification\n") :]
        self.assertIn("steps.system_one.outcome == 'failure'", failure_gate)
        # The evidence and the loud summary always run.
        self.assertIn("      - name: Report System One smoke result\n", workflow)
        self.assertIn("      - name: Upload System One smoke evidence\n", workflow)

    def test_changed_pin_repair_and_verification_both_run_the_smoke(self) -> None:
        wrapper = WRAPPER.read_text(encoding="utf-8")
        build = wrapper[wrapper.index("run_full_build() {") :]
        build = build[: build.index("\nrun_certification() {")]
        self.assertIn("scripts/skippy-system-one-smoke.sh", build)
        # The changed pin's cadence authorizes the pinned artifact.
        self.assertIn("SYSTEMONE_SMOKE_CADENCE=llama-bump", build)
        # `run_full_build` is the shared gate for repair turns and for the
        # independent verification pass, so both are covered.
        self.assertIn("run_candidate_gates() {", wrapper)
        self.assertIn("run_full_build || return 1", wrapper)
        self.assertIn("if ! run_candidate_gates; then", wrapper)

    def test_the_smoke_never_claims_family_certification(self) -> None:
        families = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))
        self.assertNotIn(
            "diffusion-gemma", {model["family"] for model in families["models"]}
        )
        family_map = json.loads(FAMILY_MAP.read_text(encoding="utf-8"))
        self.assertNotIn("diffusion-gemma", family_map)

    def test_the_pinned_artifact_matches_the_published_identity(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        rows = {row["id"]: row for row in registry["artifacts"]}
        self.assertIn("skippy-system-one-smoke", registry["suites"])
        row = rows["family-diffusion-gemma"]
        artifact = row["artifact"]
        self.assertEqual("unsloth/diffusiongemma-26B-A4B-it-GGUF", artifact["repo"])
        self.assertEqual("f4183a2c7a354128d02545752303c4354d165bf0", artifact["revision"])
        self.assertEqual(1, len(artifact["files"]))
        pinned = artifact["files"][0]
        self.assertEqual("diffusiongemma-26B-A4B-it-Q4_K_M.gguf", pinned["path"])
        self.assertEqual(16806810208, pinned["size_bytes"])
        self.assertEqual(
            "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5", pinned["sha256"]
        )
        # The contract fixture is reused, not re-pinned under a second identity.
        self.assertIn("skippy-system-one-smoke", rows["family-qwen3-dense"]["suites"])

        manifest = json.loads(SUITE_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual("skippy-system-one-smoke", manifest["suite"])
        self.assertEqual(
            {"family-qwen3-dense", "family-diffusion-gemma"},
            {row["id"] for row in manifest["artifacts"]},
        )
        for cadence in ("llama-bump", "manual-full", "nightly", "manual"):
            self.assertIn(cadence, row["cadences"])

    def test_the_smoke_resolves_pins_through_the_shared_manifest_contract(self) -> None:
        smoke = SMOKE.read_text(encoding="utf-8")
        self.assertIn("resolve-test-model-manifest.py", smoke)
        self.assertIn("--require-single-file", smoke)
        self.assertIn("--verify-root", smoke)

    def test_the_smoke_script_is_directly_executable(self) -> None:
        # The workflow step and the repair wrapper invoke it directly rather
        # than through `bash`, so the executable bit is part of the contract.
        self.assertTrue(
            os.access(SMOKE, os.X_OK),
            f"{SMOKE} must be committed executable",
        )

    def test_driver_reports_normalization_tolerance_explicitly(self) -> None:
        driver = DRIVER.read_text(encoding="utf-8")
        self.assertIn("PROBABILITY_TOLERANCE", driver)
        self.assertIn("does not sum to 1", driver)


if __name__ == "__main__":
    unittest.main()
