from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
GENERATE = SCRIPTS / "release-notes-generate.sh"
HOOK = ROOT / "scripts" / "hooks" / "commit-msg"


def load(name, filename):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVENTIONAL = load("conventional_commit", "check-conventional-commit.py")
CLASSIFY = load("release_notes_classify", "release-notes-classify.py")
REGROUP = load("release_notes_regroup", "release-notes-regroup.py")


def entry(pr, subject="a change", author="someone"):
    return f"* {subject} by @{author} in https://github.com/Mesh-LLM/mesh-llm/pull/{pr}"


BODY = "\n".join(
    [
        "## What's Changed",
        entry(1, "feat(skippy): add a thing"),
        entry(2, "fix: repair a thing"),
        entry(3, "Something without a prefix"),
        "",
        "## New Contributors",
        "* @someone made their first contribution",
        "",
        "**Full Changelog**: https://github.com/Mesh-LLM/mesh-llm/compare/v1...v2",
        "",
    ]
)


class ConventionalCommitTest(unittest.TestCase):
    def test_accepts_conventional_subjects(self):
        for subject in [
            "fix(skippy): restore recurrent prefix reuse",
            "feat: expose a tokenizer capability",
            "chore(deps)!: drop the old runtime",
            "ci: publish the SDK smoke cache policy",
        ]:
            self.assertEqual(CONVENTIONAL.check_subject(subject), [], subject)

    def test_rejects_non_conventional_subjects(self):
        for subject in [
            "Fix split serving: stages above layer 0 fail to load",
            "CI: pin Depot audit to merged policy",
            "",
        ]:
            self.assertTrue(CONVENTIONAL.check_subject(subject), subject)

    def test_rejects_unknown_type(self):
        problems = CONVENTIONAL.check_subject("task: do a thing")
        self.assertTrue(any("unknown type" in p or "not Conventional" in p for p in problems))

    def test_rejects_trailing_period_and_leading_capital(self):
        self.assertTrue(CONVENTIONAL.check_subject("fix: repair the thing."))
        self.assertTrue(CONVENTIONAL.check_subject("fix: Repair the thing"))

    def test_rejects_overlong_subject(self):
        problems = CONVENTIONAL.check_subject("fix: " + "x" * CONVENTIONAL.MAX_SUBJECT)
        self.assertTrue(any("characters" in p for p in problems))

    def test_exempts_git_and_release_authored_subjects(self):
        for subject in [
            "Merge branch 'main' into feature",
            'Revert "fix: repair a thing"',
            "fixup! fix: repair a thing",
            "v0.76.0: prepare release source",
        ]:
            self.assertEqual(CONVENTIONAL.check_subject(subject), [], subject)

    def test_ignores_squash_merge_pr_suffix(self):
        self.assertEqual(
            CONVENTIONAL.check_subject("fix(ui): use advertised capacity (#1746)"), []
        )

    def test_every_type_maps_to_a_section(self):
        sections = set(CLASSIFY.SECTION_ORDER) | {"Internal"}
        for kind, section in CONVENTIONAL.TYPES.items():
            self.assertIn(section, sections, kind)

    def test_hook_rejects_bad_message(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("Bad subject\n")
            path = handle.name
        result = subprocess.run([str(HOOK), path], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Conventional Commits", result.stderr)


class ClassifyTest(unittest.TestCase):
    def plan_for(self, commits, prs=None):
        prs = prs or sorted(commits)
        built = {
            pr: {"subject": subject, "trailers": trailers}
            for pr, (subject, trailers) in commits.items()
        }
        plan, unclassified = CLASSIFY.build_plan(prs, built, "1.0.0", "2026-01-01")
        return plan, unclassified

    def sections(self, plan):
        found = {}
        for section in plan["sections"]:
            prs = list(section.get("prs", []))
            for group in section.get("groups", []):
                prs += group["prs"]
            found[section["title"]] = prs
        return found

    def test_type_drives_the_section(self):
        plan, _ = self.plan_for(
            {
                1: ("feat: add a thing", {}),
                2: ("fix: repair a thing", {}),
                3: ("perf: speed a thing up", {}),
                4: ("security: reject hostile input", {}),
            }
        )
        found = self.sections(plan)
        self.assertEqual(found["Added"], [1])
        self.assertEqual(found["Fixed"], [2])
        self.assertEqual(found["Changed"], [3])
        self.assertEqual(found["Security"], [4])

    def test_breaking_change_is_changed_not_added(self):
        plan, _ = self.plan_for({1: ("feat!: replace the subsystem", {})})
        self.assertEqual(self.sections(plan)["Changed"], [1])
        plan, _ = self.plan_for(
            {1: ("feat: replace it", {"breaking change": "the old flag is gone"})}
        )
        self.assertEqual(self.sections(plan)["Changed"], [1])

    def test_release_notes_trailer_overrides_the_type(self):
        plan, _ = self.plan_for(
            {1: ("fix: redact provider health details", {"release-notes": "Security"})}
        )
        self.assertEqual(self.sections(plan)["Security"], [1])

    def test_tooling_scopes_are_internal_whatever_the_type(self):
        plan, _ = self.plan_for(
            {
                1: ("fix(ci): make CUDA checks hermetic", {}),
                2: ("feat(bench): run a new suite", {}),
                3: ("fix(skippy): restore prefix reuse", {}),
            }
        )
        internal = [pr for group in plan["internal"]["groups"] for pr in group["prs"]]
        self.assertEqual(sorted(internal), [1, 2])
        self.assertEqual(self.sections(plan)["Fixed"], [3])

    def test_non_conventional_entries_are_not_guessed(self):
        plan, unclassified = self.plan_for(
            {1: ("Stop persisting KV cache state to disk", {})}
        )
        self.assertEqual(unclassified, 1)
        self.assertEqual(self.sections(plan)["Other changes"], [1])

    def test_missing_commit_is_not_guessed(self):
        plan, unclassified = CLASSIFY.build_plan([7], {}, "1.0.0", "2026-01-01")
        self.assertEqual(unclassified, 1)
        self.assertEqual(self.sections(plan)["Other changes"], [7])

    def test_small_scopes_do_not_earn_subheadings(self):
        self.assertIsNone(CLASSIFY.subgroups({"skippy": [1, 2], "ui": [3]}))
        groups = CLASSIFY.subgroups(
            {"skippy": list(range(10)), "ui": list(range(10, 20)), "cli": [99]}
        )
        titles = [group["title"] for group in groups]
        self.assertEqual(titles, ["skippy", "ui", "Other"])


class RegroupTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.body = self.dir / "body.md"
        self.body.write_text(BODY, encoding="utf-8")

    def render(self, plan):
        plan_path = self.dir / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        out = self.dir / "out.md"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "release-notes-regroup.py"),
                "--body", str(self.body),
                "--plan", str(plan_path),
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        return result, out

    def test_preserves_every_pull_request_and_its_credit(self):
        plan = {
            "version": "1.0.0",
            "sections": [{"title": "Added", "prs": [1, 2, 3]}],
        }
        result, out = self.render(plan)
        self.assertEqual(result.returncode, 0, result.stderr)
        rendered = out.read_text(encoding="utf-8")
        keep = lambda text: [
            line
            for line in text.splitlines()
            if line.startswith("* ") and "/pull/" in line
        ]
        original, emitted = keep(BODY), keep(rendered)
        self.assertEqual(len(original), len(emitted))
        # The subject may lose its type prefix; the credit tail never changes.
        credits = lambda lines: sorted(line[line.index(" by @"):] for line in lines)
        self.assertEqual(credits(original), credits(emitted))

    def test_strips_the_type_prefix_and_sentence_cases_the_subject(self):
        _, out = self.render(
            {"version": "1.0.0", "sections": [{"title": "Added", "prs": [1, 2, 3]}]}
        )
        rendered = out.read_text(encoding="utf-8")
        self.assertIn("* Add a thing by @someone", rendered)
        self.assertIn("* Repair a thing by @someone", rendered)
        self.assertNotIn("feat(skippy):", rendered)
        self.assertNotIn("* fix:", rendered)

    def test_leaves_an_unprefixed_subject_exactly_as_written(self):
        _, out = self.render(
            {"version": "1.0.0", "sections": [{"title": "Added", "prs": [1, 2, 3]}]}
        )
        self.assertIn("* Something without a prefix by @someone", out.read_text(encoding="utf-8"))

    def test_preserves_the_tail(self):
        _, out = self.render(
            {"version": "1.0.0", "sections": [{"title": "Added", "prs": [1, 2, 3]}]}
        )
        rendered = out.read_text(encoding="utf-8")
        self.assertIn("## New Contributors", rendered)
        self.assertIn("**Full Changelog**", rendered)

    def test_refuses_a_plan_that_drops_an_entry(self):
        result, _ = self.render(
            {"version": "1.0.0", "sections": [{"title": "Added", "prs": [1, 2]}]}
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing from the plan", result.stderr)

    def test_refuses_a_plan_that_duplicates_an_entry(self):
        result, _ = self.render(
            {
                "version": "1.0.0",
                "sections": [
                    {"title": "Added", "prs": [1, 2, 3]},
                    {"title": "Fixed", "prs": [1]},
                ],
            }
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("more than one section", result.stderr)

    def test_refuses_a_plan_with_an_unknown_pr(self):
        result, _ = self.render(
            {"version": "1.0.0", "sections": [{"title": "Added", "prs": [1, 2, 3, 42]}]}
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("not in the release body", result.stderr)

    def test_internal_section_is_collapsed_with_a_count(self):
        plan = {
            "version": "1.0.0",
            "sections": [{"title": "Added", "prs": [1]}],
            "internal": {
                "summary": "internal work",
                "groups": [{"title": "CI", "prs": [2, 3]}],
            },
        }
        _, out = self.render(plan)
        rendered = out.read_text(encoding="utf-8")
        self.assertIn("<details>", rendered)
        self.assertIn("(2 changes)</summary>", rendered)
        # GitHub needs a blank line after <summary> to render the Markdown.
        self.assertIn("</summary>\n\n", rendered)


class WorkflowContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        cls.job = cls.workflow["jobs"]["release_notes"]
        cls.script = GENERATE.read_text(encoding="utf-8")

    def test_runs_after_publish_and_only_for_stable_releases(self):
        self.assertEqual(self.job["needs"], ["metadata", "publish"])
        condition = self.job["if"]
        self.assertIn("needs.publish.result == 'success'", condition)
        self.assertIn("prerelease != 'true'", condition)

    def test_declares_least_privilege_and_a_timeout(self):
        self.assertEqual(self.job["permissions"], {"contents": "write"})
        self.assertIn("timeout-minutes", self.job)

    def test_checkout_is_pinned_and_credential_free(self):
        checkout = self.job["steps"][0]
        self.assertRegex(checkout["uses"], r"^actions/checkout@[0-9a-f]{40}$")
        self.assertFalse(checkout["with"]["persist-credentials"])
        self.assertEqual(checkout["with"]["fetch-depth"], 0)

    def test_publishing_step_owns_the_token(self):
        step = next(s for s in self.job["steps"] if s.get("id") == "regroup")
        self.assertEqual(step["env"]["GH_TOKEN"], "${{ secrets.GITHUB_TOKEN }}")
        self.assertEqual(step["run"].strip(), "scripts/release-notes-generate.sh")

    def test_evidence_is_uploaded_with_a_pinned_action(self):
        upload = self.job["steps"][-1]
        self.assertRegex(upload["uses"], r"^actions/upload-artifact@[0-9a-f]{40}$")

    def test_agent_failures_never_fail_the_job(self):
        # Every agent guard returns rather than exiting, so an unreachable,
        # unauthenticated, out-of-quota, or wrong agent keeps the job green.
        for guard in [
            "AGENT_MODEL unset",
            "opencode not installed",
            "no agent credentials",
            "agent probe failed",
            "agent review turn failed",
            "agent produced no plan",
            "agent plan failed validation",
        ]:
            self.assertIn(guard, self.script)
        agent_block = self.script[self.script.index("agent_reachable()"):]
        self.assertNotIn("exit 1", agent_block.split("# ── Publish")[0])

    def test_agent_never_receives_publish_credentials(self):
        self.assertEqual(self.script.count("env -u GH_TOKEN -u GITHUB_TOKEN"), 2)

    def test_deterministic_pass_gates_before_publishing(self):
        self.assertIn("refusing to publish", self.script)
        self.assertLess(
            self.script.index("verify_entries"), self.script.index("gh release edit")
        )



class TrailerDenyListTest(unittest.TestCase):
    def denied(self, line):
        return CONVENTIONAL.check_trailers([line])

    def test_rejects_agent_attribution_addresses(self):
        for line in [
            "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>",
            "Co-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
            "Co-authored-by: CodeRabbit <noreply@coderabbit.ai>",
        ]:
            self.assertTrue(self.denied(line), line)

    def test_rejects_relay_identity_domains(self):
        for line in [
            "Co-authored-by: scama <a1860575018c46@meshllm.communities.buzz.xyz>",
            "Co-authored-by: Paul Hogan <5004d00b7537@meshllm.communities.buzz.xyz>",
        ]:
            self.assertTrue(self.denied(line), line)

    def test_rejects_bot_accounts(self):
        self.assertTrue(
            self.denied(
                "Co-authored-by: coderabbitai[bot] "
                "<136622811+coderabbitai[bot]@users.noreply.github.com>"
            )
        )

    def test_rejects_model_names_whatever_the_address(self):
        for name in ["Claude", "ChatGPT", "Sisyphus", "Astra", "Sol", "Luna", "Terra"]:
            line = f"Co-authored-by: {name} <someone@example.com>"
            self.assertTrue(self.denied(line), line)

    def test_matches_whole_name_tokens_only(self):
        # "Sol" is denied; "Solomon" is a person.
        self.assertEqual(
            CONVENTIONAL.check_trailers(
                ["Co-authored-by: Solomon Grundy <solomon@example.com>"]
            ),
            [],
        )

    def test_allows_human_trailers(self):
        for line in [
            "Co-authored-by: Real Person <real@example.com>",
            "Signed-off-by: Michael Neale <14976+michaelneale@users.noreply.github.com>",
            "Reviewed-by: Someone Else <someone@example.com>",
        ]:
            self.assertEqual(CONVENTIONAL.check_trailers([line]), [], line)

    def test_ignores_non_trailer_body_lines(self):
        self.assertEqual(
            CONVENTIONAL.check_trailers(
                ["This change was suggested by Claude in review.", "BREAKING CHANGE: x"]
            ),
            [],
        )

    def test_hook_rejects_a_denied_trailer_on_a_valid_subject(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(
                "fix: repair a thing\n\n"
                "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
            )
            path = handle.name
        result = subprocess.run([str(HOOK), path], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("agent attribution address", result.stderr)


class SubjectNormalizationTest(unittest.TestCase):
    def subject(self, line):
        return REGROUP.render_entry(line).split(" by @")[0][2:]

    def entry(self, subject):
        return f"* {subject} by @x in https://github.com/Mesh-LLM/mesh-llm/pull/1"

    def test_strips_type_scope_and_breaking_marker(self):
        cases = {
            "feat(skippy): load SafeTensors checkpoints directly":
                "Load SafeTensors checkpoints directly",
            "fix: repair a thing": "Repair a thing",
            "chore(deps)!: drop the old runtime": "Drop the old runtime",
            "perf(kv): keep recording off the inference path":
                "Keep recording off the inference path",
        }
        for original, expected in cases.items():
            self.assertEqual(self.subject(self.entry(original)), expected)

    def test_strips_legacy_and_uppercase_prefixes(self):
        self.assertEqual(
            self.subject(self.entry("task: refine logging console UX")),
            "Refine logging console UX",
        )
        self.assertEqual(
            self.subject(self.entry("CI: pin Depot audit to merged policy")),
            "Pin Depot audit to merged policy",
        )

    def test_never_strips_an_unrecognised_word_before_a_colon(self):
        # "Durable KV prefix cache:" is a sentence, not a type.
        for subject in [
            "Durable KV prefix cache: agent prefixes survive eviction",
            "skippy-quantize: compose-mtp splices an MTP draft",
            "Fix split serving: stages above layer 0 fail to load",
        ]:
            self.assertEqual(self.subject(self.entry(subject)), subject)

    def test_credit_is_never_touched(self):
        line = (
            "* feat(ui): render LaTeX by @ndizazzo in "
            "https://github.com/Mesh-LLM/mesh-llm/pull/1466"
        )
        self.assertTrue(
            REGROUP.render_entry(line).endswith(
                " by @ndizazzo in https://github.com/Mesh-LLM/mesh-llm/pull/1466"
            )
        )

    def test_a_subject_that_is_only_a_prefix_is_left_alone(self):
        self.assertEqual(self.subject(self.entry("fix:")), "fix:")

class OverrideHardeningTest(unittest.TestCase):
    def classify(self, subject, trailers):
        return CLASSIFY.classify({"subject": subject, "trailers": trailers})

    def test_invalid_override_is_not_silently_ignored(self):
        # A typo must not fall through and land a security fix in Fixed.
        self.assertEqual(
            self.classify("fix: redact health details", {"release-notes": "Secuirty"}),
            (None, None),
        )

    def test_valid_override_still_wins(self):
        section, _ = self.classify(
            "fix: redact health details", {"release-notes": "Security"}
        )
        self.assertEqual(section, "Security")

    def test_internal_override_survives_a_non_conventional_subject(self):
        plan, _ = CLASSIFY.build_plan(
            [1],
            {1: {"subject": "Some non-conventional subject",
                 "trailers": {"release-notes": "Internal"}}},
            "1.0.0",
            "2026-01-01",
        )
        groups = [g["title"] for g in plan["internal"]["groups"]]
        self.assertEqual(groups, ["Refactors, docs, and hygiene"])


class PlanMetadataTest(unittest.TestCase):
    """Plan text can be agent-authored, so it must not reach the body unchecked."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.body = self.dir / "body.md"
        self.body.write_text(BODY, encoding="utf-8")

    def check(self, plan, extra=()):
        path = self.dir / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "release-notes-regroup.py"),
             "--body", str(self.body), "--plan", str(path), "--check", *extra],
            capture_output=True, text=True,
        )

    def all_prs(self, **plan):
        plan.setdefault("sections", [{"title": "Added", "prs": [1, 2, 3]}])
        return plan

    def test_rejects_an_unknown_section(self):
        result = self.check(self.all_prs(sections=[{"title": "Sponsors", "prs": [1, 2, 3]}]))
        self.assertEqual(result.returncode, 1)
        self.assertIn("unknown section", result.stderr)

    def test_rejects_markup_in_a_group_title(self):
        result = self.check(self.all_prs(sections=[{
            "title": "Added",
            "groups": [{"title": "See [click](http://evil.example)", "prs": [1, 2, 3]}],
        }]))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a plain heading", result.stderr)

    def test_rejects_a_multiline_version(self):
        result = self.check(self.all_prs(version="1.0\n## Injected"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a plain version string", result.stderr)

    def test_rejects_markup_in_the_internal_summary(self):
        result = self.check(self.all_prs(
            sections=[{"title": "Added", "prs": [1, 2]}],
            internal={"summary": "<img src=x onerror=1>",
                      "groups": [{"title": "CI", "prs": [3]}]},
        ))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a plain heading", result.stderr)

    def test_accepts_an_ordinary_plan(self):
        result = self.check(self.all_prs(version="0.76.0", date="2026-09-10"))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_metadata_from_overrides_plan_authored_version(self):
        trusted = self.dir / "trusted.json"
        trusted.write_text(json.dumps({"version": "0.76.0", "date": "2026-09-10"}))
        result = self.check(
            self.all_prs(version="9.9.9", date="1999-01-01"),
            extra=["--metadata-from", str(trusted)],
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class PublishGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = GENERATE.read_text(encoding="utf-8")

    def test_manual_runs_require_explicit_approval(self):
        self.assertIn('"${GITHUB_ACTIONS:-}" != "true"', self.script)
        self.assertIn('"${RELEASE_NOTES_APPROVED:-}" != "true"', self.script)
        self.assertLess(
            self.script.index("RELEASE_NOTES_APPROVED"),
            self.script.index("gh release edit"),
        )

    def test_agent_plan_renders_with_deterministic_metadata(self):
        self.assertIn("--metadata-from", self.script)


if __name__ == "__main__":
    unittest.main()
