"""Tests for state-run day filtering and insights breakdowns (report.py + cli stats)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from harness import report, sandbox
from harness.cli import main as cli_main
from harness.sandbox import load_workspace_lessons, record_repair_lesson


def _make_run(state_dir: Path, task_id: str, status: str = "succeeded") -> Path:
    run_dir = state_dir / "runs" / f"{task_id}-{status}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps({"task_id": task_id, "status": status, "model": "test-model"}),
        encoding="utf-8",
    )
    return run_dir


class DaysFilteringTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / "state"
        (self.state_dir / "runs").mkdir(parents=True)
        self.now = time.time()

        old_dir = _make_run(self.state_dir, "old-task")
        recent_dir = _make_run(self.state_dir, "recent-task")
        os.utime(old_dir, (self.now - 11 * 86400, self.now - 11 * 86400))
        os.utime(recent_dir, (self.now - 0.5 * 86400, self.now - 0.5 * 86400))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_scan_state_runs_days_filters_old_runs(self) -> None:
        all_runs = report.scan_state_runs(self.state_dir)
        self.assertEqual(len(all_runs), 2)
        recent = report.scan_state_runs(self.state_dir, days=7, now=self.now)
        self.assertEqual(
            sorted(run["task_id"] for run in recent), ["recent-task"]
        )

    def test_compute_state_insights_days_filters_old_runs(self) -> None:
        insights = report.compute_state_insights(self.state_dir, days=7, now=self.now)
        self.assertEqual(insights["task_funnel"]["total_tasks"], 1)

        full = report.compute_state_insights(self.state_dir, now=self.now)
        self.assertEqual(full["task_funnel"]["total_tasks"], 2)

    def test_format_state_insights_markdown_days(self) -> None:
        text = report.format_state_insights_markdown(self.state_dir, days=7, now=self.now)
        self.assertNotIn("## Recent Window (Last 7 Days)", text)
        self.assertIn("- Total tasks: 1", text)
        self.assertNotIn("- Total tasks: 2", text)

    def test_cli_stats_days(self) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = cli_main(
                [
                    "stats",
                    "--insights",
                    "--days",
                    "7",
                    "--state-dir",
                    str(self.state_dir),
                ]
            )
        self.assertEqual(exit_code, 0)
        output = buffer.getvalue()
        # The stats table is model-aggregated: only the recent run survives
        # the --days filter, so the run count is 1 (not 2).
        self.assertIn("| ALL | 1 |", output)
        self.assertNotIn("| ALL | 2 |", output)


class RecentWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / "state"
        (self.state_dir / "runs").mkdir(parents=True)
        self.now = time.time()

        old_dir = _make_run(self.state_dir, "old-task")
        recent_dir = _make_run(self.state_dir, "recent-task", status="succeeded")
        os.utime(old_dir, (self.now - 11 * 86400, self.now - 11 * 86400))
        os.utime(recent_dir, (self.now - 0.5 * 86400, self.now - 0.5 * 86400))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_recent_window_when_days_none(self) -> None:
        text = report.format_state_insights_markdown(self.state_dir, now=self.now)
        self.assertIn("## Recent Window (Last 7 Days)", text)
        # Recent window counts only the recent run.
        self.assertIn("- Runs: 1", text)
        self.assertIn("- Pass@1: 100.0%", text)
        self.assertIn("- Resolved: 1", text)
        self.assertIn("- Unresolved: 0", text)
        # All-time funnel counts both runs.
        self.assertIn("- Total tasks: 2", text)

    def test_no_recent_window_when_days_given(self) -> None:
        text = report.format_state_insights_markdown(self.state_dir, days=7, now=self.now)
        self.assertNotIn("## Recent Window (Last 7 Days)", text)


class FunnelBreakdownTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / "state"
        (self.state_dir / "runs").mkdir(parents=True)
        self.now = time.time()

        # Unresolved active: two attempts, recent.
        _make_run(self.state_dir, "active-task", status="blocked")
        second = _make_run(self.state_dir, "active-task", status="failed")
        os.utime(second, (self.now - 0.5 * 86400, self.now - 0.5 * 86400))
        # Unresolved stale single-shot: one attempt, >7d old.
        stale = _make_run(self.state_dir, "stale-task", status="failed")
        os.utime(stale, (self.now - 10 * 86400, self.now - 10 * 86400))
        # Unresolved but recent single attempt -> active via recent cutoff.
        probe = _make_run(self.state_dir, "probe-task", status="failed")
        os.utime(probe, (self.now - 0.5 * 86400, self.now - 0.5 * 86400))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_funnel_breakdown(self) -> None:
        text = report.format_state_insights_markdown(self.state_dir, now=self.now)
        self.assertIn("- Unresolved: 3", text)
        self.assertIn("- Active / multi-attempt: 2", text)
        self.assertIn("- Single-shot / probe / stale (>7d, 1 attempt): 1", text)

    def test_funnel_counts_in_insights_dict(self) -> None:
        insights = report.compute_state_insights(self.state_dir, now=self.now)
        funnel = insights["task_funnel"]
        self.assertEqual(funnel["unresolved_tasks"], 3)
        self.assertEqual(funnel["unresolved_active_tasks"], 2)
        self.assertEqual(funnel["unresolved_stale_single_shot_tasks"], 1)


class RepairLessonsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / "state"
        self.state_dir.mkdir(parents=True)
        self.task_dir = Path(self._tmp.name) / "repo"
        (self.task_dir / "pkg").mkdir(parents=True)
        self.mod = self.task_dir / "pkg" / "mod.py"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _task(self, model: str = "test-model") -> dict:
        return {"task_id": "t1", "directory": str(self.task_dir), "model": model}

    def test_record_repair_lesson_dedupes_and_extracts_signature(self) -> None:
        log = (
            "================= short test summary info =================\n"
            "FAILED tests/test_x.py::test_a\n"
            "E   AssertionError: expected 1 got 2\n"
        )
        record_repair_lesson(self.state_dir, self._task(), log, [str(self.mod)])
        record_repair_lesson(self.state_dir, self._task(), log, [str(self.mod)])
        path = self.state_dir / "lessons.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["error_signature"], "E   AssertionError: expected 1 got 2")
        self.assertEqual(entries[0]["files"], ["pkg/mod.py"])
        self.assertEqual(entries[0]["task_dir"], str(self.task_dir))
        self.assertEqual(entries[0]["model"], "test-model")
        self.assertIn("timestamp", entries[0])

    def test_record_repair_lesson_skips_logs_without_signature(self) -> None:
        # Noise-only logs (passing pytest output, random text) produce no
        # lesson at all: no fallback to the first non-empty line anymore.
        record_repair_lesson(self.state_dir, self._task(), "\nrandom failure text\n", [])
        record_repair_lesson(self.state_dir, self._task(), "225 deselected in 0.04s\n", [])
        record_repair_lesson(
            self.state_dir,
            self._task(),
            ".... [100%]\n"
            "/opt/homebrew/.../tempfile.py:484: ResourceWarning: Implicitly cleaning up\n"
            "312 passed in 48s\n",
            [],
        )
        self.assertFalse((self.state_dir / "lessons.jsonl").exists())

    def test_lesson_signature_extracts_exception_after_traceback(self) -> None:
        log = (
            "Traceback (most recent call last):\n"
            '  File "pkg/mod.py", line 3, in foo\n'
            "    return val['k']\n"
            "KeyError: 'k'\n"
        )
        self.assertEqual(sandbox._lesson_error_signature(log), "KeyError: 'k'")
        self.assertEqual(sandbox._lesson_error_signature(".... [100%]\n"), "")
        self.assertEqual(sandbox._lesson_error_signature("225 deselected in 0.04s\n"), "")

    def test_load_workspace_lessons_filters_by_task_dir(self) -> None:
        other_dir = Path(self._tmp.name) / "other"
        other_dir.mkdir()
        record_repair_lesson(self.state_dir, self._task(), "E   TypeError: bad type\n", [str(self.mod)])
        record_repair_lesson(
            self.state_dir,
            {"task_id": "t2", "directory": str(other_dir), "model": "m"},
            "E   ImportError: nope\n",
            [str(other_dir / "a.py")],
        )
        text = load_workspace_lessons(
            self.state_dir, self.task_dir, files=[str(self.mod)]
        )
        self.assertIn("### Workspace Guardrails (from prior repairs)", text)
        self.assertIn("TypeError", text)
        self.assertNotIn("ImportError", text)
        self.assertEqual(
            load_workspace_lessons(self.state_dir, Path(self._tmp.name) / "nope"), ""
        )

    def test_load_workspace_lessons_respects_limits_and_prioritizes_files(self) -> None:
        for i in range(5):
            record_repair_lesson(
                self.state_dir, self._task(), f"E   TypeError: err{i}\n", []
            )
        text = load_workspace_lessons(self.state_dir, self.task_dir)
        self.assertLessEqual(len(text), 600)
        self.assertEqual(text.count("\n- "), 3)
        self.assertIn("err4", text)
        self.assertNotIn("err0", text)

        # A lesson sharing a file with `files` jumps ahead of others.
        record_repair_lesson(
            self.state_dir, self._task(), "E   TypeError: shared-file lesson\n", [str(self.mod)]
        )
        text = load_workspace_lessons(self.state_dir, self.task_dir, files=[str(self.mod)])
        self.assertLess(
            text.index("shared-file lesson"), text.index("E   TypeError: err4")
        )


class LessonsMarkdownTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / "state"
        (self.state_dir / "runs").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_markdown_shows_lesson_count(self) -> None:
        record_repair_lesson(
            self.state_dir, {"task_id": "t", "directory": "/tmp/x", "model": "m"},
            "E   TypeError: boom\n", [],
        )
        record_repair_lesson(
            self.state_dir, {"task_id": "t", "directory": "/tmp/y", "model": "m"},
            "E   ImportError: gone\n", [],
        )
        text = report.format_state_insights_markdown(self.state_dir)
        self.assertIn("- Recorded repair lessons: 2", text)

    def test_markdown_omits_lessons_when_empty(self) -> None:
        (self.state_dir / "lessons.jsonl").write_text("", encoding="utf-8")
        text = report.format_state_insights_markdown(self.state_dir)
        self.assertNotIn("Recorded repair lessons", text)


class BackfillRepairLessonsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name) / "state"
        (self.state_dir / "runs").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _repaired_run(
        self,
        task_id: str,
        *,
        status: str = "validation_passed_review_required",
        repaired: bool = True,
        salvaged_from: str | None = None,
        before_log: str = "E   TypeError: boom\n",
        after_log: str = "E   ValueError: other\n",
    ) -> Path:
        run_dir = self.state_dir / "runs" / f"{task_id}-{status}"
        run_dir.mkdir(parents=True, exist_ok=True)
        hashes_before = {"/tmp/proj/pkg/mod.py": "aaa"}
        hashes_after = {"/tmp/proj/pkg/mod.py": "bbb"}
        record: dict = {
            "task_id": task_id,
            "status": status,
            "model": "test-model",
            "repaired": repaired,
            "task_dir": "/tmp/proj",
            "hashes_before": hashes_before,
            "hashes_after": hashes_after if repaired else dict(hashes_before),
        }
        if salvaged_from is not None:
            record["salvaged_from"] = salvaged_from
        (run_dir / "result.json").write_text(json.dumps(record), encoding="utf-8")
        (run_dir / "test.log").write_text(after_log, encoding="utf-8")
        (run_dir / "test_before_repair.log").write_text(before_log, encoding="utf-8")
        return run_dir

    def test_dry_run_finds_eligible_runs_without_writing_lessons(self) -> None:
        self._repaired_run("t1")
        found = report.backfill_repair_lessons(self.state_dir, dry_run=True)
        self.assertEqual(len(found), 1)
        rec = found[0]
        self.assertEqual(rec["task_id"], "t1")
        self.assertEqual(rec["task_dir"], "/tmp/proj")
        self.assertEqual(rec["files"], ["pkg/mod.py"])
        self.assertEqual(
            rec["error_signature"],
            sandbox._lesson_error_signature("E   TypeError: boom\n"),
        )
        self.assertNotEqual(
            rec["error_signature"],
            sandbox._lesson_error_signature("E   ValueError: other\n"),
        )
        self.assertFalse((self.state_dir / "lessons.jsonl").exists())

    def test_backfill_writes_deduplicated_lessons(self) -> None:
        self._repaired_run("t1")
        first = report.backfill_repair_lessons(self.state_dir, dry_run=False)
        self.assertEqual(len(first), 1)
        lessons = self.state_dir / "lessons.jsonl"
        self.assertTrue(lessons.is_file())
        lines = [
            ln for ln in lessons.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        self.assertEqual(len(lines), 1)
        # Second pass: everything already recorded, nothing new.
        second = report.backfill_repair_lessons(self.state_dir, dry_run=False)
        self.assertEqual(second, [])
        lines = [
            ln for ln in lessons.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        self.assertEqual(len(lines), 1)

    def test_backfill_includes_fix_summary_from_changes_patch(self) -> None:
        run_dir = self._repaired_run("t-fix")
        (run_dir / "changes.patch").write_text(
            "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
            "@@ -1,3 +1,4 @@ def foo():\n"
            " import json\n"
            "+return val.get('k')\n",
            encoding="utf-8",
        )
        results = report.backfill_repair_lessons(self.state_dir)
        self.assertEqual(len(results), 1)
        path = self.state_dir / "lessons.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(entries), 1)
        expected = sandbox._extract_patch_fix_summary(
            (run_dir / "changes.patch").read_text(encoding="utf-8")
        )
        self.assertEqual(entries[0].get("fix_summary"), expected)
        self.assertIn("in def foo", expected)

    def test_backfill_skips_runs_without_error_signature(self) -> None:
        run_dir = self._repaired_run(
            "t-passing",
            before_log=".... [100%]\n225 deselected in 0.04s\n",
        )
        del run_dir
        results = report.backfill_repair_lessons(self.state_dir)
        self.assertEqual(results, [])
        self.assertFalse((self.state_dir / "lessons.jsonl").exists())

    def test_non_repaired_and_blocked_runs_are_skipped(self) -> None:
        self._repaired_run("t-ok")
        self._repaired_run("t-norepair", repaired=False)
        self._repaired_run("t-blocked", status="validation_failed", repaired=False)
        found = report.backfill_repair_lessons(self.state_dir, dry_run=True)
        self.assertEqual([rec["task_id"] for rec in found], ["t-ok"])

    def test_backfill_micro_repair_attempted_and_prior_failed_attempt(self) -> None:
        # (a) micro_repair_attempted run (no `repaired` flag) with a preserved
        # pre-repair log is backfilled using that log's signature.
        run_dir = self.state_dir / "runs" / "t-mra-validation_passed_review_required"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(
            json.dumps({
                "task_id": "t-mra",
                "status": "validation_passed_review_required",
                "model": "test-model",
                "micro_repair_attempted": True,
                "task_dir": "/tmp/proj",
                "hashes_before": {"/tmp/proj/pkg/mod.py": "aaa"},
                "hashes_after": {"/tmp/proj/pkg/mod.py": "bbb"},
            }),
            encoding="utf-8",
        )
        (run_dir / "test.log").write_text("1 passed in 0.01s\n", encoding="utf-8")
        (run_dir / "test_before_repair.log").write_text(
            "E   TypeError: boom\n", encoding="utf-8"
        )

        # (b) blocked attempt followed by a passing run: the lesson comes from
        # the earlier failed test.log signature.
        blocked_dir = self.state_dir / "runs" / "t-prior-validation_failed"
        blocked_dir.mkdir(parents=True, exist_ok=True)
        (blocked_dir / "result.json").write_text(
            json.dumps({
                "task_id": "t-prior",
                "status": "validation_failed",
                "model": "test-model",
            }),
            encoding="utf-8",
        )
        (blocked_dir / "test.log").write_text(
            "E   AssertionError: bad value\n", encoding="utf-8"
        )
        passed_dir = self.state_dir / "runs" / "t-prior-validation_passed_review_required"
        passed_dir.mkdir(parents=True, exist_ok=True)
        (passed_dir / "result.json").write_text(
            json.dumps({
                "task_id": "t-prior",
                "status": "validation_passed_review_required",
                "model": "test-model",
                "task_dir": "/tmp/proj-b",
                "hashes_before": {"/tmp/proj-b/other.py": "aaa"},
                "hashes_after": {"/tmp/proj-b/other.py": "bbb"},
            }),
            encoding="utf-8",
        )
        (passed_dir / "test.log").write_text("1 passed in 0.01s\n", encoding="utf-8")

        found = report.backfill_repair_lessons(self.state_dir, dry_run=True)
        self.assertEqual(
            [rec["task_id"] for rec in found], ["t-mra", "t-prior"]
        )
        self.assertEqual(
            found[0]["error_signature"],
            sandbox._lesson_error_signature("E   TypeError: boom\n"),
        )
        self.assertEqual(
            found[1]["error_signature"],
            sandbox._lesson_error_signature("E   AssertionError: bad value\n"),
        )

    def test_cli_backfill_lessons(self) -> None:
        self._repaired_run("t1")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main([
                "stats", "--state-dir", str(self.state_dir), "--backfill-lessons",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("Backfilled repair lessons:", buf.getvalue())
        self.assertTrue((self.state_dir / "lessons.jsonl").is_file())


class ProjectIdentityTest(unittest.TestCase):
    """H-84: project/subproject identity for lessons and repo memory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)

    def test_resolve_project_identity_mappings(self) -> None:
        # analytics-gitops worktrees share one project id
        pid, subs = report.resolve_project_identity(
            "/work/medvidi/worktrees/gitops-abc/ai/serving")
        self.assertEqual(pid, "medvidi/analytics-gitops")
        self.assertIn("ai/serving", subs)
        pid2, subs2 = report.resolve_project_identity(
            "/work/medvidi/analytics-gitops", files=["ai/local_copilot/x.py"])
        self.assertEqual(pid2, "medvidi/analytics-gitops")
        self.assertIn("ai/local_copilot", subs2)
        # Small Model Lab
        pid3, subs3 = report.resolve_project_identity(
            str(self.base / "Small Model Lab"), files=["sml/tools/foo.py"])
        self.assertEqual(pid3, "personal/small-model-lab")
        self.assertIn("sml", subs3)
        # Symlinked Harness worktree resolves to personal/harness
        real = self.base / "Harness"
        (real / "harness").mkdir(parents=True)
        link = self.base / "work" / "medvidi" / "Harness"
        link.parent.mkdir(parents=True)
        link.symlink_to(real)
        pid4, subs4 = report.resolve_project_identity(str(link / "harness"),
                                                      files=["harness/report.py"])
        self.assertEqual(pid4, "personal/harness")
        self.assertIn("harness", subs4)

    def test_load_workspace_lessons_project_isolation(self) -> None:
        from harness.sandbox import resolve_project_identity  # noqa: F401
        state_dir = self.base / "state"
        state_dir.mkdir()
        dir_a = "/work/medvidi/worktrees/gitops-a"
        dir_b = "/work/medvidi/worktrees/gitops-b"

        def lesson(directory, sig, files):
            record_repair_lesson(
                state_dir,
                {"task_id": "t", "directory": directory, "model": "m"},
                f"E   {sig}\n", files,
            )

        lesson(dir_a, "ImportError: serving", ["ai/serving/x.py"])
        lesson(dir_a, "ValueError: copilot", ["ai/local_copilot/z.py"])
        lesson(dir_a, "TypeError: airflow", ["airflow/dags/w.py"])

        # Same subproject across different worktrees matches (tier 1).
        text = sandbox.load_workspace_lessons(
            state_dir, dir_b, files=["ai/serving/y.py"])
        self.assertIn("ImportError: serving", text)
        self.assertNotIn("ValueError: copilot", text)
        self.assertNotIn("TypeError: airflow", text)

        # medvidi lessons never leak into other projects.
        sml_dir = self.base / "Small Model Lab"
        sml_dir.mkdir()
        self.assertEqual(
            sandbox.load_workspace_lessons(
                state_dir, str(sml_dir), files=["sml/main.py"]), "")

    def test_build_repo_memory_hint_no_cross_project_fallback(self) -> None:
        state_dir = self.base / "state"
        run_dir = state_dir / "runs" / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps(
            {"task_id": "t1", "status": "blocked", "reason": "step_limit"}),
            encoding="utf-8")
        (run_dir / "task.json").write_text(json.dumps(
            {"task_id": "t1", "directory": "/work/medvidi/analytics-gitops"}),
            encoding="utf-8")

        # Fewer than min_occurrences runs for the queried project: no hint,
        # and never a fallback to unrelated workspace runs.
        self.assertEqual(report.build_repo_memory_hint(
            state_dir, directory="/work/medvidi/analytics-gitops",
            min_occurrences=2), "")

        # A second run for the same project produces the hint.
        run_dir2 = state_dir / "runs" / "run2"
        run_dir2.mkdir(parents=True)
        (run_dir2 / "result.json").write_text(json.dumps(
            {"task_id": "t2", "status": "blocked", "reason": "step_limit"}),
            encoding="utf-8")
        (run_dir2 / "task.json").write_text(json.dumps(
            {"task_id": "t2", "directory": "/work/medvidi/worktrees/gitops-x"}),
            encoding="utf-8")
        hint = report.build_repo_memory_hint(
            state_dir, directory="/work/medvidi/analytics-gitops",
            min_occurrences=2)
        self.assertIn("step_limit", hint)

    def test_backfill_repair_lessons_includes_project_id(self) -> None:
        state_dir = self.base / "state"
        task_dir = self.base / "repo"
        task_dir.mkdir()
        run_dir = state_dir / "runs" / "r1"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps({
            "task_id": "t1", "status": "validation_passed_review_required",
            "repaired": True, "task_dir": str(task_dir),
            "hashes_before": {str(task_dir / "a.py"): "x"},
            "hashes_after": {str(task_dir / "a.py"): "y"},
        }), encoding="utf-8")
        (run_dir / "test_before_repair.log").write_text(
            "E   TypeError: bad\n", encoding="utf-8")
        results = report.backfill_repair_lessons(state_dir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["project_id"], str(task_dir.resolve()))
        self.assertIn("subprojects", results[0])


if __name__ == "__main__":
    unittest.main()