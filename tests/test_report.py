"""Tests for state-run day filtering and insights breakdowns (report.py + cli stats)."""

from __future__ import annotations

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

    def test_record_repair_lesson_falls_back_to_first_line(self) -> None:
        record_repair_lesson(self.state_dir, self._task(), "\nrandom failure text\n", [])
        path = self.state_dir / "lessons.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(entries[0]["error_signature"], "random failure text")

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


if __name__ == "__main__":
    unittest.main()
