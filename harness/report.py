"""Benchmark run summarization and Markdown report rendering.

The helpers in this module are deliberately tolerant of raw runner JSON:
field lookups accept the common key spellings produced by the guard runner
and the guard proxy, and ``proxy_summary`` fills any gaps that the run
record itself does not carry.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from harness.guards import (
    check_readonly_sql_tool_calls,
    compact_stale_reads,
    detect_duplicate_tool_calls,
    estimate_messages_tokens,
)
from harness.routing import apply_provider_routing

__all__ = [
    "summarize_run",
    "aggregate_benchmark_runs",
    "format_benchmark_markdown",
    "scan_state_runs",
    "format_state_stats_markdown",
    "compute_state_insights",
    "build_repo_memory_hint",
    "format_state_insights_markdown",
    "simulate_replay",
    "format_replay_markdown",
    "export_sml_dataset",
    "purge_ephemeral_test_runs",
    "backfill_salvaged_runs",
    "backfill_repair_lessons",
]

# Blocked reasons considered salvageable when the test actually passed and a
# non-empty patch exists (H-57 retroactive post-edit salvage backfill).
SALVAGEABLE_BLOCK_REASONS = frozenset(
    {
        "step_limit",
        "worker_timeout",
        "repeated_tool_error",
        "repeated_file_read_without_write",
        "model_error",
    }
)

# Statuses treated as a Pass@1 success (after lowercasing).
PASS_STATUSES = frozenset(
    {
        "pass",
        "passed",
        "success",
        "succeeded",
        "ok",
        "complete",
        "completed",
        "done",
        "0",
        "true",
        "validation_passed_review_required",
        "configuration_verified_no_model_call",
    }
)

_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "y", "on"})


def _first(mapping: dict, keys, default=None):
    """Return the first present, non-None value among ``keys``."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _to_number(value):
    """Coerce ``value`` to int/float, or None when not numeric."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _to_int(value) -> int:
    number = _to_number(value)
    if number is None:
        return 0
    return int(number)


def _to_float(value) -> float:
    number = _to_number(value)
    if number is None:
        return 0.0
    return float(number)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


_EPHEMERAL_TEST_TASK_IDS = frozenset({"h42-test", "h42-val"})

_EPHEMERAL_PATH_PREFIXES = (
    "/var/folders/",
    "/private/var/folders/",
    "/tmp/pytest-of-",
)


def _is_ephemeral_test_run(result: dict) -> bool:
    """Return True for throwaway test runs (h42-test/h42-val touching temp paths)."""
    task_id = str(result.get("task_id") or "")
    if task_id not in _EPHEMERAL_TEST_TASK_IDS:
        return False
    candidates = list(
        _to_str_list(_first(result, ("files", "allowed_files", "target_files")))
    )
    for key in ("hashes_before", "hashes_after"):
        value = result.get(key)
        if isinstance(value, dict):
            candidates.extend(str(path) for path in value)
    return any(
        path.startswith(_EPHEMERAL_PATH_PREFIXES) for path in candidates if path
    )


def purge_ephemeral_test_runs(state_dir: str | Path) -> list[str]:
    """Delete run directories for ephemeral test runs; return their run ids."""
    runs_dir = Path(state_dir) / "runs"
    if not runs_dir.is_dir():
        return []

    run_dirs: list[tuple[float, Path]] = []
    for child in runs_dir.iterdir():
        try:
            if not child.is_dir() or not (child / "result.json").is_file():
                continue
            run_dirs.append((child.stat().st_mtime, child))
        except OSError:
            continue
    run_dirs.sort(key=lambda entry: entry[0])

    purged: list[str] = []
    for _, run_dir in run_dirs:
        try:
            with (run_dir / "result.json").open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict):
            continue
        if not _is_ephemeral_test_run(result):
            continue
        run_id = str(_first(result, ("run_id", "runId", "id")) or run_dir.name)
        shutil.rmtree(run_dir)
        purged.append(run_id)
    return purged


def _to_str_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def summarize_run(result: dict, proxy_summary: dict | None = None) -> dict:
    """Normalize one run record into the canonical benchmark shape.

    Values present in ``result`` win; ``proxy_summary`` (guard-proxy
    telemetry) only fills fields the record does not carry. Missing numeric
    fields become 0, missing booleans False, missing lists empty, and an
    unknown status becomes ``"unknown"`` (counted as a failure).
    """
    if not isinstance(result, dict):
        result = {}
    proxy = proxy_summary if isinstance(proxy_summary, dict) else {}

    task_id = _first(result, ("task_id", "taskId", "id", "task"))
    if task_id is None:
        task_id = _first(proxy, ("task_id", "taskId", "id"))

    status = _first(result, ("status", "result", "outcome", "final_status", "exit_status"))
    if status is None:
        status = _first(proxy, ("status", "outcome"))
    status = str(status).strip().lower() if status is not None else "unknown"

    model = _first(result, ("model", "model_id", "modelId", "model_name"))
    if model is None:
        model = _first(proxy, ("model", "model_id"))

    steps = _first(result, ("steps", "step_count", "model_steps", "n_steps", "num_steps", "total_steps"))
    if steps is None:
        steps = _first(proxy, ("steps", "step_count"))

    cost = _first(result, ("cost_usd", "total_cost_usd", "cost", "costUSD", "usd_cost", "spend_usd"))
    if cost is None:
        cost = _first(proxy, ("cost_usd", "cost"))

    micro_repair = _first(result, ("micro_repair_used", "micro_repair_attempted", "micro_repair", "used_micro_repair", "microRepairUsed"))
    if micro_repair is None:
        # Proxy may report either a flag or a repair-event count.
        count = _first(proxy, ("micro_repair_count", "micro_repairs", "micro_repair"))
        micro_repair = True if (count is not None and _to_int(count) > 0) else _first(proxy, ("micro_repair_used",))

    dispatched = _first(result, ("dispatched_count", "dispatched", "dispatch_count", "dispatches", "num_dispatched"))
    if dispatched is None:
        dispatched = _first(proxy, ("dispatched_count", "dispatched", "dispatches"))

    blocked = _first(result, ("blocked_count", "blocked", "block_count", "num_blocked", "denied_count"))
    if blocked is None:
        blocked = _first(proxy, ("blocked_count", "blocked"))

    sticky = _first(result, ("sticky_provider", "stickyProvider", "sticky", "pinned_provider"))
    if sticky is None:
        sticky = _first(proxy, ("sticky_provider", "sticky", "pinned_provider"))

    degraded = _first(result, ("degraded_providers", "degradedProviders", "degraded", "degraded_list"))
    if degraded is None:
        degraded = _first(proxy, ("degraded_providers", "degraded"))

    return {
        "task_id": task_id,
        "status": status,
        "model": model,
        "steps": _to_number(steps) if steps is not None else 0,
        "cost_usd": _to_float(cost),
        "micro_repair_used": _to_bool(micro_repair) if micro_repair is not None else False,
        "dispatched_count": _to_int(dispatched),
        "blocked_count": _to_int(blocked),
        "sticky_provider": sticky,
        "degraded_providers": _to_str_list(degraded),
    }


def aggregate_benchmark_runs(runs: list[dict]) -> dict:
    """Aggregate normalized run records into benchmark summary metrics.

    Every entry is passed through :func:`summarize_run`; missing numeric
    fields count as 0 and non-passing statuses count against Pass@1.
    """
    normalized = [summarize_run(run) for run in (runs or [])]
    total = len(normalized)

    pass_count = sum(1 for run in normalized if run["status"] in PASS_STATUSES)
    steps = [float(run["steps"]) for run in normalized]
    costs = [run["cost_usd"] for run in normalized]

    return {
        "total_runs": total,
        "pass_count": pass_count,
        "pass_rate": (pass_count / total) if total else 0.0,
        "median_steps": float(statistics.median(steps)) if steps else 0.0,
        "median_cost_usd": float(statistics.median(costs)) if costs else 0.0,
        "total_cost_usd": float(sum(costs)),
        "micro_repair_count": sum(1 for run in normalized if run["micro_repair_used"]),
    }


def format_benchmark_markdown(baseline_runs: list[dict], guarded_runs: list[dict]) -> str:
    """Render a GitHub-flavored Markdown comparison of baseline vs guarded runs."""
    baseline = aggregate_benchmark_runs(baseline_runs or [])
    guarded = aggregate_benchmark_runs(guarded_runs or [])

    rows = [
        (
            "Pass@1 (%)",
            f"{baseline['pass_rate'] * 100:.1f}",
            f"{guarded['pass_rate'] * 100:.1f}",
        ),
        (
            "Median Steps",
            f"{baseline['median_steps']:.1f}",
            f"{guarded['median_steps']:.1f}",
        ),
        (
            "Median Cost (USD)",
            f"{baseline['median_cost_usd']:.4f}",
            f"{guarded['median_cost_usd']:.4f}",
        ),
        (
            "Total Cost (USD)",
            f"{baseline['total_cost_usd']:.4f}",
            f"{guarded['total_cost_usd']:.4f}",
        ),
        (
            "Auto-Repaired Runs",
            str(baseline["micro_repair_count"]),
            str(guarded["micro_repair_count"]),
        ),
    ]

    lines = [
        "# Benchmark Comparison: Baseline vs Guarded Harness",
        "",
        f"Runs compared: baseline {baseline['total_runs']}, guarded {guarded['total_runs']}.",
        "",
        "| Metric | Baseline (Raw Agent) | Guarded (H-1..H-5 Harness) |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {metric} | {baseline_value} | {guarded_value} |" for metric, baseline_value, guarded_value in rows)
    return "\n".join(lines)


# Request-record keys aggregated when the sibling ``requests.json`` holds a
# list of guard-proxy request records instead of a ready-made summary dict.
_PROXY_COST_KEYS = ("cost_usd", "total_cost_usd", "cost", "costUSD", "spend_usd", "usd_cost")
_PROXY_STEPS_KEYS = ("steps", "step_count", "model_steps")
_PROXY_DISPATCHED_KEYS = ("dispatched_count", "dispatched", "dispatch_count", "dispatches")
_PROXY_BLOCKED_KEYS = ("blocked_count", "blocked", "block_count", "denied_count")
_PROXY_MICRO_REPAIR_KEYS = ("micro_repair_count", "micro_repairs", "micro_repair", "used_micro_repair")


def _summarize_proxy_requests(requests: list) -> dict:
    """Aggregate a list of guard-proxy request records into a summary dict."""
    total_cost = 0.0
    total_steps = 0
    total_dispatched = 0
    total_blocked = 0
    total_micro_repairs = 0
    for request in requests:
        if not isinstance(request, dict):
            continue
        total_cost += _to_float(_first(request, _PROXY_COST_KEYS))
        total_steps += _to_int(_first(request, _PROXY_STEPS_KEYS))
        total_dispatched += _to_int(_first(request, _PROXY_DISPATCHED_KEYS))
        total_blocked += _to_int(_first(request, _PROXY_BLOCKED_KEYS))
        total_micro_repairs += _to_int(_first(request, _PROXY_MICRO_REPAIR_KEYS))
    return {
        "cost_usd": total_cost,
        "steps": total_steps,
        "dispatched_count": total_dispatched,
        "blocked_count": total_blocked,
        "micro_repair_count": total_micro_repairs,
    }


def _load_proxy_summary(path: Path) -> dict | None:
    """Load the optional sibling ``requests.json`` as a proxy summary.

    Accepts either a ready-made summary object or a list of request records
    (aggregated into one summary). Returns ``None`` when the file is absent,
    unreadable, or malformed.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return _summarize_proxy_requests(payload)
    return None


def scan_state_runs(
    state_dir: str | Path,
    *,
    days: int | None = None,
    now: float | None = None,
) -> list[dict]:
    """Scan a live state directory for guard-runner run records.

    Walks ``<state_dir>/runs/<run>/result.json`` directories oldest-first by
    run-directory mtime, loading each run's optional sibling
    ``requests.json`` guard-proxy telemetry as ``proxy_summary``. A missing
    ``runs/`` directory yields ``[]``; run directories without a readable
    ``result.json`` JSON object are skipped. When ``days`` is given, only
    runs whose directory mtime falls within the last ``days`` days are kept.
    """
    runs_dir = Path(state_dir) / "runs"
    if not runs_dir.is_dir():
        return []

    cutoff = None
    if days is not None:
        cutoff = (time.time() if now is None else now) - days * 86400.0

    run_dirs: list[tuple[float, Path]] = []
    for child in runs_dir.iterdir():
        try:
            if not child.is_dir() or not (child / "result.json").is_file():
                continue
            mtime = child.stat().st_mtime
            if cutoff is not None and mtime < cutoff:
                continue
            run_dirs.append((mtime, child))
        except OSError:
            continue
    run_dirs.sort(key=lambda entry: entry[0])

    runs: list[dict] = []
    for _, run_dir in run_dirs:
        try:
            with (run_dir / "result.json").open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict):
            continue
        if _is_ephemeral_test_run(result):
            continue
        runs.append(summarize_run(result, _load_proxy_summary(run_dir / "requests.json")))
    return runs


def format_state_stats_markdown(
    state_dir: str | Path,
    *,
    days: int | None = None,
    now: float | None = None,
) -> str:
    """Render a Markdown stats table for runs in a live state directory.

    Runs are grouped by model (an unknown model shows as ``unknown``) with a
    trailing ``ALL`` summary row covering every run. ``days`` limits the
    scan to runs from the last N days.
    """
    runs = scan_state_runs(state_dir, days=days, now=now)
    if not runs:
        return "No runs found."

    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(str(run.get("model") or "unknown"), []).append(run)

    def _row(label: str, group_runs: list[dict]) -> str:
        agg = aggregate_benchmark_runs(group_runs)
        blocked_total = sum(_to_int(run.get("blocked_count")) for run in group_runs)
        return (
            f"| {label} | {agg['total_runs']} | {agg['pass_rate'] * 100:.1f} "
            f"| {agg['median_steps']:.1f} | {agg['median_cost_usd']:.4f} "
            f"| {agg['total_cost_usd']:.4f} | {blocked_total} "
            f"| {agg['micro_repair_count']} |"
        )

    lines = [
        "# Live State Directory Stats",
        "",
        "| Model | Runs | Pass@1 (%) | Median Steps | Median Cost (USD) "
        "| Total Cost (USD) | Blocked Requests | Auto-Repairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_row(model, groups[model]) for model in sorted(groups))
    lines.append(_row("ALL", runs))
    return "\n".join(lines)


_RESOLVED_STATUSES = frozenset(
    {"validation_passed_review_required", "patch_ready_review_required", "succeeded"}
)

_READ_TOOL_NAMES = frozenset({"read", "read_file"})
_EDIT_TOOL_NAMES = frozenset({"edit", "write", "edit_file", "write_file", "apply_patch"})


def _iter_run_records(state_dir: str | Path, *, days: int | None = None, now: float | None = None):
    """Yield ``(run_dir, result_dict)`` for readable state runs, oldest-first by mtime.

    When ``days`` is given, only runs whose directory mtime falls within the
    last ``days`` days are yielded.
    """
    runs_dir = Path(state_dir) / "runs"
    if not runs_dir.is_dir():
        return
    cutoff = None
    if days is not None:
        cutoff = (time.time() if now is None else now) - days * 86400.0
    run_dirs: list[tuple[float, Path]] = []
    for child in runs_dir.iterdir():
        try:
            if not child.is_dir() or not (child / "result.json").is_file():
                continue
            mtime = child.stat().st_mtime
            if cutoff is not None and mtime < cutoff:
                continue
            run_dirs.append((mtime, child))
        except OSError:
            continue
    run_dirs.sort(key=lambda entry: entry[0])
    for _, run_dir in run_dirs:
        try:
            with (run_dir / "result.json").open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict):
            continue
        if _is_ephemeral_test_run(result):
            continue
        if result.get("status") == "configuration_verified_no_model_call":
            continue
        yield run_dir, result


def _load_tool_spans(run_dir: Path) -> list[dict]:
    """Load TOOL-kind spans from a run's traces.jsonl, sorted by start time."""
    traces_path = run_dir / "traces.jsonl"
    if not traces_path.is_file():
        return []
    spans: list[dict] = []
    try:
        with traces_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    span = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(span, dict):
                    continue
                attrs = span.get("attributes") or {}
                if not isinstance(attrs, dict):
                    continue
                if attrs.get("openinference.span.kind") != "TOOL":
                    continue
                spans.append(span)
    except OSError:
        return []
    spans.sort(key=lambda span: span.get("start_time_ns", 0))
    return spans


def compute_state_insights(
    state_dir: str | Path,
    *,
    days: int | None = None,
    now: float | None = None,
    _include_recent: bool = True,
) -> dict:
    """Compute task-funnel, blocked-reason and tool-error insights from a state directory."""
    task_order: list[str] = []
    task_runs: dict[str, list[dict]] = {}
    task_max_mtime: dict[str, float] = {}

    blocked_counter: Counter[str] = Counter()
    tool_error_counter: Counter[tuple[str, str]] = Counter()
    reads_before_edit_samples: list[int] = []
    zero_edit_failed_runs = 0

    ref_now = time.time() if now is None else now

    for run_dir, result in _iter_run_records(state_dir, days=days, now=now):
        task_id = str(result.get("task_id") or run_dir.name)
        if task_id not in task_runs:
            task_order.append(task_id)
            task_runs[task_id] = []
        task_runs[task_id].append(result)
        try:
            mtime = run_dir.stat().st_mtime
        except OSError:
            mtime = 0.0
        task_max_mtime[task_id] = max(task_max_mtime.get(task_id, 0.0), mtime)

        status = str(result.get("status") or "")
        if status == "blocked":
            blocked_counter[str(result.get("reason") or "unknown")] += 1

        spans = _load_tool_spans(run_dir)
        if not spans:
            continue
        for span in spans:
            if span.get("status") != "ERROR":
                continue
            attrs = span.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            tool_name = str(attrs.get("tool.name") or span.get("name") or "unknown")
            error_text = str(
                attrs.get("output.value")
                or attrs.get("error.message")
                or span.get("status_message")
                or "error"
            )
            error_prefix = error_text.splitlines()[0][:80] if error_text else "error"
            tool_error_counter[(tool_name, error_prefix)] += 1

        reads = 0
        seen_edit = False
        for span in spans:
            attrs = span.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            tool_name = str(attrs.get("tool.name") or span.get("name") or "unknown")
            lowered = tool_name.lower()
            if lowered in _EDIT_TOOL_NAMES:
                seen_edit = True
                break
            if lowered in _READ_TOOL_NAMES:
                reads += 1
        if seen_edit:
            reads_before_edit_samples.append(reads)
        elif status == "blocked":
            zero_edit_failed_runs += 1

    recent_cutoff = ref_now - 7 * 86400.0
    funnel = {
        "total_tasks": len(task_order),
        "resolved_tasks": 0,
        "first_try_ok": 0,
        "retry_or_escalation_ok": 0,
        "micro_repair_ok": 0,
        "salvaged_ok": 0,
        "unresolved_tasks": 0,
        "unresolved_active_tasks": 0,
        "unresolved_stale_single_shot_tasks": 0,
    }
    for task_id in task_order:
        runs = task_runs[task_id]
        first_resolved_idx = next(
            (idx for idx, run in enumerate(runs) if run.get("status") in _RESOLVED_STATUSES),
            None,
        )
        if first_resolved_idx is None:
            funnel["unresolved_tasks"] += 1
            if len(runs) >= 2 or task_max_mtime.get(task_id, 0.0) >= recent_cutoff:
                funnel["unresolved_active_tasks"] += 1
            else:
                funnel["unresolved_stale_single_shot_tasks"] += 1
            continue
        funnel["resolved_tasks"] += 1
        if first_resolved_idx > 0:
            funnel["retry_or_escalation_ok"] += 1
            continue
        run = runs[first_resolved_idx]
        micro_repair_attempted = bool(run.get("micro_repair_attempted")) or bool(run.get("micro_repair_used"))
        auto_repairs = _to_int(run.get("auto_repairs"))
        salvaged = bool(run.get("salvaged")) or bool(run.get("salvaged_from"))
        salvage_reason = run.get("salvage_reason")
        if micro_repair_attempted or auto_repairs > 0:
            funnel["micro_repair_ok"] += 1
        elif salvaged or salvage_reason:
            funnel["salvaged_ok"] += 1
        else:
            funnel["first_try_ok"] += 1

    avg_reads = (
        sum(reads_before_edit_samples) / len(reads_before_edit_samples)
        if reads_before_edit_samples
        else 0.0
    )

    insights = {
        "task_funnel": funnel,
        "top_blocked_reasons": blocked_counter.most_common(10),
        "top_tool_errors": tool_error_counter.most_common(10),
        "avg_reads_before_first_edit": float(avg_reads),
        "zero_edit_failed_runs": zero_edit_failed_runs,
    }
    if days is None and _include_recent:
        insights["recent_7d"] = _recent_7d_insights(state_dir, ref_now)
    return insights


def _recent_7d_insights(state_dir: str | Path, ref_now: float) -> dict:
    """Compute the compact last-7-days snapshot embedded in the full insights report."""
    recent_runs = scan_state_runs(state_dir, days=7, now=ref_now)
    recent_insights = compute_state_insights(
        state_dir, days=7, now=ref_now, _include_recent=False
    )
    return {
        "runs": len(recent_runs),
        "pass_rate": (
            aggregate_benchmark_runs(recent_runs)["pass_rate"] if recent_runs else 0.0
        ),
        "resolved_tasks": recent_insights["task_funnel"]["resolved_tasks"],
        "unresolved_tasks": recent_insights["task_funnel"]["unresolved_tasks"],
        "top_blocked_reasons": recent_insights["top_blocked_reasons"][:5],
    }


def format_state_insights_markdown(
    state_dir: str | Path,
    *,
    days: int | None = None,
    now: float | None = None,
) -> str:
    """Render the state-directory insights report as Markdown."""
    insights = compute_state_insights(state_dir, days=days, now=now)
    funnel = insights["task_funnel"]
    lines: list[str] = []
    if days is None and "recent_7d" in insights:
        recent = insights["recent_7d"]
        lines.extend(
            [
                "## Recent Window (Last 7 Days)",
                "",
                f"- Runs: {recent['runs']}",
                f"- Pass@1: {recent['pass_rate'] * 100:.1f}%",
                f"- Resolved: {recent['resolved_tasks']}",
                f"- Unresolved: {recent['unresolved_tasks']}",
                "- Top Blocked Reasons: "
                + (
                    ", ".join(
                        f"{reason} ({count})" for reason, count in recent["top_blocked_reasons"]
                    )
                    if recent["top_blocked_reasons"]
                    else "None"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Task Resolution Funnel",
            "",
            f"- Total tasks: {funnel['total_tasks']}",
            f"- Resolved: {funnel['resolved_tasks']}",
            f"  - First try: {funnel['first_try_ok']}",
            f"  - Retry/escalation: {funnel['retry_or_escalation_ok']}",
            f"  - Micro repair: {funnel['micro_repair_ok']}",
            f"  - Salvaged: {funnel['salvaged_ok']}",
            f"- Unresolved: {funnel['unresolved_tasks']}",
            f"  - Active / multi-attempt: {funnel['unresolved_active_tasks']}",
            f"  - Single-shot / probe / stale (>7d, 1 attempt): {funnel['unresolved_stale_single_shot_tasks']}",
            "",
            "## Top Blocked Reasons",
            "",
        ]
    )
    if insights["top_blocked_reasons"]:
        for reason, count in insights["top_blocked_reasons"]:
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Top Tool Errors", ""])
    if insights["top_tool_errors"]:
        for (tool_name, error_prefix), count in insights["top_tool_errors"]:
            lines.append(f"- {tool_name}: {error_prefix} ({count})")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            f"Average reads before first edit: {insights['avg_reads_before_first_edit']:.2f}",
            f"Zero-edit failed runs: {insights['zero_edit_failed_runs']}",
        ]
    )
    lessons_path = Path(state_dir) / "lessons.jsonl"
    if lessons_path.is_file():
        lesson_count = sum(
            1
            for line in lessons_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        )
        if lesson_count > 0:
            lines.append(f"- Recorded repair lessons: {lesson_count}")
    return "\n".join(lines)


def resolve_project_identity(task_dir, files=()):
    """H-84: re-export canonical project/submodule identity resolver."""
    from harness.sandbox import resolve_project_identity as _resolve
    return _resolve(task_dir, files=files)


def build_repo_memory_hint(
    state_dir: str | Path,
    directory: str | Path | None = None,
    *,
    min_occurrences: int = 2,
    max_chars: int = 600,
) -> str:
    """Learn a short hint from prior state runs for the same workspace directory.

    Counts recurring blocked reasons and tool errors (plus the average
    reads-before-first-edit ratio) and returns a prompt suffix, or ``''`` when
    nothing recurred at least ``min_occurrences`` times.
    """
    records = list(_iter_run_records(state_dir))
    if not records:
        return ""

    selected: list[tuple[Path, dict]] = []
    want_project: str | None = None
    if directory is not None:
        from harness.sandbox import resolve_project_identity
        want_project, _ = resolve_project_identity(directory)
        saw_any_task_dir = False
        for run_dir, result in records:
            try:
                task_path = run_dir / "task.json"
                if not task_path.is_file():
                    continue
                with task_path.open("r", encoding="utf-8") as handle:
                    task = json.load(handle)
            except (OSError, ValueError):
                continue
            if not isinstance(task, dict):
                continue
            task_dir = str(task.get("origin_directory") or task.get("directory") or task.get("task_dir") or "")
            if not task_dir:
                continue
            saw_any_task_dir = True
            run_project, _ = resolve_project_identity(
                task_dir, files=task.get("origin_files") or task.get("files") or ()
            )
            if run_project == want_project:
                selected.append((run_dir, result))
        if len(selected) < min_occurrences:
            if not saw_any_task_dir:
                selected = records
            else:
                # H-84: never fall back to unrelated workspace runs — a new
                # project (e.g. Small Model Lab) must not inherit global stats.
                return ""
    else:
        selected = records

    blocked_counter: Counter[str] = Counter()
    tool_error_counter: Counter[tuple[str, str]] = Counter()
    reads_before_edit_samples: list[int] = []

    for run_dir, result in selected:
        if str(result.get("status") or "") == "blocked":
            blocked_counter[str(result.get("reason") or "unknown")] += 1

        spans = _load_tool_spans(run_dir)
        if not spans:
            continue
        for span in spans:
            if span.get("status") != "ERROR":
                continue
            attrs = span.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            tool_name = str(attrs.get("tool.name") or span.get("name") or "unknown")
            error_text = str(
                attrs.get("output.value")
                or attrs.get("error.message")
                or span.get("status_message")
                or "error"
            )
            error_prefix = error_text.splitlines()[0][:80] if error_text else "error"
            tool_error_counter[(tool_name, error_prefix)] += 1

        reads = 0
        seen_edit = False
        for span in spans:
            attrs = span.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            tool_name = str(attrs.get("tool.name") or span.get("name") or "unknown")
            lowered = tool_name.lower()
            if lowered in _EDIT_TOOL_NAMES:
                seen_edit = True
                break
            if lowered in _READ_TOOL_NAMES:
                reads += 1
        if seen_edit:
            reads_before_edit_samples.append(reads)

    bullets: list[str] = []
    for reason, count in blocked_counter.most_common():
        if count >= min_occurrences:
            bullets.append(
                f"- Recurring block reason: {reason} ({count}x) — avoid repeating this failure mode."
            )
    for (tool_name, error_prefix), count in tool_error_counter.most_common():
        if count >= min_occurrences:
            bullets.append(f"- Recurring tool error in {tool_name}: {error_prefix} ({count}x).")
    if reads_before_edit_samples:
        avg_reads = sum(reads_before_edit_samples) / len(reads_before_edit_samples)
        if avg_reads >= 4.0:
            bullets.append(
                "- High read-before-edit ratio — batch file reads in turn 1 and edit directly."
            )

    if not bullets:
        return ""
    hint = "\n\n[REPO MEMORY HINT — learned from prior runs in this workspace]:\n" + "\n".join(
        bullets[:5]
    )
    return hint[:max_chars]


def _compact_context_window(messages: list[dict], *, max_context_tokens: int) -> list[dict]:
    """Invoke ``harness.guards.compact_context_window`` when available, with a safe fallback."""
    from harness import guards as _guards

    fn = getattr(_guards, "compact_context_window", None)
    if callable(fn):
        result = fn(messages, max_context_tokens=max_context_tokens)
        if isinstance(result, tuple):
            return list(result[0])
        return list(result)

    compacted, _ = compact_stale_reads(
        list(messages),
        turn_index=max(5, len(messages)),
        min_turn=5,
    )
    if max_context_tokens <= 0:
        return compacted
    while len(compacted) > 2 and estimate_messages_tokens(compacted) > max_context_tokens:
        drop_idx = 1 if (compacted[0].get("role") == "system") else 0
        compacted.pop(drop_idx)
    return compacted


def _load_replay_episodes(records_or_state_dir: list[dict] | str | Path) -> list[dict]:
    """Normalize a list of episode dicts or a state_dir path into episode dicts."""
    if isinstance(records_or_state_dir, (str, Path)):
        path = Path(records_or_state_dir)
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            try:
                payload = json.loads(text)
            except ValueError:
                episodes: list[dict] = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict):
                        episodes.append(item)
                return episodes
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            if isinstance(payload, dict):
                for key in ("episodes", "runs", "records"):
                    if isinstance(payload.get(key), list):
                        return [item for item in payload[key] if isinstance(item, dict)]
                return [payload]
            return []

        runs_dir = path / "runs"
        if not runs_dir.is_dir():
            return []
        run_dirs: list[tuple[float, Path]] = []
        for child in runs_dir.iterdir():
            try:
                if child.is_dir():
                    run_dirs.append((child.stat().st_mtime, child))
            except OSError:
                continue
        run_dirs.sort(key=lambda entry: entry[0])

        episodes = []
        for _, run_dir in run_dirs:
            result = None
            result_path = run_dir / "result.json"
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    result = None
                if isinstance(result, dict) and isinstance(result.get("messages"), list):
                    episodes.append(result)
                    continue
            requests_path = run_dir / "requests.json"
            appended = False
            if requests_path.is_file():
                try:
                    req_payload = json.loads(requests_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    req_payload = None
                if isinstance(req_payload, dict) and isinstance(req_payload.get("messages"), list):
                    episodes.append(req_payload)
                    appended = True
                elif isinstance(req_payload, list):
                    with_msgs = [
                        item for item in req_payload
                        if isinstance(item, dict) and isinstance(item.get("messages"), list)
                    ]
                    if with_msgs:
                        episodes.append(with_msgs[-1])
                        appended = True
            if appended:
                continue

            # Fall back to worker.log JSONL (tool_use lines) as synthetic episodes.
            worker_log = run_dir / "worker.log"
            if not worker_log.is_file():
                continue
            try:
                worker_text = worker_log.read_text(encoding="utf-8")
            except OSError:
                continue
            messages: list[dict] = []
            for line in worker_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") != "tool_use":
                    continue
                part = obj.get("part")
                if not isinstance(part, dict):
                    continue
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                tool_name = str(part.get("tool") or "unknown")
                raw_input = state.get("input", part.get("input"))
                if raw_input is None:
                    raw_input = {}
                raw_output = state.get("output", state.get("error", part.get("output")))
                call_id = f"{run_dir.name}:tool:{len(messages)}"
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(raw_input, default=str),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": str(raw_output if raw_output is not None else ""),
                })
            if messages:
                episodes.append({
                    "model": (result or {}).get("model") if isinstance(result, dict) else None,
                    "messages": messages,
                })
        return episodes

    return [item for item in (records_or_state_dir or []) if isinstance(item, dict)]


def simulate_replay(
    records_or_state_dir: list[dict] | str | Path,
    *,
    max_context_tokens: int = 6000,
    fallback_teacher_model: str | None = None,
    upstream_mode: str = "openrouter",
) -> dict:
    """Run offline A/B guard and routing simulation over recorded episodes or a state_dir."""
    episodes = _load_replay_episodes(records_or_state_dir)

    total_episodes = len(episodes)
    loop_warnings_triggered = 0
    sql_violations_caught = 0
    escalated_to_teacher_count = 0
    tokens_before = 0
    tokens_after = 0

    for record in episodes:
        raw_messages = record.get("messages")
        messages = [m for m in raw_messages if isinstance(m, dict)] if isinstance(raw_messages, list) else []
        model = _first(record, ("model", "model_id", "modelId", "model_name"))

        loop_warnings_triggered += len(detect_duplicate_tool_calls(messages))
        sql_violations_caught += len(check_readonly_sql_tool_calls(messages))

        before = estimate_messages_tokens(messages) if messages else 0
        compacted = _compact_context_window(messages, max_context_tokens=max_context_tokens)
        after = estimate_messages_tokens(compacted) if compacted else 0
        tokens_before += before
        tokens_after += after

        req_payload: dict = {"messages": compacted}
        if model is not None:
            req_payload["model"] = model
        raw_body = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        routed_bytes = apply_provider_routing(
            raw_body,
            "/chat/completions",
            default_model=model,
            turn_index=max(1, len(messages)),
            upstream_mode=upstream_mode,
            fallback_teacher_model=fallback_teacher_model,
        )
        try:
            routed = json.loads(routed_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            routed = {}
        if (
            fallback_teacher_model
            and isinstance(routed, dict)
            and routed.get("model") == fallback_teacher_model
            and model != fallback_teacher_model
        ):
            escalated_to_teacher_count += 1

    tokens_saved = max(0, tokens_before - tokens_after)
    token_reduction_pct = round((tokens_saved / tokens_before) * 100.0, 2) if tokens_before > 0 else 0.0

    return {
        "total_episodes": total_episodes,
        "loop_warnings_triggered": loop_warnings_triggered,
        "sql_violations_caught": sql_violations_caught,
        "escalated_to_teacher_count": escalated_to_teacher_count,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_saved,
        "token_reduction_pct": token_reduction_pct,
    }


def format_replay_markdown(summary: dict) -> str:
    """Render a GitHub-flavored Markdown report table for replay simulation results."""
    data = summary if isinstance(summary, dict) else {}
    rows = [
        ("Total Episodes", str(_to_int(data.get("total_episodes")))),
        ("Loop Warnings Triggered", str(_to_int(data.get("loop_warnings_triggered")))),
        ("SQL Violations Caught", str(_to_int(data.get("sql_violations_caught")))),
        ("Escalated to Teacher", str(_to_int(data.get("escalated_to_teacher_count")))),
        ("Tokens Before", str(_to_int(data.get("tokens_before")))),
        ("Tokens After", str(_to_int(data.get("tokens_after")))),
        ("Tokens Saved", str(_to_int(data.get("tokens_saved")))),
        ("Token Reduction (%)", f"{_to_float(data.get('token_reduction_pct')):.2f}%"),
    ]
    lines = [
        "# Offline Replay Simulation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {metric} | {value} |" for metric, value in rows)
    return "\n".join(lines)


def _load_run_messages(run_dir: Path) -> list:
    """Extract the ``messages`` list from the last usable record in requests.jsonl."""
    requests_path = run_dir / "requests.jsonl"
    if not requests_path.is_file():
        return []
    messages: list = []
    try:
        with requests_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                candidate = record
                if isinstance(candidate, dict):
                    for key in ("request", "body", "payload"):
                        nested = candidate.get(key)
                        if isinstance(nested, dict) and isinstance(nested.get("messages"), list):
                            candidate = nested
                            break
                if isinstance(candidate, dict) and isinstance(candidate.get("messages"), list):
                    messages = candidate["messages"]
    except OSError:
        return []
    return messages


_GUARD_INTERVENTION_MARKERS = (
    "[HARNESS_EDIT_HINT",
    "[HARNESS_READ_HINT",
    "[HARNESS_READ_BUDGET_WARNING",
    "[GUARD LOOP WARNING",
    "[AST PREFLIGHT ERROR",
    "[EDIT OSCILLATION WARNING",
)


def _message_text(msg) -> str:
    """Render a message record (dict, string, or arbitrary JSON) as text."""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    try:
        return json.dumps(msg, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(msg)


def _matched_guard_marker(msg):
    """Return the guard-intervention marker contained in ``msg``, or None."""
    content = _message_text(msg)
    for marker in _GUARD_INTERVENTION_MARKERS:
        if marker in content:
            return marker
    return None


def _is_clean_sft_run(result: dict, messages: list[dict]) -> bool:
    """Return False for runs tainted by micro-repair, salvage, or guard hints."""
    if not isinstance(result, dict):
        result = {}
    if result.get("micro_repair_attempted") or result.get("salvaged_from"):
        return False
    for msg in messages or []:
        if _matched_guard_marker(msg) is not None:
            return False
    return True


def _two_session_evidence(run_dir: Path) -> bool:
    """Return True when worker.log/requests.json shows two distinct sessions."""
    for name in ("requests.json", "worker.log"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        session_ids = set(
            re.findall(r"session[\"']?\s*[:=]\s*[\"']([A-Za-z0-9_-]+)", text)
        )
        if len(session_ids) >= 2:
            return True
    return False


def _lesson_error_signature(log_text: str) -> str:
    """Extract an error signature line from a failed test log."""
    if not log_text:
        return ""
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line.lower().startswith("traceback (most recent call last)"):
            for candidate in lines[index + 1:]:
                if candidate.startswith(("E   ", "E:")):
                    return candidate[4:].strip() or candidate
            break
    return lines[-1][:300] if lines else ""


def _diff_before_after_files(before_path: Path, after_path: Path) -> str:
    """Build a textual diff marker from files.before.json -> files.before_repair.json."""
    try:
        before = json.loads(before_path.read_text(encoding="utf-8", errors="replace"))
        after = json.loads(after_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return ""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return ""
    lines = []
    for path_key in sorted(set(before) | set(after)):
        if before.get(path_key) != after.get(path_key):
            lines.append(f"--- {path_key} (before)\n+++ {path_key} (after)")
            if path_key in after:
                lines.append(str(after[path_key]))
            else:
                lines.append("(file removed)")
    return "\n".join(lines)


def export_sml_dataset(
    state_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    mode: str = "sft",
    clean_only: bool = False,
    max_steps: int | None = None,
) -> list[dict]:
    """Export Small Model Lab fine-tuning (SFT) or preference (DPO) records.

    Scans ``<state_dir>/runs/*/`` oldest-first and produces JSONL-ready dicts:
    ``mode="sft"`` yields one record per successful run with a patch or
    transcript; ``mode="dpo"`` pairs each retry-to-success transition
    (rejected blocked run followed by a chosen passing run) within a task
    group. When ``output_path`` is given, records are atomically written as
    JSONL; the records are also returned as a list of dicts.

    With ``clean_only=True``, runs carrying micro-repair/salvage flags or any
    guard-intervention marker are filtered out. With ``max_steps``, only runs
    with a positive step count at or below the cap are kept.
    """
    if mode not in ("sft", "dpo", "sft-chat", "grpo", "probes", "repair-sft"):
        raise ValueError(
            f"Unsupported mode: {mode!r}; expected one of "
            "'sft', 'dpo', 'sft-chat', 'grpo', 'probes', 'repair-sft'"
        )

    records: list[dict] = []
    if mode == "sft":
        for run_dir, result in _iter_run_records(state_dir):
            summary = summarize_run(result, _load_proxy_summary(run_dir / "requests.json"))
            run_id = _first(result, ("run_id", "runId", "id")) or run_dir.name
            task_id = summary.get("task_id")
            status = str(summary.get("status") or "").lower()
            files = _to_str_list(_first(result, ("files", "allowed_files", "target_files")))
            prompt_value = _first(result, ("prompt", "task_prompt", "instruction", "description"))
            if (not files or prompt_value is None) and (run_dir / "task.json").is_file():
                try:
                    t_meta = json.loads((run_dir / "task.json").read_text(encoding="utf-8", errors="replace"))
                except (OSError, ValueError):
                    t_meta = {}
                if isinstance(t_meta, dict):
                    if not files:
                        files = _to_str_list(_first(t_meta, ("origin_files", "files", "allowed_files", "target_files")))
                    if prompt_value is None:
                        prompt_value = _first(t_meta, ("prompt", "task_prompt", "instruction", "description"))
            prompt = str(prompt_value) if prompt_value is not None else ""
            patch_path = run_dir / "changes.patch"
            patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.is_file() else ""
            messages = _load_run_messages(run_dir)
            passed = summary.get("passed") is True or status in PASS_STATUSES
            has_content = bool(patch.strip()) or bool(messages)
            if not (passed and has_content):
                continue
            steps = summary.get("steps", 0)
            if max_steps is not None and (steps <= 0 or steps > max_steps):
                continue
            if clean_only and not _is_clean_sft_run(result, messages):
                continue
            records.append({
                "task_id": task_id,
                "run_id": run_id,
                "model": summary.get("model"),
                "steps": summary.get("steps", 0),
                "cost_usd": summary.get("cost_usd", 0.0),
                "files": files,
                "prompt": prompt,
                "patch": patch,
                "messages": messages,
            })
        dataset = records
    elif mode == "sft-chat":
        for run_dir, result in _iter_run_records(state_dir):
            summary = summarize_run(result, _load_proxy_summary(run_dir / "requests.json"))
            run_id = _first(result, ("run_id", "runId", "id")) or run_dir.name
            task_id = summary.get("task_id")
            status = str(summary.get("status") or "").lower()
            messages = _load_run_messages(run_dir)
            passed = summary.get("passed") is True or status in PASS_STATUSES
            if not (passed and messages):
                continue
            steps = summary.get("steps", 0)
            if max_steps is not None and (steps <= 0 or steps > max_steps):
                continue
            if clean_only and not _is_clean_sft_run(result, messages):
                continue
            mask = [False] * len(messages)
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                j = i + 1
                tainted = False
                while (
                    j < len(messages)
                    and isinstance(messages[j], dict)
                    and messages[j].get("role") in ("tool", "user")
                ):
                    if _matched_guard_marker(messages[j]) is not None:
                        tainted = True
                        break
                    j += 1
                mask[i] = not tainted
            records.append({
                "task_id": task_id,
                "run_id": run_id,
                "model": summary.get("model"),
                "steps": steps,
                "cost_usd": summary.get("cost_usd", 0.0),
                "messages": messages,
                "assistant_turn_mask": mask,
            })
        dataset = records
    elif mode == "grpo":
        groups: dict[object, list[dict]] = {}
        group_order: list[object] = []
        for run_dir, result in _iter_run_records(state_dir):
            summary = summarize_run(result, _load_proxy_summary(run_dir / "requests.json"))
            run_id = _first(result, ("run_id", "runId", "id")) or run_dir.name
            task_id = summary.get("task_id")
            status = str(summary.get("status") or "").lower()
            files = _to_str_list(_first(result, ("files", "allowed_files", "target_files")))
            prompt_value = _first(result, ("prompt", "task_prompt", "instruction", "description"))
            if (not files or prompt_value is None) and (run_dir / "task.json").is_file():
                try:
                    t_meta = json.loads((run_dir / "task.json").read_text(encoding="utf-8", errors="replace"))
                except (OSError, ValueError):
                    t_meta = {}
                if isinstance(t_meta, dict):
                    if not files:
                        files = _to_str_list(_first(t_meta, ("origin_files", "files", "allowed_files", "target_files")))
                    if prompt_value is None:
                        prompt_value = _first(t_meta, ("prompt", "task_prompt", "instruction", "description"))
            prompt = str(prompt_value) if prompt_value is not None else ""
            patch_path = run_dir / "changes.patch"
            patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.is_file() else ""
            messages = _load_run_messages(run_dir)
            steps = summary.get("steps", 0)
            if max_steps is not None and (steps <= 0 or steps > max_steps):
                continue
            if clean_only and not _is_clean_sft_run(result, messages):
                continue
            key = task_id if task_id else (tuple(sorted(files)), " ".join(prompt.split()).lower())
            passed = summary.get("passed") is True or status in PASS_STATUSES
            guard_interventions = max(
                _to_int(result.get("guard_interventions")),
                sum(1 for msg in messages if _matched_guard_marker(msg) is not None),
            )
            sql_or_egress_violations = max(
                _to_int(result.get("sql_or_egress_violations")),
                max(
                    _to_int(result.get("sql_violations")),
                    len(check_readonly_sql_tool_calls(messages)),
                ) + _to_int(result.get("egress_violations")),
            )
            r_process = max(0.0, 1.0 - 0.25 * guard_interventions - 0.5 * sql_or_egress_violations)
            evidence_tool_ok = any(
                isinstance(msg, dict)
                and msg.get("role") == "tool"
                and bool(_message_text(msg).strip())
                and msg.get("is_error") is not True
                and not _message_text(msg).lower().startswith(
                    ("error", "traceback", "[ast preflight error")
                )
                for msg in messages
            )
            r_evidence = 1.0 if steps > 0 and (
                bool(patch.strip()) or result.get("test_return_code") == 0 or evidence_tool_ok
            ) else 0.0
            r_outcome = 1.0 if passed else 0.0
            r_total = round(r_outcome * r_evidence * r_process, 6)
            entry = {
                "task_id": task_id,
                "run_id": run_id,
                "model": summary.get("model"),
                "steps": steps,
                "prompt": prompt,
                "patch": patch,
                "messages": messages,
                "r_outcome": r_outcome,
                "r_evidence": r_evidence,
                "r_process": r_process,
                "r_total": r_total,
            }
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(entry)
        for key in group_order:
            totals = [entry["r_total"] for entry in groups[key]]
            if len(totals) >= 2:
                mean = sum(totals) / len(totals)
                std = math.sqrt(sum((x - mean) ** 2 for x in totals) / len(totals))
                for entry in groups[key]:
                    entry["group_advantage"] = round((entry["r_total"] - mean) / (std + 1e-6), 6)
            else:
                for entry in groups[key]:
                    entry["group_advantage"] = 0.0
            records.extend(groups[key])
        dataset = records
    elif mode == "probes":
        groups: dict[object, list[dict]] = {}
        group_order: list[object] = []
        for run_dir, result in _iter_run_records(state_dir):
            summary = summarize_run(result, _load_proxy_summary(run_dir / "requests.json"))
            status = str(summary.get("status") or "").lower()
            files = _to_str_list(_first(result, ("files", "allowed_files", "target_files")))
            prompt_value = _first(result, ("prompt", "task_prompt", "instruction", "description"))
            if (not files or prompt_value is None) and (run_dir / "task.json").is_file():
                try:
                    t_meta = json.loads((run_dir / "task.json").read_text(encoding="utf-8", errors="replace"))
                except (OSError, ValueError):
                    t_meta = {}
                if isinstance(t_meta, dict):
                    if not files:
                        files = _to_str_list(_first(t_meta, ("origin_files", "files", "allowed_files", "target_files")))
                    if prompt_value is None:
                        prompt_value = _first(t_meta, ("prompt", "task_prompt", "instruction", "description"))
            prompt = str(prompt_value) if prompt_value is not None else ""
            task_id = summary.get("task_id")
            key = task_id if task_id else (tuple(sorted(files)), " ".join(prompt.split()).lower())
            steps = summary.get("steps", 0)
            if max_steps is not None and (steps <= 0 or steps > max_steps):
                continue
            entry = {
                "task_id": task_id,
                "run_id": _first(result, ("run_id", "runId", "id")) or run_dir.name,
                "passed": summary.get("passed") is True or status in PASS_STATUSES,
                "messages": _load_run_messages(run_dir),
            }
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(entry)
        for key in group_order:
            group = groups[key]
            for run_index, entry in enumerate(group):
                messages = entry["messages"]
                for i, msg in enumerate(messages):
                    if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
                        continue
                    following = messages[i + 1] if i + 1 < len(messages) else None
                    if not (isinstance(following, dict) and following.get("role") in ("user", "tool")):
                        continue
                    marker = _matched_guard_marker(following)
                    if marker is None:
                        continue
                    anchor = None
                    for j in range(i + 1, len(messages)):
                        candidate = messages[j]
                        if not (isinstance(candidate, dict) and candidate.get("role") == "assistant"):
                            continue
                        nxt = messages[j + 1] if j + 1 < len(messages) else None
                        if nxt is None or _matched_guard_marker(nxt) is None:
                            anchor = candidate
                            break
                    if anchor is None:
                        for later in group[run_index + 1:]:
                            if not later["passed"]:
                                continue
                            later_messages = later["messages"]
                            for j, candidate in enumerate(later_messages):
                                if not (isinstance(candidate, dict) and candidate.get("role") == "assistant"):
                                    continue
                                nxt = later_messages[j + 1] if j + 1 < len(later_messages) else None
                                if nxt is None or _matched_guard_marker(nxt) is None:
                                    anchor = candidate
                                    break
                            if anchor is not None:
                                break
                    if anchor is None:
                        continue
                    records.append({
                        "task_id": entry["task_id"],
                        "run_id": entry["run_id"],
                        "intervention_type": marker.lstrip("[").rstrip(":").strip(),
                        "positive_anchor_turn": anchor,
                        "negative_target_turn": msg,
                    })
        dataset = records
    elif mode == "dpo":
        groups: dict[object, list[dict]] = {}
        group_order: list[object] = []
        for run_dir, result in _iter_run_records(state_dir):
            summary = summarize_run(result, _load_proxy_summary(run_dir / "requests.json"))
            status = str(summary.get("status") or "").lower()
            files = _to_str_list(_first(result, ("files", "allowed_files", "target_files")))
            prompt_value = _first(result, ("prompt", "task_prompt", "instruction", "description"))
            if (not files or prompt_value is None) and (run_dir / "task.json").is_file():
                try:
                    t_meta = json.loads((run_dir / "task.json").read_text(encoding="utf-8", errors="replace"))
                except (OSError, ValueError):
                    t_meta = {}
                if isinstance(t_meta, dict):
                    if not files:
                        files = _to_str_list(_first(t_meta, ("origin_files", "files", "allowed_files", "target_files")))
                    if prompt_value is None:
                        prompt_value = _first(t_meta, ("prompt", "task_prompt", "instruction", "description"))
            prompt = str(prompt_value) if prompt_value is not None else ""
            patch_path = run_dir / "changes.patch"
            patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.is_file() else ""
            task_id = summary.get("task_id")
            key = task_id if task_id else (tuple(sorted(files)), " ".join(prompt.split()).lower())
            entry = {
                "run_id": _first(result, ("run_id", "runId", "id")) or run_dir.name,
                "task_id": task_id,
                "prompt": prompt,
                "patch": patch,
                "status": status,
                "passed": summary.get("passed") is True,
                "model": summary.get("model"),
                "steps": summary.get("steps", 0),
                "result": result,
                "run_dir": run_dir,
                "reason": str(
                    _first(result, ("reason", "blocked_reason", "error_reason", "failure_reason"))
                    or ""
                ).strip().lower(),
            }
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(entry)

        for key in group_order:
            pending_rejected: dict | None = None
            for entry in groups[key]:
                is_rejected = entry["status"] == "blocked" and (
                    bool(entry["patch"].strip())
                    or entry["reason"] in ("validation_failed", "repeated_tool_error")
                )
                is_chosen = (
                    entry["status"] == "validation_passed_review_required"
                    or entry["passed"]
                    or entry["status"] in PASS_STATUSES
                ) and bool(entry["patch"].strip())
                if is_chosen and max_steps is not None and (
                    entry["steps"] <= 0 or entry["steps"] > max_steps
                ):
                    is_chosen = False
                if is_chosen and clean_only:
                    messages = _load_run_messages(Path(entry["run_dir"]))
                    if not _is_clean_sft_run(entry["result"], messages):
                        is_chosen = False
                if is_chosen and (
                    entry["result"].get("micro_repair_attempted") is True
                    or entry["result"].get("repaired") is True
                ):
                    run_dir_p = Path(entry["run_dir"])
                    result_p = entry["result"]
                    has_patch = bool(entry["patch"].strip())
                    before_patch_path = run_dir_p / "before_repair.patch"
                    pre_evidence = (
                        (run_dir_p / "test_before_repair.log").is_file()
                        or before_patch_path.is_file()
                        or (run_dir_p / "files.before_repair.json").is_file()
                        or (
                            (run_dir_p / "files.before.json").is_file()
                            and (run_dir_p / "files.latest.json").is_file()
                        )
                        or bool(result_p.get("micro_repair_session_id"))
                        or _two_session_evidence(run_dir_p)
                    )
                    if has_patch and pre_evidence:
                        rejected_patch = ""
                        if before_patch_path.is_file():
                            rejected_patch = before_patch_path.read_text(
                                encoding="utf-8", errors="replace"
                            )
                        elif (
                            (run_dir_p / "files.before.json").is_file()
                            and (run_dir_p / "files.before_repair.json").is_file()
                        ):
                            rejected_patch = _diff_before_after_files(
                                run_dir_p / "files.before.json",
                                run_dir_p / "files.before_repair.json",
                            )
                        failed_log = ""
                        if (run_dir_p / "test_before_repair.log").is_file():
                            failed_log = (run_dir_p / "test_before_repair.log").read_text(
                                encoding="utf-8", errors="replace"
                            )
                        records.append({
                            "task_id": entry.get("task_id"),
                            "prompt": entry["prompt"],
                            "chosen_patch": entry["patch"],
                            "rejected_patch": rejected_patch,
                            "rejected_reason": entry["reason"] or "micro_repair_triggered",
                            "failed_test_log": failed_log,
                            "chosen_model": entry["model"],
                            "rejected_model": entry["model"],
                            "chosen_run_id": entry["run_id"],
                            "rejected_run_id": entry["run_id"],
                            "pair_source": "intra_run_micro_repair",
                        })
                if is_rejected:
                    pending_rejected = entry
                    continue
                if is_chosen and pending_rejected is not None:
                    rejected, pending_rejected = pending_rejected, None
                    records.append({
                        "task_id": entry.get("task_id") or rejected.get("task_id"),
                        "prompt": entry["prompt"] or rejected["prompt"],
                        "chosen_patch": entry["patch"],
                        "rejected_patch": rejected["patch"],
                        "chosen_model": entry["model"],
                        "rejected_model": rejected["model"],
                        "rejected_reason": rejected["reason"],
                        "chosen_run_id": entry["run_id"],
                        "rejected_run_id": rejected["run_id"],
                        "pair_source": "cross_run_retry",
                    })
        dataset = records
    elif mode == "repair-sft":
        if True:
            run_records = list(_iter_run_records(state_dir))
            prior_failed_logs: dict[str, str] = {}
            for run_dir, result in run_records:
                summary = summarize_run(
                    result, _load_proxy_summary(run_dir / "requests.json")
                )
                status = str(summary.get("status") or "").lower()
                if summary.get("passed") is True or status in PASS_STATUSES:
                    continue
                task_id_f = summary.get("task_id")
                if task_id_f is None:
                    continue
                reason = str(
                    _first(
                        result,
                        ("reason", "blocked_reason", "error_reason", "failure_reason"),
                    )
                    or ""
                ).strip().lower()
                log_text = ""
                if (run_dir / "test.log").is_file():
                    log_text = (run_dir / "test.log").read_text(
                        encoding="utf-8", errors="replace"
                    )
                if log_text or reason == "validation_failed":
                    prior_failed_logs.setdefault(str(task_id_f), log_text)

            for run_dir, result in run_records:
                summary = summarize_run(
                    result, _load_proxy_summary(run_dir / "requests.json")
                )
                run_id = _first(result, ("run_id", "runId", "id")) or run_dir.name
                task_id = summary.get("task_id")
                status = str(summary.get("status") or "").lower()
                passed = summary.get("passed") is True or status in PASS_STATUSES
                if not passed:
                    continue
                steps = summary.get("steps", 0)
                if max_steps is not None and (steps <= 0 or steps > max_steps):
                    continue
                if clean_only and not _is_clean_sft_run(
                    result, _load_run_messages(run_dir)
                ):
                    continue
                repaired = (
                    result.get("micro_repair_attempted") is True
                    or result.get("repaired") is True
                    or (task_id is not None and str(task_id) in prior_failed_logs)
                )
                if not repaired:
                    continue
                patch_path = run_dir / "changes.patch"
                patch = (
                    patch_path.read_text(encoding="utf-8", errors="replace")
                    if patch_path.is_file()
                    else ""
                )
                if not patch.strip():
                    continue
                if (run_dir / "test_before_repair.log").is_file():
                    failed_test_log = (run_dir / "test_before_repair.log").read_text(
                        encoding="utf-8", errors="replace"
                    )
                else:
                    failed_test_log = (
                        prior_failed_logs.get(str(task_id), "")
                        if task_id is not None
                        else ""
                    )
                error_signature = (
                    _lesson_error_signature(failed_test_log)
                    if failed_test_log
                    else ""
                )
                files = _to_str_list(
                    _first(result, ("files", "allowed_files", "target_files"))
                )
                task_dir = _first(result, ("origin_directory", "task_dir", "directory"))
                prompt_value = _first(
                    result, ("prompt", "task_prompt", "instruction", "description")
                )
                if (not task_dir or not files or prompt_value is None) and (run_dir / "task.json").is_file():
                    try:
                        task_json = json.loads(
                            (run_dir / "task.json").read_text(
                                encoding="utf-8", errors="replace"
                            )
                        )
                    except (OSError, ValueError):
                        task_json = {}
                    if isinstance(task_json, dict):
                        if not task_dir:
                            task_dir = _first(task_json, ("origin_directory", "directory", "task_dir"))
                        if not files:
                            files = _to_str_list(_first(task_json, ("origin_files", "files", "allowed_files", "target_files")))
                        if prompt_value is None:
                            prompt_value = _first(task_json, ("prompt", "task_prompt", "instruction", "description"))
                if not files and isinstance(result.get("hashes_after"), dict):
                    files = [str(k) for k in result["hashes_after"].keys()]
                prompt = str(prompt_value) if prompt_value is not None else ""
                project_id, _ = resolve_project_identity(task_dir, files)
                records.append({
                    "task_id": task_id,
                    "run_id": run_id,
                    "project_id": project_id,
                    "model": summary.get("model"),
                    "prompt": prompt,
                    "failed_test_log": failed_test_log,
                    "error_signature": error_signature,
                    "repair_patch": patch,
                    "messages": _load_run_messages(run_dir),
                })
            dataset = records
        dataset = records

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(out_path.parent), delete=False
        ) as tmp:
            tmp_name = tmp.name
            for record in dataset:
                tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_name, out_path)
    return dataset


def backfill_salvaged_runs(state_dir: str | Path, *, dry_run: bool = False) -> list[dict]:
    """Retroactively salvage blocked runs whose test actually passed (H-57).

    Scans ``<state_dir>/runs/*/result.json`` and flips eligible runs from
    ``status="blocked"`` to ``status="validation_passed_review_required"``
    (adding ``salvaged_from="retroactive_<reason>"``) when all hold:

    1. the block reason is in :data:`SALVAGEABLE_BLOCK_REASONS`;
    2. ``test_return_code`` is strictly integer ``0``;
    3. ``hashes_before`` / ``hashes_after`` are non-empty dicts that differ;
    4. a non-empty ``changes.patch`` exists in the run directory.

    With ``dry_run=True`` nothing is written; matched records are only
    returned. Each record: ``run_id``, ``task_id``, ``old_reason``,
    ``salvaged_from``.
    """
    state_path = Path(state_dir)
    runs_dir = state_path / "runs"
    if not runs_dir.is_dir():
        return []

    run_dirs = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: (d.stat().st_mtime, d.name),
    )

    salvaged: list[dict] = []
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict):
            continue

        reason = str(result.get("reason") or "")
        if result.get("status") != "blocked":
            continue
        if reason not in SALVAGEABLE_BLOCK_REASONS:
            continue

        rc = result.get("test_return_code")
        if not (isinstance(rc, int) and not isinstance(rc, bool) and rc == 0):
            continue

        hashes_before = result.get("hashes_before")
        hashes_after = result.get("hashes_after")
        if not (
            isinstance(hashes_before, dict) and bool(hashes_before)
            and isinstance(hashes_after, dict) and bool(hashes_after)
        ):
            continue
        if hashes_after == hashes_before:
            continue

        patch_path = run_dir / "changes.patch"
        if not (patch_path.is_file() and patch_path.stat().st_size > 0):
            continue

        salvaged_from = f"retroactive_{reason}"
        salvaged.append({
            "run_id": run_dir.name,
            "task_id": result.get("task_id"),
            "old_reason": result.get("reason"),
            "salvaged_from": salvaged_from,
        })
        if dry_run:
            continue

        result["status"] = "validation_passed_review_required"
        result["salvaged_from"] = salvaged_from
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(run_dir), delete=False
        ) as tmp:
            tmp_name = tmp.name
            json.dump(result, tmp, ensure_ascii=False, indent=2)
        os.replace(tmp_name, result_path)

    return salvaged


def backfill_repair_lessons(state_dir: str | Path, *, dry_run: bool = False) -> list[dict]:
    """Retroactively backfill repair lessons from micro_repair runs (H-82).

    Scans ``<state_dir>/runs/*/result.json`` in chronological order and, for
    every successful run that was repaired (``repaired`` flag or a
    ``micro_repair`` ``salvaged_from`` marker), extracts the failed-test log,
    derives an error signature, and records a repair lesson (deduplicated by
    ``(task_dir, error_signature)``) unless ``dry_run=True``.

    Returns one record per processed run: ``run_id``, ``task_id``,
    ``task_dir``, ``error_signature``, ``files``.
    """
    from harness.sandbox import (
        _extract_patch_fix_summary,
        _lesson_error_signature,
        record_repair_lesson,
        resolve_project_identity,
    )

    state_path = Path(state_dir)
    runs_dir = state_path / "runs"
    if not runs_dir.is_dir():
        return []

    run_dirs = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: (d.stat().st_mtime, d.name),
    )

    lessons_path = state_path / "lessons.jsonl"
    seen: set[tuple[str, str]] = set()
    if lessons_path.is_file():
        try:
            for line in lessons_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    rec_project, _ = resolve_project_identity(
                        str(record.get("task_dir") or ""),
                        [str(f) for f in (record.get("files") or [])],
                    )
                    seen.add((
                        str(record.get("project_id") or "") or rec_project,
                        str(record.get("error_signature") or ""),
                    ))
        except OSError:
            pass

    results: list[dict] = []
    prior_failed_logs: dict[str, str] = {}
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict):
            continue

        if result.get("status") != "validation_passed_review_required":
            log_text = ""
            if (run_dir / "test.log").is_file():
                try:
                    log_text = (run_dir / "test.log").read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    log_text = ""
                if _lesson_error_signature(log_text) != "test_failure":
                    prior_failed_logs[str(result.get("task_id") or "")] = log_text
            continue

        task_id = str(result.get("task_id") or "")
        eligible = (
            result.get("repaired") is True
            or result.get("micro_repair_attempted") is True
            or str(result.get("salvaged_from") or "").startswith("micro_repair")
            or task_id in prior_failed_logs
        )
        if not eligible:
            continue

        hashes_before = result.get("hashes_before") or {}
        hashes_after = result.get("hashes_after") or {}
        if not isinstance(hashes_before, dict):
            hashes_before = {}
        if not isinstance(hashes_after, dict):
            hashes_after = {}

        repaired_files = [k for k in hashes_after if hashes_before.get(k) != hashes_after.get(k)]
        if not repaired_files:
            repaired_files = [k for k in hashes_before]

        task_dir = str(result.get("task_dir") or result.get("directory") or "")
        if not task_dir:
            manifest_path = run_dir / "escalate.manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict):
                    task_dir = str(manifest.get("task_dir") or manifest.get("directory") or "")
            except (OSError, ValueError):
                pass
        if not task_dir:
            try:
                task = json.loads((run_dir / "task.json").read_text(encoding="utf-8"))
                if isinstance(task, dict):
                    task_dir = str(task.get("task_dir") or task.get("directory") or "")
            except (OSError, ValueError):
                pass
        if not task_dir and hashes_before:
            try:
                cp = os.path.commonpath(list(hashes_before.keys()))
                task_dir = cp if os.path.isdir(cp) else str(Path(cp).parent)
            except ValueError:
                pass

        if (run_dir / "test_before_repair.log").is_file():
            try:
                failed_log = (run_dir / "test_before_repair.log").read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                failed_log = ""
        else:
            failed_log = ""
            if (run_dir / "test.log").is_file():
                try:
                    failed_log = (run_dir / "test.log").read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    failed_log = ""
            if task_id in prior_failed_logs:
                # The run's own test.log is post-repair output (often a plain
                # "N passed" summary); the earlier failed attempt's log carries
                # the actionable error signature.
                failed_log = prior_failed_logs[task_id]

        sig = _lesson_error_signature(failed_log)
        if not sig or sig == "test_failure":
            # Passing pytest output (e.g. "312 passed in 48s" or
            # "225 deselected in 0.04s") carries no actionable error
            # signature; skip it instead of polluting lessons.
            continue
        if not (run_dir / "test_before_repair.log").is_file():
            if task_id not in prior_failed_logs and not any(
                marker in failed_log
                for marker in ("E   ", "Traceback", "FAILED", "ERROR")
            ):
                continue
        project_id, subprojects = resolve_project_identity(task_dir, repaired_files)
        if (project_id, sig) in seen:
            continue

        rel_files: list[str] = []
        if task_dir:
            task_path = Path(task_dir)
            for f in repaired_files:
                try:
                    rel_files.append(str(Path(f).relative_to(task_path)))
                except ValueError:
                    rel_files.append(str(f))
        else:
            rel_files = [str(f) for f in repaired_files]

        seen.add((project_id, sig))
        results.append({
            "run_id": run_dir.name,
            "task_id": result.get("task_id"),
            "task_dir": task_dir,
            "project_id": project_id,
            "subprojects": sorted(subprojects),
            "error_signature": sig,
            "files": rel_files,
        })

        if dry_run:
            continue

        record_repair_lesson(
            state_path,
            {
                "task_id": result.get("task_id", ""),
                "task_dir": task_dir,
                "model": result.get("model"),
            },
            failed_log,
            repaired_files,
            patch_text=(
                (run_dir / "changes.patch").read_text(encoding="utf-8", errors="replace")
                if (run_dir / "changes.patch").is_file() else ""
            ),
        )

    return results
