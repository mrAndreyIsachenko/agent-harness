"""Benchmark run summarization and Markdown report rendering.

The helpers in this module are deliberately tolerant of raw runner JSON:
field lookups accept the common key spellings produced by the guard runner
and the guard proxy, and ``proxy_summary`` fills any gaps that the run
record itself does not carry.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile
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


def scan_state_runs(state_dir: str | Path) -> list[dict]:
    """Scan a live state directory for guard-runner run records.

    Walks ``<state_dir>/runs/<run>/result.json`` directories oldest-first by
    run-directory mtime, loading each run's optional sibling
    ``requests.json`` guard-proxy telemetry as ``proxy_summary``. A missing
    ``runs/`` directory yields ``[]``; run directories without a readable
    ``result.json`` JSON object are skipped.
    """
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


def format_state_stats_markdown(state_dir: str | Path) -> str:
    """Render a Markdown stats table for runs in a live state directory.

    Runs are grouped by model (an unknown model shows as ``unknown``) with a
    trailing ``ALL`` summary row covering every run.
    """
    runs = scan_state_runs(state_dir)
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


def _iter_run_records(state_dir: str | Path):
    """Yield ``(run_dir, result_dict)`` for readable state runs, oldest-first by mtime."""
    runs_dir = Path(state_dir) / "runs"
    if not runs_dir.is_dir():
        return
    run_dirs: list[tuple[float, Path]] = []
    for child in runs_dir.iterdir():
        try:
            if not child.is_dir() or not (child / "result.json").is_file():
                continue
            run_dirs.append((child.stat().st_mtime, child))
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


def compute_state_insights(state_dir: str | Path) -> dict:
    """Compute task-funnel, blocked-reason and tool-error insights from a state directory."""
    task_order: list[str] = []
    task_runs: dict[str, list[dict]] = {}

    blocked_counter: Counter[str] = Counter()
    tool_error_counter: Counter[tuple[str, str]] = Counter()
    reads_before_edit_samples: list[int] = []

    for run_dir, result in _iter_run_records(state_dir):
        task_id = str(result.get("task_id") or run_dir.name)
        if task_id not in task_runs:
            task_order.append(task_id)
            task_runs[task_id] = []
        task_runs[task_id].append(result)

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

    funnel = {
        "total_tasks": len(task_order),
        "resolved_tasks": 0,
        "first_try_ok": 0,
        "retry_or_escalation_ok": 0,
        "micro_repair_ok": 0,
        "salvaged_ok": 0,
        "unresolved_tasks": 0,
    }
    for task_id in task_order:
        runs = task_runs[task_id]
        first_resolved_idx = next(
            (idx for idx, run in enumerate(runs) if run.get("status") in _RESOLVED_STATUSES),
            None,
        )
        if first_resolved_idx is None:
            funnel["unresolved_tasks"] += 1
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

    return {
        "task_funnel": funnel,
        "top_blocked_reasons": blocked_counter.most_common(10),
        "top_tool_errors": tool_error_counter.most_common(10),
        "avg_reads_before_first_edit": float(avg_reads),
    }


def format_state_insights_markdown(state_dir: str | Path) -> str:
    """Render the state-directory insights report as Markdown."""
    insights = compute_state_insights(state_dir)
    funnel = insights["task_funnel"]
    lines = [
        "## Task Resolution Funnel",
        "",
        f"- Total tasks: {funnel['total_tasks']}",
        f"- Resolved: {funnel['resolved_tasks']}",
        f"  - First try: {funnel['first_try_ok']}",
        f"  - Retry/escalation: {funnel['retry_or_escalation_ok']}",
        f"  - Micro repair: {funnel['micro_repair_ok']}",
        f"  - Salvaged: {funnel['salvaged_ok']}",
        f"- Unresolved: {funnel['unresolved_tasks']}",
        "",
        "## Top Blocked Reasons",
        "",
    ]
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
        ]
    )
    return "\n".join(lines)


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
    if directory is not None:
        target = str(Path(directory).resolve())
        for run_dir, result in records:
            try:
                task_path = run_dir / "task.json"
                if not task_path.is_file():
                    continue
                with task_path.open("r", encoding="utf-8") as handle:
                    task = json.load(handle)
            except (OSError, ValueError):
                continue
            if isinstance(task, dict) and task.get("directory") == target:
                selected.append((run_dir, result))
        if len(selected) < min_occurrences:
            selected = records
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


def _is_clean_sft_run(result: dict, messages: list[dict]) -> bool:
    """Return False for runs tainted by micro-repair, salvage, or guard hints."""
    if not isinstance(result, dict):
        result = {}
    if result.get("micro_repair_attempted") or result.get("salvaged_from"):
        return False
    for msg in messages or []:
        if isinstance(msg, str):
            content = msg
        elif isinstance(msg, dict) and isinstance(msg.get("content"), str):
            content = msg["content"]
        else:
            try:
                content = json.dumps(msg, ensure_ascii=False)
            except (TypeError, ValueError):
                content = str(msg)
        if any(marker in content for marker in _GUARD_INTERVENTION_MARKERS):
            return False
    return True


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
    if mode not in ("sft", "dpo"):
        raise ValueError(f"Unsupported mode: {mode!r}; expected 'sft' or 'dpo'")

    records: list[dict] = []
    if mode == "sft":
        for run_dir, result in _iter_run_records(state_dir):
            summary = summarize_run(result, _load_proxy_summary(run_dir / "requests.json"))
            run_id = _first(result, ("run_id", "runId", "id")) or run_dir.name
            task_id = summary.get("task_id")
            status = str(summary.get("status") or "").lower()
            files = _to_str_list(_first(result, ("files", "allowed_files", "target_files")))
            prompt_value = _first(result, ("prompt", "task_prompt", "instruction", "description"))
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
    else:  # mode == "dpo"
        groups: dict[object, list[dict]] = {}
        group_order: list[object] = []
        for run_dir, result in _iter_run_records(state_dir):
            summary = summarize_run(result, _load_proxy_summary(run_dir / "requests.json"))
            status = str(summary.get("status") or "").lower()
            files = _to_str_list(_first(result, ("files", "allowed_files", "target_files")))
            prompt_value = _first(result, ("prompt", "task_prompt", "instruction", "description"))
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
                    })
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
